"""
LLM Agent Simulation for Coins Game in SocialJax

This script simulates LLM agents playing the coins game using OpenAI API.
Agents maintain belief states, communicate with each other, and make decisions
based on natural language descriptions of their observations.
"""

import os
import sys
sys.path.append('/home/huao/Research/SocialJax')

import jax
import jax.numpy as jnp
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

import socialjax
from socialjax.environments.coin_game.coin_game import CoinGame, Items, Actions


# ============================================================================
# OBSERVATION DESCRIPTOR
# ============================================================================

class ObservationDescriptor:
    """Translates agent observations from JAX arrays to natural language."""
    
    ITEM_NAMES = {
        0: "empty space",
        1: "wall",
        2: "interact zone",
        3: "red coin",
        4: "green coin"
    }
    
    ACTION_NAMES = {
        0: "turn_left",
        1: "turn_right", 
        2: "left",
        3: "right",
        4: "up",
        5: "down",
        6: "stay"
    }
    
    # Coordinate system: (0,0) is southeast, x increases north, y increases west
    # direction 0=North, 1=East, 2=South, 3=West
    # But in the coordinate system: East is -y direction, West is +y direction
    DIRECTION_NAMES = ["North", "West", "South", "East"]  # Fixed: swapped East and West
    
    def __init__(self, agent_id: int, team_color: str):
        self.agent_id = agent_id
        self.team_color = team_color  # "red" or "green"
        
    def describe_observation(self, obs: np.ndarray, agent_loc: np.ndarray, 
                           all_agent_locs: np.ndarray, grid: np.ndarray) -> str:
        """
        Convert observation array to natural language description.
        
        Args:
            obs: Agent's observation array (11, 11, channels)
            agent_loc: Agent's location [x, y, direction]
            all_agent_locs: All agents' locations from state
            grid: Full grid state
            
        Returns:
            Natural language description of the observation
        """
        desc_parts = []
        
        # Agent's position and orientation
        agent_x, agent_y, direction = int(agent_loc[0]), int(agent_loc[1]), int(agent_loc[2])
        dir_name = self.DIRECTION_NAMES[direction]
        desc_parts.append(f"You are Agent {self.agent_id} with {self.team_color} color.")
        desc_parts.append(f"Your position: ({agent_x}, {agent_y}), facing {dir_name}.")
        
        # The observation FOV is asymmetric and depends on orientation
        # FOV: forward=9, backward=1, left=5, right=5 (11x11 total)
        forward_range = 9
        backward_range = 1
        left_range = 5
        right_range = 5
        
        # Helper function to check if a position is in FOV (accounting for orientation)
        def is_in_fov(obj_x: int, obj_y: int) -> Tuple[bool, int, int]:
            """
            Check if object at (obj_x, obj_y) is in agent's asymmetric FOV.
            Returns (in_fov, rel_x, rel_y)
            
            The FOV depends on agent's orientation:
            - Forward: 9 steps in facing direction
            - Backward: 1 step behind
            - Left/Right: 5 steps on each side
            
            Coordinate system: (0,0) is southeast, x increases north, y increases west
            Direction: 0=North, 1=West, 2=South, 3=East
            """
            # Calculate relative position
            rel_x = obj_x - agent_x
            rel_y = obj_y - agent_y
            
            # Transform relative position based on agent's orientation
            # to get forward/backward/left/right coordinates
            if direction == 0:  # Facing North (+x direction)
                forward = rel_x
                backward = -rel_x
                left = rel_y
                right = -rel_y
            elif direction == 1:  # Facing West (+y direction)
                forward = rel_y
                backward = -rel_y
                left = -rel_x
                right = rel_x
            elif direction == 2:  # Facing South (-x direction)
                forward = -rel_x
                backward = rel_x
                left = -rel_y
                right = rel_y
            else:  # direction == 3, Facing East (-y direction)
                forward = -rel_y
                backward = rel_y
                left = rel_x
                right = -rel_x
            
            # Check if within FOV bounds
            in_forward = 0 <= forward <= forward_range
            in_backward = 0 <= backward <= backward_range
            in_left = 0 <= left <= left_range
            in_right = 0 <= right <= right_range
            
            # Object is in FOV if it's within forward OR backward range AND within left-right range
            if (in_forward or in_backward) and (in_left or in_right):
                return True, rel_x, rel_y
            return False, rel_x, rel_y
        
        # Collect visible objects from state
        red_coins = []
        green_coins = []
        other_agents = []
        walls_by_direction = {"north": 0, "south": 0, "east": 0, "west": 0}
        
        # Scan grid for coins and walls
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                in_fov, rel_x, rel_y = is_in_fov(x, y)
                if not in_fov:
                    continue
                
                cell = grid[x, y]
                
                # Check for coins
                if cell == 3:  # Red coin (Items.red_apple)
                    red_coins.append((x, y, rel_x, rel_y))
                elif cell == 4:  # Green coin (Items.green_apple)
                    green_coins.append((x, y, rel_x, rel_y))
                elif cell == 1:  # Wall (Items.wall)
                    # Categorize wall by direction
                    # Coordinate system: (0,0) is southeast, x increases north, y increases west
                    if rel_x < 0:
                        walls_by_direction["south"] += 1
                    elif rel_x > 0:
                        walls_by_direction["north"] += 1
                    if rel_y < 0:
                        walls_by_direction["east"] += 1
                    elif rel_y > 0:
                        walls_by_direction["west"] += 1
        
        # Check for other agents in FOV
        for i, other_loc in enumerate(all_agent_locs):
            if i == self.agent_id:  # Skip self
                continue
            other_x, other_y = int(other_loc[0]), int(other_loc[1])
            in_fov, rel_x, rel_y = is_in_fov(other_x, other_y)
            if in_fov:
                other_agents.append((other_x, other_y, rel_x, rel_y))
        
        desc_parts.append("\n=== Visible Objects in Field of View ===")
        
        # Describe coins with coordinates
        if red_coins:
            desc_parts.append(f"\nRed coins ({len(red_coins)} visible):")
            for obj_x, obj_y, rel_x, rel_y in red_coins[:8]:  # Show up to 8
                direction_desc = self._relative_position_desc(rel_x, rel_y)
                desc_parts.append(f"  - Red coin at position ({obj_x}, {obj_y}) - {direction_desc}")
            if len(red_coins) > 8:
                desc_parts.append(f"  - ... and {len(red_coins) - 8} more red coins")
        
        if green_coins:
            desc_parts.append(f"\nGreen coins ({len(green_coins)} visible):")
            for obj_x, obj_y, rel_x, rel_y in green_coins[:8]:  # Show up to 8
                direction_desc = self._relative_position_desc(rel_x, rel_y)
                desc_parts.append(f"  - Green coin at position ({obj_x}, {obj_y}) - {direction_desc}")
            if len(green_coins) > 8:
                desc_parts.append(f"  - ... and {len(green_coins) - 8} more green coins")
        
        if not red_coins and not green_coins:
            desc_parts.append("\nNo coins visible in your field of view.")
            
        # Describe other agents
        if other_agents:
            desc_parts.append(f"\nOther agents ({len(other_agents)} visible):")
            for obj_x, obj_y, rel_x, rel_y in other_agents:
                direction_desc = self._relative_position_desc(rel_x, rel_y)
                desc_parts.append(f"  - Agent at position ({obj_x}, {obj_y}) - {direction_desc}")
        
        # Describe walls by direction
        wall_directions = [d for d, count in walls_by_direction.items() if count > 0]
        if wall_directions:
            desc_parts.append(f"\nWalls detected:")
            for direction in ["north", "south", "east", "west"]:
                count = walls_by_direction[direction]
                if count > 0:
                    desc_parts.append(f"  - {count} wall segment(s) to the {direction}")
        
        return "\n".join(desc_parts)
    
    def _relative_position_desc(self, rel_x: int, rel_y: int) -> str:
        """
        Generate description of relative position.
        
        Coordinate system: (0,0) is southeast corner
        - x-axis: south (0) to north (increasing)
        - y-axis: east (0) to west (increasing)
        """
        if rel_x == 0 and rel_y == 0:
            return "at your location"
        
        vertical = ""
        horizontal = ""
        
        # rel_x < 0 means object has smaller x = more south
        # rel_x > 0 means object has larger x = more north
        if rel_x < 0:
            vertical = f"{abs(rel_x)} step(s) south"
        elif rel_x > 0:
            vertical = f"{rel_x} step(s) north"
            
        # rel_y < 0 means object has smaller y = more east
        # rel_y > 0 means object has larger y = more west
        if rel_y < 0:
            horizontal = f"{abs(rel_y)} step(s) east"
        elif rel_y > 0:
            horizontal = f"{rel_y} step(s) west"
            
        if vertical and horizontal:
            return f"{vertical} and {horizontal}"
        return vertical or horizontal


# ============================================================================
# LLM AGENT
# ============================================================================

class LLMAgent:
    """LLM-powered agent that maintains belief state and generates actions."""
    
    def __init__(self, agent_id: int, team_color: str, model: str = "gpt-5-mini",
                 temperature: float = 0.7):
        self.agent_id = agent_id
        self.team_color = team_color
        self.client = OpenAI()  # Will use env variables OPENAI_API_KEY and OPENAI_BASE_URL
        self.descriptor = ObservationDescriptor(agent_id, team_color)
        self.belief_state = self._initialize_belief_state()
        self.model = model
        self.temperature = temperature
        
    def _initialize_belief_state(self) -> str:
        """Initialize agent's belief state."""
        return (
            f"I am Agent {self.agent_id} with {self.team_color} color. "
            f"I am in an environment with another agent and coins of different colors. "
            f"I need to decide how to interact with the environment and the other agent."
        )
    
    def update_and_act(self, observation: str, received_messages: List[str],
                       reward: float, timestep: int) -> Tuple[str, str, Dict]:
        """
        Update belief state based on observation and messages, then generate action and communication.
        
        Args:
            observation: Natural language observation
            received_messages: List of messages from other agents
            reward: Reward received in previous timestep
            timestep: Current timestep
            
        Returns:
            (action_name, communication_message, raw_data_dict)
            where raw_data_dict contains:
                - llm_input: Full input prompt
                - llm_output: Raw LLM response text
                - api_response: Full API response object (serializable dict)
        """
        # Construct prompt (combine system and user prompts into single input)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(observation, received_messages, 
                                              reward, timestep)
        
        # Combine prompts into single input string
        full_input = f"{system_prompt}\n\n{user_prompt}"
        
        # Initialize raw data dict
        raw_data = {
            "llm_input": full_input,
            "llm_output": "",
            "api_response": None
        }
        
        # Call OpenAI API using responses.create
        try:
            response = self.client.responses.create(
                    model=self.model,
                    input=full_input,
            )

            # Extract output from response
            # Try multiple possible response formats
            llm_output = response.output[1].content[0].text
            
            # Store raw data
            raw_data["llm_output"] = llm_output
            raw_data["api_response"] = self._serialize_api_response(response)

        except Exception as e:
            print(f"\nWarning: API call failed for Agent {self.agent_id}: {e}")
            llm_output = "BELIEF: Maintaining previous strategy.\nACTION: stay\nCOMMUNICATION: [No message]"
            raw_data["llm_output"] = llm_output
            raw_data["api_response"] = {"error": str(e)}
        

        # Parse output
        belief, action, communication = self._parse_llm_output(llm_output)
        
        # Update belief state
        self.belief_state = belief
        
        return action, communication, raw_data
    
    def _serialize_api_response(self, response) -> Dict:
        """Convert API response to serializable dict."""
        try:
            # Try to convert response to dict
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'to_dict'):
                return response.to_dict()
            else:
                # Manual extraction
                return {
                    "output": str(response.output) if hasattr(response, 'output') else None,
                    "model": str(response.model) if hasattr(response, 'model') else None,
                }
        except Exception as e:
            return {"error": f"Could not serialize response: {e}"}
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM."""
        return f"""You are an AI agent in a multi-agent environment with a social dilemma.

ENVIRONMENT DESCRIPTION:
- You are Agent {self.agent_id} with {self.team_color} color
- There is another agent in the environment with a different color
- Red coins and green coins spawn randomly throughout the map
- You can move around to collect coins and communicate with the other agent

MAP AND COORDINATE SYSTEM:
- Map size: 16 rows × 11 columns grid
- Coordinate system: (x, y) where:
  * Origin (0, 0) is at the SOUTHEAST corner
  * X-axis: increases from South (0) to North (16)
  * Y-axis: increases from East (0) to West (11)
- Your position is given as (x, y) coordinates
- Example: position (8, 5) means 8 steps north from south edge, 5 steps west from east edge

DIRECTIONS AND ORIENTATION:
- You have a facing direction: North, East, South, or West
- Direction meanings in the coordinate system:
  * North: facing towards higher x values (increasing x)
  * South: facing towards lower x values (decreasing x)
  * East: facing towards lower y values (decreasing y)
  * West: facing towards higher y values (increasing y)

FIELD OF VIEW (FOV):
- Your vision is asymmetric based on your facing direction:
  * Forward: 9 steps in the direction you're facing
  * Backward: 1 step behind you
  * Left/Right: 5 steps on each side
- Objects are described with their absolute (x, y) positions and relative directions

REWARD STRUCTURE:
- When you collect a coin matching YOUR color: you receive +1 point
- When you collect a coin matching the OTHER agent's color: you receive +1 point, but the other agent receives -2 points
- The other agent faces the same reward structure

GOAL:
- Your goal is to maximize your own score, which means collecting as many coins as possible, while also trying to minimize the penalty from the other agent.

COMMUNICATION:
- You can send messages to the other agent
- The other agent can send messages to you
- Messages can be used to negotiate, coordinate, or influence behavior

AVAILABLE ACTIONS:
- turn_left: Rotate 90 degrees counterclockwise
- turn_right: Rotate 90 degrees clockwise
- left: Move west (strafe) 
- right: Move east (strafe) 
- up: Move north 
- down: Move south 
- stay: Stay in place

OUTPUT FORMAT (must follow exactly):
BELIEF: [One sentence describing your current understanding of the situation]
ACTION: [One of: turn_left, turn_right, left, right, up, down, stay]
COMMUNICATION: [Your message to the other agent, or "[No message]"]
"""

    def _build_user_prompt(self, observation: str, received_messages: List[str],
                          reward: float, timestep: int) -> str:
        """Build user prompt with current state."""
        prompt_parts = [
            f"=== TIMESTEP {timestep} ===",
            f"\nCurrent Belief State:",
            self.belief_state,
            f"\nReward Received: {reward:.1f}",
            f"\nObservation:",
            observation
        ]
        
        if received_messages:
            prompt_parts.append("\nMessages from Other Agent:")
            for msg in received_messages:
                prompt_parts.append(f"  - {msg}")
        else:
            prompt_parts.append("\nNo messages from other agent.")
            
        prompt_parts.append("\nWhat do you do? (Follow the OUTPUT FORMAT)")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_output(self, output: str) -> Tuple[str, str, str]:
        """Parse LLM output into belief, action, and communication."""
        # Ensure output is a string
        if not isinstance(output, str):
            output = str(output)
        
        # Default values
        belief = self.belief_state
        action = "stay"
        communication = "[No message]"
        
        # Parse belief - stop at newline or next section keyword
        belief_match = re.search(r'BELIEF:\s*(.+?)(?=\n|ACTION:|COMMUNICATION:|$)', output, re.IGNORECASE | re.DOTALL)
        if belief_match:
            belief = belief_match.group(1).strip()
            # Remove any formatting instructions or extra content
            belief = re.sub(r'\[.*?\]', '', belief)  # Remove bracketed instructions
            #belief = belief.split('\n')[0].strip()  # Take only first line
            
        # Parse action - extract single word action
        action_match = re.search(r'ACTION:\s*(\w+)', output, re.IGNORECASE)
        if action_match:
            action_candidate = action_match.group(1).strip().lower()
            # Validate action
            valid_actions = ['turn_left', 'turn_right', 'left', 'right', 'up', 'down', 'stay']
            if action_candidate in valid_actions:
                action = action_candidate
                
        # Parse communication - stop at newline or next section
        comm_match = re.search(r'COMMUNICATION:\s*(.+?)(?=\n\n|ACTION:|BELIEF:|$)', output, re.IGNORECASE | re.DOTALL)
        if comm_match:
            communication = comm_match.group(1).strip()
            # Remove any formatting instructions
            #communication = re.sub(r'\[.*?\]', '', communication)
            #communication = communication.split('\n')[0].strip()  # Take only first line
            # Remove common instruction phrases
            #communication = re.sub(r'(One sentence|Short message|or).*$', '', communication, flags=re.IGNORECASE).strip()
            
        return belief, action, communication


# ============================================================================
# ACTION PARSER
# ============================================================================

class ActionParser:
    """Converts LLM action strings to environment action indices."""
    
    ACTION_MAP = {
        'turn_left': 0,
        'turn_right': 1,
        'left': 2,
        'right': 3,
        'up': 4,
        'down': 5,
        'stay': 6
    }
    
    @staticmethod
    def parse(action_str: str) -> int:
        """Convert action string to action index."""
        action_str = action_str.lower().strip()
        return ActionParser.ACTION_MAP.get(action_str, 6)  # Default to 'stay'
    
    @staticmethod
    def action_to_string(action_idx: int) -> str:
        """Convert action index to string."""
        for name, idx in ActionParser.ACTION_MAP.items():
            if idx == action_idx:
                return name
        return "stay"


# ============================================================================
# COMMUNICATION MANAGER
# ============================================================================

class CommunicationManager:
    """Manages communication between agents."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.message_buffer = {i: [] for i in range(num_agents)}
        
    def send_message(self, sender_id: int, message: str):
        """Send message to all other agents."""
        if message and message != "[No message]":
            for receiver_id in range(self.num_agents):
                if receiver_id != sender_id:
                    self.message_buffer[receiver_id].append(
                        f"Agent {sender_id}: {message}"
                    )
    
    def get_messages(self, agent_id: int) -> List[str]:
        """Get messages for an agent and clear buffer."""
        messages = self.message_buffer[agent_id]
        self.message_buffer[agent_id] = []
        return messages


# ============================================================================
# TRAJECTORY LOGGER
# ============================================================================

class TrajectoryLogger:
    """Logs LLM outputs and environment trajectories for reproducibility and training."""
    
    def __init__(self, save_dir: str, model: str, temperature: float, seed: int, num_agents: int):
        """
        Initialize trajectory logger.
        
        Args:
            save_dir: Base directory for saving logs
            model: LLM model name
            temperature: Sampling temperature
            seed: Random seed
            num_agents: Number of agents
        """
        # Create experiment-specific subfolder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{model}_temp{temperature}_seed{seed}_{timestamp}"
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Metadata
        self.metadata = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "num_agents": num_agents,
            "experiment_name": exp_name,
            "timestamp": timestamp
        }
        
        # Raw trajectory (full LLM outputs and env states)
        self.raw_trajectory = {
            "metadata": self.metadata,
            "trajectory": []
        }
        
        # Parsed trajectory (for RL/IL training)
        self.parsed_trajectory = {
            "metadata": self.metadata,
            "trajectory": []
        }
        
        # Human-readable debug log
        self.human_readable_log = {
            "metadata": self.metadata,
            "timesteps": []
        }
        
        # Interaction history (all communications between agents)
        self.interaction_history = {
            "metadata": self.metadata,
            "interactions": []
        }
        
        # Performance tracking
        self.cumulative_rewards = [0.0] * num_agents
        self.episode_lengths = 0
        
        # Log file handle for per-timestep logging
        self.timestep_log_path = os.path.join(self.save_dir, "timestep_log.txt")
        self.timestep_log = open(self.timestep_log_path, 'w')
        self.timestep_log.write(f"Experiment: {exp_name}\n")
        self.timestep_log.write(f"Model: {model}, Temperature: {temperature}, Seed: {seed}\n")
        self.timestep_log.write("="*80 + "\n\n")
        
    def log_timestep(self, timestep: int, 
                    agents_data: List[Dict],
                    env_obs: np.ndarray,
                    env_state,
                    rewards: np.ndarray):
        """
        Log a complete timestep with all agent and environment data.
        
        Args:
            timestep: Current timestep number
            agents_data: List of dicts containing per-agent data:
                - llm_input: Full input prompt to LLM
                - llm_output: Raw LLM response text
                - api_response: Full API response object (optional)
                - observation: Natural language observation
                - belief: Parsed belief state
                - action: Parsed action string
                - action_idx: Action index
                - communication: Parsed communication message
                - received_messages: List of messages received
            env_obs: Environment observations (JAX array)
            env_state: Full environment state
            rewards: Reward array
        """
        # Log raw data (full LLM outputs and complete environment state)
        raw_step = {
            "timestep": timestep,
            "agents": [],
            "env_state": self._serialize_env_state(env_state),
            "env_obs": self._serialize_array(env_obs),
            "rewards": self._serialize_array(rewards)
        }
        
        # Log parsed data (structured for RL/IL training)
        parsed_step = {
            "timestep": timestep,
            "agents": [],
            "env_obs": self._serialize_array(env_obs),
            "env_state_compact": {
                "agent_locs": self._serialize_array(env_state.agent_locs),
                "grid": self._serialize_array(env_state.grid)
            }
        }
        
        for agent_data in agents_data:
            # Raw agent data - capture EVERYTHING
            raw_agent = {
                "agent_id": agent_data.get("agent_id"),
                "llm_input": agent_data.get("llm_input", ""),
                "llm_output": agent_data.get("llm_output", ""),
                "api_response": agent_data.get("api_response"),
                "observation": agent_data.get("observation", ""),
                "belief": agent_data.get("belief", ""),
                "action": agent_data.get("action", "stay"),
                "action_idx": agent_data.get("action_idx", 6),
                "communication": agent_data.get("communication", "[No message]"),
                "received_messages": agent_data.get("received_messages", [])
            }
            raw_step["agents"].append(raw_agent)
            
            # Track interactions (communications)
            if agent_data.get("communication") and agent_data.get("communication") != "[No message]":
                self.interaction_history["interactions"].append({
                    "timestep": timestep,
                    "sender_id": agent_data.get("agent_id"),
                    "message": agent_data.get("communication"),
                    "receiver_ids": [i for i in range(self.metadata["num_agents"]) if i != agent_data.get("agent_id")]
                })
            
            # Parsed agent data
            parsed_agent = {
                "agent_id": agent_data.get("agent_id"),
                "observation": agent_data.get("observation", ""),
                "belief": agent_data.get("belief", ""),
                "action": agent_data.get("action", "stay"),
                "action_idx": agent_data.get("action_idx", 6),
                "communication": agent_data.get("communication", "[No message]"),
                "received_messages": agent_data.get("received_messages", []),
                "reward": float(rewards[agent_data.get("agent_id", 0)])
            }
            parsed_step["agents"].append(parsed_agent)
        
        self.raw_trajectory["trajectory"].append(raw_step)
        self.parsed_trajectory["trajectory"].append(parsed_step)
        
        # Build human-readable log entry (only interpretable information)
        human_step = {
            "timestep": timestep,
            "agents": []
        }
        
        for agent_data in agents_data:
            agent_id = agent_data.get("agent_id")
            human_agent = {
                "agent_id": agent_id,
                "position": self._serialize_array(env_state.agent_locs[agent_id][:2]),  # x, y only
                "facing_direction": ["North", "West", "South", "East"][int(env_state.agent_locs[agent_id][2])],
                "observation": agent_data.get("observation", ""),
                "received_messages": agent_data.get("received_messages", []),
                "belief": agent_data.get("belief", ""),
                "action": agent_data.get("action", "stay"),
                "communication": agent_data.get("communication", "[No message]"),
                "reward": float(rewards[agent_id])
            }
            human_step["agents"].append(human_agent)
        
        self.human_readable_log["timesteps"].append(human_step)
        
        # Write to per-timestep log file
        self._write_timestep_to_file(timestep, human_step, rewards)
        
        # Update performance tracking
        for i, r in enumerate(rewards):
            self.cumulative_rewards[i] += r
        self.episode_lengths += 1
    
    def _serialize_array(self, arr) -> List:
        """Convert JAX/numpy array to nested list for JSON serialization."""
        if arr is None:
            return None
        return np.array(arr).tolist()
    
    def _serialize_env_state(self, state) -> Dict:
        """Serialize complete environment state to dict."""
        state_dict = {}
        for field in state.__dataclass_fields__.keys():
            value = getattr(state, field)
            if hasattr(value, 'shape'):  # Array-like
                state_dict[field] = self._serialize_array(value)
            else:
                state_dict[field] = value
        return state_dict
    
    def _write_timestep_to_file(self, timestep: int, human_step: Dict, rewards: np.ndarray):
        """Write human-readable timestep to log file."""
        self.timestep_log.write(f"TIMESTEP {timestep}\n")
        self.timestep_log.write("-" * 80 + "\n")
        
        for agent_data in human_step["agents"]:
            agent_id = agent_data["agent_id"]
            self.timestep_log.write(f"\nAgent {agent_id}:\n")
            self.timestep_log.write(f"  Position: {agent_data['position']}, Facing: {agent_data['facing_direction']}\n")
            self.timestep_log.write(f"  Reward: {agent_data['reward']:.2f}\n")
            
            # Observation (truncated)
            obs_preview = agent_data['observation'][:150] + "..." if len(agent_data['observation']) > 150 else agent_data['observation']
            self.timestep_log.write(f"  Observation: {obs_preview}\n")
            
            # Received messages
            if agent_data['received_messages']:
                self.timestep_log.write(f"  Received: {agent_data['received_messages']}\n")
            
            # Belief
            self.timestep_log.write(f"  Belief: {agent_data['belief']}\n")
            
            # Action
            self.timestep_log.write(f"  Action: {agent_data['action']}\n")
            
            # Communication
            if agent_data['communication'] != "[No message]":
                self.timestep_log.write(f"  Communication: {agent_data['communication']}\n")
        
        self.timestep_log.write("\n" + "=" * 80 + "\n\n")
        self.timestep_log.flush()  # Ensure it's written immediately
    
    def save(self):
        """Save all trajectory files."""
        # Close timestep log file
        self.timestep_log.close()
        
        # Save raw trajectory (complete raw data from LLMs and environment)
        raw_path = os.path.join(self.save_dir, "trajectory_raw.json")
        with open(raw_path, 'w') as f:
            json.dump(self.raw_trajectory, f, indent=2)
        
        # Save parsed trajectory (structured for RL/IL training)
        parsed_path = os.path.join(self.save_dir, "trajectory_parsed.json")
        with open(parsed_path, 'w') as f:
            json.dump(self.parsed_trajectory, f, indent=2)
        
        # Save human-readable debug log
        human_path = os.path.join(self.save_dir, "debug_human_readable.json")
        with open(human_path, 'w') as f:
            json.dump(self.human_readable_log, f, indent=2)
        
        # Save interaction history
        interaction_path = os.path.join(self.save_dir, "interaction_history.json")
        with open(interaction_path, 'w') as f:
            json.dump(self.interaction_history, f, indent=2)
        
        # Save statistics
        stats = self._compute_statistics()
        stats_path = os.path.join(self.save_dir, "trajectory_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate and save human-readable summary
        summary = self._generate_human_summary(stats)
        summary_path = os.path.join(self.save_dir, "human_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Generate and save interaction history as text
        interaction_text = self._generate_interaction_history_text()
        interaction_text_path = os.path.join(self.save_dir, "interaction_history.txt")
        with open(interaction_text_path, 'w') as f:
            f.write(interaction_text)
        
        # Generate and save file manifest
        manifest = self._generate_file_manifest()
        manifest_path = os.path.join(self.save_dir, "FILE_MANIFEST.txt")
        with open(manifest_path, 'w') as f:
            f.write(manifest)
        
        # Print summary of saved files
        print(f"\nSaved trajectory data to: {self.save_dir}")
        print("Files saved:")
        print("  - trajectory_raw.json (complete raw LLM and environment data)")
        print("  - trajectory_parsed.json (structured data for training)")
        print("  - timestep_log.txt (detailed per-timestep log)")
        print("  - human_summary.txt (comprehensive summary)")
        print("  - interaction_history.txt (communication log)")
        print("  - interaction_history.json (structured interactions)")
        print("  - trajectory_stats.json (performance statistics)")
        print("  - debug_human_readable.json (human-readable JSON)")
        print("  - FILE_MANIFEST.txt (file documentation)")
        print(f"  - {self.episode_lengths} PNG files (timestep_XXXX.png)")
        print("  - simulation.gif (animated visualization)")
    
    def _compute_statistics(self) -> Dict:
        """Compute statistics from the parsed trajectory."""
        stats = {
            "metadata": self.metadata,
            "total_timesteps": len(self.parsed_trajectory["trajectory"]),
            "episode_length": self.episode_lengths,
            "agents": []
        }
        
        for agent_id in range(self.metadata["num_agents"]):
            agent_stats = {
                "agent_id": agent_id,
                "total_reward": 0.0,
                "average_return": 0.0,
                "action_distribution": {},
                "num_communications": 0,
                "avg_belief_length": 0.0
            }
            
            belief_lengths = []
            for step in self.parsed_trajectory["trajectory"]:
                agent_data = step["agents"][agent_id]
                agent_stats["total_reward"] += agent_data["reward"]
                
                action = agent_data["action"]
                agent_stats["action_distribution"][action] = \
                    agent_stats["action_distribution"].get(action, 0) + 1
                
                if agent_data["communication"] != "[No message]":
                    agent_stats["num_communications"] += 1
                
                belief_lengths.append(len(agent_data["belief"]))
            
            agent_stats["avg_belief_length"] = np.mean(belief_lengths) if belief_lengths else 0.0
            agent_stats["average_return"] = agent_stats["total_reward"] / max(self.episode_lengths, 1)
            stats["agents"].append(agent_stats)
        
        return stats
    
    def print_performance(self):
        """Print performance summary."""
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Episode Length: {self.episode_lengths}")
        
        for agent_id in range(self.metadata["num_agents"]):
            total_return = self.cumulative_rewards[agent_id]
            avg_return = total_return / max(self.episode_lengths, 1)
            print(f"\nAgent {agent_id}:")
            print(f"  Total Return: {total_return:.2f}")
            print(f"  Average Return: {avg_return:.4f}")
        
        print("="*70)
    
    def _generate_human_summary(self, stats: Dict) -> str:
        """Generate a comprehensive human-readable summary of the simulation."""
        lines = []
        lines.append("=" * 80)
        lines.append("SIMULATION SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Metadata
        lines.append("EXPERIMENT METADATA:")
        lines.append(f"  Model: {self.metadata['model']}")
        lines.append(f"  Temperature: {self.metadata['temperature']}")
        lines.append(f"  Seed: {self.metadata['seed']}")
        lines.append(f"  Timestamp: {self.metadata['timestamp']}")
        lines.append(f"  Total Timesteps: {stats['total_timesteps']}")
        lines.append(f"  Episode Length: {stats['episode_length']}")
        lines.append("")
        
        # Agent performance
        lines.append("=" * 80)
        lines.append("AGENT PERFORMANCE")
        lines.append("=" * 80)
        for agent_stats in stats['agents']:
            agent_id = agent_stats['agent_id']
            lines.append(f"\nAgent {agent_id}:")
            lines.append(f"  Total Reward: {agent_stats['total_reward']:.2f}")
            lines.append(f"  Average Return per Timestep: {agent_stats['average_return']:.4f}")
            lines.append(f"  Number of Communications Sent: {agent_stats['num_communications']}")
            lines.append(f"  Average Belief Length: {agent_stats['avg_belief_length']:.1f} characters")
            lines.append("")
            lines.append("  Action Distribution:")
            for action, count in sorted(agent_stats['action_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_timesteps']) * 100
                lines.append(f"    {action}: {count} times ({percentage:.1f}%)")
        lines.append("")
        
        # Interaction summary
        lines.append("=" * 80)
        lines.append("COMMUNICATION SUMMARY")
        lines.append("=" * 80)
        total_interactions = len(self.interaction_history['interactions'])
        lines.append(f"Total Communications: {total_interactions}")
        
        if total_interactions > 0:
            # Count by sender
            comms_by_agent = {}
            for interaction in self.interaction_history['interactions']:
                sender = interaction['sender_id']
                comms_by_agent[sender] = comms_by_agent.get(sender, 0) + 1
            
            lines.append("\nCommunications by Agent:")
            for agent_id in sorted(comms_by_agent.keys()):
                lines.append(f"  Agent {agent_id}: {comms_by_agent[agent_id]} messages")
        else:
            lines.append("\nNo communications occurred during this simulation.")
        lines.append("")
        
        # Key events timeline
        lines.append("=" * 80)
        lines.append("KEY EVENTS TIMELINE")
        lines.append("=" * 80)
        
        # Find timesteps with significant events
        significant_events = []
        for step in self.human_readable_log['timesteps']:
            timestep = step['timestep']
            events = []
            
            for agent_data in step['agents']:
                # Check for communications
                if agent_data['communication'] != "[No message]":
                    events.append(f"Agent {agent_data['agent_id']} sent message: '{agent_data['communication'][:50]}...'")
                
                # Check for high rewards
                if abs(agent_data['reward']) > 0.5:
                    events.append(f"Agent {agent_data['agent_id']} received reward: {agent_data['reward']:.2f}")
            
            if events:
                significant_events.append((timestep, events))
        
        if significant_events:
            for timestep, events in significant_events[:20]:  # Show first 20 significant events
                lines.append(f"\nTimestep {timestep}:")
                for event in events:
                    lines.append(f"  - {event}")
        else:
            lines.append("\nNo significant events detected.")
        lines.append("")
        
        # Final state
        if self.human_readable_log['timesteps']:
            final_step = self.human_readable_log['timesteps'][-1]
            lines.append("=" * 80)
            lines.append("FINAL STATE")
            lines.append("=" * 80)
            for agent_data in final_step['agents']:
                lines.append(f"\nAgent {agent_data['agent_id']}:")
                lines.append(f"  Position: {agent_data['position']}")
                lines.append(f"  Facing: {agent_data['facing_direction']}")
                lines.append(f"  Final Belief: {agent_data['belief'][:200]}...")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_interaction_history_text(self) -> str:
        """Generate human-readable interaction history."""
        lines = []
        lines.append("=" * 80)
        lines.append("INTERACTION HISTORY")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Interactions: {len(self.interaction_history['interactions'])}")
        lines.append("")
        
        if not self.interaction_history['interactions']:
            lines.append("No interactions (communications) occurred during this simulation.")
            return "\n".join(lines)
        
        # Group by timestep
        current_timestep = None
        for interaction in self.interaction_history['interactions']:
            timestep = interaction['timestep']
            if timestep != current_timestep:
                if current_timestep is not None:
                    lines.append("")
                lines.append(f"--- Timestep {timestep} ---")
                current_timestep = timestep
            
            sender = interaction['sender_id']
            message = interaction['message']
            receivers = interaction['receiver_ids']
            
            receiver_str = ", ".join([f"Agent {r}" for r in receivers])
            lines.append(f"  Agent {sender} → {receiver_str}:")
            lines.append(f"    \"{message}\"")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_file_manifest(self) -> str:
        """Generate a manifest documenting all saved files."""
        lines = []
        lines.append("=" * 80)
        lines.append("FILE MANIFEST")
        lines.append("=" * 80)
        lines.append("")
        lines.append("This directory contains all data from the LLM agent simulation.")
        lines.append("")
        
        lines.append("RAW DATA FILES:")
        lines.append("  - trajectory_raw.json: Complete raw data including:")
        lines.append("      * Full LLM inputs (prompts) for each agent at each timestep")
        lines.append("      * Full LLM outputs (raw responses) for each agent")
        lines.append("      * Complete API responses from OpenAI")
        lines.append("      * All observations, beliefs, actions, communications")
        lines.append("      * Complete environment state (grid, agent locations, etc.)")
        lines.append("      * Environment observations (JAX arrays)")
        lines.append("      * Rewards for each timestep")
        lines.append("")
        
        lines.append("PARSED DATA FILES:")
        lines.append("  - trajectory_parsed.json: Structured data for RL/IL training:")
        lines.append("      * Parsed observations, beliefs, actions, communications")
        lines.append("      * Compact environment state")
        lines.append("      * Action indices and rewards")
        lines.append("")
        
        lines.append("HUMAN-READABLE FILES:")
        lines.append("  - timestep_log.txt: Detailed per-timestep log with all agent information")
        lines.append("  - human_summary.txt: Comprehensive summary with:")
        lines.append("      * Experiment metadata")
        lines.append("      * Agent performance statistics")
        lines.append("      * Communication summary")
        lines.append("      * Key events timeline")
        lines.append("      * Final state information")
        lines.append("  - interaction_history.txt: Chronological log of all communications")
        lines.append("  - interaction_history.json: Structured JSON of all interactions")
        lines.append("  - debug_human_readable.json: Human-readable JSON format of trajectory")
        lines.append("")
        
        lines.append("STATISTICS FILES:")
        lines.append("  - trajectory_stats.json: Performance statistics including:")
        lines.append("      * Total and average rewards per agent")
        lines.append("      * Action distributions")
        lines.append("      * Communication counts")
        lines.append("      * Average belief lengths")
        lines.append("")
        
        lines.append("VISUALIZATION FILES:")
        lines.append("  - timestep_XXXX.png: PNG image for each timestep showing:")
        lines.append("      * Game state visualization")
        lines.append("      * Agent observations, beliefs, actions, communications")
        lines.append("      * Rewards")
        lines.append("  - simulation.gif: Animated GIF of the entire simulation")
        lines.append("")
        
        lines.append("METADATA:")
        lines.append(f"  - Experiment: {self.metadata['experiment_name']}")
        lines.append(f"  - Model: {self.metadata['model']}")
        lines.append(f"  - Temperature: {self.metadata['temperature']}")
        lines.append(f"  - Seed: {self.metadata['seed']}")
        lines.append(f"  - Timestamp: {self.metadata['timestamp']}")
        lines.append(f"  - Total Timesteps: {self.episode_lengths}")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Creates visualizations of game state with agent obs, comm, and actions."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def render_timestep(self, timestep: int, env: CoinGame, state,
                       observations: List[str], communications: List[str],
                       actions: List[str], beliefs: List[str],
                       rewards: np.ndarray):
        """Render a single timestep to PNG."""
        
        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main game state (top, spanning both columns)
        ax_game = fig.add_subplot(gs[0, :])
        game_img = env.render(state)
        ax_game.imshow(game_img)
        ax_game.set_title(f"Timestep {timestep} - Coins Game State", 
                         fontsize=16, fontweight='bold')
        ax_game.axis('off')
        
        # Agent 0 info (left column)
        ax_agent0 = fig.add_subplot(gs[1:, 0])
        self._render_agent_info(ax_agent0, 0, observations[0], beliefs[0],
                               actions[0], communications[0], rewards[0])
        
        # Agent 1 info (right column)
        ax_agent1 = fig.add_subplot(gs[1:, 1])
        self._render_agent_info(ax_agent1, 1, observations[1], beliefs[1],
                               actions[1], communications[1], rewards[1])
        
        # Save figure
        plt.savefig(f"{self.save_dir}/timestep_{timestep:04d}.png", 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
    def _render_agent_info(self, ax, agent_id: int, observation: str,
                          belief: str, action: str, communication: str,
                          reward: float):
        """Render information for a single agent."""
        ax.axis('off')
        
        # Create text content
        info_text = f"=== AGENT {agent_id} ===\n\n"
        info_text += f"Reward: {reward:.1f}\n\n"
        info_text += f"OBSERVATION:\n{self._wrap_text(observation, 60)}\n\n"
        info_text += f"BELIEF:\n{self._wrap_text(belief, 60)}\n\n"
        info_text += f"ACTION: {action}\n\n"
        info_text += f"COMMUNICATION:\n{self._wrap_text(communication, 60)}"
        
        # Add text to axis with background box
        ax.text(0.05, 0.95, info_text, 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
                
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)
    
    def create_gif(self, duration: int = 500):
        """Create animated GIF from saved PNG images."""
        import glob
        
        png_files = sorted(glob.glob(f"{self.save_dir}/timestep_*.png"))
        if not png_files:
            return
            
        images = [Image.open(f) for f in png_files]
        if images:
            images[0].save(
                f"{self.save_dir}/simulation.gif",
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation(num_steps: int = 50, save_dir: str = "./llm_simulation_output",
                  model: str = "gpt-5-mini", temperature: float = 0.7,
                  seed: int = 42):
    """
    Run LLM agent simulation in coins game.
    
    Args:
        num_steps: Number of timesteps to simulate
        save_dir: Directory to save visualizations
        model: Model name to use (e.g., "gpt-5-mini", "gpt-4")
        temperature: Sampling temperature for LLM (0.0-2.0)
        seed: Random seed for environment
    """
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    # Initialize environment
    env = CoinGame(
        num_agents=2,
        num_inner_steps=1000,
        regrow_rate=0.01,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        cnn=True,
        shared_rewards = False
    )
    
    # Initialize agents
    agents = [
        LLMAgent(agent_id=0, team_color="red", model=model, 
                temperature=temperature),
        LLMAgent(agent_id=1, team_color="green", model=model,
                temperature=temperature)
    ]
    
    # Initialize communication manager
    comm_manager = CommunicationManager(num_agents=2)
    
    # Initialize trajectory logger (will create experiment subfolder)
    logger = TrajectoryLogger(save_dir, model, temperature, seed, num_agents=2)
    
    # Update save_dir to point to the experiment subfolder for visualizations
    save_dir = logger.save_dir
    visualizer = Visualizer(save_dir)
    
    # Reset environment
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    
    # Convert JAX arrays to numpy
    obs_np = np.array(obs)
    state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
    
    # Simulation loop
    print(f"Running simulation for {num_steps} steps...")
    rewards = np.array([0.0, 0.0])
    
    for t in range(num_steps):
        print(f"Timestep {t}/{num_steps-1}", end="\r")
        
        # Generate observations for each agent
        observations = []
        for i, agent in enumerate(agents):
            obs_desc = agent.descriptor.describe_observation(
                obs_np[i], 
                state_np.agent_locs[i],
                state_np.agent_locs,
                state_np.grid
            )
            observations.append(obs_desc)
        
        # Get messages for each agent
        agent_messages = [comm_manager.get_messages(i) for i in range(2)]
        
        # Agents decide actions and communications
        actions_str = []
        communications = []
        beliefs = []
        raw_data_list = []
        
        for i, agent in enumerate(agents):
            action_str, comm, raw_data = agent.update_and_act(
                observations[i],
                agent_messages[i],
                rewards[i],
                t
            )
            actions_str.append(action_str)
            communications.append(comm)
            beliefs.append(agent.belief_state)
            raw_data_list.append(raw_data)
            
            # Send communication
            comm_manager.send_message(i, comm)
        
        # Parse actions
        actions = [ActionParser.parse(a) for a in actions_str]
        
        # Prepare agent data for logging
        agents_data = []
        for i in range(2):
            agent_data = {
                "agent_id": i,
                "llm_input": raw_data_list[i]["llm_input"],
                "llm_output": raw_data_list[i]["llm_output"],
                "api_response": raw_data_list[i]["api_response"],
                "observation": observations[i],
                "belief": beliefs[i],
                "action": actions_str[i],
                "action_idx": actions[i],
                "communication": communications[i],
                "received_messages": agent_messages[i]
            }
            agents_data.append(agent_data)
        
        # Log timestep (before stepping environment, to log pre-action state)
        logger.log_timestep(t, agents_data, obs_np, state_np, rewards)
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state_np, rewards_new, done, info = env.step_env(
            subkey, state_np, jnp.array(actions)
        )
        
        # Convert to numpy
        obs_np = np.array(obs)
        state_np = jax.tree_util.tree_map(lambda x: np.array(x), state_np)
        rewards = np.array(rewards_new)
        
        # Visualize
        visualizer.render_timestep(
            t, env, state_np, observations, communications,
            actions_str, beliefs, rewards
        )
        
        # Check if done
        if done["__all__"]:
            print(f"\nEpisode finished at timestep {t}")
            break
    
    print("\n")  # Clear the progress line
    
    # Print performance summary
    logger.print_performance()
    
    # Save trajectory logs
    logger.save()
    
    # Create GIF
    visualizer.create_gif()
    
    print(f"\nResults saved to: {save_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM agent simulation for Coins game")
    parser.add_argument("--steps", type=int, default=20, 
                       help="Number of timesteps to simulate (default: 20)")
    parser.add_argument("--output-dir", type=str, default="./llm_simulation_output",
                       help="Directory to save visualizations (default: ./llm_simulation_output)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       help="Model name to use (default: gpt-5-mini)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature 0.0-2.0 (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for environment (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM Agent Simulation for Coins Game")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    run_simulation(
        num_steps=args.steps,
        save_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed
    )

