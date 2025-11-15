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
import time
from datetime import datetime

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
# EMBEDDING UTILITIES
# ============================================================================

def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-large", dimensions: int = 64) -> List[float]:
    """Get embedding vector for text using OpenAI API."""
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Warning: Embedding failed: {e}")
        return [0.0] * dimensions


def find_closest_coin_in_fov(agent_x: int, agent_y: int, agent_direction: int, 
                              grid: np.ndarray, agent_color: str) -> Optional[str]:
    """
    Find the closest coin in the agent's field of view.
    
    Returns:
        "red", "green", or None if no coin in FOV
    """
    # FOV parameters
    forward_range = 9
    backward_range = 1
    left_range = 5
    right_range = 5
    
    def is_in_fov(obj_x: int, obj_y: int) -> Tuple[bool, float]:
        """Check if object is in FOV and return distance."""
        rel_x = obj_x - agent_x
        rel_y = obj_y - agent_y
        
        # Transform based on direction
        if agent_direction == 0:  # North
            forward = rel_x
            backward = -rel_x
            left = rel_y
            right = -rel_y
        elif agent_direction == 1:  # West
            forward = rel_y
            backward = -rel_y
            left = -rel_x
            right = rel_x
        elif agent_direction == 2:  # South
            forward = -rel_x
            backward = rel_x
            left = -rel_y
            right = rel_y
        else:  # East
            forward = -rel_y
            backward = rel_y
            left = rel_x
            right = -rel_x
        
        in_forward = 0 <= forward <= forward_range
        in_backward = 0 <= backward <= backward_range
        in_left = 0 <= left <= left_range
        in_right = 0 <= right <= right_range
        
        if (in_forward or in_backward) and (in_left or in_right):
            distance = np.sqrt(rel_x**2 + rel_y**2)
            return True, distance
        return False, float('inf')
    
    closest_coin = None
    closest_distance = float('inf')
    
    # Scan grid for coins
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            cell = grid[x, y]
            if cell == 3:  # Red coin
                in_fov, dist = is_in_fov(x, y)
                if in_fov and dist < closest_distance:
                    closest_distance = dist
                    closest_coin = "red"
            elif cell == 4:  # Green coin
                in_fov, dist = is_in_fov(x, y)
                if in_fov and dist < closest_distance:
                    closest_distance = dist
                    closest_coin = "green"
    
    return closest_coin


def construct_semantic_key(agent_color: str, agent_x: int, agent_y: int, 
                           closest_coin_color: Optional[str], agent_id: int, 
                           action: str) -> Tuple:
    """
    Construct semantic representation key: (agent_color, agent_x, agent_y, 
    closest_coin_color_in_fov, agent, action).
    
    No normalization - stores original values.
    """
    return (agent_color, int(agent_x), int(agent_y), 
            closest_coin_color if closest_coin_color else "none", 
            int(agent_id), action)


# ============================================================================
# LLM AGENT
# ============================================================================

class LLMAgent:
    """LLM-powered agent that maintains belief state and generates actions."""
    
    def __init__(self, agent_id: int, team_color: str, model: str = "gpt-5-mini",
                 temperature: float = 0.7, reasoning: str = None):
        self.agent_id = agent_id
        self.team_color = team_color
        self.model = model
        self.temperature = temperature
        self.reasoning = reasoning  # For GPT-5.1: "none", "low", "medium", "high"
        
        # Use different endpoint and key for GPT-5.1
        if self.model == "gpt-5.1":
            gpt51_url = os.getenv("GPT_51_URL")
            gpt51_key = os.getenv("GPT_51_KEY")
            if gpt51_url and gpt51_key:
                self.client = OpenAI(
                    api_key=gpt51_key,
                    base_url=gpt51_url
                )
            else:
                raise ValueError(
                    "GPT-5.1 requires GPT_51_URL and GPT_51_KEY environment variables. "
                    f"GPT_51_URL={'set' if gpt51_url else 'not set'}, "
                    f"GPT_51_KEY={'set' if gpt51_key else 'not set'}"
                )
        else:
            # Default client uses OPENAI_API_KEY and OPENAI_BASE_URL
            self.client = OpenAI()
        
        self.descriptor = ObservationDescriptor(agent_id, team_color)
        self.belief_state = self._initialize_belief_state()
        
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
        start_time = time.time()
        try:
            # Build API call parameters
            # Format: reasoning={ "effort": "low" | "medium" | "high" }
            # Note: "none" means no reasoning parameter (default behavior)
            api_params = {
                "model": self.model,
                "input": full_input,
            }
            
            # Add reasoning parameter for GPT-5.1
            # Only pass reasoning parameter for "low", "medium", or "high"
            # "none" or None means use default (no reasoning parameter)
            # Example: reasoning={ "effort": "low" }
            if self.model == "gpt-5.1" and self.reasoning is not None and self.reasoning != "none":
                if self.reasoning in ["low", "medium", "high"]:
                    api_params["reasoning"] = {"effort": self.reasoning}
            
            response = self.client.responses.create(**api_params)

            # Extract output from response
            # Try multiple possible response formats
            llm_output = response.output[1].content[0].text
            
            # Store raw data
            raw_data["llm_output"] = llm_output
            raw_data["api_response"] = self._serialize_api_response(response)
            
            # Extract token usage if available
            elapsed_time = time.time() - start_time
            raw_data["api_time"] = elapsed_time
            raw_data["token_usage"] = self._extract_token_usage(response)

        except Exception as e:
            print(f"\nWarning: API call failed for Agent {self.agent_id}: {e}")
            llm_output = "BELIEF: Maintaining previous strategy.\nACTION: stay\nCOMMUNICATION: [No message]"
            raw_data["llm_output"] = llm_output
            raw_data["api_response"] = {"error": str(e)}
            raw_data["api_time"] = time.time() - start_time
            raw_data["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        

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
    
    def _extract_token_usage(self, response) -> Dict:
        """Extract token usage from API response."""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'model_dump'):
                    return usage.model_dump()
                elif hasattr(usage, 'to_dict'):
                    return usage.to_dict()
                else:
                    return {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage, 'completion_tokens', 0),
                        "total_tokens": getattr(usage, 'total_tokens', 0)
                    }
            else:
                return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e:
            return {"error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
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
BELIEF: [One sentence describing your current understanding of the situation including current positon ,next goal, and general game strategy.]
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
    """Simplified logger that stores all data in a single JSON file."""
    
    def __init__(self, save_dir: str, model: str, temperature: float, seed: int, num_agents: int, embedding_client: OpenAI):
        """
        Initialize trajectory logger.
        
        Args:
            save_dir: Base directory for saving logs
            model: LLM model name
            temperature: Sampling temperature
            seed: Random seed
            num_agents: Number of agents
            embedding_client: OpenAI client for embeddings
        """
        # Create experiment-specific subfolder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{model}_temp{temperature}_seed{seed}_{timestamp}"
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create subfolder for state visualizations
        self.state_vis_dir = os.path.join(self.save_dir, "state_visualizations")
        os.makedirs(self.state_vis_dir, exist_ok=True)
        
        # Metadata
        self.metadata = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "num_agents": num_agents,
            "experiment_name": exp_name,
            "timestamp": timestamp
        }
        
        # Single trajectory data structure
        self.trajectory = {
            "metadata": self.metadata,
            "trajectory": []
        }
        
        # Performance tracking
        self.cumulative_rewards = [0.0] * num_agents
        self.episode_lengths = 0
        self.total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_api_time = 0.0
        
        # Embedding client (not used during collection, but kept for compatibility)
        self.embedding_client = embedding_client
        
        # Performance profiling
        self.timestep_times = []
        self.api_call_times = []
        self.observation_times = []
        self.logging_times = []
        
        # Paths for incremental saving
        self.temp_json_path = os.path.join(self.save_dir, "trajectory_temp.json")
        self.last_saved_timestep = -1
        
    def log_timestep(self, timestep: int, 
                    agents_data: List[Dict],
                    env_obs: np.ndarray,
                    env_state,
                    rewards: np.ndarray,
                    cumulative_own_eaten: np.ndarray = None,
                    cumulative_other_eaten: np.ndarray = None,
                    agent_colors: List[str] = None,
                    observation_state=None):
        """
        Log a complete timestep with all agent and environment data, including embeddings.
        
        Args:
            timestep: Current timestep number
            agents_data: List of dicts containing per-agent data
            env_obs: Environment observations (JAX array) - observations at timestep t
            env_state: Full environment state - state at t+1 (result of action at t)
            rewards: Reward array - rewards at t+1 (result of action at t)
            cumulative_own_eaten: Cumulative own-color coins eaten
            cumulative_other_eaten: Cumulative other-color coins eaten
            agent_colors: List of agent colors ["red", "green"]
            observation_state: State at timestep t (the state the observation describes)
        """
        grid = np.array(env_state.grid)
        red_coins = np.sum(grid == 3)
        green_coins = np.sum(grid == 4)
        
        # Use observation_state for semantic key construction (state at t), env_state for logging (state at t+1)
        state_for_semantic = observation_state if observation_state is not None else env_state
        
        step_data = {
            "timestep": timestep,
            "agents": [],
            "env_state": self._serialize_env_state(env_state),  # State at t+1
            "env_obs": self._serialize_array(env_obs),
            "rewards": self._serialize_array(rewards),  # Rewards at t+1
            "coins_in_env": {"red": int(red_coins), "green": int(green_coins)},
            "accumulated_rewards": [float(r) for r in self.cumulative_rewards],
            "cumulative_own_eaten": cumulative_own_eaten.tolist() if cumulative_own_eaten is not None else None,
            "cumulative_other_eaten": cumulative_other_eaten.tolist() if cumulative_other_eaten is not None else None
        }
        
        for agent_data in agents_data:
            agent_id = agent_data.get("agent_id")
            # Use observation_state for semantic key (state at t that observation describes)
            semantic_grid = np.array(state_for_semantic.grid)
            agent_x = int(state_for_semantic.agent_locs[agent_id][0])
            agent_y = int(state_for_semantic.agent_locs[agent_id][1])
            agent_direction = int(state_for_semantic.agent_locs[agent_id][2])
            agent_color = agent_colors[agent_id] if agent_colors else ("red" if agent_id == 0 else "green")
            
            # Find closest coin in FOV (using state at t)
            closest_coin = find_closest_coin_in_fov(agent_x, agent_y, agent_direction, semantic_grid, agent_color)
            
            # Construct semantic key
            semantic_key = construct_semantic_key(
                agent_color, agent_x, agent_y, closest_coin, agent_id, 
                agent_data.get("action", "stay")
            )
            
            # Save original text instead of embeddings (embeddings can be computed later)
            belief_text = agent_data.get("belief", "")
            comm_text = agent_data.get("communication", "[No message]")
            
            # Extract token usage and time
            token_usage = agent_data.get("token_usage", {})
            api_time = agent_data.get("api_time", 0.0)
            
            # Track API call times
            if api_time > 0:
                self.api_call_times.append(api_time)
            
            # Update totals
            if isinstance(token_usage, dict):
                self.total_token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.total_token_usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
                self.total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)
            self.total_api_time += api_time
            
            agent_entry = {
                "agent_id": agent_id,
                "agent_color": agent_color,
                "observation": agent_data.get("observation", ""),
                "belief": belief_text,  # Original text, no embedding
                "action": agent_data.get("action", "stay"),
                "action_idx": agent_data.get("action_idx", 6),
                "communication": comm_text,  # Original text, no embedding
                "received_messages": agent_data.get("received_messages", []),
                "reward": float(rewards[agent_id]),
                "semantic_key": semantic_key,
                "token_usage": token_usage,
                "api_time": api_time,
                "position": [agent_x, agent_y],
                "direction": agent_direction
            }
            step_data["agents"].append(agent_entry)
        
        self.trajectory["trajectory"].append(step_data)
        
        # Update performance tracking
        for i, r in enumerate(rewards):
            self.cumulative_rewards[i] += r
        self.episode_lengths += 1
        
        # Incremental save after each timestep
        self._save_incremental()
    
    def _save_incremental(self):
        """Save trajectory incrementally to temp file after each timestep."""
        try:
            # Add current metadata
            self.trajectory["metadata"]["total_timesteps"] = self.episode_lengths
            self.trajectory["metadata"]["total_token_usage"] = self.total_token_usage
            self.trajectory["metadata"]["total_api_time"] = self.total_api_time
            
            # Save to temp file (use compact format for speed)
            with open(self.temp_json_path, 'w') as f:
                json.dump(self.trajectory, f, separators=(',', ':'))  # Compact format, faster
            
            self.last_saved_timestep = self.episode_lengths - 1
        except Exception as e:
            print(f"Warning: Failed to save incremental data: {e}")
    
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
    
    def save(self):
        """Save trajectory data to single JSON file and summary text file."""
        # Add summary statistics to trajectory metadata
        self.trajectory["metadata"]["total_timesteps"] = self.episode_lengths
        self.trajectory["metadata"]["total_token_usage"] = self.total_token_usage
        self.trajectory["metadata"]["total_api_time"] = self.total_api_time
        
        # Add performance profiling data
        if self.timestep_times:
            self.trajectory["metadata"]["performance"] = {
                "avg_timestep_time": np.mean(self.timestep_times),
                "avg_api_time": np.mean(self.api_call_times) if self.api_call_times else 0.0,
                "avg_observation_time": np.mean(self.observation_times) if self.observation_times else 0.0,
                "avg_logging_time": np.mean(self.logging_times) if self.logging_times else 0.0,
                "total_timesteps": len(self.timestep_times)
            }
        
        # Save single JSON file (rename from temp if exists)
        json_path = os.path.join(self.save_dir, "trajectory.json")
        if os.path.exists(self.temp_json_path):
            import shutil
            shutil.move(self.temp_json_path, json_path)
        else:
            with open(json_path, 'w') as f:
                json.dump(self.trajectory, f, indent=2)
        
        # Generate and save summary text file
        summary = self._generate_summary()
        summary_path = os.path.join(self.save_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\nSaved trajectory data to: {self.save_dir}")
        print("Files saved:")
        print("  - trajectory.json (complete trajectory with text data)")
        print("  - summary.txt (aggregated statistics)")
        print(f"  - state_visualizations/ (state visualization images)")
        print("  - simulation.gif (animated visualization)")
        
        # Print performance summary with bottleneck identification
        if self.timestep_times:
            print("\nPerformance Summary:")
            avg_timestep = np.mean(self.timestep_times)
            print(f"  Average timestep time: {avg_timestep:.3f}s")
            
            if self.api_call_times:
                avg_api = np.mean(self.api_call_times)
                api_percentage = (sum(self.api_call_times) / sum(self.timestep_times)) * 100
                print(f"  Average API call time: {avg_api:.3f}s ({api_percentage:.1f}% of total)")
                print(f"    → API calls are the main bottleneck" if api_percentage > 50 else "")
            
            if self.observation_times:
                avg_obs = np.mean(self.observation_times)
                obs_percentage = (sum(self.observation_times) / sum(self.timestep_times)) * 100
                print(f"  Average observation generation time: {avg_obs:.3f}s ({obs_percentage:.1f}% of total)")
            
            if self.logging_times:
                avg_log = np.mean(self.logging_times)
                log_percentage = (sum(self.logging_times) / sum(self.timestep_times)) * 100
                print(f"  Average logging time: {avg_log:.3f}s ({log_percentage:.1f}% of total)")
            
            # Identify bottleneck
            percentages = {}
            if self.api_call_times:
                percentages["API calls"] = (sum(self.api_call_times) / sum(self.timestep_times)) * 100
            if self.observation_times:
                percentages["Observation generation"] = (sum(self.observation_times) / sum(self.timestep_times)) * 100
            if self.logging_times:
                percentages["Logging"] = (sum(self.logging_times) / sum(self.timestep_times)) * 100
            
            if percentages:
                bottleneck = max(percentages.items(), key=lambda x: x[1])
                print(f"\n  ⚠️  Performance Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}% of total time)")
    
    def _generate_summary(self) -> str:
        """Generate summary text file with aggregated statistics."""
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
        lines.append(f"  Total Timesteps: {self.episode_lengths}")
        lines.append("")
        
        # Token usage and time
        lines.append("=" * 80)
        lines.append("TOKEN USAGE AND TIME")
        lines.append("=" * 80)
        lines.append(f"  Total Prompt Tokens: {self.total_token_usage['prompt_tokens']}")
        lines.append(f"  Total Completion Tokens: {self.total_token_usage['completion_tokens']}")
        lines.append(f"  Total Tokens: {self.total_token_usage['total_tokens']}")
        lines.append(f"  Total API Time: {self.total_api_time:.2f} seconds")
        lines.append(f"  Average Time per Timestep: {self.total_api_time / max(self.episode_lengths, 1):.3f} seconds")
        lines.append("")
        
        # Performance profiling
        if self.timestep_times:
            lines.append("=" * 80)
            lines.append("PERFORMANCE PROFILING")
            lines.append("=" * 80)
            lines.append(f"  Average Timestep Time: {np.mean(self.timestep_times):.3f} seconds")
            if self.api_call_times:
                lines.append(f"  Average API Call Time: {np.mean(self.api_call_times):.3f} seconds")
                lines.append(f"  Total API Calls: {len(self.api_call_times)}")
                lines.append(f"  API Time Percentage: {(sum(self.api_call_times) / sum(self.timestep_times) * 100):.1f}%")
            if self.observation_times:
                lines.append(f"  Average Observation Generation Time: {np.mean(self.observation_times):.3f} seconds")
            if self.logging_times:
                lines.append(f"  Average Logging Time: {np.mean(self.logging_times):.3f} seconds")
            lines.append("")
        
        # Agent performance
        lines.append("=" * 80)
        lines.append("AGENT PERFORMANCE")
        lines.append("=" * 80)
        
        # Compute statistics from trajectory
        agent_stats = {i: {
            "total_reward": 0.0,
            "num_communications": 0,
            "action_distribution": {},
            "total_own_eaten": 0,
            "total_other_eaten": 0
        } for i in range(self.metadata["num_agents"])}
        
        for step in self.trajectory["trajectory"]:
            for agent_data in step["agents"]:
                agent_id = agent_data["agent_id"]
                agent_stats[agent_id]["total_reward"] += agent_data["reward"]
                
                if agent_data["communication"] != "[No message]":
                    agent_stats[agent_id]["num_communications"] += 1
                
                action = agent_data["action"]
                agent_stats[agent_id]["action_distribution"][action] = \
                    agent_stats[agent_id]["action_distribution"].get(action, 0) + 1
        
        # Get final cumulative eaten values
        if self.trajectory["trajectory"]:
            final_step = self.trajectory["trajectory"][-1]
            if final_step.get("cumulative_own_eaten"):
                for i, val in enumerate(final_step["cumulative_own_eaten"]):
                    agent_stats[i]["total_own_eaten"] = val
            if final_step.get("cumulative_other_eaten"):
                for i, val in enumerate(final_step["cumulative_other_eaten"]):
                    agent_stats[i]["total_other_eaten"] = val
        
        for agent_id in range(self.metadata["num_agents"]):
            stats = agent_stats[agent_id]
            lines.append(f"\nAgent {agent_id}:")
            lines.append(f"  Total Reward: {stats['total_reward']:.2f}")
            lines.append(f"  Average Return per Timestep: {stats['total_reward'] / max(self.episode_lengths, 1):.4f}")
            lines.append(f"  Number of Communications Sent: {stats['num_communications']}")
            lines.append(f"  Total Own-Color Coins Eaten: {stats['total_own_eaten']:.0f}")
            lines.append(f"  Total Other-Color Coins Eaten: {stats['total_other_eaten']:.0f}")
            lines.append("  Action Distribution:")
            for action, count in sorted(stats['action_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                percentage = (count / self.episode_lengths) * 100 if self.episode_lengths > 0 else 0
                lines.append(f"    {action}: {count} times ({percentage:.1f}%)")
        lines.append("")
        
        # Environment metrics
        lines.append("=" * 80)
        lines.append("ENVIRONMENT METRICS")
        lines.append("=" * 80)
        if self.trajectory["trajectory"]:
            # Average coins in environment
            total_red = sum(step["coins_in_env"]["red"] for step in self.trajectory["trajectory"])
            total_green = sum(step["coins_in_env"]["green"] for step in self.trajectory["trajectory"])
            avg_red = total_red / len(self.trajectory["trajectory"])
            avg_green = total_green / len(self.trajectory["trajectory"])
            lines.append(f"  Average Red Coins in Environment: {avg_red:.2f}")
            lines.append(f"  Average Green Coins in Environment: {avg_green:.2f}")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
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
                  model: str = "gpt-5.1", temperature: float = 0.7,
                  seed: int = 42, reasoning: str = "medium"):
    """
    Run LLM agent simulation in coins game.
    
    Args:
        num_steps: Number of timesteps to simulate
        save_dir: Directory to save visualizations
        model: Model name to use (e.g., "gpt-5.1", "gpt-5-mini", "o3")
        temperature: Sampling temperature for LLM (0.0-2.0)
        seed: Random seed for environment
        reasoning: Reasoning effort level for GPT-5.1 ("low", "medium", "high", or None for default)
    """
    
    # Check API keys based on model
    if model == "gpt-5.1":
        gpt51_url = os.getenv("GPT_51_URL")
        gpt51_key = os.getenv("GPT_51_KEY")
        if not gpt51_url or not gpt51_key:
            raise ValueError(
                "GPT-5.1 requires GPT_51_URL and GPT_51_KEY environment variables. "
                f"GPT_51_URL={'set' if gpt51_url else 'NOT SET'}, "
                f"GPT_51_KEY={'set' if gpt51_key else 'NOT SET'}"
            )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    # Initialize environment
    env = CoinGame(
        num_agents=2,
        num_inner_steps=1000,
        regrow_rate=0.0005,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        cnn=True,
        shared_rewards = False
    )
    
    # Initialize agents
    agents = [
        LLMAgent(agent_id=0, team_color="red", model=model, 
                temperature=temperature, reasoning=reasoning),
        LLMAgent(agent_id=1, team_color="green", model=model,
                temperature=temperature, reasoning=reasoning)
    ]
    
    # Initialize communication manager
    comm_manager = CommunicationManager(num_agents=2)
    
    # Initialize OpenAI client for embeddings
    embedding_client = OpenAI()
    
    # Create display name for logging (include reasoning if specified)
    if model == "gpt-5.1" and reasoning and reasoning != "none":
        model_display_name = f"{model}_reasoning-{reasoning}"
    else:
        model_display_name = model
    
    # Initialize trajectory logger (will create experiment subfolder)
    logger = TrajectoryLogger(save_dir, model_display_name, temperature, seed, num_agents=2, embedding_client=embedding_client)
    
    # Update save_dir to point to the experiment subfolder for visualizations
    save_dir = logger.save_dir
    visualizer = Visualizer(logger.state_vis_dir)  # Save visualizations in subfolder
    
    # Reset environment
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    
    # Convert JAX arrays to numpy
    obs_np = np.array(obs)
    state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
    
    # Simulation loop
    print(f"Running simulation for {num_steps} steps...")
    rewards = np.array([0.0, 0.0])
    
    # Track cumulative coin eating statistics
    cumulative_own_eaten = np.array([0.0, 0.0])
    cumulative_other_eaten = np.array([0.0, 0.0])
    
    for t in range(num_steps):
        timestep_start = time.time()
        print(f"Timestep {t}/{num_steps-1}", end="\r")
        
        # Generate observations for each agent
        obs_start = time.time()
        observations = []
        for i, agent in enumerate(agents):
            obs_desc = agent.descriptor.describe_observation(
                obs_np[i], 
                state_np.agent_locs[i],
                state_np.agent_locs,
                state_np.grid
            )
            observations.append(obs_desc)
        obs_time = time.time() - obs_start
        logger.observation_times.append(obs_time)
        
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
        
        # Step environment FIRST (action at t leads to state at t+1)
        key, subkey = jax.random.split(key)
        obs_new, state_new, rewards_new, done, info = env.step_env(
            subkey, state_np, jnp.array(actions)
        )
        
        # Convert to numpy
        obs_new_np = np.array(obs_new)
        state_new_np = jax.tree_util.tree_map(lambda x: np.array(x), state_new)
        rewards_new_np = np.array(rewards_new)
        
        # Update coin eating statistics
        # info["eat_own_coins"] contains coins of own color eaten in this step
        if "eat_own_coins" in info:
            own_eaten_this_step = np.array(info["eat_own_coins"])
            cumulative_own_eaten += own_eaten_this_step
        
        # Calculate other-color coins eaten from rewards
        # Reward structure: +1 for any coin, but causes -2 to other agent if it's their color
        # So if agent i gets +1 and agent j gets -2, agent i ate agent j's coin
        for i in range(2):
            j = 1 - i  # Other agent
            # If agent i got positive reward and agent j got negative reward, i ate j's coin
            if rewards_new_np[i] > 0 and rewards_new_np[j] < 0:
                cumulative_other_eaten[i] += 1
        
        # Prepare agent data for logging (observation at t describes state at t, action at t leads to state at t+1)
        agents_data = []
        agent_colors = ["red", "green"]
        for i in range(2):
            agent_data = {
                "agent_id": i,
                "llm_input": raw_data_list[i]["llm_input"],
                "llm_output": raw_data_list[i]["llm_output"],
                "api_response": raw_data_list[i]["api_response"],
                "observation": observations[i],  # Observation at t describing state at t
                "belief": beliefs[i],
                "action": actions_str[i],  # Action at t
                "action_idx": actions[i],
                "communication": communications[i],
                "received_messages": agent_messages[i],
                "token_usage": raw_data_list[i].get("token_usage", {}),
                "api_time": raw_data_list[i].get("api_time", 0.0)
            }
            agents_data.append(agent_data)
        
        # Log timestep AFTER stepping (observation at t, action at t, state at t+1, rewards at t+1)
        # Note: observation at t describes state at t, action at t leads to state at t+1
        # Pass state_np (state at t) for semantic key construction, state_new_np (state at t+1) for logging
        log_start = time.time()
        logger.log_timestep(t, agents_data, obs_np, state_new_np, rewards_new_np,
                           cumulative_own_eaten, cumulative_other_eaten, agent_colors, observation_state=state_np)
        log_time = time.time() - log_start
        logger.logging_times.append(log_time)
        
        # Visualize with new state
        visualizer.render_timestep(
            t, env, state_new_np, observations, communications,
            actions_str, beliefs, rewards_new_np
        )
        
        # Update state for next iteration
        obs_np = obs_new_np
        state_np = state_new_np
        rewards = rewards_new_np
        
        # Track total timestep time
        timestep_time = time.time() - timestep_start
        logger.timestep_times.append(timestep_time)
        
        # Check if done
        if done["__all__"]:
            print(f"\nEpisode finished at timestep {t}")
            break
    
    print("\n")  # Clear the progress line
    
    # Print performance summary
    logger.print_performance()
    
    # Save trajectory logs
    logger.save()
    
    # Create GIF in main save_dir
    visualizer.create_gif()
    # Move GIF to main save_dir
    import shutil
    gif_src = os.path.join(logger.state_vis_dir, "simulation.gif")
    gif_dst = os.path.join(logger.save_dir, "simulation.gif")
    if os.path.exists(gif_src):
        shutil.move(gif_src, gif_dst)
    
    print(f"\nResults saved to: {save_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM agent simulation for Coins game")
    parser.add_argument("--steps", type=int, default=1000, 
                       help="Number of timesteps to simulate (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="./llm_simulation_output",
                       help="Directory to save visualizations (default: ./llm_simulation_output)")
    parser.add_argument("--model", type=str, default="gpt-5.1",
                       help="Model name to use (default: gpt-5.1)")
    parser.add_argument("--reasoning", type=str, default="medium",
                       choices=["low", "medium", "high", "none"],
                       help="Reasoning effort level for GPT-5.1 (default: medium)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature 0.0-2.0 (default: 0.0)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for environment (default: None, will run 10 seeds)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of runs with different seeds (default: 10)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM Agent Simulation for Coins Game")
    print("=" * 70)
    print(f"Model: {args.model}")
    if args.model == "gpt-5.1":
        print(f"Reasoning: {args.reasoning}")
    print(f"Temperature: {args.temperature}")
    print(f"Steps per run: {args.steps}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Run multiple times with different seeds
    if args.seed is not None:
        # Single run with specified seed
        run_simulation(
            num_steps=args.steps,
            save_dir=args.output_dir,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            reasoning=args.reasoning if args.model == "gpt-5.1" else None
        )
    else:
        # Multiple runs with different seeds
        seeds = list(range(1, 1 + args.num_runs))
        for i, seed in enumerate(seeds):
            print(f"\n{'='*70}")
            print(f"Run {i+1}/{args.num_runs} with seed {seed}")
            print(f"{'='*70}")
            run_simulation(
                num_steps=args.steps,
                save_dir=args.output_dir,
                model=args.model,
                temperature=args.temperature,
                seed=seed,
                reasoning=args.reasoning if args.model == "gpt-5.1" else None
            )

