"""
LLM Agent Simulation for Coop Mining Game in SocialJax

This script simulates LLM agents playing the coop mining game using OpenAI API.
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
from socialjax.environments.coop_mining.coop_mining import CoopMining, Items, Actions


# ============================================================================
# OBSERVATION DESCRIPTOR
# ============================================================================

class ObservationDescriptor:
    """Translates agent observations from JAX arrays to natural language."""
    
    ITEM_NAMES = {
        0: "empty space",
        1: "wall",
        2: "ore_wait",
        3: "spawn_point",
        4: "iron ore",
        5: "gold ore",
        6: "gold ore (partially mined, flashing)"
    }
    
    ACTION_NAMES = {
        0: "turn_left",
        1: "turn_right", 
        2: "left",
        3: "right",
        4: "up",
        5: "down",
        6: "stay",
        7: "mine"
    }
    
    # Coordinate system: (0,0) is at top-left, row increases down, col increases right
    # direction 0=North, 1=East, 2=South, 3=West
    DIRECTION_NAMES = ["North", "East", "South", "West"]
    
    def __init__(self, agent_id: int, team_color: str):
        self.agent_id = agent_id
        self.team_color = team_color  # For compatibility, but not used in coop_mining
        
    def describe_observation(self, obs: np.ndarray, agent_loc: np.ndarray, 
                           all_agent_locs: np.ndarray, grid: np.ndarray,
                           received_messages: List[str] = None) -> str:
        """
        Convert observation array to natural language description.
        
        Args:
            obs: Agent's observation array (11, 11, channels)
            agent_loc: Agent's location [row, col, direction]
            all_agent_locs: All agents' locations from state
            grid: Full grid state
            received_messages: List of messages from other agents to append to observation
            
        Returns:
            Natural language description of the observation
        """
        desc_parts = []
        
        # Agent's position and orientation
        agent_row, agent_col, direction = int(agent_loc[0]), int(agent_loc[1]), int(agent_loc[2])
        desc_parts.append(f"You are Agent {self.agent_id}.")
        desc_parts.append(f"Your position: row {agent_row}, col {agent_col}.")
        
        # The observation FOV is asymmetric and depends on orientation
        # FOV: forward=9, backward=1, left=5, right=5 (11x11 total)
        forward_range = 9
        backward_range = 1
        left_range = 5
        right_range = 5
        
        # Helper function to check if a position is in FOV (accounting for orientation)
        def is_in_fov(obj_row: int, obj_col: int) -> Tuple[bool, int, int]:
            """
            Check if object at (obj_row, obj_col) is in agent's asymmetric FOV.
            Returns (in_fov, rel_row, rel_col)
            
            The FOV depends on agent's orientation:
            - Forward: 9 steps in facing direction
            - Backward: 1 step behind
            - Left/Right: 5 steps on each side
            
            Coordinate system: (0,0) is top-left, row increases south, col increases east
            Direction: 0=North, 1=East, 2=South, 3=West
            """
            # Calculate relative position
            rel_row = obj_row - agent_row
            rel_col = obj_col - agent_col
            
            # Transform relative position based on agent's orientation
            # to get forward/backward/left/right coordinates
            if direction == 0:  # Facing North
                forward = -rel_row
                backward = rel_row
                left = -rel_col
                right = rel_col
            elif direction == 1:  # Facing East
                forward = rel_col
                backward = -rel_col
                left = rel_row
                right = -rel_row
            elif direction == 2:  # Facing South
                forward = rel_row
                backward = -rel_row
                left = rel_col
                right = -rel_col
            else:  # direction == 3, Facing West
                forward = -rel_col
                backward = rel_col
                left = -rel_row
                right = rel_row
            
            # Check if within FOV bounds
            in_forward = 0 <= forward <= forward_range
            in_backward = 0 <= backward <= backward_range
            in_left = 0 <= left <= left_range
            in_right = 0 <= right <= right_range
            
            # Object is in FOV if it's within forward OR backward range AND within left-right range
            if (in_forward or in_backward) and (in_left or in_right):
                return True, rel_row, rel_col
            return False, rel_row, rel_col
        
        # Collect visible objects from state
        iron_ores = []
        gold_ores = []
        gold_partials = []
        other_agents = []
        
        # Scan grid for ores
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                in_fov, rel_row, rel_col = is_in_fov(row, col)
                if not in_fov:
                    continue
                
                cell = grid[row, col]
                
                # Check for ores
                if cell == Items.iron_ore:
                    iron_ores.append((row, col, rel_row, rel_col))
                elif cell == Items.gold_ore:
                    gold_ores.append((row, col, rel_row, rel_col))
                elif cell == Items.gold_partial:
                    gold_partials.append((row, col, rel_row, rel_col))
        
        # Check for other agents in FOV
        for i, other_loc in enumerate(all_agent_locs):
            if i == self.agent_id:  # Skip self
                continue
            other_row, other_col = int(other_loc[0]), int(other_loc[1])
            in_fov, rel_row, rel_col = is_in_fov(other_row, other_col)
            if in_fov:
                other_agents.append((other_row, other_col, rel_row, rel_col))
        
        desc_parts.append("\n=== Visible Objects in Field of View ===")
        
        # Describe ores with coordinates
        if iron_ores:
            desc_parts.append(f"\nIron ore ({len(iron_ores)} visible):")
            for obj_row, obj_col, rel_row, rel_col in iron_ores[:8]:  # Show up to 8
                direction_desc = self._relative_position_desc(rel_row, rel_col)
                desc_parts.append(f"  - Iron ore at position (row {obj_row}, col {obj_col}) - {direction_desc}")
            if len(iron_ores) > 8:
                desc_parts.append(f"  - ... and {len(iron_ores) - 8} more iron ores")
        
        if gold_ores:
            desc_parts.append(f"\nGold ore ({len(gold_ores)} visible):")
            for obj_row, obj_col, rel_row, rel_col in gold_ores[:8]:  # Show up to 8
                direction_desc = self._relative_position_desc(rel_row, rel_col)
                desc_parts.append(f"  - Gold ore at position (row {obj_row}, col {obj_col}) - {direction_desc}")
            if len(gold_ores) > 8:
                desc_parts.append(f"  - ... and {len(gold_ores) - 8} more gold ores")
        
        if gold_partials:
            desc_parts.append(f"\nGold ore (partially mined, flashing) ({len(gold_partials)} visible):")
            for obj_row, obj_col, rel_row, rel_col in gold_partials[:8]:  # Show up to 8
                direction_desc = self._relative_position_desc(rel_row, rel_col)
                desc_parts.append(f"  - Partially mined gold ore at position (row {obj_row}, col {obj_col}) - {direction_desc}")
            if len(gold_partials) > 8:
                desc_parts.append(f"  - ... and {len(gold_partials) - 8} more partially mined gold ores")
        
        if not iron_ores and not gold_ores and not gold_partials:
            desc_parts.append("\nNo ores visible in your field of view.")
            
        # Describe other agents
        if other_agents:
            desc_parts.append(f"\nOther agents ({len(other_agents)} visible):")
            for obj_row, obj_col, rel_row, rel_col in other_agents:
                direction_desc = self._relative_position_desc(rel_row, rel_col)
                desc_parts.append(f"  - Agent at position (row {obj_row}, col {obj_col}) - {direction_desc}")
        
        # Append messages to observation if provided
        if received_messages:
            desc_parts.append("\n=== Messages from Other Agents ===")
            for msg in received_messages:
                desc_parts.append(f"  - {msg}")
        elif received_messages is not None:
            desc_parts.append("\n=== Messages from Other Agents ===")
            desc_parts.append("  - No messages received.")
        
        return "\n".join(desc_parts)
    
    def _relative_position_desc(self, rel_row: int, rel_col: int) -> str:
        """
        Generate description of relative position using up/down/left/right.
        
        Coordinate system: (0,0) is top-left corner
        - row-axis: up (0) to down (increasing)
        - col-axis: left (0) to right (increasing)
        """
        if rel_row == 0 and rel_col == 0:
            return "at your location"
        
        vertical = ""
        horizontal = ""
        
        # rel_row < 0 means object has smaller row = more up
        # rel_row > 0 means object has larger row = more down
        if rel_row < 0:
            vertical = f"{abs(rel_row)} step(s) up"
        elif rel_row > 0:
            vertical = f"{rel_row} step(s) down"
            
        # rel_col < 0 means object has smaller col = more left
        # rel_col > 0 means object has larger col = more right
        if rel_col < 0:
            horizontal = f"{abs(rel_col)} step(s) left"
        elif rel_col > 0:
            horizontal = f"{rel_col} step(s) right"
            
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


def find_closest_ore_in_fov(agent_row: int, agent_col: int, agent_direction: int, 
                              grid: np.ndarray) -> Optional[str]:
    """
    Find the closest ore in the agent's field of view.
    
    Returns:
        "iron", "gold", "gold_partial", or None if no ore in FOV
    """
    # FOV parameters
    forward_range = 9
    backward_range = 1
    left_range = 5
    right_range = 5
    
    def is_in_fov(obj_row: int, obj_col: int) -> Tuple[bool, float]:
        """Check if object is in FOV and return distance."""
        rel_row = obj_row - agent_row
        rel_col = obj_col - agent_col
        
        # Transform based on direction
        if agent_direction == 0:  # North
            forward = -rel_row
            backward = rel_row
            left = -rel_col
            right = rel_col
        elif agent_direction == 1:  # East
            forward = rel_col
            backward = -rel_col
            left = rel_row
            right = -rel_row
        elif agent_direction == 2:  # South
            forward = rel_row
            backward = -rel_row
            left = rel_col
            right = -rel_col
        else:  # West
            forward = -rel_col
            backward = rel_col
            left = -rel_row
            right = rel_row
        
        in_forward = 0 <= forward <= forward_range
        in_backward = 0 <= backward <= backward_range
        in_left = 0 <= left <= left_range
        in_right = 0 <= right <= right_range
        
        if (in_forward or in_backward) and (in_left or in_right):
            distance = np.sqrt(rel_row**2 + rel_col**2)
            return True, distance
        return False, float('inf')
    
    closest_ore = None
    closest_distance = float('inf')
    
    # Scan grid for ores
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell = grid[row, col]
            if cell == Items.iron_ore:
                in_fov, dist = is_in_fov(row, col)
                if in_fov and dist < closest_distance:
                    closest_distance = dist
                    closest_ore = "iron"
            elif cell == Items.gold_ore:
                in_fov, dist = is_in_fov(row, col)
                if in_fov and dist < closest_distance:
                    closest_distance = dist
                    closest_ore = "gold"
            elif cell == Items.gold_partial:
                in_fov, dist = is_in_fov(row, col)
                if in_fov and dist < closest_distance:
                    closest_distance = dist
                    closest_ore = "gold_partial"
    
    return closest_ore


def count_agents_in_fov(agent_id: int, agent_row: int, agent_col: int, 
                        agent_direction: int, all_agent_locs: np.ndarray) -> int:
    """
    Count the number of other agents in the agent's field of view.
    
    Returns:
        Number of other agents visible in FOV
    """
    # FOV parameters
    forward_range = 9
    backward_range = 1
    left_range = 5
    right_range = 5
    
    def is_in_fov(obj_row: int, obj_col: int) -> bool:
        """Check if object is in FOV."""
        rel_row = obj_row - agent_row
        rel_col = obj_col - agent_col
        
        # Transform based on direction
        if agent_direction == 0:  # North
            forward = -rel_row
            backward = rel_row
            left = -rel_col
            right = rel_col
        elif agent_direction == 1:  # East
            forward = rel_col
            backward = -rel_col
            left = rel_row
            right = -rel_row
        elif agent_direction == 2:  # South
            forward = rel_row
            backward = -rel_row
            left = rel_col
            right = -rel_col
        else:  # West
            forward = -rel_col
            backward = rel_col
            left = -rel_row
            right = rel_row
        
        in_forward = 0 <= forward <= forward_range
        in_backward = 0 <= backward <= backward_range
        in_left = 0 <= left <= left_range
        in_right = 0 <= right <= right_range
        
        return (in_forward or in_backward) and (in_left or in_right)
    
    count = 0
    for i, other_loc in enumerate(all_agent_locs):
        if i == agent_id:  # Skip self
            continue
        other_row, other_col = int(other_loc[0]), int(other_loc[1])
        if is_in_fov(other_row, other_col):
            count += 1
    
    return count


def construct_semantic_key(agent_id: int, agent_row: int, agent_col: int, 
                           closest_ore_type: Optional[str], num_agents_in_fov: int, 
                           action: str) -> Tuple:
    """
    Construct semantic representation key: (agent_id, agent_row, agent_col, 
    closest_ore_type_in_fov, num_agents_in_fov, action).
    
    No normalization - stores original values.
    """
    return (int(agent_id), int(agent_row), int(agent_col), 
            closest_ore_type if closest_ore_type else "none", 
            int(num_agents_in_fov), action)


# ============================================================================
# LLM AGENT
# ============================================================================

class LLMAgent:
    """LLM-powered agent that maintains belief state and generates actions."""
    
    def __init__(self, agent_id: int, team_color: str, model: str = "gpt-5-mini",
                 temperature: float = 0.7, reasoning: str = None):
        self.agent_id = agent_id
        self.team_color = team_color  # For compatibility
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
            f"I am Agent {self.agent_id}. "
            f"I am in a cooperative mining environment with other agents. "
            f"I need to decide whether to mine iron ore individually or coordinate with others to mine gold ore."
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
            api_params = {
                "model": self.model,
                "input": full_input,
            }
            
            # Add reasoning parameter for GPT-5.1
            if self.model == "gpt-5.1" and self.reasoning is not None and self.reasoning != "none":
                if self.reasoning in ["low", "medium", "high"]:
                    api_params["reasoning"] = {"effort": self.reasoning}
            
            response = self.client.responses.create(**api_params)

            # Extract output from response
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
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'to_dict'):
                return response.to_dict()
            else:
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
        return f"""You are an AI agent in a cooperative mining environment.

ENVIRONMENT DESCRIPTION:
- You are Agent {self.agent_id} in a multi-agent mining environment
- Two types of ore spawn randomly in empty spaces: iron ore (gray) and gold ore (yellow)
- You are equipped with a mining beam that attempts to extract ore in front of you
- Iron ore can be mined individually and provides a reward of +1 upon extraction
- Gold ore requires group coordination: it must be mined by exactly two players within a time window of 3 timesteps
- When you mine gold ore, it flashes (becomes "gold_partial") to indicate it is ready to be mined by another player
- If no other player helps within 3 timesteps, or if too many players try to mine, the gold ore reverts back to normal
- Gold ore yields a reward of +8 to each of the two miners when successfully extracted

MAP AND COORDINATE SYSTEM:
- Map size: 27 rows × 27 columns grid
- Coordinate system: (row, col) where:
  * Origin (0, 0) is at the TOP-LEFT corner
  * Row-axis: increases from top (0) to bottom (27) - "up" means decreasing row, "down" means increasing row
  * Col-axis: increases from left (0) to right (27) - "left" means decreasing col, "right" means increasing col
- Your position is given as (row, col) coordinates

FIELD OF VIEW (FOV):
- Your vision is asymmetric:
  * Forward: 9 steps ahead
  * Backward: 1 step behind
  * Left/Right: 5 steps on each side
- Objects are described with their absolute (row, col) positions and relative directions using up/down/left/right

REWARD STRUCTURE:
- Mining iron ore: +1 point (reliable, no coordination needed)
- Mining gold ore successfully (with another player): +8 points each (higher payoff, requires coordination)
- Mining gold ore alone: 0 points (no reward if no one else helps)
- Mining has an opportunity cost: while mining gold, you're not mining iron

STRATEGIC CONSIDERATIONS:
- Mining iron has a reliable payoff without needing to coordinate with others
- Mining gold has an opportunity cost (not mining iron), but if no one else helps, you get no reward
- However, if two players stick together (spatially) and go around mining gold, they will both receive higher reward than if they were mining iron
- Selfish agents tend to mine iron ore more, while cooperative agents will try to cooperate and mine gold ore

COMMUNICATION:
- You can send messages to other agents
- Other agents can send messages to you
- Messages can be used to negotiate, coordinate mining locations, or influence behavior

AVAILABLE ACTIONS:
- turn_left: Rotate 90 degrees counterclockwise
- turn_right: Rotate 90 degrees clockwise
- up: Move up (decrease row value)
- down: Move down (increase row value)
- left: Move left (decrease col value)
- right: Move right (increase col value)
- stay: Stay in place
- mine: Activate mining beam to extract ore in front of you (up to 3 tiles ahead)

OUTPUT FORMAT (must follow exactly):
BELIEF: [One sentence describing your current understanding of the situation including current position, next goal, and general game strategy.]
ACTION: [One of: turn_left, turn_right, up, down, left, right, stay, mine]
COMMUNICATION: [Your message to other agents, or "[No message]"]
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
            prompt_parts.append("\nMessages from Other Agents:")
            for msg in received_messages:
                prompt_parts.append(f"  - {msg}")
        else:
            prompt_parts.append("\nNo messages from other agents.")
            
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
            belief = re.sub(r'\[.*?\]', '', belief)  # Remove bracketed instructions
            
        # Parse action - extract single word action
        action_match = re.search(r'ACTION:\s*(\w+)', output, re.IGNORECASE)
        if action_match:
            action_candidate = action_match.group(1).strip().lower()
            # Validate action (accept both absolute and relative for backward compatibility)
            valid_actions = ['turn_left', 'turn_right', 'up', 'down', 'left', 'right', 'stay', 'mine',
                           'step_left', 'step_right', 'forward', 'backward']  # Legacy support
            if action_candidate in valid_actions:
                action = action_candidate
                
        # Parse communication - stop at newline or next section
        comm_match = re.search(r'COMMUNICATION:\s*(.+?)(?=\n\n|ACTION:|BELIEF:|$)', output, re.IGNORECASE | re.DOTALL)
        if comm_match:
            communication = comm_match.group(1).strip()
            
        return belief, action, communication


# ============================================================================
# ACTION PARSER
# ============================================================================

class ActionParser:
    """Converts LLM action strings to environment action indices."""
    
    # Map absolute directions to environment actions
    # The environment uses relative actions, so we need agent direction to convert
    ABSOLUTE_ACTION_MAP = {
        'up': 4,      # forward when facing north, maps to forward
        'down': 5,    # backward when facing north, maps to backward  
        'left': 2,    # step_left
        'right': 3,   # step_right
        'turn_left': 0,
        'turn_right': 1,
        'stay': 6,
        'mine': 7
    }
    
    # Legacy relative actions for backward compatibility
    RELATIVE_ACTION_MAP = {
        'turn_left': 0,
        'turn_right': 1,
        'step_left': 2,
        'step_right': 3,
        'forward': 4,
        'backward': 5,
        'stay': 6,
        'mine': 7
    }
    
    @staticmethod
    def parse(action_str: str, agent_direction: int = None) -> int:
        """
        Convert action string to action index.
        
        For absolute directions (up, down, left, right), we need to convert
        to relative actions based on agent's current direction.
        """
        action_str = action_str.lower().strip()
        
        # Try absolute direction first
        if action_str in ActionParser.ABSOLUTE_ACTION_MAP:
            abs_action = action_str
            if abs_action in ['up', 'down', 'left', 'right'] and agent_direction is not None:
                # Convert absolute direction to relative action based on agent's facing direction
                # direction: 0=North, 1=East, 2=South, 3=West
                # Up = decreasing row (north), Down = increasing row (south)
                # Left = decreasing col (west), Right = increasing col (east)
                if abs_action == 'up':
                    # Up means decreasing row (north)
                    if agent_direction == 0:  # Facing north
                        return 4  # forward (north = up)
                    elif agent_direction == 1:  # Facing east
                        return 2  # step_left (north = up)
                    elif agent_direction == 2:  # Facing south
                        return 5  # backward (north = up)
                    else:  # Facing west (3)
                        return 3  # step_right (north = up)
                elif abs_action == 'down':
                    # Down means increasing row (south)
                    if agent_direction == 0:  # Facing north
                        return 5  # backward (south = down)
                    elif agent_direction == 1:  # Facing east
                        return 3  # step_right (south = down)
                    elif agent_direction == 2:  # Facing south
                        return 4  # forward (south = down)
                    else:  # Facing west (3)
                        return 2  # step_left (south = down)
                elif abs_action == 'left':
                    # Left means decreasing col (west)
                    if agent_direction == 0:  # Facing north
                        return 2  # step_left (west = left)
                    elif agent_direction == 1:  # Facing east
                        return 5  # backward (west = left)
                    elif agent_direction == 2:  # Facing south
                        return 3  # step_right (west = left)
                    else:  # Facing west (3)
                        return 4  # forward (west = left)
                elif abs_action == 'right':
                    # Right means increasing col (east)
                    if agent_direction == 0:  # Facing north
                        return 3  # step_right (east = right)
                    elif agent_direction == 1:  # Facing east
                        return 4  # forward (east = right)
                    elif agent_direction == 2:  # Facing south
                        return 2  # step_left (east = right)
                    else:  # Facing west (3)
                        return 5  # backward (east = right)
            else:
                # turn_left, turn_right, stay, mine don't need conversion
                return ActionParser.ABSOLUTE_ACTION_MAP[abs_action]
        
        # Fall back to relative actions for backward compatibility
        if action_str in ActionParser.RELATIVE_ACTION_MAP:
            return ActionParser.RELATIVE_ACTION_MAP[action_str]
        
        # Default to 'stay'
        return 6
    
    @staticmethod
    def action_to_string(action_idx: int) -> str:
        """Convert action index to string."""
        for name, idx in ActionParser.ABSOLUTE_ACTION_MAP.items():
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
        # self.embedding_client = embedding_client
        
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
                    cumulative_iron_mined: np.ndarray = None,
                    cumulative_gold_mined: np.ndarray = None,
                    observation_state=None):
        """
        Log a complete timestep with all agent and environment data.
        
        Args:
            timestep: Current timestep number
            agents_data: List of dicts containing per-agent data
            env_obs: Environment observations (JAX array) - observations at timestep t
            env_state: Full environment state - state at t+1 (result of action at t)
            rewards: Reward array - rewards at t+1 (result of action at t)
            cumulative_iron_mined: Cumulative iron ore mined
            cumulative_gold_mined: Cumulative gold ore mined
            observation_state: State at timestep t (the state the observation describes)
        """
        grid = np.array(env_state.grid)
        iron_ores = np.sum(grid == Items.iron_ore)
        gold_ores = np.sum(grid == Items.gold_ore)
        gold_partials = np.sum(grid == Items.gold_partial)
        
        # Use observation_state for semantic key construction (state at t), env_state for logging (state at t+1)
        state_for_semantic = observation_state if observation_state is not None else env_state
        
        step_data = {
            "timestep": timestep,
            "agents": [],
            "env_state": self._serialize_env_state(env_state),  # State at t+1
            "env_obs": self._serialize_array(env_obs),
            "rewards": self._serialize_array(rewards),  # Rewards at t+1
            "ores_in_env": {"iron": int(iron_ores), "gold": int(gold_ores), "gold_partial": int(gold_partials)},
            "accumulated_rewards": [float(r) for r in self.cumulative_rewards],
            "cumulative_iron_mined": cumulative_iron_mined.tolist() if cumulative_iron_mined is not None else None,
            "cumulative_gold_mined": cumulative_gold_mined.tolist() if cumulative_gold_mined is not None else None
        }
        
        for agent_data in agents_data:
            agent_id = agent_data.get("agent_id")
            # Use observation_state for semantic key (state at t that observation describes)
            semantic_grid = np.array(state_for_semantic.grid)
            agent_row = int(state_for_semantic.agent_locs[agent_id][0])
            agent_col = int(state_for_semantic.agent_locs[agent_id][1])
            agent_direction = int(state_for_semantic.agent_locs[agent_id][2])
            
            # Find closest ore in FOV (using state at t)
            closest_ore = find_closest_ore_in_fov(agent_row, agent_col, agent_direction, semantic_grid)
            
            # Count agents in FOV (using state at t)
            num_agents_in_fov = count_agents_in_fov(
                agent_id, agent_row, agent_col, agent_direction, 
                state_for_semantic.agent_locs
            )
            
            # Construct semantic key
            semantic_key = construct_semantic_key(
                agent_id, agent_row, agent_col, closest_ore, num_agents_in_fov,
                agent_data.get("action", "stay")
            )
            
            # Save original text instead of embeddings
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
                "observation": agent_data.get("observation", ""),
                "belief": belief_text,
                "action": agent_data.get("action", "stay"),
                "action_idx": agent_data.get("action_idx", 6),
                "communication": comm_text,
                "received_messages": agent_data.get("received_messages", []),
                "reward": float(rewards[agent_id]),
                "semantic_key": semantic_key,
                "token_usage": token_usage,
                "api_time": api_time,
                "position": [agent_row, agent_col],
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
        
        # Print performance summary
        if self.timestep_times:
            print("\nPerformance Summary:")
            avg_timestep = np.mean(self.timestep_times)
            print(f"  Average timestep time: {avg_timestep:.3f}s")
            
            if self.api_call_times:
                avg_api = np.mean(self.api_call_times)
                api_percentage = (sum(self.api_call_times) / sum(self.timestep_times)) * 100
                print(f"  Average API call time: {avg_api:.3f}s ({api_percentage:.1f}% of total)")
    
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
        
        # Agent performance
        lines.append("=" * 80)
        lines.append("AGENT PERFORMANCE")
        lines.append("=" * 80)
        
        # Compute statistics from trajectory
        agent_stats = {i: {
            "total_reward": 0.0,
            "num_communications": 0,
            "action_distribution": {},
            "total_iron_mined": 0,
            "total_gold_mined": 0
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
        
        # Get final cumulative mined values
        if self.trajectory["trajectory"]:
            final_step = self.trajectory["trajectory"][-1]
            if final_step.get("cumulative_iron_mined"):
                for i, val in enumerate(final_step["cumulative_iron_mined"]):
                    agent_stats[i]["total_iron_mined"] = val
            if final_step.get("cumulative_gold_mined"):
                for i, val in enumerate(final_step["cumulative_gold_mined"]):
                    agent_stats[i]["total_gold_mined"] = val
        
        for agent_id in range(self.metadata["num_agents"]):
            stats = agent_stats[agent_id]
            lines.append(f"\nAgent {agent_id}:")
            lines.append(f"  Total Reward: {stats['total_reward']:.2f}")
            lines.append(f"  Average Return per Timestep: {stats['total_reward'] / max(self.episode_lengths, 1):.4f}")
            lines.append(f"  Number of Communications Sent: {stats['num_communications']}")
            lines.append(f"  Total Iron Ore Mined: {stats['total_iron_mined']:.0f}")
            lines.append(f"  Total Gold Ore Mined: {stats['total_gold_mined']:.0f}")
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
            # Average ores in environment
            total_iron = sum(step["ores_in_env"]["iron"] for step in self.trajectory["trajectory"])
            total_gold = sum(step["ores_in_env"]["gold"] for step in self.trajectory["trajectory"])
            total_gold_partial = sum(step["ores_in_env"]["gold_partial"] for step in self.trajectory["trajectory"])
            avg_iron = total_iron / len(self.trajectory["trajectory"])
            avg_gold = total_gold / len(self.trajectory["trajectory"])
            avg_gold_partial = total_gold_partial / len(self.trajectory["trajectory"])
            lines.append(f"  Average Iron Ore in Environment: {avg_iron:.2f}")
            lines.append(f"  Average Gold Ore in Environment: {avg_gold:.2f}")
            lines.append(f"  Average Partially Mined Gold Ore in Environment: {avg_gold_partial:.2f}")
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
        
    def render_timestep(self, timestep: int, env: CoopMining, state,
                       observations: List[str], communications: List[str],
                       actions: List[str], beliefs: List[str],
                       rewards: np.ndarray, plain_observations: List[str] = None):
        """
        Render a single timestep to PNG with all agents (up to 6).
        
        Args:
            plain_observations: Observations without communication messages
        """
        num_agents = len(observations)
        
        # Create figure with grid layout - adjust size based on number of agents
        # Layout: game state on top, then agents in rows of 3
        rows = 1 + ((num_agents + 2) // 3)  # +1 for game state, +2 to round up
        fig = plt.figure(figsize=(24, 4 + rows * 3))
        gs = GridSpec(rows, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Main game state (top, spanning all columns)
        def materialize(x):
            """Convert JAX array to numpy, ensuring it's concrete."""
            if hasattr(x, '__array__'):
                if hasattr(x, 'block_until_ready'):
                    x = x.block_until_ready()
                return np.array(x)
            return x
        
        state_np_materialized = jax.tree_util.tree_map(materialize, state)
        state_concrete = jax.tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, (np.ndarray, int, float)) else x, state_np_materialized)
        
        game_img = env.render(state_concrete)
        game_img = jax.block_until_ready(game_img)
        game_img_np = np.array(game_img)
        ax_game = fig.add_subplot(gs[0, :])
        ax_game.imshow(game_img_np)
        ax_game.set_title(f"Timestep {timestep} - Coop Mining State", 
                         fontsize=16, fontweight='bold')
        ax_game.axis('off')
        
        # Render all agents (up to 6)
        for i in range(min(num_agents, 6)):
            row = 1 + (i // 3)
            col = i % 3
            ax_agent = fig.add_subplot(gs[row, col])
            
            # Use plain observation if available, otherwise use full observation
            obs_to_show = plain_observations[i] if plain_observations and i < len(plain_observations) else observations[i]
            
            self._render_agent_info(ax_agent, i, obs_to_show, beliefs[i] if i < len(beliefs) else "",
                                   actions[i] if i < len(actions) else "stay", 
                                   communications[i] if i < len(communications) else "[No message]",
                                   rewards[i] if i < len(rewards) else 0.0)
        
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
                  seed: int = 42, reasoning: str = "medium", num_agents: int = 2):
    """
    Run LLM agent simulation in coop mining game.
    
    Args:
        num_steps: Number of timesteps to simulate
        save_dir: Directory to save visualizations
        model: Model name to use (e.g., "gpt-5.1", "gpt-5-mini", "o3")
        temperature: Sampling temperature for LLM (0.0-2.0)
        seed: Random seed for environment
        reasoning: Reasoning effort level for GPT-5.1 ("low", "medium", "high", or None for default)
        num_agents: Number of agents in the environment
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
    # Set JAX to use CPU for some operations to reduce GPU memory pressure
    # We'll use GPU for stepping but CPU for rendering
    env = CoopMining(
        num_agents=num_agents,
        num_inner_steps=1000,
        shared_rewards=False,
        cnn=True,
        jit=True
    )
    
    # Configure JAX memory management
    # Preallocate a smaller amount of memory to avoid OOM
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Initialize agents
    agents = [
        LLMAgent(agent_id=i, team_color=f"agent_{i}", model=model, 
                temperature=temperature, reasoning=reasoning)
        for i in range(num_agents)
    ]
    
    # Initialize communication manager
    comm_manager = CommunicationManager(num_agents=num_agents)
    
    # Initialize OpenAI client for embeddings
    # embedding_client = OpenAI()
    
    # Create display name for logging (include reasoning if specified)
    if model == "gpt-5.1" and reasoning and reasoning != "none":
        model_display_name = f"{model}_reasoning-{reasoning}"
    else:
        model_display_name = model
    
    # Initialize trajectory logger (will create experiment subfolder)
    logger = TrajectoryLogger(save_dir, model_display_name, temperature, seed, num_agents=num_agents)
    
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
    rewards = np.array([0.0] * num_agents)
    
    # Track cumulative mining statistics
    cumulative_iron_mined = np.array([0.0] * num_agents)
    cumulative_gold_mined = np.array([0.0] * num_agents)
    
    for t in range(num_steps):
        timestep_start = time.time()
        print(f"Timestep {t}/{num_steps-1}", end="\r")
        
        # Get messages for each agent (from previous timestep)
        agent_messages = [comm_manager.get_messages(i) for i in range(num_agents)]
        
        # Generate observations for each agent (with messages appended)
        obs_start = time.time()
        observations = []
        plain_observations = []  # Observations without communication
        for i, agent in enumerate(agents):
            # Plain observation without messages
            plain_obs = agent.descriptor.describe_observation(
                obs_np[i], 
                state_np.agent_locs[i],
                state_np.agent_locs,
                state_np.grid,
                received_messages=None  # No messages
            )
            plain_observations.append(plain_obs)
            
            # Full observation with messages
            obs_desc = agent.descriptor.describe_observation(
                obs_np[i], 
                state_np.agent_locs[i],
                state_np.agent_locs,
                state_np.grid,
                received_messages=agent_messages[i]  # Append messages to observation
            )
            observations.append(obs_desc)
        obs_time = time.time() - obs_start
        logger.observation_times.append(obs_time)
        
        # Collect ALL actions and communications first (don't send messages yet)
        actions_str = []
        communications = []
        beliefs = []
        raw_data_list = []
        
        for i, agent in enumerate(agents):
            # Use observations without separate messages since they're already included
            action_str, comm, raw_data = agent.update_and_act(
                observations[i],
                [],  # Messages already in observation, pass empty list
                rewards[i],
                t
            )
            actions_str.append(action_str)
            communications.append(comm)
            beliefs.append(agent.belief_state)
            raw_data_list.append(raw_data)
        
        # Parse actions (need agent direction for absolute direction conversion)
        actions = []
        for i, action_str in enumerate(actions_str):
            agent_direction = int(state_np.agent_locs[i][2])
            actions.append(ActionParser.parse(action_str, agent_direction=agent_direction))
        
        # Now step environment with all collected actions
        # Convert state back to JAX for stepping (JIT functions need JAX arrays)
        state_jax = jax.tree_util.tree_map(lambda x: jnp.array(x), state_np)
        
        # Step environment FIRST (action at t leads to state at t+1)
        key, subkey = jax.random.split(key)
        obs_new, state_new, rewards_new, done, info = env.step_env(
            subkey, state_jax, jnp.array(actions)
        )
        
        # Convert to numpy immediately to free GPU memory
        # Use block_until_ready to ensure computation completes before conversion
        obs_new = jax.block_until_ready(obs_new)
        state_new = jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), state_new)
        rewards_new = jax.block_until_ready(rewards_new)
        
        obs_new_np = np.array(obs_new)
        state_new_np = jax.tree_util.tree_map(lambda x: np.array(x), state_new)
        rewards_new_np = np.array(rewards_new)
        
        # Clear JAX computation cache periodically to free GPU memory
        if t % 10 == 0:
            jax.clear_backends()
        
        # Send all communications (they'll be available in next timestep)
        for i, comm in enumerate(communications):
            comm_manager.send_message(i, comm)
        
        # Update mining statistics from info
        # Note: The environment doesn't directly track this, so we'll infer from rewards
        # Iron gives +1, gold gives +8 (but only if successfully mined with partner)
        for i in range(num_agents):
            if rewards_new_np[i] >= 1.0:
                if rewards_new_np[i] >= 8.0:
                    cumulative_gold_mined[i] += 1
                elif rewards_new_np[i] >= 1.0:
                    cumulative_iron_mined[i] += 1
        
        # Prepare agent data for logging
        agents_data = []
        for i in range(num_agents):
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
                "received_messages": agent_messages[i],
                "token_usage": raw_data_list[i].get("token_usage", {}),
                "api_time": raw_data_list[i].get("api_time", 0.0)
            }
            agents_data.append(agent_data)
        
        # Log timestep AFTER stepping
        log_start = time.time()
        logger.log_timestep(t, agents_data, obs_np, state_new_np, rewards_new_np,
                           cumulative_iron_mined, cumulative_gold_mined, observation_state=state_np)
        log_time = time.time() - log_start
        logger.logging_times.append(log_time)
        
        # Visualize with new state (convert back to JAX for rendering, but use concrete values)
        # Only render every Nth timestep to save memory and time
        if t % 5 == 0 or t == num_steps - 1:
            try:
                # Convert numpy state back to JAX for rendering
                state_for_render = jax.tree_util.tree_map(lambda x: jnp.array(x), state_new_np)
                visualizer.render_timestep(
                    t, env, state_for_render, observations, communications,
                    actions_str, beliefs, rewards_new_np, plain_observations=plain_observations
                )
                # Clear the render state to free memory
                del state_for_render
            except Exception as e:
                # If rendering fails (e.g., due to JAX tracing issues), skip this frame
                print(f"\nWarning: Failed to render timestep {t}: {e}")
                pass
        
        # Update state for next iteration
        obs_np = obs_new_np
        state_np = state_new_np
        rewards = rewards_new_np
        
        # Explicitly delete JAX arrays to free GPU memory
        del obs_new, state_new, rewards_new, state_jax
        
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
    
    parser = argparse.ArgumentParser(description="Run LLM agent simulation for Coop Mining game")
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
    parser.add_argument("--num-agents", type=int, default=2,
                       help="Number of agents (default: 2)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM Agent Simulation for Coop Mining Game")
    print("=" * 70)
    print(f"Model: {args.model}")
    if args.model == "gpt-5.1":
        print(f"Reasoning: {args.reasoning}")
    print(f"Temperature: {args.temperature}")
    print(f"Steps per run: {args.steps}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Number of agents: {args.num_agents}")
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
            reasoning=args.reasoning if args.model == "gpt-5.1" else None,
            num_agents=args.num_agents
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
                reasoning=args.reasoning if args.model == "gpt-5.1" else None,
                num_agents=args.num_agents
            )

