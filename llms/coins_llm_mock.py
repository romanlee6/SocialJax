"""
Mock LLM Agent Simulation for Coins Game - No API Calls Required

This script provides a mock version of the LLM agent simulation that doesn't
require OpenAI API calls. It uses rule-based agents that follow the same
interface and output format as the LLM agents.

Useful for testing the pipeline and visualization without API costs.
"""

import sys
sys.path.append('/home/huao/Research/SocialJax')

import numpy as np
from typing import List, Tuple
import random

# Import everything from main simulation except LLMAgent
from coins_llm_simulation import (
    ObservationDescriptor,
    ActionParser,
    CommunicationManager,
    Visualizer,
    CoinGame,
    jax,
    jnp
)


class MockLLMAgent:
    """Mock agent that uses simple rules instead of LLM API."""
    
    def __init__(self, agent_id: int, team_color: str):
        self.agent_id = agent_id
        self.team_color = team_color
        self.descriptor = ObservationDescriptor(agent_id, team_color)
        self.belief_state = (
            f"I am Agent {self.agent_id} with {self.team_color} color. "
            f"I am in an environment with another agent and need to navigate the social dilemma."
        )
        
    def update_and_act(self, observation: str, received_messages: List[str],
                       reward: float, timestep: int) -> Tuple[str, str]:
        """
        Mock decision making using simple rules.
        
        Returns:
            (action_name, communication_message)
        """
        # Simple rule-based logic
        action = self._decide_action(observation)
        belief = self._update_belief(observation, reward)
        communication = self._generate_communication(observation, action, timestep)
        
        self.belief_state = belief
        
        return action, communication
    
    def _decide_action(self, observation: str) -> str:
        """Decide action based on observation keywords."""
        
        # Look for team coin
        team_coin_keyword = f"{self.team_color} coin"
        
        if "north" in observation.lower() and team_coin_keyword in observation.lower():
            return "up"
        elif "south" in observation.lower() and team_coin_keyword in observation.lower():
            return "down"
        elif "east" in observation.lower() and team_coin_keyword in observation.lower():
            return "right"
        elif "west" in observation.lower() and team_coin_keyword in observation.lower():
            return "left"
        else:
            # Random exploration
            actions = ["up", "down", "left", "right", "turn_left", "turn_right"]
            return random.choice(actions)
    
    def _update_belief(self, observation: str, reward: float) -> str:
        """Update belief state based on observation and reward."""
        
        if reward > 0:
            return f"I collected a coin successfully! Looking for more coins to collect."
        elif reward < 0:
            return f"The other agent collected my color's coin, negatively affecting me. I should adapt my strategy."
        
        team_coin_keyword = f"{self.team_color} coin"
        
        if team_coin_keyword in observation.lower():
            return f"I can see a {self.team_color} coin nearby. Moving towards it."
        else:
            return f"Exploring the area to find coins and navigate the social dilemma."
    
    def _generate_communication(self, observation: str, action: str, 
                               timestep: int) -> str:
        """Generate communication message."""
        
        # Communicate occasionally
        if timestep % 3 != 0:
            return "[No message]"
        
        team_coin_keyword = f"{self.team_color} coin"
        
        if team_coin_keyword in observation.lower():
            return f"I see a {self.team_color} coin! Going to collect it."
        elif "coin" in observation.lower():
            # Seeing other agent's coin
            other_color = "green" if self.team_color == "red" else "red"
            return f"I see a {other_color} coin. Considering whether to take it."
        else:
            return f"Exploring, no coins visible yet."


def run_mock_simulation(num_steps: int = 20, 
                       save_dir: str = "./llm_mock_simulation_output"):
    """
    Run mock LLM agent simulation without API calls.
    
    Args:
        num_steps: Number of timesteps to simulate
        save_dir: Directory to save visualizations
    """
    
    # Initialize environment
    print("Initializing environment...")
    env = CoinGame(
        num_agents=2,
        num_inner_steps=1000,
        regrow_rate=0.01,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        cnn=True
    )
    
    # Initialize mock agents
    print("Initializing mock agents...")
    agents = [
        MockLLMAgent(agent_id=0, team_color="red"),
        MockLLMAgent(agent_id=1, team_color="green")
    ]
    
    # Initialize communication manager
    comm_manager = CommunicationManager(num_agents=2)
    
    # Initialize visualizer
    visualizer = Visualizer(save_dir)
    
    # Reset environment
    print("Resetting environment...")
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    
    # Convert JAX arrays to numpy
    obs_np = np.array(obs)
    state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
    
    # Simulation loop
    print(f"Running mock simulation for {num_steps} steps...")
    rewards = np.array([0.0, 0.0])
    
    for t in range(num_steps):
        print(f"\n=== Timestep {t} ===")
        
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
        
        for i, agent in enumerate(agents):
            action_str, comm = agent.update_and_act(
                observations[i],
                agent_messages[i],
                rewards[i],
                t
            )
            actions_str.append(action_str)
            communications.append(comm)
            beliefs.append(agent.belief_state)
            
            # Send communication
            comm_manager.send_message(i, comm)
            
            print(f"Agent {i} - Action: {action_str}, Comm: {comm[:50]}...")
        
        # Parse actions
        actions = [ActionParser.parse(a) for a in actions_str]
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state_np, rewards_new, done, info = env.step_env(
            subkey, state_np, jnp.array(actions)
        )
        
        # Convert to numpy
        obs_np = np.array(obs)
        state_np = jax.tree_util.tree_map(lambda x: np.array(x), state_np)
        rewards = np.array(rewards_new)
        
        print(f"Rewards: {rewards}")
        
        # Visualize
        visualizer.render_timestep(
            t, env, state_np, observations, communications,
            actions_str, beliefs, rewards
        )
        
        # Check if done
        if done["__all__"]:
            print(f"\nEpisode finished at timestep {t}")
            break
    
    # Create GIF
    print("\nCreating GIF animation...")
    visualizer.create_gif()
    
    print(f"\nMock simulation complete! Results saved to: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run mock LLM agent simulation (no API calls)"
    )
    parser.add_argument("--steps", type=int, default=20, 
                       help="Number of timesteps to simulate")
    parser.add_argument("--output-dir", type=str, 
                       default="./llm_mock_simulation_output",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    run_mock_simulation(
        num_steps=args.steps,
        save_dir=args.output_dir
    )

