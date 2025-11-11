"""
Example usage scripts for the LLM agent simulation.

Shows different ways to run and customize the simulation.
"""

import os
import sys
sys.path.append('/home/huao/Research/SocialJax')


def example_1_basic_run():
    """Example 1: Basic simulation run with default settings."""
    from coins_llm_simulation import run_simulation
    
    print("=" * 60)
    print("Example 1: Basic LLM Agent Simulation")
    print("=" * 60)
    
    # Make sure to set your API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Using mock simulation instead.")
        from coins_llm_mock import run_mock_simulation
        run_mock_simulation(num_steps=10, save_dir="./example_1_output")
    else:
        run_simulation(
            num_steps=10,
            api_key=api_key,
            save_dir="./example_1_output"
        )


def example_2_mock_run():
    """Example 2: Run mock simulation without API calls."""
    from coins_llm_mock import run_mock_simulation
    
    print("=" * 60)
    print("Example 2: Mock Agent Simulation (No API Calls)")
    print("=" * 60)
    
    run_mock_simulation(
        num_steps=30,
        save_dir="./example_2_output"
    )


def example_3_custom_agents():
    """Example 3: Customize agent behavior by subclassing."""
    from coins_llm_simulation import (
        CoinGame, ObservationDescriptor, ActionParser,
        CommunicationManager, Visualizer
    )
    import jax
    import jax.numpy as jnp
    import numpy as np
    
    print("=" * 60)
    print("Example 3: Custom Rule-Based Agents")
    print("=" * 60)
    
    class GreedyAgent:
        """Agent that greedily pursues nearest coin of any color."""
        
        def __init__(self, agent_id: int, team_color: str):
            self.agent_id = agent_id
            self.team_color = team_color
            self.descriptor = ObservationDescriptor(agent_id, team_color)
            self.belief_state = f"I am a greedy agent collecting any coins I see."
            
        def update_and_act(self, observation, received_messages, reward, timestep):
            # Simplified greedy strategy
            action = "stay"
            
            if "coin" in observation.lower():
                if "north" in observation.lower():
                    action = "up"
                elif "south" in observation.lower():
                    action = "down"
                elif "east" in observation.lower():
                    action = "right"
                elif "west" in observation.lower():
                    action = "left"
            else:
                # Random walk
                import random
                action = random.choice(["up", "down", "left", "right"])
            
            self.belief_state = f"Pursuing coins greedily. Last reward: {reward}"
            communication = f"I'm at position searching for coins" if timestep % 5 == 0 else "[No message]"
            
            return action, communication
    
    # Initialize environment
    env = CoinGame(num_agents=2, num_inner_steps=1000, regrow_rate=0.01, cnn=True)
    
    # Initialize custom agents
    agents = [
        GreedyAgent(agent_id=0, team_color="red"),
        GreedyAgent(agent_id=1, team_color="green")
    ]
    
    comm_manager = CommunicationManager(num_agents=2)
    visualizer = Visualizer("./example_3_output")
    
    # Run simulation
    key = jax.random.PRNGKey(123)
    obs, state = env.reset(key)
    obs_np = np.array(obs)
    state_np = jax.tree_map(lambda x: np.array(x), state)
    rewards = np.array([0.0, 0.0])
    
    for t in range(15):
        print(f"Timestep {t}")
        
        # Get observations
        observations = []
        for i, agent in enumerate(agents):
            obs_desc = agent.descriptor.describe_observation(
                obs_np[i], state_np.agent_locs[i],
                state_np.agent_locs, state_np.grid
            )
            observations.append(obs_desc)
        
        agent_messages = [comm_manager.get_messages(i) for i in range(2)]
        
        # Get actions
        actions_str = []
        communications = []
        beliefs = []
        
        for i, agent in enumerate(agents):
            action_str, comm = agent.update_and_act(
                observations[i], agent_messages[i], rewards[i], t
            )
            actions_str.append(action_str)
            communications.append(comm)
            beliefs.append(agent.belief_state)
            comm_manager.send_message(i, comm)
        
        actions = [ActionParser.parse(a) for a in actions_str]
        
        # Step
        key, subkey = jax.random.split(key)
        obs, state_np, rewards_new, done, info = env.step_env(
            subkey, state_np, jnp.array(actions)
        )
        
        obs_np = np.array(obs)
        state_np = jax.tree_map(lambda x: np.array(x), state_np)
        rewards = np.array(rewards_new)
        
        # Visualize
        visualizer.render_timestep(
            t, env, state_np, observations, communications,
            actions_str, beliefs, rewards
        )
        
        if done["__all__"]:
            break
    
    visualizer.create_gif()
    print("Custom agent simulation complete!")


def example_4_analysis():
    """Example 4: Analyze agent behavior from saved outputs."""
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Example 4: Analyze Saved Simulations")
    print("=" * 60)
    
    # This assumes you've run a simulation and have output
    output_dirs = ["./example_1_output", "./example_2_output", "./example_3_output"]
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            print(f"Skipping {output_dir} - not found")
            continue
            
        png_files = sorted(glob.glob(f"{output_dir}/timestep_*.png"))
        print(f"\n{output_dir}: Found {len(png_files)} timestep images")
        
        if os.path.exists(f"{output_dir}/simulation.gif"):
            gif = Image.open(f"{output_dir}/simulation.gif")
            print(f"  - GIF has {gif.n_frames} frames")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run example simulations")
    parser.add_argument("--example", type=int, default=2, choices=[1, 2, 3, 4],
                       help="Which example to run (1-4)")
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_run,
        2: example_2_mock_run,
        3: example_3_custom_agents,
        4: example_4_analysis
    }
    
    print("\nRunning example...")
    examples[args.example]()
    print("\nExample complete!")

