"""
Benchmark script to compare performance of different LLMs on Coins Game.

This script runs multiple LLM models on the same coins game environment with
fixed seeds to ensure fair comparison. It tracks:
- Performance metrics (rewards, episode length)
- Time cost per model
- Token usage (input/output tokens)

Results are saved both as individual detailed logs and as aggregate benchmark summary.
"""

import os
import sys
sys.path.append('/home/huao/Research/SocialJax')

import time
import json
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np

import jax
import jax.numpy as jnp

from coins_llm_simulation import (
    CoinGame, LLMAgent, CommunicationManager, 
    ActionParser, Visualizer
)


class BenchmarkLogger:
    """Tracks benchmark metrics across multiple model runs."""
    
    def __init__(self, base_dir: str):
        """Initialize benchmark logger.
        
        Args:
            base_dir: Base directory for saving all benchmark results
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.benchmark_dir = os.path.join(base_dir, f"benchmark_{timestamp}")
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        self.benchmark_results = {
            "timestamp": timestamp,
            "models": []
        }
        
        print(f"\nBenchmark results will be saved to: {self.benchmark_dir}\n")
    
    def add_model_result(self, model_name: str, result: Dict):
        """Add result for a single model run."""
        self.benchmark_results["models"].append({
            "model": model_name,
            **result
        })
    
    def save_summary(self):
        """Save benchmark summary with aggregate statistics."""
        # Save detailed benchmark results
        detailed_path = os.path.join(self.benchmark_dir, "benchmark_detailed.json")
        with open(detailed_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        # Create human-readable summary
        summary_path = os.path.join(self.benchmark_dir, "benchmark_summary.txt")
        with open(summary_path, 'w') as f:
            self._write_summary_report(f)
        
        # Create comparison table
        table_path = os.path.join(self.benchmark_dir, "benchmark_table.csv")
        self._write_comparison_table(table_path)
        
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {self.benchmark_dir}")
        print(f"  - benchmark_detailed.json: Complete results with all metrics")
        print(f"  - benchmark_summary.txt: Human-readable summary report")
        print(f"  - benchmark_table.csv: Comparison table (importable to Excel/Sheets)")
        print(f"{'='*80}\n")
    
    def _write_summary_report(self, f):
        """Write human-readable summary report."""
        f.write("="*80 + "\n")
        f.write("LLM BENCHMARK SUMMARY - Coins Game\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {self.benchmark_results['timestamp']}\n")
        f.write(f"Number of Models: {len(self.benchmark_results['models'])}\n")
        f.write("="*80 + "\n\n")
        
        for model_result in self.benchmark_results["models"]:
            f.write(f"MODEL: {model_result['model']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {model_result['status']}\n")
            
            if model_result['status'] == 'success':
                f.write(f"\nPerformance:\n")
                f.write(f"  Episode Length: {model_result['episode_length']} steps\n")
                f.write(f"  Total Time: {model_result['total_time_seconds']:.2f} seconds\n")
                f.write(f"  Time per Step: {model_result['time_per_step']:.3f} seconds\n")
                
                f.write(f"\nAgent Performance:\n")
                for agent_metrics in model_result['agents']:
                    agent_id = agent_metrics['agent_id']
                    f.write(f"  Agent {agent_id}:\n")
                    f.write(f"    Total Reward: {agent_metrics['total_reward']:.2f}\n")
                    f.write(f"    Average Reward: {agent_metrics['avg_reward']:.4f}\n")
                    f.write(f"    Communications Sent: {agent_metrics['num_communications']}\n")
                
                f.write(f"\nToken Usage (Total):\n")
                f.write(f"  Input Tokens: {model_result['total_input_tokens']:,}\n")
                f.write(f"  Output Tokens: {model_result['total_output_tokens']:,}\n")
                f.write(f"  Total Tokens: {model_result['total_tokens']:,}\n")
                
                f.write(f"\nToken Usage (Average per Agent per Step):\n")
                f.write(f"  Input Tokens: {model_result['avg_input_tokens_per_step']:.1f}\n")
                f.write(f"  Output Tokens: {model_result['avg_output_tokens_per_step']:.1f}\n")
                
                if 'action_distributions' in model_result:
                    f.write(f"\nAction Distributions:\n")
                    for agent_id, action_dist in enumerate(model_result['action_distributions']):
                        f.write(f"  Agent {agent_id}: {action_dist}\n")
            else:
                f.write(f"\nError: {model_result.get('error', 'Unknown error')}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def _write_comparison_table(self, filepath: str):
        """Write CSV comparison table."""
        with open(filepath, 'w') as f:
            # Header
            f.write("Model,Status,Episode_Length,Total_Time_s,Time_per_Step_s,")
            f.write("Agent0_Total_Reward,Agent0_Avg_Reward,Agent1_Total_Reward,Agent1_Avg_Reward,")
            f.write("Total_Input_Tokens,Total_Output_Tokens,Total_Tokens,")
            f.write("Avg_Input_Tokens_per_Step,Avg_Output_Tokens_per_Step\n")
            
            # Data rows
            for model_result in self.benchmark_results["models"]:
                model = model_result['model']
                status = model_result['status']
                
                if status == 'success':
                    f.write(f"{model},{status},")
                    f.write(f"{model_result['episode_length']},")
                    f.write(f"{model_result['total_time_seconds']:.2f},")
                    f.write(f"{model_result['time_per_step']:.3f},")
                    
                    # Agent metrics
                    for agent_metrics in model_result['agents']:
                        f.write(f"{agent_metrics['total_reward']:.2f},")
                        f.write(f"{agent_metrics['avg_reward']:.4f},")
                    
                    # Token usage
                    f.write(f"{model_result['total_input_tokens']},")
                    f.write(f"{model_result['total_output_tokens']},")
                    f.write(f"{model_result['total_tokens']},")
                    f.write(f"{model_result['avg_input_tokens_per_step']:.1f},")
                    f.write(f"{model_result['avg_output_tokens_per_step']:.1f}\n")
                else:
                    # Failed run - fill with N/A
                    f.write(f"{model},{status}," + "N/A," * 12 + "\n")


class BenchmarkRunner:
    """Runs benchmark simulations across multiple models."""
    
    def __init__(self, model_configs: List[Dict], num_steps: int, seed: int, 
                 base_output_dir: str, temperature: float = 0.7):
        """
        Initialize benchmark runner.
        
        Args:
            model_configs: List of model config dicts, each with 'model' (str) and optionally 'reasoning' (str)
            num_steps: Number of simulation steps for each model
            seed: Environment seed (fixed for fair comparison)
            base_output_dir: Base directory for all outputs
            temperature: Sampling temperature for LLMs
        """
        self.model_configs = model_configs
        self.num_steps = num_steps
        self.seed = seed
        self.temperature = temperature
        self.base_output_dir = base_output_dir
        
        self.logger = BenchmarkLogger(base_output_dir)
    
    def run_single_model(self, model_config: Dict) -> Dict:
        """
        Run simulation for a single model and collect metrics.
        
        Args:
            model_config: Dict with 'model' (str) and optionally 'reasoning' (str)
            
        Returns:
            Dictionary with performance metrics, timing, and token usage
        """
        model_name = model_config['model']
        reasoning = model_config.get('reasoning')
        
        # Create display name for logging
        # "none" or None means default behavior (no reasoning parameter)
        if reasoning and reasoning != "none":
            display_name = f"{model_name}_reasoning-{reasoning}"
        else:
            display_name = model_name
        
        print(f"\n{'='*80}")
        print(f"Running benchmark for model: {display_name}")
        print(f"{'='*80}")
        
        result = {
            "model": display_name,
            "status": "running",
            "seed": self.seed,
            "num_steps": self.num_steps,
            "temperature": self.temperature,
            "reasoning": reasoning
        }
        
        try:
            # Create model-specific output directory
            model_output_dir = os.path.join(
                self.logger.benchmark_dir, 
                f"{display_name.replace('/', '_')}_seed{self.seed}"
            )
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Initialize environment
            env = CoinGame(
                num_agents=2,
                num_inner_steps=1000,
                regrow_rate=0.01,
                payoff_matrix=[[1, 1, -2], [1, 1, -2]],
                cnn=True,
                shared_rewards=False
            )
            
            # Initialize agents
            agents = [
                LLMAgent(agent_id=0, team_color="red", model=model_name, 
                        temperature=self.temperature, reasoning=reasoning),
                LLMAgent(agent_id=1, team_color="green", model=model_name,
                        temperature=self.temperature, reasoning=reasoning)
            ]
            
            # Initialize communication manager
            comm_manager = CommunicationManager(num_agents=2)
            
            # Initialize visualizer
            visualizer = Visualizer(model_output_dir)
            
            # Reset environment
            key = jax.random.PRNGKey(self.seed)
            obs, state = env.reset(key)
            
            # Convert JAX arrays to numpy
            obs_np = np.array(obs)
            state_np = jax.tree_util.tree_map(lambda x: np.array(x), state)
            
            # Tracking variables
            cumulative_rewards = [0.0, 0.0]
            action_counts = [{}, {}]
            communication_counts = [0, 0]
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Trajectory storage for detailed logging
            trajectory_log = {
                "model": display_name,
                "seed": self.seed,
                "temperature": self.temperature,
                "timesteps": []
            }
            
            # Simulation loop
            rewards = np.array([0.0, 0.0])
            start_time = time.time()
            
            for t in range(self.num_steps):
                print(f"  Step {t+1}/{self.num_steps}", end="\r")
                
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
                    
                    # Extract token usage from API response
                    if raw_data.get("api_response") and isinstance(raw_data["api_response"], dict):
                        api_resp = raw_data["api_response"]
                        usage = api_resp.get("usage", {})
                        if usage:
                            total_input_tokens += usage.get("input_tokens", 0)
                            total_output_tokens += usage.get("output_tokens", 0)
                    
                    # Send communication
                    comm_manager.send_message(i, comm)
                    
                    # Track communications
                    if comm != "[No message]":
                        communication_counts[i] += 1
                
                # Parse actions
                actions = [ActionParser.parse(a) for a in actions_str]
                
                # Track actions
                for i, action_str in enumerate(actions_str):
                    action_counts[i][action_str] = action_counts[i].get(action_str, 0) + 1
                
                # Log timestep data
                timestep_data = {
                    "timestep": t,
                    "agents": [
                        {
                            "agent_id": i,
                            "observation": observations[i][:100] + "...",  # Truncate for brevity
                            "belief": beliefs[i][:100] + "...",
                            "action": actions_str[i],
                            "communication": communications[i],
                            "reward": float(rewards[i])
                        }
                        for i in range(2)
                    ]
                }
                trajectory_log["timesteps"].append(timestep_data)
                
                # Update cumulative rewards
                cumulative_rewards[0] += float(rewards[0])
                cumulative_rewards[1] += float(rewards[1])
                
                # Step environment
                key, subkey = jax.random.split(key)
                obs, state_np, rewards_new, done, info = env.step_env(
                    subkey, state_np, jnp.array(actions)
                )
                
                # Convert to numpy
                obs_np = np.array(obs)
                state_np = jax.tree_util.tree_map(lambda x: np.array(x), state_np)
                rewards = np.array(rewards_new)
                
                # Visualize (save every 5th frame to reduce storage)
                if t % 5 == 0 or t == self.num_steps - 1:
                    visualizer.render_timestep(
                        t, env, state_np, observations, communications,
                        actions_str, beliefs, rewards
                    )
                
                # Check if done
                if done["__all__"]:
                    print(f"\n  Episode finished early at timestep {t}")
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n  Completed in {total_time:.2f} seconds")
            
            # Calculate metrics
            episode_length = t + 1
            
            # Compile result
            result.update({
                "status": "success",
                "episode_length": episode_length,
                "total_time_seconds": total_time,
                "time_per_step": total_time / episode_length,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "avg_input_tokens_per_step": total_input_tokens / episode_length,
                "avg_output_tokens_per_step": total_output_tokens / episode_length,
                "agents": [
                    {
                        "agent_id": i,
                        "total_reward": cumulative_rewards[i],
                        "avg_reward": cumulative_rewards[i] / episode_length,
                        "num_communications": communication_counts[i]
                    }
                    for i in range(2)
                ],
                "action_distributions": action_counts
            })
            
            # Save trajectory log
            trajectory_path = os.path.join(model_output_dir, "trajectory_log.json")
            with open(trajectory_path, 'w') as f:
                json.dump(trajectory_log, f, indent=2)
            
            # Create GIF
            visualizer.create_gif(duration=500)
            
            print(f"  Model output saved to: {model_output_dir}")
            
        except Exception as e:
            print(f"\n  ERROR: Benchmark failed for {display_name}: {e}")
            import traceback
            traceback.print_exc()
            result.update({
                "status": "failed",
                "error": str(e)
            })
        
        return result
    
    def run_benchmark(self):
        """Run benchmark for all models."""
        print(f"\n{'='*80}")
        print("STARTING LLM BENCHMARK")
        print(f"{'='*80}")
        model_names = [f"{cfg['model']}" + (f"_reasoning-{cfg.get('reasoning')}" if cfg.get('reasoning') and cfg.get('reasoning') != "none" else "") 
                      for cfg in self.model_configs]
        print(f"Models to test: {', '.join(model_names)}")
        print(f"Number of steps: {self.num_steps}")
        print(f"Environment seed: {self.seed}")
        print(f"Temperature: {self.temperature}")
        print(f"{'='*80}\n")
        
        for model_config in self.model_configs:
            result = self.run_single_model(model_config)
            self.logger.add_model_result(result['model'], result)
        
        # Save summary
        self.logger.save_summary()
        
        # Print final comparison
        self._print_final_comparison()
    
    def _print_final_comparison(self):
        """Print final comparison table to console."""
        print(f"\n{'='*80}")
        print("BENCHMARK COMPARISON")
        print(f"{'='*80}\n")
        
        # Print table header
        print(f"{'Model':<20} {'Status':<10} {'Time(s)':<10} {'Agent0 Reward':<15} {'Agent1 Reward':<15} {'Total Tokens':<15}")
        print("-" * 90)
        
        # Print rows
        for model_result in self.logger.benchmark_results["models"]:
            model = model_result['model']
            status = model_result['status']
            
            if status == 'success':
                time_s = f"{model_result['total_time_seconds']:.2f}"
                agent0_reward = f"{model_result['agents'][0]['total_reward']:.2f}"
                agent1_reward = f"{model_result['agents'][1]['total_reward']:.2f}"
                total_tokens = f"{model_result['total_tokens']:,}"
                
                print(f"{model:<20} {status:<10} {time_s:<10} {agent0_reward:<15} {agent1_reward:<15} {total_tokens:<15}")
            else:
                print(f"{model:<20} {status:<10} {'N/A':<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        print(f"\n{'='*80}\n")


def main():
    """Main entry point for benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark multiple LLMs on Coins Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python benchmark_llms.py
  python benchmark_llms.py --models gpt-5 gpt-5-mini o3 --steps 30 --seed 123
        """
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        default=None,
        help="List of models to benchmark (if not specified, uses default set: gpt-5.1 with reasoning levels, gpt-5-mini, gpt-5-nano, o3)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=20,
        help="Number of simulation steps (default: 20)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Environment seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./llm_benchmarks",
        help="Base output directory (default: ./llm_benchmarks)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature for LLMs (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the benchmark.")
        sys.exit(1)
    
    # Build model configurations
    if args.models is None:
        # Default: GPT-5.1 with default (none) and different reasoning levels, plus comparison models
        # Note: "none" means no reasoning parameter is passed (default behavior)
        model_configs = [
            {"model": "gpt-5.1"},  # Default: no reasoning parameter (effort="none" by default)
            {"model": "gpt-5.1", "reasoning": "low"},
            {"model": "gpt-5.1", "reasoning": "medium"},
            {"model": "gpt-5.1", "reasoning": "high"},
            {"model": "gpt-5-mini"},
            {"model": "gpt-5-nano"},
            {"model": "o3"},
        ]
    else:
        # Parse simple model names (backward compatibility)
        model_configs = [{"model": m} for m in args.models]
    
    # Check if GPT-5.1 is being used and verify environment variables
    has_gpt51 = any(cfg.get("model") == "gpt-5.1" for cfg in model_configs)
    if has_gpt51:
        gpt51_url = os.getenv("GPT_51_URL")
        gpt51_key = os.getenv("GPT_51_KEY")
        if not gpt51_url or not gpt51_key:
            print("ERROR: GPT-5.1 model requires GPT_51_URL and GPT_51_KEY environment variables.")
            print(f"  GPT_51_URL: {'set' if gpt51_url else 'NOT SET'}")
            print(f"  GPT_51_KEY: {'set' if gpt51_key else 'NOT SET'}")
            print("Please set both environment variables before running the benchmark.")
            sys.exit(1)
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        model_configs=model_configs,
        num_steps=args.steps,
        seed=args.seed,
        base_output_dir=args.output_dir,
        temperature=args.temperature
    )
    
    runner.run_benchmark()


if __name__ == "__main__":
    main()

