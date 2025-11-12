#!/usr/bin/env python3
"""
Collect multiple runs of LLM simulation with different random seeds.

This script runs the coins game simulation multiple times with different seeds,
collects all results, and generates a comparative summary.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse

# Add project root and current directory to path
sys.path.append('/home/huao/Research/SocialJax')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import run_simulation from the same directory
from coins_llm_simulation import run_simulation


def run_single_simulation(seed: int, model: str = "o3", temperature: float = 0.7,
                         num_steps: int = 20, base_output_dir: str = "./llm_simulation_output") -> str:
    """
    Run a single simulation and return the save directory path.
    
    Args:
        seed: Random seed for the simulation
        model: Model name to use
        temperature: Sampling temperature
        num_steps: Number of timesteps
        base_output_dir: Base directory for outputs
        
    Returns:
        Path to the experiment directory
    """
    print(f"\n{'='*70}")
    print(f"Running simulation with seed {seed}")
    print(f"{'='*70}")
    
    # Run simulation
    run_simulation(
        num_steps=num_steps,
        save_dir=base_output_dir,
        model=model,
        temperature=temperature,
        seed=seed
    )
    
    # Find the created experiment directory
    # The TrajectoryLogger creates a subdirectory with timestamp
    # We'll find the most recent one matching our pattern
    base_path = Path(base_output_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Output directory not found: {base_output_dir}")
    
    # Find directories matching the pattern
    pattern = f"{model}_temp{temperature}_seed{seed}_*"
    matching_dirs = sorted(base_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not matching_dirs:
        raise FileNotFoundError(f"Could not find experiment directory for seed {seed}")
    
    return str(matching_dirs[0])


def load_statistics(experiment_dir: str) -> Dict:
    """Load statistics from an experiment directory."""
    stats_path = Path(experiment_dir) / "trajectory_stats.json"
    if not stats_path.exists():
        return None
    
    with open(stats_path, 'r') as f:
        return json.load(f)


def load_summary(experiment_dir: str) -> str:
    """Load human-readable summary from an experiment directory."""
    summary_path = Path(experiment_dir) / "human_summary.txt"
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return f.read()


def create_comparative_summary(run_directories: List[str], seeds: List[int],
                              model: str, output_dir: str) -> str:
    """
    Create a comparative summary of multiple runs.
    
    Args:
        run_directories: List of experiment directory paths
        seeds: List of seeds used for each run
        model: Model name
        output_dir: Directory to save the summary
        
    Returns:
        Path to the summary file
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("COMPARATIVE SUMMARY: Multiple Runs")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Model: {model}")
    summary_lines.append(f"Number of Runs: {len(run_directories)}")
    summary_lines.append(f"Seeds: {seeds}")
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Collect statistics from all runs
    all_stats = []
    for i, (exp_dir, seed) in enumerate(zip(run_directories, seeds)):
        stats = load_statistics(exp_dir)
        if stats:
            stats['seed'] = seed
            stats['experiment_dir'] = exp_dir
            all_stats.append(stats)
    
    if not all_stats:
        summary_lines.append("ERROR: Could not load statistics from any run.")
        summary_path = Path(output_dir) / "comparative_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
        return str(summary_path)
    
    # Overall statistics
    summary_lines.append("=" * 80)
    summary_lines.append("OVERALL STATISTICS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Average episode length
    avg_episode_length = sum(s['episode_length'] for s in all_stats) / len(all_stats)
    summary_lines.append(f"Average Episode Length: {avg_episode_length:.2f} timesteps")
    summary_lines.append("")
    
    # Per-agent statistics across runs
    num_agents = all_stats[0]['metadata']['num_agents']
    for agent_id in range(num_agents):
        summary_lines.append(f"Agent {agent_id} - Across All Runs:")
        summary_lines.append("-" * 80)
        
        # Collect rewards
        total_rewards = [s['agents'][agent_id]['total_reward'] for s in all_stats]
        avg_rewards = [s['agents'][agent_id]['average_return'] for s in all_stats]
        
        summary_lines.append(f"  Total Rewards: {total_rewards}")
        summary_lines.append(f"  Average Total Reward: {sum(total_rewards) / len(total_rewards):.2f}")
        summary_lines.append(f"  Min Total Reward: {min(total_rewards):.2f}")
        summary_lines.append(f"  Max Total Reward: {max(total_rewards):.2f}")
        summary_lines.append("")
        
        summary_lines.append(f"  Average Returns per Timestep: {avg_rewards}")
        summary_lines.append(f"  Mean Average Return: {sum(avg_rewards) / len(avg_rewards):.4f}")
        summary_lines.append(f"  Min Average Return: {min(avg_rewards):.4f}")
        summary_lines.append(f"  Max Average Return: {max(avg_rewards):.4f}")
        summary_lines.append("")
        
        # Communication statistics
        comm_counts = [s['agents'][agent_id]['num_communications'] for s in all_stats]
        summary_lines.append(f"  Communications Sent: {comm_counts}")
        summary_lines.append(f"  Average Communications: {sum(comm_counts) / len(comm_counts):.1f}")
        summary_lines.append("")
        
        # Action distributions (aggregate)
        action_dist = {}
        for stats in all_stats:
            for action, count in stats['agents'][agent_id]['action_distribution'].items():
                action_dist[action] = action_dist.get(action, 0) + count
        
        summary_lines.append("  Aggregated Action Distribution:")
        total_actions = sum(action_dist.values())
        for action, count in sorted(action_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            summary_lines.append(f"    {action}: {count} times ({percentage:.1f}%)")
        summary_lines.append("")
    
    # Per-run details
    summary_lines.append("=" * 80)
    summary_lines.append("PER-RUN DETAILS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for i, (exp_dir, seed, stats) in enumerate(zip(run_directories, seeds, all_stats)):
        summary_lines.append(f"Run {i+1} - Seed {seed}:")
        summary_lines.append(f"  Directory: {exp_dir}")
        summary_lines.append(f"  Episode Length: {stats['episode_length']} timesteps")
        summary_lines.append("")
        
        for agent_id in range(num_agents):
            agent_stats = stats['agents'][agent_id]
            summary_lines.append(f"  Agent {agent_id}:")
            summary_lines.append(f"    Total Reward: {agent_stats['total_reward']:.2f}")
            summary_lines.append(f"    Average Return: {agent_stats['average_return']:.4f}")
            summary_lines.append(f"    Communications: {agent_stats['num_communications']}")
        summary_lines.append("")
    
    # Directory listing
    summary_lines.append("=" * 80)
    summary_lines.append("EXPERIMENT DIRECTORIES")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    for i, (exp_dir, seed) in enumerate(zip(run_directories, seeds)):
        summary_lines.append(f"Run {i+1} (Seed {seed}): {exp_dir}")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    # Save summary
    summary_path = Path(output_dir) / "comparative_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    
    return str(summary_path)


def create_summary_json(run_directories: List[str], seeds: List[int],
                       model: str, output_dir: str) -> str:
    """Create a JSON summary of all runs."""
    summary_data = {
        "metadata": {
            "model": model,
            "num_runs": len(run_directories),
            "seeds": seeds,
            "generated": datetime.now().isoformat()
        },
        "runs": []
    }
    
    for exp_dir, seed in zip(run_directories, seeds):
        stats = load_statistics(exp_dir)
        run_data = {
            "seed": seed,
            "experiment_dir": exp_dir,
            "statistics": stats
        }
        summary_data["runs"].append(run_data)
    
    json_path = Path(output_dir) / "comparative_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return str(json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Collect multiple runs of LLM simulation with different seeds"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds to use (default: 42 123 456)")
    parser.add_argument("--model", type=str, default="o3",
                       help="Model name to use (default: o3)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of timesteps per run (default: 20)")
    parser.add_argument("--output-dir", type=str, default="./llm_simulation_output",
                       help="Base output directory (default: ./llm_simulation_output)")
    parser.add_argument("--collect-dir", type=str, default=None,
                       help="Directory to save comparative summary (default: creates collection dir)")
    
    args = parser.parse_args()
    
    # Create collection directory
    if args.collect_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        collect_dir = Path(args.output_dir) / f"collection_{args.model}_{timestamp}"
    else:
        collect_dir = Path(args.collect_dir)
    
    collect_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Collecting Multiple Runs")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Steps per run: {args.steps}")
    print(f"Seeds: {args.seeds}")
    print(f"Collection directory: {collect_dir}")
    print("=" * 70)
    
    # Run simulations
    run_directories = []
    for seed in args.seeds:
        try:
            exp_dir = run_single_simulation(
                seed=seed,
                model=args.model,
                temperature=args.temperature,
                num_steps=args.steps,
                base_output_dir=args.output_dir
            )
            run_directories.append(exp_dir)
            print(f"\n✓ Completed run with seed {seed}")
            print(f"  Saved to: {exp_dir}")
        except Exception as e:
            print(f"\n✗ Failed run with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    if not run_directories:
        print("\nERROR: No successful runs completed.")
        return 1
    
    # Create comparative summary
    print(f"\n{'='*70}")
    print("Creating Comparative Summary")
    print(f"{'='*70}")
    
    summary_path = create_comparative_summary(
        run_directories=run_directories,
        seeds=args.seeds[:len(run_directories)],
        model=args.model,
        output_dir=str(collect_dir)
    )
    
    json_path = create_summary_json(
        run_directories=run_directories,
        seeds=args.seeds[:len(run_directories)],
        model=args.model,
        output_dir=str(collect_dir)
    )
    
    # Save run directory list
    runs_manifest = {
        "metadata": {
            "model": args.model,
            "temperature": args.temperature,
            "steps": args.steps,
            "num_runs": len(run_directories),
            "generated": datetime.now().isoformat()
        },
        "runs": [
            {"seed": seed, "directory": exp_dir}
            for seed, exp_dir in zip(args.seeds[:len(run_directories)], run_directories)
        ]
    }
    
    manifest_path = collect_dir / "runs_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(runs_manifest, f, indent=2)
    
    print(f"\n✓ Comparative summary saved to: {summary_path}")
    print(f"✓ JSON summary saved to: {json_path}")
    print(f"✓ Runs manifest saved to: {manifest_path}")
    print(f"\nCollection directory: {collect_dir}")
    print("\n" + "=" * 70)
    print("Collection Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

