"""
Analyze and visualize benchmark results.

This script loads benchmark results and creates visualizations comparing
different LLMs across various metrics.
"""

import json
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(benchmark_dir: str) -> dict:
    """Load benchmark results from directory."""
    detailed_path = os.path.join(benchmark_dir, "benchmark_detailed.json")
    
    if not os.path.exists(detailed_path):
        raise FileNotFoundError(f"Benchmark results not found: {detailed_path}")
    
    with open(detailed_path, 'r') as f:
        return json.load(f)


def plot_performance_comparison(results: dict, output_dir: str):
    """Create bar chart comparing model performance."""
    models = []
    agent0_rewards = []
    agent1_rewards = []
    
    for model_result in results['models']:
        if model_result['status'] == 'success':
            models.append(model_result['model'])
            agent0_rewards.append(model_result['agents'][0]['total_reward'])
            agent1_rewards.append(model_result['agents'][1]['total_reward'])
    
    if not models:
        print("No successful runs to plot")
        return
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, agent0_rewards, width, label='Agent 0', alpha=0.8)
    ax.bar(x + width/2, agent1_rewards, width, label='Agent 1', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Total Rewards by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'performance_comparison.png')}")


def plot_efficiency_comparison(results: dict, output_dir: str):
    """Create charts comparing time and token efficiency."""
    models = []
    time_per_step = []
    tokens_per_step = []
    
    for model_result in results['models']:
        if model_result['status'] == 'success':
            models.append(model_result['model'])
            time_per_step.append(model_result['time_per_step'])
            total_tokens_per_step = (
                model_result['avg_input_tokens_per_step'] + 
                model_result['avg_output_tokens_per_step']
            )
            tokens_per_step.append(total_tokens_per_step)
    
    if not models:
        print("No successful runs to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Time efficiency
    ax1.bar(models, time_per_step, alpha=0.8, color='skyblue')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time per Step (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Efficiency: Seconds per Step', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Token efficiency
    ax2.bar(models, tokens_per_step, alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tokens per Step', fontsize=12, fontweight='bold')
    ax2.set_title('Token Efficiency: Total Tokens per Step', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'efficiency_comparison.png')}")


def plot_reward_vs_efficiency(results: dict, output_dir: str):
    """Create scatter plot showing reward vs efficiency trade-off."""
    models = []
    avg_rewards = []
    time_per_step = []
    tokens_per_step = []
    
    for model_result in results['models']:
        if model_result['status'] == 'success':
            models.append(model_result['model'])
            # Average reward across both agents
            avg_reward = np.mean([
                model_result['agents'][0]['total_reward'],
                model_result['agents'][1]['total_reward']
            ])
            avg_rewards.append(avg_reward)
            time_per_step.append(model_result['time_per_step'])
            total_tokens_per_step = (
                model_result['avg_input_tokens_per_step'] + 
                model_result['avg_output_tokens_per_step']
            )
            tokens_per_step.append(total_tokens_per_step)
    
    if not models:
        print("No successful runs to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward vs Time
    ax1.scatter(time_per_step, avg_rewards, s=100, alpha=0.6)
    for i, model in enumerate(models):
        ax1.annotate(model, (time_per_step[i], avg_rewards[i]), 
                    fontsize=9, ha='right', va='bottom')
    ax1.set_xlabel('Time per Step (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Total Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Time Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Reward vs Tokens
    ax2.scatter(tokens_per_step, avg_rewards, s=100, alpha=0.6, color='coral')
    for i, model in enumerate(models):
        ax2.annotate(model, (tokens_per_step[i], avg_rewards[i]), 
                    fontsize=9, ha='right', va='bottom')
    ax2.set_xlabel('Tokens per Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Total Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Token Usage Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff_analysis.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'tradeoff_analysis.png')}")


def print_summary_table(results: dict):
    """Print a formatted summary table to console."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY TABLE")
    print("="*100)
    
    # Header
    print(f"{'Model':<20} {'Status':<10} {'Avg Reward':<12} {'Time/Step':<12} {'Tokens/Step':<12} {'Communications':<15}")
    print("-"*100)
    
    for model_result in results['models']:
        model = model_result['model']
        status = model_result['status']
        
        if status == 'success':
            avg_reward = np.mean([
                model_result['agents'][0]['total_reward'],
                model_result['agents'][1]['total_reward']
            ])
            time_per_step = model_result['time_per_step']
            tokens_per_step = (
                model_result['avg_input_tokens_per_step'] + 
                model_result['avg_output_tokens_per_step']
            )
            total_comms = sum([
                model_result['agents'][i]['num_communications']
                for i in range(len(model_result['agents']))
            ])
            
            print(f"{model:<20} {status:<10} {avg_reward:<12.2f} {time_per_step:<12.3f} {tokens_per_step:<12.1f} {total_comms:<15}")
        else:
            print(f"{model:<20} {status:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
    
    print("="*100 + "\n")


def generate_analysis_report(benchmark_dir: str):
    """Generate complete analysis with visualizations."""
    print(f"\nAnalyzing benchmark results from: {benchmark_dir}\n")
    
    # Load results
    results = load_benchmark_results(benchmark_dir)
    
    # Print summary
    print_summary_table(results)
    
    # Create visualizations
    print("Generating visualizations...")
    
    try:
        plot_performance_comparison(results, benchmark_dir)
        plot_efficiency_comparison(results, benchmark_dir)
        plot_reward_vs_efficiency(results, benchmark_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Visualizations saved to: {benchmark_dir}")
        print("  - performance_comparison.png: Bar chart of rewards by model")
        print("  - efficiency_comparison.png: Time and token efficiency charts")
        print("  - tradeoff_analysis.png: Reward vs efficiency scatter plots")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_benchmark_results.py path/to/benchmark_2025-11-10_12-00-00
  python analyze_benchmark_results.py --auto  # Analyze most recent benchmark
        """
    )
    
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        help="Path to benchmark directory containing benchmark_detailed.json"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find and analyze the most recent benchmark"
    )
    parser.add_argument(
        "--base-dir",
        default="./llm_benchmarks",
        help="Base directory containing benchmarks (used with --auto)"
    )
    
    args = parser.parse_args()
    
    # Determine benchmark directory
    if args.auto:
        # Find most recent benchmark
        base_path = Path(args.base_dir)
        if not base_path.exists():
            print(f"Error: Base directory not found: {args.base_dir}")
            sys.exit(1)
        
        benchmark_dirs = sorted(
            [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("benchmark_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not benchmark_dirs:
            print(f"Error: No benchmark directories found in {args.base_dir}")
            sys.exit(1)
        
        benchmark_dir = str(benchmark_dirs[0])
        print(f"Auto-detected most recent benchmark: {benchmark_dir}")
    
    elif args.benchmark_dir:
        benchmark_dir = args.benchmark_dir
    
    else:
        print("Error: Please specify a benchmark directory or use --auto")
        parser.print_help()
        sys.exit(1)
    
    # Validate directory
    if not os.path.exists(benchmark_dir):
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        sys.exit(1)
    
    # Generate analysis
    generate_analysis_report(benchmark_dir)


if __name__ == "__main__":
    main()

