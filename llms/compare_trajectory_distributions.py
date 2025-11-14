#!/usr/bin/env python3
"""
Compare observation/state/action distributions between LLM trajectories and RL policy trajectories.

This script analyzes whether episode length (100 steps for LLM vs 1000 steps for RL) 
has a significant impact on the distributions of observations, states, and actions.

Usage:
    # Compare with RL episodes matching LLM length (100 steps)
    python compare_trajectory_distributions.py \\
        --llm-trajectory llms/llm_simulation_output/o3_temp0.7_seed456_2025-11-12_19-10-53 \\
        --rl-checkpoint checkpoints/lgtom/coin_game_seed42_ps.pkl \\
        --rl-config algorithms/LG-TOM/config/lgtom_cnn_coins.yaml \\
        --num-episodes 10 \\
        --output-dir comparison_results
    
    # Compare with RL episodes using training length (1000 steps)
    python compare_trajectory_distributions.py \\
        --llm-trajectory llms/llm_simulation_output/o3_temp0.7_seed456_2025-11-12_19-10-53 \\
        --rl-checkpoint checkpoints/lgtom/coin_game_seed42_ps.pkl \\
        --rl-config algorithms/LG-TOM/config/lgtom_cnn_coins.yaml \\
        --num-episodes 10 \\
        --rl-episode-length 1000 \\
        --output-dir comparison_results
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
import yaml

# Add project root to path
sys.path.append('/home/huao/Research/SocialJax')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import socialjax
from socialjax.environments.coin_game.coin_game import CoinGame
from omegaconf import OmegaConf

# Import RL algorithm components
# Need to import from the file directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "lgtom_cnn_coins", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                 "algorithms/LG-TOM/lgtom_cnn_coins.py")
)
lgtom_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lgtom_module)

ActorCriticComm = lgtom_module.ActorCriticComm
load_params = lgtom_module.load_params
aggregate_communication = lgtom_module.aggregate_communication
import distrax


class TrajectoryComparator:
    """Compare distributions between LLM and RL trajectories."""
    
    def __init__(self, llm_trajectory_dir: str, rl_checkpoint: str, rl_config: str,
                 num_episodes: int = 10, rl_episode_length: int = None,
                 output_dir: str = "comparison_results"):
        """
        Initialize comparator.
        
        Args:
            llm_trajectory_dir: Directory containing LLM trajectory JSON files
            rl_checkpoint: Path to RL policy checkpoint (.pkl file)
            rl_config: Path to RL config YAML file
            num_episodes: Number of RL episodes to generate for comparison
            rl_episode_length: Length of RL episodes (None = match LLM trajectory length)
            output_dir: Directory to save comparison results
        """
        self.llm_trajectory_dir = llm_trajectory_dir
        self.rl_checkpoint = rl_checkpoint
        self.rl_config = rl_config
        self.num_episodes = num_episodes
        self.rl_episode_length = rl_episode_length
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load RL config
        with open(rl_config, 'r') as f:
            self.config = OmegaConf.load(rl_config)
        
        # Set default environment settings to match LLM simulation
        # LLM uses: regrow_rate=0.0005, num_inner_steps=1000, num_agents=2, cnn=True, shared_rewards=False
        self.env_kwargs = {
            "num_agents": 2,
            "num_inner_steps": 1000,
            "regrow_rate": 0.0005,  # Default regrow_rate from CoinGame
            "payoff_matrix": [[1, 1, -2], [1, 1, -2]],
            "cnn": True,
            "shared_rewards": False,
            "jit": True
        }
        
        # Override with config if present
        if "ENV_KWARGS" in self.config:
            for key, value in self.config.ENV_KWARGS.items():
                if key not in ["regrow_rate"]:  # Always use default regrow_rate
                    self.env_kwargs[key] = value
        
        print(f"Environment settings: {self.env_kwargs}")
        
    def load_llm_trajectories(self) -> Dict[str, Any]:
        """Load LLM trajectories from JSON file."""
        trajectory_path = os.path.join(self.llm_trajectory_dir, "trajectory_parsed.json")
        
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"LLM trajectory not found: {trajectory_path}")
        
        print(f"Loading LLM trajectories from: {trajectory_path}")
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        trajectory = data.get('trajectory', [])
        
        print(f"Loaded LLM trajectory:")
        print(f"  Model: {metadata.get('model', 'unknown')}")
        print(f"  Seed: {metadata.get('seed', 'unknown')}")
        print(f"  Num timesteps: {len(trajectory)}")
        
        return {
            'metadata': metadata,
            'trajectory': trajectory
        }
    
    def extract_llm_distributions(self, llm_data: Dict) -> Dict[str, np.ndarray]:
        """Extract observation, state, and action distributions from LLM trajectories."""
        trajectory = llm_data['trajectory']
        
        observations = []
        actions = []
        states_grid = []
        states_agent_locs = []
        rewards = []
        
        for timestep_data in trajectory:
            # Extract observations (env_obs)
            env_obs = timestep_data.get('env_obs', [])
            if env_obs:
                # Stack observations for all agents
                obs_array = np.array(env_obs)  # Shape: (num_agents, height, width, channels)
                observations.append(obs_array)
            
            # Extract actions (action_idx)
            agents = timestep_data.get('agents', [])
            timestep_actions = []
            timestep_rewards = []
            for agent in agents:
                action_idx = agent.get('action_idx', None)
                if action_idx is not None:
                    timestep_actions.append(action_idx)
                reward = agent.get('reward', 0.0)
                timestep_rewards.append(reward)
            
            if timestep_actions:
                actions.append(np.array(timestep_actions))
            if timestep_rewards:
                rewards.append(np.array(timestep_rewards))
            
            # Extract state information
            env_state_compact = timestep_data.get('env_state_compact', {})
            if env_state_compact:
                grid = np.array(env_state_compact.get('grid', []))
                agent_locs = np.array(env_state_compact.get('agent_locs', []))
                if grid.size > 0:
                    states_grid.append(grid)
                if agent_locs.size > 0:
                    states_agent_locs.append(agent_locs)
        
        return {
            'observations': np.array(observations) if observations else np.array([]),
            'actions': np.array(actions) if actions else np.array([]),
            'states_grid': np.array(states_grid) if states_grid else np.array([]),
            'states_agent_locs': np.array(states_agent_locs) if states_agent_locs else np.array([]),
            'rewards': np.array(rewards) if rewards else np.array([]),
            'num_timesteps': len(trajectory)
        }
    
    def generate_rl_trajectories(self, seed: int = 42) -> Dict[str, Any]:
        """Generate trajectories using RL policy."""
        print(f"\nGenerating {self.num_episodes} RL episodes...")
        
        # Load LLM trajectory length once
        if not hasattr(self, '_llm_length'):
            llm_data = self.load_llm_trajectories()
            self._llm_length = len(llm_data['trajectory'])
        
        # Load checkpoint
        print(f"Loading RL checkpoint from: {self.rl_checkpoint}")
        params = load_params(self.rl_checkpoint)
        
        # Initialize environment
        env = CoinGame(**self.env_kwargs)
        
        # Initialize network
        if self.config.get("PARAMETER_SHARING", True):
            network = ActorCriticComm(
                action_dim=env.action_space().n,
                comm_dim=self.config.get("COMM_DIM", 64),
                num_protos=self.config.get("NUM_PROTOS", 10),
                hidden_dim=self.config.get("HIDDEN_DIM", 128),
                activation=self.config.get("ACTIVATION", "relu")
            )
        else:
            raise NotImplementedError("Non-parameter sharing not yet implemented in this script")
        
        # Storage for all episodes
        all_observations = []
        all_actions = []
        all_states_grid = []
        all_states_agent_locs = []
        all_rewards = []
        
        # Generate multiple episodes
        key = jax.random.PRNGKey(seed)
        
        for episode in range(self.num_episodes):
            print(f"  Generating episode {episode + 1}/{self.num_episodes}...")
            
            # Initialize hidden states and communication
            hidden_states = jnp.zeros((env.num_agents, self.config.get("HIDDEN_DIM", 128)))
            prev_comm = jnp.zeros((env.num_agents, self.config.get("COMM_DIM", 64)))
            
            # Reset environment
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
            
            episode_observations = []
            episode_actions = []
            episode_states_grid = []
            episode_states_agent_locs = []
            episode_rewards = []
            
            # Run episode (use specified length or match LLM trajectory length)
            if self.rl_episode_length is not None:
                max_steps = min(self.rl_episode_length, env.num_inner_steps)
            else:
                # Match LLM trajectory length for fair comparison
                if not hasattr(self, '_llm_length'):
                    llm_data = self.load_llm_trajectories()
                    self._llm_length = len(llm_data['trajectory'])
                max_steps = min(self._llm_length, env.num_inner_steps)
            
            done = False
            for step in range(max_steps):
                # Store observation
                obs_np = np.array(obs)
                episode_observations.append(obs_np)
                
                # Store state
                state_grid = np.array(state.grid)
                state_agent_locs = np.array(state.agent_locs)
                episode_states_grid.append(state_grid)
                episode_states_agent_locs.append(state_agent_locs)
                
                # Stack observations for network
                obs_batch = jnp.stack([obs[a] for a in env.agents])
                
                # Forward pass
                key, action_key = jax.random.split(key)
                
                action_logits, comm_vectors, _, _, new_hidden_states, _, _ = network.apply(
                    params,
                    obs_batch,
                    prev_comm,
                    hidden_states,
                    train_mode=False,
                    rngs={'gumbel': action_key}
                )
                
                # Sample actions
                key, sample_key = jax.random.split(key)
                pi = distrax.Categorical(logits=action_logits)
                actions = pi.sample(seed=sample_key)
                
                # Store actions
                actions_np = np.array(actions)
                episode_actions.append(actions_np)
                
                # Aggregate communication
                # aggregate_communication expects (num_envs, num_agents, comm_dim)
                comm_expanded = jnp.expand_dims(comm_vectors, axis=0)  # (1, num_agents, comm_dim)
                aggregated_comm = aggregate_communication(
                    comm_expanded,
                    env.num_agents,
                    comm_mode=self.config.get("COMM_MODE", "avg")
                ).squeeze(0)  # (num_agents, comm_dim)
                
                # Update states
                hidden_states = new_hidden_states
                prev_comm = aggregated_comm
                
                # Execute actions
                key, step_key = jax.random.split(key)
                env_act = {env.agents[i]: int(actions[i]) for i in range(env.num_agents)}
                obs, state, reward, done, info = env.step(step_key, state, [v for v in env_act.values()])
                done = done["__all__"]
                
                # Store rewards
                reward_np = np.array([reward[agent] for agent in env.agents])
                episode_rewards.append(reward_np)
                
                if done:
                    break
            
            # Append episode data
            all_observations.extend(episode_observations)
            all_actions.extend(episode_actions)
            all_states_grid.extend(episode_states_grid)
            all_states_agent_locs.extend(episode_states_agent_locs)
            all_rewards.extend(episode_rewards)
        
        return {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'states_grid': np.array(all_states_grid),
            'states_agent_locs': np.array(all_states_agent_locs),
            'rewards': np.array(all_rewards),
            'num_timesteps': len(all_observations)
        }
    
    def compare_distributions(self, llm_dist: Dict, rl_dist: Dict) -> Dict[str, Any]:
        """Compare distributions between LLM and RL trajectories."""
        results = {}
        
        print("\n" + "="*80)
        print("COMPARING DISTRIBUTIONS")
        print("="*80)
        
        # Compare observations
        if llm_dist['observations'].size > 0 and rl_dist['observations'].size > 0:
            print("\n1. Comparing OBSERVATIONS")
            obs_llm = llm_dist['observations']
            obs_rl = rl_dist['observations']
            
            # Flatten observations for comparison
            obs_llm_flat = obs_llm.flatten()
            obs_rl_flat = obs_rl.flatten()
            
            results['observations'] = self._compare_arrays(
                obs_llm_flat, obs_rl_flat, "observations"
            )
        
        # Compare actions
        if llm_dist['actions'].size > 0 and rl_dist['actions'].size > 0:
            print("\n2. Comparing ACTIONS")
            actions_llm = llm_dist['actions'].flatten()
            actions_rl = rl_dist['actions'].flatten()
            
            results['actions'] = self._compare_arrays(
                actions_llm, actions_rl, "actions"
            )
            
            # Action distribution (histogram)
            unique_llm, counts_llm = np.unique(actions_llm, return_counts=True)
            unique_rl, counts_rl = np.unique(actions_rl, return_counts=True)
            
            results['actions']['llm_distribution'] = {
                'unique': unique_llm.tolist(),
                'counts': counts_llm.tolist(),
                'probabilities': (counts_llm / len(actions_llm)).tolist()
            }
            results['actions']['rl_distribution'] = {
                'unique': unique_rl.tolist(),
                'counts': counts_rl.tolist(),
                'probabilities': (counts_rl / len(actions_rl)).tolist()
            }
        
        # Compare rewards
        if llm_dist['rewards'].size > 0 and rl_dist['rewards'].size > 0:
            print("\n3. Comparing REWARDS")
            rewards_llm = llm_dist['rewards'].flatten()
            rewards_rl = rl_dist['rewards'].flatten()
            
            results['rewards'] = self._compare_arrays(
                rewards_llm, rewards_rl, "rewards"
            )
        
        # Compare state grids
        if llm_dist['states_grid'].size > 0 and rl_dist['states_grid'].size > 0:
            print("\n4. Comparing STATE GRIDS")
            grid_llm = llm_dist['states_grid'].flatten()
            grid_rl = rl_dist['states_grid'].flatten()
            
            results['states_grid'] = self._compare_arrays(
                grid_llm, grid_rl, "state_grids"
            )
        
        # Compare agent locations
        if llm_dist['states_agent_locs'].size > 0 and rl_dist['states_agent_locs'].size > 0:
            print("\n5. Comparing AGENT LOCATIONS")
            locs_llm = llm_dist['states_agent_locs'].flatten()
            locs_rl = rl_dist['states_agent_locs'].flatten()
            
            results['states_agent_locs'] = self._compare_arrays(
                locs_llm, locs_rl, "agent_locations"
            )
        
        return results
    
    def _compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, name: str) -> Dict:
        """Compare two arrays using statistical tests."""
        stats_dict = {}
        
        # Basic statistics
        stats_dict['llm_mean'] = float(np.mean(arr1))
        stats_dict['rl_mean'] = float(np.mean(arr2))
        stats_dict['llm_std'] = float(np.std(arr1))
        stats_dict['rl_std'] = float(np.std(arr2))
        stats_dict['llm_min'] = float(np.min(arr1))
        stats_dict['rl_min'] = float(np.min(arr2))
        stats_dict['llm_max'] = float(np.max(arr1))
        stats_dict['rl_max'] = float(np.max(arr2))
        
        print(f"  {name}:")
        print(f"    LLM - Mean: {stats_dict['llm_mean']:.4f}, Std: {stats_dict['llm_std']:.4f}")
        print(f"    RL  - Mean: {stats_dict['rl_mean']:.4f}, Std: {stats_dict['rl_std']:.4f}")
        
        # Kolmogorov-Smirnov test (non-parametric test for distribution equality)
        ks_statistic, ks_pvalue = stats.ks_2samp(arr1, arr2)
        stats_dict['ks_statistic'] = float(ks_statistic)
        stats_dict['ks_pvalue'] = float(ks_pvalue)
        
        print(f"    KS test: statistic={ks_statistic:.4f}, p-value={ks_pvalue:.6f}")
        if ks_pvalue < 0.05:
            print(f"    -> Distributions are significantly different (p < 0.05)")
        else:
            print(f"    -> Distributions are not significantly different (p >= 0.05)")
        
        # Mann-Whitney U test (non-parametric test for distribution equality)
        try:
            mw_statistic, mw_pvalue = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
            stats_dict['mw_statistic'] = float(mw_statistic)
            stats_dict['mw_pvalue'] = float(mw_pvalue)
            print(f"    Mann-Whitney U test: statistic={mw_statistic:.4f}, p-value={mw_pvalue:.6f}")
        except Exception as e:
            print(f"    Mann-Whitney U test failed: {e}")
            stats_dict['mw_statistic'] = None
            stats_dict['mw_pvalue'] = None
        
        return stats_dict
    
    def visualize_comparisons(self, llm_dist: Dict, rl_dist: Dict, 
                            comparison_results: Dict):
        """Create visualization plots comparing distributions."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM vs RL Trajectory Distribution Comparison', fontsize=16)
        
        # 1. Observations distribution
        if llm_dist['observations'].size > 0 and rl_dist['observations'].size > 0:
            ax = axes[0, 0]
            obs_llm = llm_dist['observations'].flatten()
            obs_rl = rl_dist['observations'].flatten()
            ax.hist(obs_llm, bins=50, alpha=0.5, label='LLM', density=True)
            ax.hist(obs_rl, bins=50, alpha=0.5, label='RL', density=True)
            ax.set_xlabel('Observation Values')
            ax.set_ylabel('Density')
            ax.set_title('Observation Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Actions distribution
        if llm_dist['actions'].size > 0 and rl_dist['actions'].size > 0:
            ax = axes[0, 1]
            actions_llm = llm_dist['actions'].flatten()
            actions_rl = rl_dist['actions'].flatten()
            
            unique_llm, counts_llm = np.unique(actions_llm, return_counts=True)
            unique_rl, counts_rl = np.unique(actions_rl, return_counts=True)
            
            x = np.arange(max(len(unique_llm), len(unique_rl)))
            width = 0.35
            
            # Create counts arrays aligned to x
            llm_counts_aligned = np.zeros(len(x))
            rl_counts_aligned = np.zeros(len(x))
            for i, u in enumerate(unique_llm):
                if u < len(x):
                    llm_counts_aligned[u] = counts_llm[i] / len(actions_llm)
            for i, u in enumerate(unique_rl):
                if u < len(x):
                    rl_counts_aligned[u] = counts_rl[i] / len(actions_rl)
            
            ax.bar(x - width/2, llm_counts_aligned, width, label='LLM', alpha=0.7)
            ax.bar(x + width/2, rl_counts_aligned, width, label='RL', alpha=0.7)
            ax.set_xlabel('Action Index')
            ax.set_ylabel('Probability')
            ax.set_title('Action Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Rewards distribution
        if llm_dist['rewards'].size > 0 and rl_dist['rewards'].size > 0:
            ax = axes[0, 2]
            rewards_llm = llm_dist['rewards'].flatten()
            rewards_rl = rl_dist['rewards'].flatten()
            ax.hist(rewards_llm, bins=50, alpha=0.5, label='LLM', density=True)
            ax.hist(rewards_rl, bins=50, alpha=0.5, label='RL', density=True)
            ax.set_xlabel('Reward Values')
            ax.set_ylabel('Density')
            ax.set_title('Reward Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. State grid distribution
        if llm_dist['states_grid'].size > 0 and rl_dist['states_grid'].size > 0:
            ax = axes[1, 0]
            grid_llm = llm_dist['states_grid'].flatten()
            grid_rl = rl_dist['states_grid'].flatten()
            ax.hist(grid_llm, bins=50, alpha=0.5, label='LLM', density=True)
            ax.hist(grid_rl, bins=50, alpha=0.5, label='RL', density=True)
            ax.set_xlabel('Grid Cell Values')
            ax.set_ylabel('Density')
            ax.set_title('State Grid Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Agent locations distribution
        if llm_dist['states_agent_locs'].size > 0 and rl_dist['states_agent_locs'].size > 0:
            ax = axes[1, 1]
            locs_llm = llm_dist['states_agent_locs'].flatten()
            locs_rl = rl_dist['states_agent_locs'].flatten()
            ax.hist(locs_llm, bins=50, alpha=0.5, label='LLM', density=True)
            ax.hist(locs_rl, bins=50, alpha=0.5, label='RL', density=True)
            ax.set_xlabel('Agent Location Values')
            ax.set_ylabel('Density')
            ax.set_title('Agent Location Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
        summary_text += f"LLM Trajectories:\n"
        summary_text += f"  Timesteps: {llm_dist['num_timesteps']}\n"
        summary_text += f"  Total samples: {sum(len(x) for x in [llm_dist['observations'], llm_dist['actions'], llm_dist['rewards']] if x.size > 0)}\n\n"
        
        summary_text += f"RL Trajectories:\n"
        summary_text += f"  Episodes: {self.num_episodes}\n"
        summary_text += f"  Timesteps: {rl_dist['num_timesteps']}\n\n"
        
        if 'actions' in comparison_results:
            summary_text += f"Action Distribution:\n"
            summary_text += f"  KS p-value: {comparison_results['actions'].get('ks_pvalue', 'N/A'):.6f}\n"
            if comparison_results['actions'].get('ks_pvalue', 1.0) < 0.05:
                summary_text += f"  → Significantly different\n"
            else:
                summary_text += f"  → Not significantly different\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'distribution_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()
    
    def save_results(self, comparison_results: Dict, llm_dist: Dict, rl_dist: Dict):
        """Save comparison results to JSON file."""
        output = {
            'llm_trajectory_dir': self.llm_trajectory_dir,
            'rl_checkpoint': self.rl_checkpoint,
            'rl_config': self.rl_config,
            'environment_settings': self.env_kwargs,
            'llm_trajectory_stats': {
                'num_timesteps': int(llm_dist['num_timesteps'])
            },
            'rl_trajectory_stats': {
                'num_episodes': self.num_episodes,
                'num_timesteps': int(rl_dist['num_timesteps']),
                'episode_length': self.rl_episode_length if self.rl_episode_length else 'matched_to_llm'
            },
            'comparison_results': comparison_results
        }
        
        output_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved results to: {output_path}")
    
    def run(self):
        """Run full comparison analysis."""
        print("="*80)
        print("TRAJECTORY DISTRIBUTION COMPARISON")
        print("="*80)
        
        # Load LLM trajectories
        llm_data = self.load_llm_trajectories()
        llm_dist = self.extract_llm_distributions(llm_data)
        
        # Generate RL trajectories
        # Use seed from LLM metadata if available
        llm_seed = llm_data['metadata'].get('seed', 42)
        rl_dist = self.generate_rl_trajectories(seed=llm_seed)
        
        # Compare distributions
        comparison_results = self.compare_distributions(llm_dist, rl_dist)
        
        # Visualize
        self.visualize_comparisons(llm_dist, rl_dist, comparison_results)
        
        # Save results
        self.save_results(comparison_results, llm_dist, rl_dist)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM and RL trajectory distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--llm-trajectory",
        type=str,
        required=True,
        help="Path to LLM trajectory directory (contains trajectory_parsed.json)"
    )
    
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        required=True,
        help="Path to RL policy checkpoint (.pkl file)"
    )
    
    parser.add_argument(
        "--rl-config",
        type=str,
        required=True,
        help="Path to RL config YAML file"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of RL episodes to generate (default: 10)"
    )
    
    parser.add_argument(
        "--rl-episode-length",
        type=int,
        default=None,
        help="Length of RL episodes in timesteps (default: None = match LLM trajectory length)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison results (default: comparison_results)"
    )
    
    args = parser.parse_args()
    
    # Create comparator and run
    comparator = TrajectoryComparator(
        llm_trajectory_dir=args.llm_trajectory,
        rl_checkpoint=args.rl_checkpoint,
        rl_config=args.rl_config,
        num_episodes=args.num_episodes,
        rl_episode_length=args.rl_episode_length,
        output_dir=args.output_dir
    )
    
    comparator.run()


if __name__ == "__main__":
    main()

