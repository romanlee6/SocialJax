"""
Replay and Visualize Trajectories from Saved JSON Files

This script reconstructs game states from saved trajectory JSON files and
creates visualizations to verify that the logs are valid and complete.

Supports both trajectory_parsed.json (compact state) and trajectory_raw.json (full state).
"""

import os
import sys
sys.path.append('/home/huao/Research/SocialJax')

import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob

import socialjax
from socialjax.environments.coin_game.coin_game import CoinGame


class TrajectoryReplayer:
    """Replays and visualizes trajectories from JSON files."""
    
    def __init__(self, trajectory_path: str, use_raw: bool = False):
        """
        Initialize trajectory replayer.
        
        Args:
            trajectory_path: Path to trajectory JSON file or directory containing it
            use_raw: If True, use trajectory_raw.json, else use trajectory_parsed.json
        """
        self.trajectory_path = self._resolve_trajectory_path(trajectory_path, use_raw)
        self.use_raw = use_raw
        
        # Load trajectory
        print(f"Loading trajectory from: {self.trajectory_path}")
        with open(self.trajectory_path, 'r') as f:
            self.trajectory_data = json.load(f)
        
        # Extract metadata
        self.metadata = self.trajectory_data.get('metadata', {})
        self.num_agents = self.metadata.get('num_agents', 2)
        self.seed = self.metadata.get('seed', 42)
        
        print(f"Loaded trajectory:")
        print(f"  Model: {self.metadata.get('model', 'unknown')}")
        print(f"  Seed: {self.seed}")
        print(f"  Num Agents: {self.num_agents}")
        print(f"  Timesteps: {len(self.trajectory_data.get('trajectory', []))}")
        
        # Initialize environment
        self._init_environment()
    
    def _resolve_trajectory_path(self, path: str, use_raw: bool) -> str:
        """Resolve path to actual trajectory file."""
        filename = 'trajectory_raw.json' if use_raw else 'trajectory_parsed.json'
        
        # If path is a directory, look for trajectory file
        if os.path.isdir(path):
            trajectory_file = os.path.join(path, filename)
            if os.path.exists(trajectory_file):
                return trajectory_file
            else:
                raise FileNotFoundError(f"No {filename} found in directory: {path}")
        
        # If path is a file, use it
        elif os.path.isfile(path):
            return path
        
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    
    def _init_environment(self):
        """Initialize CoinGame environment for rendering."""
        self.env = CoinGame(
            num_agents=self.num_agents,
            num_inner_steps=1000,
            regrow_rate=0.01,
            payoff_matrix=[[1, 1, -2], [1, 1, -2]],
            cnn=True,
            shared_rewards=False
        )
        print("Environment initialized for rendering")
    
    def _reconstruct_state(self, timestep_data: Dict) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Reconstruct game state from timestep data.
        
        Args:
            timestep_data: Data for a single timestep
            
        Returns:
            (grid, agent_locs, agent_info_list)
        """
        if self.use_raw:
            # Full state available in raw trajectory
            env_state = timestep_data.get('env_state', {})
            grid = np.array(env_state.get('grid', []))
            agent_locs = np.array(env_state.get('agent_locs', []))
        else:
            # Compact state in parsed trajectory
            env_state_compact = timestep_data.get('env_state_compact', {})
            grid = np.array(env_state_compact.get('grid', []))
            agent_locs = np.array(env_state_compact.get('agent_locs', []))
        
        # Extract agent info (observations, beliefs, actions, communications)
        agent_info_list = []
        for agent_data in timestep_data.get('agents', []):
            agent_info = {
                'agent_id': agent_data.get('agent_id', 0),
                'observation': agent_data.get('observation', ''),
                'belief': agent_data.get('belief', ''),
                'action': agent_data.get('action', 'stay'),
                'communication': agent_data.get('communication', '[No message]'),
                'reward': agent_data.get('reward', 0.0)
            }
            agent_info_list.append(agent_info)
        
        return grid, agent_locs, agent_info_list
    
    def _create_mock_state(self, grid: np.ndarray, agent_locs: np.ndarray):
        """
        Create a mock environment state for rendering.
        
        Args:
            grid: Grid state array
            agent_locs: Agent locations array
            
        Returns:
            Mock state object compatible with env.render()
        """
        # Create a simple class to hold state
        class MockState:
            pass
        
        state = MockState()
        state.grid = jnp.array(grid)
        state.agent_locs = jnp.array(agent_locs)
        
        return state
    
    def render_timestep(self, timestep: int, output_dir: str, show_previous_actions: bool = True) -> bool:
        """
        Render a single timestep.
        
        Args:
            timestep: Timestep index
            output_dir: Directory to save rendered image
            show_previous_actions: If True, show actions from previous timestep that led to current state.
                                   If False, show current actions (agent's response to current state).
            
        Returns:
            True if successful, False otherwise
        """
        if timestep >= len(self.trajectory_data['trajectory']):
            print(f"Timestep {timestep} out of range")
            return False
        
        timestep_data = self.trajectory_data['trajectory'][timestep]
        
        # Reconstruct state for current timestep
        grid, agent_locs, agent_info_list = self._reconstruct_state(timestep_data)
        
        # If show_previous_actions, get agent info from previous timestep
        # This shows "what actions led to this state"
        if show_previous_actions and timestep > 0:
            prev_timestep_data = self.trajectory_data['trajectory'][timestep - 1]
            _, _, agent_info_list = self._reconstruct_state(prev_timestep_data)
            info_label = "Actions that led to this state (from previous timestep)"
        else:
            info_label = "Agent responses to current state"
        
        # Validate state
        if grid.size == 0 or agent_locs.size == 0:
            print(f"Warning: Invalid state at timestep {timestep}")
            return False
        
        # Create mock state for rendering
        state = self._create_mock_state(grid, agent_locs)
        
        # Render game state
        game_img = self.env.render(state)
        
        # Create figure with agent info
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main game state (top, spanning both columns)
        ax_game = fig.add_subplot(gs[0, :])
        ax_game.imshow(game_img)
        title = f"Timestep {timestep} - Reconstructed from Trajectory\n({info_label})"
        ax_game.set_title(title, fontsize=14, fontweight='bold')
        ax_game.axis('off')
        
        # Agent info panels
        for i, agent_info in enumerate(agent_info_list[:2]):  # Show up to 2 agents
            col = i % 2
            ax = fig.add_subplot(gs[1:, col])
            self._render_agent_info(ax, agent_info)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"replay_timestep_{timestep:04d}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return True
    
    def _render_agent_info(self, ax, agent_info: Dict):
        """Render information for a single agent."""
        ax.axis('off')
        
        agent_id = agent_info['agent_id']
        observation = agent_info['observation']
        belief = agent_info['belief']
        action = agent_info['action']
        communication = agent_info['communication']
        reward = agent_info['reward']
        
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
    
    def replay_all(self, output_dir: str, frame_skip: int = 1, show_previous_actions: bool = True):
        """
        Replay entire trajectory and save visualizations.
        
        Args:
            output_dir: Directory to save rendered frames
            frame_skip: Only render every Nth frame (1 = render all)
            show_previous_actions: If True, show actions from previous timestep that led to current state
        """
        trajectory = self.trajectory_data['trajectory']
        num_timesteps = len(trajectory)
        
        print(f"\nReplaying {num_timesteps} timesteps...")
        print(f"Frame skip: {frame_skip} (rendering every {frame_skip} frame(s))")
        print(f"Action alignment: {'Previous actions (what led here)' if show_previous_actions else 'Current actions (response to state)'}")
        print(f"Output directory: {output_dir}")
        
        success_count = 0
        fail_count = 0
        
        for t in range(num_timesteps):
            if t % frame_skip == 0 or t == num_timesteps - 1:
                print(f"  Rendering timestep {t}/{num_timesteps-1}", end="\r")
                if self.render_timestep(t, output_dir, show_previous_actions=show_previous_actions):
                    success_count += 1
                else:
                    fail_count += 1
        
        print(f"\n\nReplay complete!")
        print(f"  Successfully rendered: {success_count} frames")
        print(f"  Failed: {fail_count} frames")
        
        return success_count, fail_count
    
    def create_gif(self, output_dir: str, duration: int = 500):
        """Create animated GIF from rendered frames."""
        print(f"\nCreating GIF animation...")
        
        png_files = sorted(glob.glob(f"{output_dir}/replay_timestep_*.png"))
        
        if not png_files:
            print("No frames found to create GIF")
            return False
        
        print(f"Found {len(png_files)} frames")
        
        images = [Image.open(f) for f in png_files]
        if images:
            gif_path = os.path.join(output_dir, "replay_animation.gif")
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved: {gif_path}")
            return True
        
        return False
    
    def validate_trajectory(self) -> Dict:
        """
        Validate that trajectory contains all necessary information.
        
        Returns:
            Dictionary with validation results
        """
        print("\n" + "="*80)
        print("VALIDATING TRAJECTORY")
        print("="*80)
        
        validation = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check metadata
        if not self.metadata:
            validation['issues'].append("Missing metadata")
            validation['valid'] = False
        
        # Check trajectory exists
        trajectory = self.trajectory_data.get('trajectory', [])
        if not trajectory:
            validation['issues'].append("Empty trajectory")
            validation['valid'] = False
            return validation
        
        validation['statistics']['num_timesteps'] = len(trajectory)
        
        # Check each timestep
        missing_grid_count = 0
        missing_agents_count = 0
        missing_agent_locs_count = 0
        
        for t, timestep_data in enumerate(trajectory):
            # Check for required fields
            if self.use_raw:
                env_state = timestep_data.get('env_state', {})
                if not env_state.get('grid'):
                    missing_grid_count += 1
                if not env_state.get('agent_locs'):
                    missing_agent_locs_count += 1
            else:
                env_state_compact = timestep_data.get('env_state_compact', {})
                if not env_state_compact.get('grid'):
                    missing_grid_count += 1
                if not env_state_compact.get('agent_locs'):
                    missing_agent_locs_count += 1
            
            agents = timestep_data.get('agents', [])
            if not agents:
                missing_agents_count += 1
        
        # Report issues
        if missing_grid_count > 0:
            validation['issues'].append(f"Missing grid in {missing_grid_count} timesteps")
            validation['valid'] = False
        
        if missing_agent_locs_count > 0:
            validation['issues'].append(f"Missing agent_locs in {missing_agent_locs_count} timesteps")
            validation['valid'] = False
        
        if missing_agents_count > 0:
            validation['warnings'].append(f"Missing agents data in {missing_agents_count} timesteps")
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"  Timesteps: {validation['statistics']['num_timesteps']}")
        print(f"  Valid: {'✓ YES' if validation['valid'] else '✗ NO'}")
        
        if validation['issues']:
            print(f"\n  Issues:")
            for issue in validation['issues']:
                print(f"    ✗ {issue}")
        
        if validation['warnings']:
            print(f"\n  Warnings:")
            for warning in validation['warnings']:
                print(f"    ⚠ {warning}")
        
        if validation['valid'] and not validation['warnings']:
            print(f"\n  ✓ All checks passed! Trajectory is valid.")
        
        print("="*80 + "\n")
        
        return validation
    
    def compare_with_original(self, original_viz_dir: str, replay_viz_dir: str):
        """
        Compare replayed visualizations with originals (if available).
        
        Args:
            original_viz_dir: Directory with original visualizations
            replay_viz_dir: Directory with replayed visualizations
        """
        print("\n" + "="*80)
        print("COMPARING WITH ORIGINAL VISUALIZATIONS")
        print("="*80)
        
        original_frames = sorted(glob.glob(f"{original_viz_dir}/timestep_*.png"))
        replay_frames = sorted(glob.glob(f"{replay_viz_dir}/replay_timestep_*.png"))
        
        print(f"\nOriginal frames: {len(original_frames)}")
        print(f"Replayed frames: {len(replay_frames)}")
        
        if not original_frames:
            print("No original frames found for comparison")
            return
        
        # Just report that comparison would require image comparison logic
        print("\nNote: Visual comparison requires manual inspection or image diff tools.")
        print("Suggested approach:")
        print(f"  1. Open frames side by side")
        print(f"  2. Original: {original_viz_dir}")
        print(f"  3. Replayed: {replay_viz_dir}")
        print(f"  4. Verify game states match visually")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay and visualize trajectories from saved JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay from directory (auto-finds trajectory_parsed.json)
  python replay_trajectory.py llm_simulation_output/
  
  # Replay from specific file
  python replay_trajectory.py llm_simulation_output/trajectory_parsed.json
  
  # Use raw trajectory (full state)
  python replay_trajectory.py llm_simulation_output/ --use-raw
  
  # Replay with frame skip (every 5th frame)
  python replay_trajectory.py llm_simulation_output/ --frame-skip 5
  
  # Show current actions instead of previous (match original visualization)
  python replay_trajectory.py llm_simulation_output/ --show-current-actions
  
  # Validate only (no rendering)
  python replay_trajectory.py llm_simulation_output/ --validate-only
        """
    )
    
    parser.add_argument(
        "trajectory_path",
        help="Path to trajectory JSON file or directory containing it"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for replayed frames (default: [trajectory_dir]/replay_output)"
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use trajectory_raw.json instead of trajectory_parsed.json"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Only render every Nth frame (default: 1, render all)"
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Don't create animated GIF"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate trajectory, don't render frames"
    )
    parser.add_argument(
        "--compare-original",
        type=str,
        default=None,
        help="Path to original visualization directory for comparison"
    )
    parser.add_argument(
        "--show-current-actions",
        action="store_true",
        help="Show current actions (agent response) instead of previous actions (what led to state)"
    )
    
    args = parser.parse_args()
    
    # Initialize replayer
    try:
        replayer = TrajectoryReplayer(args.trajectory_path, use_raw=args.use_raw)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate trajectory
    validation = replayer.validate_trajectory()
    
    if not validation['valid']:
        print("\n⚠ WARNING: Trajectory validation failed!")
        print("Some timesteps may not render correctly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    if args.validate_only:
        print("Validation complete. Exiting (--validate-only flag set).")
        sys.exit(0)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: create replay_output in same directory as trajectory
        if os.path.isdir(args.trajectory_path):
            output_dir = os.path.join(args.trajectory_path, "replay_output")
        else:
            output_dir = os.path.join(os.path.dirname(args.trajectory_path), "replay_output")
    
    # Replay trajectory
    print("\n" + "="*80)
    print("REPLAYING TRAJECTORY")
    print("="*80)
    
    # Determine action alignment
    show_previous_actions = not args.show_current_actions
    
    success_count, fail_count = replayer.replay_all(
        output_dir, 
        frame_skip=args.frame_skip,
        show_previous_actions=show_previous_actions
    )
    
    if success_count == 0:
        print("\n❌ No frames were successfully rendered!")
        sys.exit(1)
    
    # Create GIF
    if not args.no_gif:
        replayer.create_gif(output_dir)
    
    # Compare with original if requested
    if args.compare_original:
        replayer.compare_with_original(args.compare_original, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("REPLAY COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Frames rendered: {success_count}")
    if not args.no_gif:
        print(f"Animation: {os.path.join(output_dir, 'replay_animation.gif')}")
    print("="*80 + "\n")
    
    # Provide next steps
    print("Next steps:")
    print(f"  1. View frames: ls {output_dir}/replay_timestep_*.png")
    if not args.no_gif:
        print(f"  2. Watch animation: open {os.path.join(output_dir, 'replay_animation.gif')}")
    print(f"  3. Compare with original visualizations (if available)")
    print()


if __name__ == "__main__":
    main()

