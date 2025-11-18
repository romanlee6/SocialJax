#!/usr/bin/env python3
"""
Build a dataset from LLM trajectories with semantic_key vectors as keys
and embedded beliefs/communications as values.

The dataset structure:
- Key: semantic_key_vector (tuple of [color_encoded, x, y, coin_color_encoded, action_idx])
- Value: {
    'belief_embedding': np.ndarray,
    'communication_embedding': np.ndarray,
    'belief_text': str,
    'communication_text': str,
    'metadata': dict
}

Semantic key transformation:
- semantic_key: (agent_color, agent_x, agent_y, closest_coin_color, agent_id, action)
- Converted to: [color_encoded, x, y, coin_color_encoded, action_idx]
  - color_encoded: 0 (red) or 1 (green)
  - coin_color_encoded: 0 (none), 1 (red), 2 (green)
  - action_idx: integer action index
  - agent_id is removed (redundant with agent_color)

Usage:
    python build_llm_dataset.py \\
        --input-dir llms/llm_simulation_output \\
        --output-dir llms/llm_datasets \\
        --game-type coins
"""

import os
import sys
import json
import argparse
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Install with: pip install openai")


class LLMDatasetBuilder:
    """Build a searchable dataset from LLM trajectories."""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        embedding_dim: int = 256,
        game_type: str = "coins",
        obs_normalization: str = "none",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            embedding_model: Name of OpenAI embedding model (default: text-embedding-3-large)
            embedding_dim: Dimension of embeddings (default: 64)
            game_type: Type of game ('coins' or 'territory')
            obs_normalization: How to normalize observations ('none', 'l2', 'minmax')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (for custom endpoints)
        """
        self.game_type = game_type
        self.obs_normalization = obs_normalization
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        
        # Initialize OpenAI client
        if HAS_OPENAI:
            print(f"Initializing OpenAI client for model: {embedding_model} (dim={embedding_dim})")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            elif base_url:
                # For custom endpoints, try to get API key from env or use default
                api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI()  # Uses OPENAI_API_KEY from environment
        else:
            raise ImportError("openai is required. Install with: pip install openai")
        
        # Dataset storage - using semantic_key_vector as key for O(1) lookup
        self.dataset: Dict[Tuple, Dict] = {}
        
        # Track multiple assignments to same key for averaging
        # Store: key -> list of embeddings (for averaging)
        self._belief_embeddings_accumulator: Dict[Tuple, List] = defaultdict(list)
        self._comm_embeddings_accumulator: Dict[Tuple, List] = defaultdict(list)
        self._repeated_key_count = 0  # Count how many times we see a repeated key
        
        # Action mapping
        self.action_to_idx = {
            'turn_left': 0, 'turn_right': 1, 'left': 2, 'right': 3,
            'up': 4, 'down': 5, 'stay': 6
        }
        
        # Color encoding
        self.color_to_idx = {'red': 0, 'green': 1}
        self.coin_color_to_idx = {'none': 0, 'red': 1, 'green': 2}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Normalized embedding vector of shape (embedding_dim,)
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=text,
                dimensions=self.embedding_dim
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # Normalize embeddings (L2 normalization)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"  Warning: Embedding failed for text '{text[:50]}...': {e}")
            # Return zero vector if embedding fails
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _semantic_key_to_vector(self, semantic_key: List) -> Tuple:
        """
        Convert semantic_key to numerical vector for exact matching.
        
        Args:
            semantic_key: [agent_color, agent_x, agent_y, closest_coin_color, agent_id, action]
            
        Returns:
            Tuple of [color_encoded, x, y, coin_color_encoded, action_idx]
        """
        if len(semantic_key) != 6:
            raise ValueError(f"Expected semantic_key of length 6, got {len(semantic_key)}")
        
        agent_color, agent_x, agent_y, closest_coin_color, agent_id, action = semantic_key
        
        # Encode color: red=0, green=1
        color_encoded = self.color_to_idx.get(agent_color.lower(), 0)
        
        # Keep x, y as integers
        x = int(agent_x)
        y = int(agent_y)
        
        # Encode coin color: none=0, red=1, green=2
        coin_color_encoded = self.coin_color_to_idx.get(closest_coin_color.lower() if closest_coin_color else 'none', 0)
        
        # Convert action to index
        if isinstance(action, str):
            action_idx = self.action_to_idx.get(action, 6)
        else:
            action_idx = int(action)
        
        # Return as tuple for dictionary key (removed agent_id as redundant)
        return (color_encoded, x, y, coin_color_encoded, action_idx)
    
    def _process_trajectory_file(self, trajectory_path: Path) -> int:
        """
        Process a single trajectory file.
        
        Returns:
            Number of entries added
        """
        print(f"Processing: {trajectory_path}")
        
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        trajectory = data.get('trajectory', [])
        if not trajectory:
            print(f"  Warning: No trajectory data found in {trajectory_path}")
            return 0
        
        entries_added = 0
        
        for timestep_data in trajectory:
            timestep = timestep_data.get('timestep', 0)
            agents = timestep_data.get('agents', [])
            
            if not agents:
                continue
            
            # Process each agent
            for agent_data in agents:
                semantic_key = agent_data.get('semantic_key', None)
                if semantic_key is None:
                    continue
                
                belief = agent_data.get('belief', '')
                communication = agent_data.get('communication', '')
                
                # Skip if no belief or communication
                if not belief and not communication:
                    continue
                
                # Convert semantic_key to numerical vector
                try:
                    key_vector = self._semantic_key_to_vector(semantic_key)
                except (ValueError, KeyError) as e:
                    print(f"  Warning: Failed to convert semantic_key {semantic_key}: {e}")
                    continue
                
                # Embed belief and communication
                belief_embedding = None
                communication_embedding = None
                
                if belief:
                    belief_embedding = self._get_embedding(belief)
                
                if communication:
                    communication_embedding = self._get_embedding(communication)
                
                # Check if key already exists (for tracking repeated keys)
                key_exists = key_vector in self.dataset
                if key_exists:
                    self._repeated_key_count += 1
                
                # Accumulate embeddings for averaging
                if belief_embedding is not None:
                    self._belief_embeddings_accumulator[key_vector].append(belief_embedding)
                if communication_embedding is not None:
                    self._comm_embeddings_accumulator[key_vector].append(communication_embedding)
                
                # Compute average embeddings
                avg_belief_embedding = None
                if key_vector in self._belief_embeddings_accumulator:
                    belief_list = self._belief_embeddings_accumulator[key_vector]
                    if belief_list:
                        # Average all belief embeddings for this key
                        avg_belief_embedding = np.mean(belief_list, axis=0).astype(np.float32)
                        # Re-normalize after averaging
                        norm = np.linalg.norm(avg_belief_embedding)
                        if norm > 0:
                            avg_belief_embedding = avg_belief_embedding / norm
                
                avg_comm_embedding = None
                if key_vector in self._comm_embeddings_accumulator:
                    comm_list = self._comm_embeddings_accumulator[key_vector]
                    if comm_list:
                        # Average all communication embeddings for this key
                        avg_comm_embedding = np.mean(comm_list, axis=0).astype(np.float32)
                        # Re-normalize after averaging
                        norm = np.linalg.norm(avg_comm_embedding)
                        if norm > 0:
                            avg_comm_embedding = avg_comm_embedding / norm
                
                # Store entry with averaged embeddings (overwrite if key exists)
                self.dataset[key_vector] = {
                    'belief_embedding': avg_belief_embedding,
                    'communication_embedding': avg_comm_embedding,
                    'belief_text': belief,
                    'communication_text': communication,
                    'metadata': {
                        'agent_id': agent_data.get('agent_id', 0),
                        'timestep': timestep,
                        'trajectory_file': str(trajectory_path),
                        'game_type': self.game_type
                    }
                }
                
                entries_added += 1
        
        return entries_added
    
    def build_dataset(
        self,
        input_dir: str,
        pattern: str = "trajectory.json",
        filter_prefix: str = "gpt-5.1"
    ) -> Dict[str, Any]:
        """
        Build dataset from all trajectory files in input directory.
        Only processes directories starting with filter_prefix.
        
        Args:
            input_dir: Directory containing trajectory files
            pattern: Filename pattern to match
            filter_prefix: Only process directories starting with this prefix
            
        Returns:
            Dataset statistics
        """
        input_path = Path(input_dir)
        
        # Resolve to absolute path if relative
        # Try resolving relative to current working directory first
        if not input_path.is_absolute():
            # First try as-is (relative to current working directory)
            if not input_path.exists():
                # Try relative to project root (parent of llms directory)
                project_root = Path(__file__).parent.parent
                alt_path = project_root / input_path
                if alt_path.exists():
                    input_path = alt_path.resolve()
                else:
                    # Try resolving from current directory
                    input_path = input_path.resolve()
            else:
                input_path = input_path.resolve()
        
        if not input_path.exists():
            # Try one more time with project root
            project_root = Path(__file__).parent.parent
            alt_path = project_root / Path(input_dir)
            if alt_path.exists():
                input_path = alt_path.resolve()
            else:
                raise ValueError(f"Input directory does not exist: {input_dir} (tried: {Path(input_dir).resolve()}, {alt_path})")
        
        # Find all trajectory files, but only in directories starting with filter_prefix
        all_trajectory_files = list(input_path.rglob(pattern))
        trajectory_files = [
            f for f in all_trajectory_files 
            if f.parent.name.startswith(filter_prefix)
        ]
        
        if not trajectory_files:
            # Provide helpful debug information
            print(f"Debug: Input path: {input_path}")
            print(f"Debug: Found {len(all_trajectory_files)} total trajectory files")
            if all_trajectory_files:
                print(f"Debug: Sample parent directories: {[f.parent.name for f in all_trajectory_files[:5]]}")
            raise ValueError(f"No trajectory files found matching '{pattern}' in directories starting with '{filter_prefix}' in {input_dir} (resolved to: {input_path})")
        
        print(f"Found {len(trajectory_files)} trajectory files (filtered by prefix '{filter_prefix}')")
        
        total_entries = 0
        for traj_file in trajectory_files:
            entries = self._process_trajectory_file(traj_file)
            total_entries += entries
            print(f"  Added {entries} entries from {traj_file.name}")
        
        # Count how many keys have multiple values
        keys_with_multiple_values = 0
        for key_vector in self.dataset.keys():
            belief_count = len(self._belief_embeddings_accumulator.get(key_vector, []))
            comm_count = len(self._comm_embeddings_accumulator.get(key_vector, []))
            if belief_count > 1 or comm_count > 1:
                keys_with_multiple_values += 1
        
        # Build statistics
        stats = {
            'total_entries': total_entries,
            'unique_keys': len(self.dataset),
            'keys_with_multiple_values': keys_with_multiple_values,
            'repeated_key_assignments': self._repeated_key_count,
            'total_trajectory_files': len(trajectory_files),
            'embedding_dim': self.embedding_dim,
            'game_type': self.game_type,
            'filter_prefix': filter_prefix
        }
        
        print(f"\nDataset Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Unique semantic_key vectors: {stats['unique_keys']}")
        print(f"  Keys with multiple values: {stats['keys_with_multiple_values']}")
        print(f"  Repeated key assignments: {stats['repeated_key_assignments']}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
        print(f"  Filter prefix: {stats['filter_prefix']}")
        
        return stats
    
    def save_dataset(
        self,
        output_path: str,
        save_faiss_index: bool = False
    ):
        """
        Save dataset to disk.
        
        Args:
            output_path: Path to save dataset (.pkl file)
            save_faiss_index: Not used (kept for compatibility)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset with semantic_key vectors as keys
        # The dataset is already a dict with tuple keys, which is pickle-serializable
        save_data = {
            'dataset': self.dataset,  # Dict[Tuple, Dict] - optimal for O(1) lookup
            'embedding_dim': self.embedding_dim,
            'game_type': self.game_type,
            'color_to_idx': self.color_to_idx,
            'coin_color_to_idx': self.coin_color_to_idx,
            'action_to_idx': self.action_to_idx
        }
        
        # Save main dataset
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved dataset to {output_path}")
        print(f"  Dataset size: {len(self.dataset)} entries")


def main():
    parser = argparse.ArgumentParser(description="Build dataset from LLM trajectories")
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing LLM trajectory files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='llms/llm_datasets',
        help='Output directory for dataset files'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='text-embedding-3-large',
        help='OpenAI embedding model name'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=256,
        help='Embedding dimension (default: 256)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (defaults to OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='OpenAI API base URL (for custom endpoints)'
    )
    parser.add_argument(
        '--game-type',
        type=str,
        choices=['coins', 'territory'],
        default='coins',
        help='Type of game'
    )
    parser.add_argument(
        '--obs-normalization',
        type=str,
        choices=['none', 'l2', 'minmax'],
        default='none',
        help='Observation normalization method'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='trajectory.json',
        help='Filename pattern to match'
    )
    parser.add_argument(
        '--filter-prefix',
        type=str,
        default='gpt-5.1',
        help='Only process directories starting with this prefix'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = LLMDatasetBuilder(
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        game_type=args.game_type,
        obs_normalization=args.obs_normalization,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    stats = builder.build_dataset(args.input_dir, pattern=args.pattern, filter_prefix=args.filter_prefix)
    
    # Save dataset
    output_path = Path(args.output_dir) / f"llm_dataset_{args.game_type}_semantic_key.pkl"
    builder.save_dataset(output_path, save_faiss_index=False)
    
    # Save statistics
    stats_path = Path(args.output_dir) / f"llm_dataset_{args.game_type}_semantic_key_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()

