#!/usr/bin/env python3
"""
Build a dataset from LLM trajectories with (obs, action) pairs as keys
and embedded beliefs/communications as values.

The dataset structure:
- Key: (observation_array, action_idx) tuple
- Value: {
    'belief_embedding': np.ndarray,
    'communication_embedding': np.ndarray,
    'belief_text': str,
    'communication_text': str,
    'metadata': dict
}

Usage:
    python build_llm_dataset.py \\
        --input-dir llms/llm_simulation_output \\
        --output-dir llms/llm_datasets \\
        --embedding-model all-MiniLM-L6-v2 \\
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
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss-cpu not installed. Install with: pip install faiss-cpu")


class LLMDatasetBuilder:
    """Build a searchable dataset from LLM trajectories."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        game_type: str = "coins",
        obs_normalization: str = "none"
    ):
        """
        Initialize dataset builder.
        
        Args:
            embedding_model: Name of sentence transformer model for embeddings
            game_type: Type of game ('coins' or 'territory')
            obs_normalization: How to normalize observations ('none', 'l2', 'minmax')
        """
        self.game_type = game_type
        self.obs_normalization = obs_normalization
        
        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        # Dataset storage
        self.dataset: Dict[Tuple, List[Dict]] = defaultdict(list)
        self.obs_action_keys: List[Tuple] = []
        self.obs_arrays: List[np.ndarray] = []
        self.action_indices: List[int] = []
        
        # Action mapping
        self.action_to_idx = {
            'turn_left': 0, 'turn_right': 1, 'left': 2, 'right': 3,
            'up': 4, 'down': 5, 'stay': 6
        }
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation array."""
        if self.obs_normalization == "none":
            return obs
        elif self.obs_normalization == "l2":
            # L2 normalization
            norm = np.linalg.norm(obs)
            if norm > 0:
                return obs / norm
            return obs
        elif self.obs_normalization == "minmax":
            # Min-max normalization
            obs_min = obs.min()
            obs_max = obs.max()
            if obs_max > obs_min:
                return (obs - obs_min) / (obs_max - obs_min)
            return obs
        else:
            raise ValueError(f"Unknown normalization: {self.obs_normalization}")
    
    def _hash_obs_action(self, obs: np.ndarray, action: int) -> str:
        """Create a hash for (obs, action) pair."""
        # Flatten and normalize for hashing
        obs_flat = obs.flatten()
        obs_hash = hashlib.md5(obs_flat.tobytes()).hexdigest()
        return f"{obs_hash}_{action}"
    
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
            env_obs = timestep_data.get('env_obs', [])
            
            if not env_obs or not agents:
                continue
            
            # Convert env_obs to numpy array
            env_obs_array = np.array(env_obs, dtype=np.float32)
            
            # Process each agent
            for agent_data in agents:
                agent_id = agent_data.get('agent_id', 0)
                action_str = agent_data.get('action', 'stay')
                action_idx = agent_data.get('action_idx', self.action_to_idx.get(action_str, 6))
                belief = agent_data.get('belief', '')
                communication = agent_data.get('communication', '')
                
                # Skip if no belief or communication
                if not belief and not communication:
                    continue
                
                # Get agent's observation (env_obs is per-agent)
                if agent_id < len(env_obs_array):
                    agent_obs = env_obs_array[agent_id]
                else:
                    # Fallback: use first agent's observation
                    agent_obs = env_obs_array[0]
                
                # Normalize observation
                agent_obs = self._normalize_observation(agent_obs)
                
                # Create key
                key = (agent_obs.copy(), action_idx)
                
                # Embed belief and communication
                belief_embedding = None
                communication_embedding = None
                
                if belief:
                    belief_embedding = self.embedding_model.encode(
                        belief, 
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                if communication:
                    communication_embedding = self.embedding_model.encode(
                        communication,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                # Store entry
                entry = {
                    'belief_embedding': belief_embedding,
                    'communication_embedding': communication_embedding,
                    'belief_text': belief,
                    'communication_text': communication,
                    'metadata': {
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'trajectory_file': str(trajectory_path),
                        'game_type': self.game_type
                    }
                }
                
                self.dataset[key].append(entry)
                self.obs_action_keys.append(key)
                self.obs_arrays.append(agent_obs)
                self.action_indices.append(action_idx)
                
                entries_added += 1
        
        return entries_added
    
    def build_dataset(
        self,
        input_dir: str,
        pattern: str = "trajectory_parsed.json"
    ) -> Dict[str, Any]:
        """
        Build dataset from all trajectory files in input directory.
        
        Args:
            input_dir: Directory containing trajectory files
            pattern: Filename pattern to match
            
        Returns:
            Dataset statistics
        """
        input_path = Path(input_dir)
        
        # Find all trajectory files
        trajectory_files = list(input_path.rglob(pattern))
        
        if not trajectory_files:
            raise ValueError(f"No trajectory files found matching '{pattern}' in {input_dir}")
        
        print(f"Found {len(trajectory_files)} trajectory files")
        
        total_entries = 0
        for traj_file in trajectory_files:
            entries = self._process_trajectory_file(traj_file)
            total_entries += entries
            print(f"  Added {entries} entries from {traj_file.name}")
        
        # Build statistics
        stats = {
            'total_entries': total_entries,
            'unique_keys': len(self.dataset),
            'total_trajectory_files': len(trajectory_files),
            'embedding_dim': self.embedding_dim,
            'game_type': self.game_type,
            'obs_shape': self.obs_arrays[0].shape if self.obs_arrays else None,
            'num_actions': len(set(self.action_indices)),
            'action_distribution': dict(zip(*np.unique(self.action_indices, return_counts=True)))
        }
        
        print(f"\nDataset Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Unique (obs, action) keys: {stats['unique_keys']}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
        print(f"  Observation shape: {stats['obs_shape']}")
        print(f"  Number of actions: {stats['num_actions']}")
        
        return stats
    
    def save_dataset(
        self,
        output_path: str,
        save_faiss_index: bool = True
    ):
        """
        Save dataset to disk.
        
        Args:
            output_path: Path to save dataset (.pkl file)
            save_faiss_index: Whether to build and save FAISS index for fast search
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        # Convert keys to serializable format
        # Note: We'll reconstruct keys from obs_action_keys when loading
        dataset_serializable = {}
        for i, (obs, action) in enumerate(self.obs_action_keys):
            # Use index as key since we have obs_action_keys list
            dataset_serializable[i] = {
                'obs': obs,
                'action': action,
                'entries': self.dataset.get((obs, action), [])
            }
        
        save_data = {
            'dataset': dataset_serializable,
            'obs_action_keys': self.obs_action_keys,
            'obs_arrays': self.obs_arrays,
            'action_indices': self.action_indices,
            'embedding_dim': self.embedding_dim,
            'game_type': self.game_type,
            'obs_normalization': self.obs_normalization
        }
        
        # Save main dataset
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved dataset to {output_path}")
        
        # Build and save FAISS index if requested
        if save_faiss_index and HAS_FAISS and len(self.obs_arrays) > 0:
            self._build_faiss_index(output_path)
    
    def _build_faiss_index(self, dataset_path: Path):
        """Build FAISS index for fast similarity search."""
        if not self.obs_arrays:
            return
        
        # Flatten observations for FAISS
        obs_dim = self.obs_arrays[0].size
        obs_matrix = np.array([obs.flatten() for obs in self.obs_arrays], dtype=np.float32)
        
        # Build FAISS index (L2 distance)
        index = faiss.IndexFlatL2(obs_dim)
        index.add(obs_matrix)
        
        # Save index
        index_path = dataset_path.parent / f"{dataset_path.stem}_faiss.index"
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
        # Also save action indices mapping
        action_mapping_path = dataset_path.parent / f"{dataset_path.stem}_action_mapping.npy"
        np.save(action_mapping_path, np.array(self.action_indices))
        print(f"Saved action mapping to {action_mapping_path}")


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
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name'
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
        default='trajectory_parsed.json',
        help='Filename pattern to match'
    )
    parser.add_argument(
        '--no-faiss',
        action='store_true',
        help='Skip building FAISS index'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = LLMDatasetBuilder(
        embedding_model=args.embedding_model,
        game_type=args.game_type,
        obs_normalization=args.obs_normalization
    )
    
    stats = builder.build_dataset(args.input_dir, pattern=args.pattern)
    
    # Save dataset
    output_path = Path(args.output_dir) / f"llm_dataset_{args.game_type}_{args.obs_normalization}.pkl"
    builder.save_dataset(output_path, save_faiss_index=not args.no_faiss)
    
    # Save statistics
    stats_path = Path(args.output_dir) / f"llm_dataset_{args.game_type}_{args.obs_normalization}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()

