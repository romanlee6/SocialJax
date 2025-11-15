#!/usr/bin/env python3
"""
Similarity search module for finding similar (obs, action) pairs in the LLM dataset.

Supports multiple similarity metrics:
- Cosine similarity
- L2/Euclidean distance
- Manhattan distance
- FAISS-based fast search

Usage:
    from similarity_search import SimilaritySearcher
    
    searcher = SimilaritySearcher.load('llms/llm_datasets/llm_dataset_coins_none.pkl')
    results = searcher.search(obs_array, action_idx, top_k=5, metric='cosine')
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimilaritySearcher:
    """Search for similar (obs, action) pairs in the dataset."""
    
    def __init__(
        self,
        dataset: Dict,
        obs_arrays: List[np.ndarray],
        action_indices: List[int],
        obs_action_keys: List[Tuple],
        embedding_dim: int,
        game_type: str,
        obs_normalization: str = "none"
    ):
        """
        Initialize similarity searcher.
        
        Args:
            dataset: Dictionary mapping (obs, action) keys to entries
            obs_arrays: List of observation arrays
            action_indices: List of action indices
            obs_action_keys: List of (obs, action) key tuples
            embedding_dim: Dimension of belief/communication embeddings
            game_type: Type of game
            obs_normalization: Normalization method used
        """
        self.dataset = dataset
        self.obs_arrays = obs_arrays
        self.action_indices = action_indices
        self.obs_action_keys = obs_action_keys
        self.embedding_dim = embedding_dim
        self.game_type = game_type
        self.obs_normalization = obs_normalization
        
        # FAISS index (optional)
        self.faiss_index = None
        self.faiss_action_mapping = None
        
        # Precompute flattened observations for faster search
        if obs_arrays:
            self.obs_dim = obs_arrays[0].size
            self.obs_matrix = np.array([obs.flatten() for obs in obs_arrays], dtype=np.float32)
        else:
            self.obs_dim = None
            self.obs_matrix = None
    
    @classmethod
    def load(cls, dataset_path: str, faiss_index_path: Optional[str] = None):
        """
        Load dataset and create searcher.
        
        Args:
            dataset_path: Path to dataset pickle file
            faiss_index_path: Optional path to FAISS index file
        """
        dataset_path = Path(dataset_path)
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct dataset from saved data
        # Use obs_action_keys directly from saved data
        obs_action_keys = data['obs_action_keys']
        obs_arrays = data['obs_arrays']
        action_indices = data['action_indices']
        
        # Rebuild dataset dictionary from indexed format
        dataset = {}
        indexed_data = data['dataset']
        for obs, action in obs_action_keys:
            # Find matching entry in indexed data
            for idx, value in indexed_data.items():
                if np.array_equal(value['obs'], obs) and value['action'] == action:
                    dataset[(obs, action)] = value['entries']
                    break
            else:
                # If not found, create empty list
                dataset[(obs, action)] = []
        
        searcher = cls(
            dataset=dataset,
            obs_arrays=obs_arrays,
            action_indices=action_indices,
            obs_action_keys=obs_action_keys,
            embedding_dim=data['embedding_dim'],
            game_type=data['game_type'],
            obs_normalization=data.get('obs_normalization', 'none')
        )
        
        # Load FAISS index if provided
        if faiss_index_path and HAS_FAISS:
            searcher.faiss_index = faiss.read_index(faiss_index_path)
            action_mapping_path = Path(faiss_index_path).parent / f"{Path(faiss_index_path).stem.replace('_faiss', '')}_action_mapping.npy"
            if action_mapping_path.exists():
                searcher.faiss_action_mapping = np.load(action_mapping_path)
        
        return searcher
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using the same method as dataset."""
        if self.obs_normalization == "none":
            return obs
        elif self.obs_normalization == "l2":
            norm = np.linalg.norm(obs)
            if norm > 0:
                return obs / norm
            return obs
        elif self.obs_normalization == "minmax":
            obs_min = obs.min()
            obs_max = obs.max()
            if obs_max > obs_min:
                return (obs - obs_min) / (obs_max - obs_min)
            return obs
        else:
            return obs
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _l2_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2/Euclidean distance between two arrays."""
        return np.linalg.norm(a.flatten() - b.flatten())
    
    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Manhattan distance between two arrays."""
        return np.sum(np.abs(a.flatten() - b.flatten()))
    
    def search(
        self,
        query_obs: np.ndarray,
        query_action: Optional[int] = None,
        top_k: int = 5,
        metric: str = "cosine",
        filter_action: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar (obs, action) pairs.
        
        Args:
            query_obs: Query observation array
            query_action: Optional query action index (for filtering)
            top_k: Number of results to return
            metric: Similarity metric ('cosine', 'l2', 'manhattan', 'faiss')
            filter_action: If True and query_action provided, only return matching actions
            
        Returns:
            List of results, each containing:
            - 'obs': observation array
            - 'action': action index
            - 'similarity': similarity score
            - 'distance': distance score (if applicable)
            - 'entries': list of dataset entries with beliefs/communications
        """
        query_obs = self._normalize_observation(query_obs.copy())
        query_obs_flat = query_obs.flatten()
        
        # Use FAISS for fast search if available
        if metric == "faiss" and self.faiss_index is not None:
            return self._search_faiss(query_obs, query_action, top_k, filter_action)
        
        # Compute similarities/distances
        similarities = []
        
        for i, (obs, action) in enumerate(self.obs_action_keys):
            # Filter by action if requested
            if filter_action and query_action is not None and action != query_action:
                continue
            
            if metric == "cosine":
                score = self._cosine_similarity(query_obs, obs)
                distance = 1.0 - score  # Convert to distance
            elif metric == "l2":
                distance = self._l2_distance(query_obs, obs)
                score = 1.0 / (1.0 + distance)  # Convert to similarity
            elif metric == "manhattan":
                distance = self._manhattan_distance(query_obs, obs)
                score = 1.0 / (1.0 + distance)  # Convert to similarity
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            similarities.append({
                'index': i,
                'obs': obs,
                'action': action,
                'similarity': float(score),
                'distance': float(distance),
                'entries': self.dataset[(obs, action)]
            })
        
        # Sort by similarity (descending) or distance (ascending)
        if metric == "cosine":
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
        else:
            similarities.sort(key=lambda x: x['distance'])
        
        return similarities[:top_k]
    
    def _search_faiss(
        self,
        query_obs: np.ndarray,
        query_action: Optional[int],
        top_k: int,
        filter_action: bool
    ) -> List[Dict[str, Any]]:
        """Fast search using FAISS index."""
        query_obs_flat = query_obs.flatten().reshape(1, -1).astype(np.float32)
        
        # Search in FAISS
        k = min(top_k * 10, len(self.obs_arrays)) if filter_action else top_k
        distances, indices = self.faiss_index.search(query_obs_flat, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.obs_action_keys):
                continue
            
            obs, action = self.obs_action_keys[idx]
            
            # Filter by action if requested
            if filter_action and query_action is not None and action != query_action:
                continue
            
            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + dist)
            
            results.append({
                'index': int(idx),
                'obs': obs,
                'action': action,
                'similarity': float(similarity),
                'distance': float(dist),
                'entries': self.dataset[(obs, action)]
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_belief(
        self,
        query_belief: str,
        embedding_model,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for entries with similar beliefs.
        
        Args:
            query_belief: Query belief text
            embedding_model: SentenceTransformer model for encoding
            top_k: Number of results to return
            
        Returns:
            List of results sorted by belief similarity
        """
        # Encode query belief
        query_embedding = embedding_model.encode(
            query_belief,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarities = []
        
        for key, entries in self.dataset.items():
            obs, action = key
            
            # Find best matching entry for this key
            best_similarity = -1.0
            best_entry = None
            
            for entry in entries:
                if entry['belief_embedding'] is not None:
                    belief_emb = entry['belief_embedding']
                    similarity = np.dot(query_embedding, belief_emb)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_entry = entry
            
            if best_entry:
                similarities.append({
                    'obs': obs,
                    'action': action,
                    'belief_similarity': float(best_similarity),
                    'entry': best_entry
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['belief_similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        action_counts = defaultdict(int)
        for action in self.action_indices:
            action_counts[action] += 1
        
        return {
            'total_entries': len(self.obs_action_keys),
            'unique_keys': len(self.dataset),
            'embedding_dim': self.embedding_dim,
            'game_type': self.game_type,
            'obs_shape': self.obs_arrays[0].shape if self.obs_arrays else None,
            'obs_dim': self.obs_dim,
            'action_distribution': dict(action_counts),
            'has_faiss_index': self.faiss_index is not None
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test similarity search")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--faiss-index', type=str, help='Path to FAISS index file')
    parser.add_argument('--query-obs', type=str, help='Path to query observation (npy file)')
    parser.add_argument('--query-action', type=int, help='Query action index')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2', 'manhattan', 'faiss'])
    
    args = parser.parse_args()
    
    # Load searcher
    searcher = SimilaritySearcher.load(args.dataset, args.faiss_index)
    
    # Print statistics
    stats = searcher.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search if query provided
    if args.query_obs:
        query_obs = np.load(args.query_obs)
        results = searcher.search(
            query_obs,
            args.query_action,
            top_k=args.top_k,
            metric=args.metric,
            filter_action=args.query_action is not None
        )
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"    Action: {result['action']}")
            print(f"    Similarity: {result['similarity']:.4f}")
            print(f"    Distance: {result['distance']:.4f}")
            print(f"    Number of entries: {len(result['entries'])}")
            if result['entries']:
                entry = result['entries'][0]
                print(f"    Belief: {entry['belief_text'][:100]}...")
                print(f"    Communication: {entry['communication_text'][:100]}...")


if __name__ == "__main__":
    main()

