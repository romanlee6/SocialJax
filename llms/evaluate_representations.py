#!/usr/bin/env python3
"""
Evaluate observation and action representations in coin and territory games.

This script analyzes:
1. Observation space coverage and diversity
2. Action distribution and patterns
3. Similarity structure in the dataset
4. Potential issues and improvements

Usage:
    python evaluate_representations.py \\
        --dataset llms/llm_datasets/llm_dataset_coins_none.pkl \\
        --output-dir evaluation_results
"""

import os
import sys
import json
import argparse
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llms.similarity_search import SimilaritySearcher


class RepresentationEvaluator:
    """Evaluate observation and action representations."""
    
    def __init__(self, searcher: SimilaritySearcher):
        """Initialize evaluator with a loaded searcher."""
        self.searcher = searcher
        self.stats = {}
    
    def analyze_observation_space(self) -> Dict[str, Any]:
        """Analyze observation space coverage and diversity."""
        print("Analyzing observation space...")
        
        obs_arrays = self.searcher.obs_arrays
        if not obs_arrays:
            return {}
        
        # Flatten observations
        obs_matrix = np.array([obs.flatten() for obs in obs_arrays])
        
        # Basic statistics
        obs_mean = np.mean(obs_matrix, axis=0)
        obs_std = np.std(obs_matrix, axis=0)
        obs_min = np.min(obs_matrix, axis=0)
        obs_max = np.max(obs_matrix, axis=0)
        
        # Sparsity analysis
        sparsity = np.mean(obs_matrix == 0)
        
        # Diversity: pairwise distances
        # Sample subset for efficiency
        sample_size = min(1000, len(obs_matrix))
        sample_indices = np.random.choice(len(obs_matrix), sample_size, replace=False)
        obs_sample = obs_matrix[sample_indices]
        
        # Compute pairwise distances
        distances = pdist(obs_sample, metric='euclidean')
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Check for duplicate observations
        unique_obs = len(np.unique(obs_matrix, axis=0))
        duplicate_ratio = 1.0 - (unique_obs / len(obs_matrix))
        
        stats = {
            'num_observations': len(obs_arrays),
            'obs_shape': obs_arrays[0].shape,
            'obs_dim': obs_arrays[0].size,
            'sparsity': float(sparsity),
            'mean_values': obs_mean.tolist(),
            'std_values': obs_std.tolist(),
            'value_range': {
                'min': float(obs_min.min()),
                'max': float(obs_max.max())
            },
            'diversity': {
                'mean_pairwise_distance': float(mean_distance),
                'std_pairwise_distance': float(std_distance),
                'min_pairwise_distance': float(min_distance),
                'max_pairwise_distance': float(max_distance)
            },
            'uniqueness': {
                'unique_observations': unique_obs,
                'duplicate_ratio': float(duplicate_ratio)
            }
        }
        
        self.stats['observation_space'] = stats
        return stats
    
    def analyze_action_distribution(self) -> Dict[str, Any]:
        """Analyze action distribution and patterns."""
        print("Analyzing action distribution...")
        
        action_indices = self.searcher.action_indices
        action_counts = defaultdict(int)
        for action in action_indices:
            action_counts[action] += 1
        
        total = len(action_indices)
        action_probs = {k: v / total for k, v in action_counts.items()}
        
        # Action names
        action_names = {
            0: 'turn_left', 1: 'turn_right', 2: 'left', 3: 'right',
            4: 'up', 5: 'down', 6: 'stay'
        }
        
        # Entropy (diversity measure)
        probs = np.array(list(action_probs.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(action_probs))
        normalized_entropy = entropy / max_entropy
        
        stats = {
            'total_actions': total,
            'action_counts': {action_names.get(k, k): v for k, v in action_counts.items()},
            'action_probabilities': {action_names.get(k, k): v for k, v in action_probs.items()},
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(normalized_entropy),
            'num_unique_actions': len(action_probs)
        }
        
        self.stats['action_distribution'] = stats
        return stats
    
    def analyze_obs_action_coupling(self) -> Dict[str, Any]:
        """Analyze coupling between observations and actions."""
        print("Analyzing observation-action coupling...")
        
        # Group observations by action
        obs_by_action = defaultdict(list)
        for i, (obs, action) in enumerate(self.searcher.obs_action_keys):
            obs_by_action[action].append(obs.flatten())
        
        # Compute mean observation for each action
        mean_obs_by_action = {}
        for action, obs_list in obs_by_action.items():
            mean_obs_by_action[action] = np.mean(obs_list, axis=0)
        
        # Compute pairwise distances between action-specific mean observations
        actions = sorted(obs_by_action.keys())
        if len(actions) > 1:
            mean_obs_matrix = np.array([mean_obs_by_action[a] for a in actions])
            action_distances = pdist(mean_obs_matrix, metric='euclidean')
            mean_action_distance = np.mean(action_distances)
        else:
            mean_action_distance = 0.0
        
        # Check if same observation leads to different actions
        obs_to_actions = defaultdict(set)
        for obs, action in self.searcher.obs_action_keys:
            obs_key = tuple(obs.flatten().round(decimals=3))  # Round for grouping
            obs_to_actions[obs_key].add(action)
        
        ambiguous_obs = sum(1 for actions in obs_to_actions.values() if len(actions) > 1)
        ambiguous_ratio = ambiguous_obs / len(obs_to_actions) if obs_to_actions else 0.0
        
        stats = {
            'observations_per_action': {k: len(v) for k, v in obs_by_action.items()},
            'mean_action_distance': float(mean_action_distance),
            'ambiguous_observations': ambiguous_obs,
            'ambiguous_ratio': float(ambiguous_ratio),
            'total_unique_obs_keys': len(obs_to_actions)
        }
        
        self.stats['obs_action_coupling'] = stats
        return stats
    
    def analyze_similarity_structure(self, sample_size: int = 500) -> Dict[str, Any]:
        """Analyze similarity structure in the dataset."""
        print("Analyzing similarity structure...")
        
        obs_arrays = self.searcher.obs_arrays
        if len(obs_arrays) == 0:
            return {}
        
        # Sample for efficiency
        sample_size = min(sample_size, len(obs_arrays))
        sample_indices = np.random.choice(len(obs_arrays), sample_size, replace=False)
        obs_sample = [obs_arrays[i] for i in sample_indices]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(obs_sample)):
            for j in range(i + 1, len(obs_sample)):
                sim = self.searcher._cosine_similarity(obs_sample[i], obs_sample[j])
                similarities.append(sim)
        
        similarities = np.array(similarities)
        
        stats = {
            'sample_size': sample_size,
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities))
        }
        
        self.stats['similarity_structure'] = stats
        return stats
    
    def identify_issues(self) -> Dict[str, List[str]]:
        """Identify potential issues with the representations."""
        print("Identifying potential issues...")
        
        issues = {
            'critical': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check observation sparsity
        if 'observation_space' in self.stats:
            sparsity = self.stats['observation_space']['sparsity']
            if sparsity > 0.9:
                issues['critical'].append(f"Very high observation sparsity ({sparsity:.2%}) - most values are zero")
            elif sparsity > 0.7:
                issues['warnings'].append(f"High observation sparsity ({sparsity:.2%})")
        
        # Check observation diversity
        if 'observation_space' in self.stats:
            dup_ratio = self.stats['observation_space']['uniqueness']['duplicate_ratio']
            if dup_ratio > 0.5:
                issues['critical'].append(f"High duplicate observation ratio ({dup_ratio:.2%}) - limited diversity")
            elif dup_ratio > 0.2:
                issues['warnings'].append(f"Moderate duplicate observation ratio ({dup_ratio:.2%})")
        
        # Check action distribution
        if 'action_distribution' in self.stats:
            norm_entropy = self.stats['action_distribution']['normalized_entropy']
            if norm_entropy < 0.3:
                issues['warnings'].append(f"Low action diversity (entropy: {norm_entropy:.2f}) - some actions rarely used")
        
        # Check observation-action coupling
        if 'obs_action_coupling' in self.stats:
            ambiguous_ratio = self.stats['obs_action_coupling']['ambiguous_ratio']
            if ambiguous_ratio > 0.3:
                issues['warnings'].append(f"High ambiguous observation ratio ({ambiguous_ratio:.2%}) - same obs can lead to different actions")
        
        # Check similarity structure
        if 'similarity_structure' in self.stats:
            mean_sim = self.stats['similarity_structure']['mean_similarity']
            if mean_sim > 0.95:
                issues['warnings'].append(f"Very high mean similarity ({mean_sim:.3f}) - observations are too similar")
            elif mean_sim < 0.1:
                issues['warnings'].append(f"Very low mean similarity ({mean_sim:.3f}) - observations are too diverse")
        
        self.stats['issues'] = issues
        return issues
    
    def propose_solutions(self) -> Dict[str, List[str]]:
        """Propose solutions for identified issues."""
        print("Proposing solutions...")
        
        solutions = {
            'observation_improvements': [],
            'action_improvements': [],
            'similarity_improvements': [],
            'general_improvements': []
        }
        
        # Observation improvements
        solutions['observation_improvements'].extend([
            "1. Use observation normalization (L2 or min-max) to improve similarity search",
            "2. Apply PCA or autoencoder to reduce observation dimensionality while preserving information",
            "3. Use attention mechanisms to focus on relevant parts of observations",
            "4. Extract spatial features (e.g., relative positions, distances) as additional features",
            "5. Use data augmentation (rotations, translations) to increase observation diversity"
        ])
        
        # Action improvements
        solutions['action_improvements'].extend([
            "1. Use action embeddings instead of discrete indices for better similarity",
            "2. Consider action sequences/history for context-aware matching",
            "3. Use action-value estimates to weight action importance",
            "4. Group similar actions (e.g., movement actions) for hierarchical matching"
        ])
        
        # Similarity improvements
        solutions['similarity_improvements'].extend([
            "1. Use learned similarity metrics (e.g., Siamese networks) instead of fixed metrics",
            "2. Combine multiple similarity metrics (cosine, L2, learned) for robust matching",
            "3. Use FAISS for efficient approximate nearest neighbor search",
            "4. Implement locality-sensitive hashing (LSH) for very large datasets",
            "5. Use graph-based similarity (e.g., k-NN graph) for structured search"
        ])
        
        # General improvements
        solutions['general_improvements'].extend([
            "1. Use contrastive learning to learn better (obs, action) representations",
            "2. Implement active learning to collect diverse trajectories",
            "3. Use curriculum learning to gradually increase task difficulty",
            "4. Combine multiple game types in a unified representation space",
            "5. Use meta-learning to adapt similarity metrics to specific tasks"
        ])
        
        self.stats['solutions'] = solutions
        return solutions
    
    def visualize(self, output_dir: Path):
        """Create visualization plots."""
        print("Creating visualizations...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Action distribution
        if 'action_distribution' in self.stats:
            action_probs = self.stats['action_distribution']['action_probabilities']
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(actions, probs)
            plt.xlabel('Action')
            plt.ylabel('Probability')
            plt.title('Action Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'action_distribution.png')
            plt.close()
        
        # 2. Observation similarity distribution
        if 'similarity_structure' in self.stats:
            # This would require storing similarity values, simplified for now
            pass
        
        # 3. PCA visualization of observations
        if len(self.searcher.obs_arrays) > 0:
            obs_matrix = np.array([obs.flatten() for obs in self.searcher.obs_arrays])
            
            # Sample for visualization
            sample_size = min(1000, len(obs_matrix))
            sample_indices = np.random.choice(len(obs_matrix), sample_size, replace=False)
            obs_sample = obs_matrix[sample_indices]
            actions_sample = [self.searcher.action_indices[i] for i in sample_indices]
            
            # PCA
            pca = PCA(n_components=2)
            obs_pca = pca.fit_transform(obs_sample)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(obs_pca[:, 0], obs_pca[:, 1], c=actions_sample, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Action')
            plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
            plt.title('Observation Space (PCA)')
            plt.tight_layout()
            plt.savefig(output_dir / 'observation_pca.png')
            plt.close()
    
    def run_full_evaluation(self, output_dir: str):
        """Run complete evaluation pipeline."""
        print("=" * 80)
        print("Running Full Representation Evaluation")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run analyses
        self.analyze_observation_space()
        self.analyze_action_distribution()
        self.analyze_obs_action_coupling()
        self.analyze_similarity_structure()
        
        # Identify issues and propose solutions
        self.identify_issues()
        self.propose_solutions()
        
        # Create visualizations
        self.visualize(output_path)
        
        # Save statistics
        stats_path = output_path / 'evaluation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"\nSaved evaluation statistics to {stats_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        
        if 'issues' in self.stats:
            issues = self.stats['issues']
            if issues['critical']:
                print("\nCritical Issues:")
                for issue in issues['critical']:
                    print(f"  - {issue}")
            
            if issues['warnings']:
                print("\nWarnings:")
                for warning in issues['warnings']:
                    print(f"  - {warning}")
        
        if 'solutions' in self.stats:
            solutions = self.stats['solutions']
            print("\nProposed Solutions:")
            for category, sols in solutions.items():
                print(f"\n{category.replace('_', ' ').title()}:")
                for sol in sols[:3]:  # Show first 3
                    print(f"  {sol}")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate observation and action representations")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset pickle file'
    )
    parser.add_argument(
        '--faiss-index',
        type=str,
        help='Path to FAISS index file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    searcher = SimilaritySearcher.load(args.dataset, args.faiss_index)
    
    # Run evaluation
    evaluator = RepresentationEvaluator(searcher)
    stats = evaluator.run_full_evaluation(args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

