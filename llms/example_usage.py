#!/usr/bin/env python3
"""
Example usage of the LLM dataset builder and similarity search system.

This script demonstrates:
1. Building a dataset from LLM trajectories
2. Searching for similar (obs, action) pairs
3. Evaluating representations
4. Using the results for downstream tasks

Usage:
    python example_usage.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llms.build_llm_dataset import LLMDatasetBuilder
from llms.similarity_search import SimilaritySearcher
from llms.evaluate_representations import RepresentationEvaluator


def example_build_dataset():
    """Example: Build dataset from LLM trajectories."""
    print("=" * 80)
    print("Example 1: Building Dataset from LLM Trajectories")
    print("=" * 80)
    
    # Initialize builder
    builder = LLMDatasetBuilder(
        embedding_model="all-MiniLM-L6-v2",
        game_type="coins",
        obs_normalization="l2"  # Use L2 normalization for better similarity search
    )
    
    # Build dataset
    input_dir = "llms/llm_simulation_output"
    stats = builder.build_dataset(input_dir, pattern="trajectory_parsed.json")
    
    # Save dataset
    output_path = "llms/llm_datasets/llm_dataset_coins_l2.pkl"
    builder.save_dataset(output_path, save_faiss_index=True)
    
    print(f"\nDataset saved to {output_path}")
    return output_path


def example_similarity_search(dataset_path: str):
    """Example: Search for similar (obs, action) pairs."""
    print("\n" + "=" * 80)
    print("Example 2: Similarity Search")
    print("=" * 80)
    
    # Load searcher
    dataset_path_obj = Path(dataset_path)
    faiss_index_path = dataset_path_obj.parent / f"{dataset_path_obj.stem}_faiss.index"
    
    searcher = SimilaritySearcher.load(
        dataset_path,
        str(faiss_index_path) if faiss_index_path.exists() else None
    )
    
    # Get a sample observation from the dataset
    if searcher.obs_arrays:
        sample_obs = searcher.obs_arrays[0]
        sample_action = searcher.action_indices[0]
        
        print(f"\nQuery observation shape: {sample_obs.shape}")
        print(f"Query action: {sample_action}")
        
        # Search using different metrics
        print("\n--- Cosine Similarity Search ---")
        results = searcher.search(
            sample_obs,
            query_action=None,
            top_k=5,
            metric="cosine"
        )
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Action: {result['action']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            if result['entries']:
                entry = result['entries'][0]
                print(f"  Belief: {entry['belief_text'][:80]}...")
                print(f"  Communication: {entry['communication_text'][:80]}...")
        
        # Search with action filtering
        print("\n--- Cosine Similarity Search (Filtered by Action) ---")
        results_filtered = searcher.search(
            sample_obs,
            query_action=sample_action,
            top_k=3,
            metric="cosine",
            filter_action=True
        )
        
        print(f"Found {len(results_filtered)} results with action {sample_action}")
        
        # FAISS search (if available)
        if searcher.faiss_index:
            print("\n--- FAISS Fast Search ---")
            results_faiss = searcher.search(
                sample_obs,
                query_action=None,
                top_k=5,
                metric="faiss"
            )
            print(f"Found {len(results_faiss)} results using FAISS")


def example_belief_search(dataset_path: str):
    """Example: Search by belief similarity."""
    print("\n" + "=" * 80)
    print("Example 3: Belief-based Search")
    print("=" * 80)
    
    # Load searcher
    searcher = SimilaritySearcher.load(dataset_path)
    
    # Load embedding model (same as used in dataset building)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Search by belief
    query_belief = "I should explore the map to find coins"
    print(f"\nQuery belief: {query_belief}")
    
    results = searcher.search_by_belief(
        query_belief,
        embedding_model,
        top_k=3
    )
    
    print(f"\nFound {len(results)} similar beliefs:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Belief similarity: {result['belief_similarity']:.4f}")
        print(f"  Action: {result['action']}")
        print(f"  Belief text: {result['entry']['belief_text'][:100]}...")
        print(f"  Communication: {result['entry']['communication_text'][:100]}...")


def example_evaluation(dataset_path: str):
    """Example: Evaluate representations."""
    print("\n" + "=" * 80)
    print("Example 4: Representation Evaluation")
    print("=" * 80)
    
    # Load searcher
    searcher = SimilaritySearcher.load(dataset_path)
    
    # Run evaluation
    evaluator = RepresentationEvaluator(searcher)
    stats = evaluator.run_full_evaluation("evaluation_results")
    
    # Print key findings
    print("\nKey Findings:")
    if 'observation_space' in stats:
        obs_stats = stats['observation_space']
        print(f"  Observation sparsity: {obs_stats['sparsity']:.2%}")
        print(f"  Duplicate ratio: {obs_stats['uniqueness']['duplicate_ratio']:.2%}")
    
    if 'action_distribution' in stats:
        action_stats = stats['action_distribution']
        print(f"  Action entropy: {action_stats['normalized_entropy']:.3f}")
        print(f"  Most common action: {max(action_stats['action_probabilities'], key=action_stats['action_probabilities'].get)}")
    
    if 'issues' in stats:
        issues = stats['issues']
        print(f"\n  Critical issues: {len(issues['critical'])}")
        print(f"  Warnings: {len(issues['warnings'])}")


def example_compare_normalizations():
    """Example: Compare different normalization methods."""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Normalization Methods")
    print("=" * 80)
    
    input_dir = "llms/llm_simulation_output"
    normalization_methods = ["none", "l2", "minmax"]
    
    for norm_method in normalization_methods:
        print(f"\n--- Testing {norm_method} normalization ---")
        
        builder = LLMDatasetBuilder(
            embedding_model="all-MiniLM-L6-v2",
            game_type="coins",
            obs_normalization=norm_method
        )
        
        stats = builder.build_dataset(input_dir, pattern="trajectory_parsed.json")
        
        # Quick similarity test
        if builder.obs_arrays:
            sample_obs = builder.obs_arrays[0]
            searcher = SimilaritySearcher(
                builder.dataset,
                builder.obs_arrays,
                builder.action_indices,
                builder.obs_action_keys,
                builder.embedding_dim,
                builder.game_type,
                norm_method
            )
            
            results = searcher.search(sample_obs, top_k=3, metric="cosine")
            print(f"  Top similarity: {results[0]['similarity']:.4f}")


def main():
    """Run all examples."""
    print("LLM Dataset and Similarity Search - Example Usage")
    print("=" * 80)
    
    # Check if dataset exists, if not build it
    dataset_path = "llms/llm_datasets/llm_dataset_coins_l2.pkl"
    
    if not Path(dataset_path).exists():
        print("Dataset not found. Building dataset...")
        dataset_path = example_build_dataset()
    else:
        print(f"Using existing dataset: {dataset_path}")
    
    # Run examples
    example_similarity_search(dataset_path)
    example_belief_search(dataset_path)
    example_evaluation(dataset_path)
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

