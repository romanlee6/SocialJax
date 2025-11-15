# LLM Trajectory Dataset and Similarity Search System

This system constructs a searchable dataset from LLM trajectories, where (observation, action) pairs serve as keys and embedded beliefs/communications serve as values. It includes evaluation tools and proposed solutions for representation improvements.

## Overview

The system consists of three main components:

1. **Dataset Builder** (`build_llm_dataset.py`): Extracts (obs, action) pairs from LLM trajectories and embeds beliefs/communications
2. **Similarity Search** (`similarity_search.py`): Implements efficient similarity search with multiple metrics
3. **Representation Evaluator** (`evaluate_representations.py`): Analyzes obs/action representations and proposes improvements

## Installation

Install required dependencies:

```bash
pip install sentence-transformers faiss-cpu scikit-learn
```

## Quick Start

### 1. Build Dataset from LLM Trajectories

```bash
python llms/build_llm_dataset.py \
    --input-dir llms/llm_simulation_output \
    --output-dir llms/llm_datasets \
    --embedding-model all-MiniLM-L6-v2 \
    --game-type coins \
    --obs-normalization l2
```

This will:
- Process all `trajectory_parsed.json` files in the input directory
- Extract (observation, action) pairs
- Embed beliefs and communications using sentence transformers
- Save dataset as pickle file
- Build FAISS index for fast similarity search

### 2. Search for Similar (Obs, Action) Pairs

```python
from llms.similarity_search import SimilaritySearcher

# Load dataset
searcher = SimilaritySearcher.load(
    'llms/llm_datasets/llm_dataset_coins_l2.pkl',
    'llms/llm_datasets/llm_dataset_coins_l2_faiss.index'
)

# Search
results = searcher.search(
    query_obs=observation_array,
    query_action=None,  # Optional: filter by action
    top_k=5,
    metric='cosine'  # or 'l2', 'manhattan', 'faiss'
)

# Access results
for result in results:
    print(f"Action: {result['action']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Belief: {result['entries'][0]['belief_text']}")
    print(f"Communication: {result['entries'][0]['communication_text']}")
```

### 3. Evaluate Representations

```bash
python llms/evaluate_representations.py \
    --dataset llms/llm_datasets/llm_dataset_coins_l2.pkl \
    --output-dir evaluation_results
```

This generates:
- Statistics on observation space coverage and diversity
- Action distribution analysis
- Observation-action coupling analysis
- Similarity structure analysis
- Identified issues and proposed solutions
- Visualization plots

## Dataset Structure

The dataset stores:

**Key**: `(observation_array, action_idx)` tuple
- `observation_array`: Normalized CNN observation (e.g., 11x11x14 for coins)
- `action_idx`: Discrete action index (0-6)

**Value**: List of entries, each containing:
- `belief_embedding`: High-dimensional vector (384-dim for all-MiniLM-L6-v2)
- `communication_embedding`: High-dimensional vector
- `belief_text`: Original belief text
- `communication_text`: Original communication text
- `metadata`: Additional info (agent_id, timestep, trajectory_file, etc.)

## Similarity Metrics

The system supports multiple similarity metrics:

1. **Cosine Similarity**: Measures angle between observation vectors
   - Best for normalized observations
   - Range: [-1, 1], higher is more similar

2. **L2/Euclidean Distance**: Measures straight-line distance
   - Good for absolute value comparisons
   - Lower is more similar

3. **Manhattan Distance**: Measures sum of absolute differences
   - Robust to outliers
   - Lower is more similar

4. **FAISS**: Fast approximate nearest neighbor search
   - Uses L2 distance with optimized indexing
   - Best for large datasets (>10K entries)

## Evaluation Results and Proposed Solutions

### Common Issues Identified

1. **High Observation Sparsity**
   - Problem: Most observation values are zero
   - Impact: Reduces information content, makes similarity search less meaningful
   - Solution: Use attention mechanisms or feature extraction to focus on non-zero regions

2. **Low Observation Diversity**
   - Problem: Many duplicate observations
   - Impact: Limited coverage of state space
   - Solution: Data augmentation (rotations, translations), active learning

3. **Action Distribution Skew**
   - Problem: Some actions rarely used
   - Impact: Limited training data for certain actions
   - Solution: Balanced sampling, action-value weighting

4. **Observation-Action Ambiguity**
   - Problem: Same observation can lead to different actions
   - Impact: Difficulty in learning deterministic policies
   - Solution: Include action history/context, use belief embeddings

### Proposed Solutions

#### Observation Improvements

1. **Normalization**: Use L2 or min-max normalization to improve similarity search
   ```bash
   --obs-normalization l2  # or minmax
   ```

2. **Dimensionality Reduction**: Apply PCA or autoencoders to reduce noise
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=50)
   obs_reduced = pca.fit_transform(obs_flat)
   ```

3. **Feature Extraction**: Extract spatial features (distances, relative positions)
   ```python
   # Extract relative positions, distances to objects
   # Add as additional channels to observation
   ```

4. **Attention Mechanisms**: Focus on relevant parts of observations
   ```python
   # Use attention to weight important observation regions
   ```

#### Action Improvements

1. **Action Embeddings**: Use learned embeddings instead of discrete indices
   ```python
   # Learn action embeddings that capture action similarity
   action_embedding = action_encoder(action_idx)
   ```

2. **Action Sequences**: Consider action history for context
   ```python
   # Include previous N actions as context
   context = [action_t-N, ..., action_t-1, action_t]
   ```

3. **Action-Value Weighting**: Weight actions by their expected value
   ```python
   # Weight similarity by action-value estimates
   weighted_similarity = similarity * action_value
   ```

#### Similarity Search Improvements

1. **Learned Metrics**: Train Siamese networks for task-specific similarity
   ```python
   # Train network to learn optimal similarity metric
   similarity = siamese_network(obs1, obs2)
   ```

2. **Multi-Metric Fusion**: Combine multiple metrics
   ```python
   # Weighted combination of cosine, L2, learned metrics
   final_similarity = w1*cosine + w2*l2 + w3*learned
   ```

3. **Hierarchical Search**: Use coarse-to-fine search strategy
   ```python
   # First find coarse matches, then refine
   ```

#### General Improvements

1. **Contrastive Learning**: Learn better representations
   ```python
   # Use contrastive loss to learn discriminative features
   ```

2. **Meta-Learning**: Adapt similarity metrics to specific tasks
   ```python
   # Learn to adapt similarity function per task
   ```

3. **Multi-Game Unified Space**: Combine coin and territory games
   ```python
   # Learn shared representation across game types
   ```

## File Structure

```
llms/
├── build_llm_dataset.py      # Dataset builder
├── similarity_search.py      # Similarity search implementation
├── evaluate_representations.py # Representation evaluation
├── example_usage.py           # Usage examples
└── LLM_DATASET_README.md      # This file

llms/llm_datasets/             # Generated datasets
├── llm_dataset_coins_l2.pkl
├── llm_dataset_coins_l2_faiss.index
└── llm_dataset_coins_l2_stats.json

evaluation_results/            # Evaluation outputs
├── evaluation_stats.json
├── action_distribution.png
└── observation_pca.png
```

## Usage Examples

See `example_usage.py` for comprehensive examples:

```bash
python llms/example_usage.py
```

## Performance Considerations

- **Dataset Size**: For datasets >100K entries, use FAISS for fast search
- **Memory**: Large datasets may require chunking or streaming
- **Embedding Model**: `all-MiniLM-L6-v2` is fast but `all-mpnet-base-v2` is more accurate
- **Normalization**: L2 normalization generally works best for similarity search

## Evaluation for Coin and Territory Games

### Coin Game Observations
- Shape: (11, 11, 14) when CNN=True
- Channels: Empty, wall, interact, red_apple, green_apple, agent channels, etc.
- Issues: High sparsity (most cells empty), limited diversity

### Territory Game Observations
- Shape: (11, 11, 13) when CNN=True
- Channels: Empty, wall, resource types, agent channels, etc.
- Issues: Similar to coin game, may have more spatial structure

### Recommendations

1. **For Coin Game**:
   - Use L2 normalization
   - Extract relative positions of coins
   - Focus on agent's field of view
   - Consider action history (cooperation patterns)

2. **For Territory Game**:
   - Use spatial features (distance to resources)
   - Include territory ownership information
   - Consider multi-agent coordination patterns
   - Use graph-based similarity for spatial relationships

## Future Work

1. Implement learned similarity metrics using Siamese networks
2. Add support for action sequence matching
3. Integrate with RL training pipeline
4. Support for multi-game unified representation space
5. Real-time similarity search API

## Troubleshooting

**Issue**: "sentence-transformers not found"
- Solution: `pip install sentence-transformers`

**Issue**: "FAISS not available"
- Solution: `pip install faiss-cpu` (or `faiss-gpu` for GPU)

**Issue**: "Out of memory when building dataset"
- Solution: Process trajectories in batches, use smaller embedding model

**Issue**: "Low similarity scores"
- Solution: Try different normalization methods, check observation preprocessing

## Citation

If you use this system, please cite:
- Sentence Transformers: Reimers & Gurevych, 2019
- FAISS: Johnson et al., 2019

