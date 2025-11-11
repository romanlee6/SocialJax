# LLM Benchmark for Coins Game

This directory contains tools for benchmarking different Large Language Models (LLMs) on the Coins Game environment.

## Files

- `coins_llm_simulation.py`: Core simulation code for LLM agents in the Coins Game
- `benchmark_llms.py`: Benchmark script to compare multiple LLMs
- `BENCHMARK_README.md`: This file

## Quick Start

### Prerequisites

1. Set your OpenAI API key (or compatible API):
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, for custom endpoints
```

2. Install dependencies (if not already installed):
```bash
pip install openai jax jaxlib numpy matplotlib pillow
```

### Running the Benchmark

**Basic usage** (runs default models with default settings):
```bash
python benchmark_llms.py
```

This will test the following models by default:
- `gpt-5`
- `gpt-5-mini`
- `o3`
- `o4-mini`
- `gpt-oss-120b`

**Custom configuration:**
```bash
python benchmark_llms.py --models gpt-5 gpt-5-mini o3 --steps 30 --seed 123
```

**All options:**
```bash
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini o3 o4-mini \
  --steps 20 \
  --seed 42 \
  --temperature 0.7 \
  --output-dir ./llm_benchmarks
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | `gpt-5 gpt-5-mini o3 o4-mini gpt-oss-120b` | List of model names to benchmark |
| `--steps` | `20` | Number of simulation steps per model |
| `--seed` | `42` | Environment seed (fixed for fair comparison) |
| `--temperature` | `0.7` | Sampling temperature for LLMs (0.0-2.0) |
| `--output-dir` | `./llm_benchmarks` | Base directory for all outputs |

## Output Structure

The benchmark creates a timestamped directory structure:

```
llm_benchmarks/
└── benchmark_YYYY-MM-DD_HH-MM-SS/
    ├── benchmark_detailed.json      # Complete results with all metrics
    ├── benchmark_summary.txt         # Human-readable summary report
    ├── benchmark_table.csv           # Comparison table (Excel/Sheets compatible)
    ├── gpt-5_seed42/
    │   ├── trajectory_log.json       # Detailed trajectory for this model
    │   ├── timestep_0000.png         # Visualizations (every 5th frame)
    │   ├── timestep_0005.png
    │   └── simulation.gif            # Animated GIF of the run
    ├── gpt-5-mini_seed42/
    │   └── ...
    └── ...
```

## Metrics Tracked

### Performance Metrics
- **Episode Length**: Number of steps completed
- **Total Reward**: Cumulative reward per agent
- **Average Reward**: Reward per step per agent
- **Communication Count**: Number of messages sent by each agent
- **Action Distribution**: Breakdown of actions taken

### Efficiency Metrics
- **Total Time**: Wall-clock time for entire simulation
- **Time per Step**: Average time per simulation step
- **Total Input Tokens**: Sum of input tokens across all API calls
- **Total Output Tokens**: Sum of output tokens across all API calls
- **Average Tokens per Step**: Token usage normalized by episode length

## Understanding Results

### Benchmark Summary (benchmark_summary.txt)

A human-readable report containing:
- Overview of all tested models
- Performance metrics for each model
- Token usage statistics
- Action distributions
- Timing information

### Benchmark Table (benchmark_table.csv)

A CSV file that can be imported into Excel or Google Sheets for further analysis:
- One row per model
- Columns: Model, Status, Time, Rewards, Token usage
- Easy to create charts and perform statistical analysis

### Benchmark Detailed (benchmark_detailed.json)

Complete JSON output with all metrics, suitable for:
- Programmatic analysis
- Integration with other tools
- Reproducibility and archival

## Example Workflow

1. **Run benchmark with default settings:**
   ```bash
   python benchmark_llms.py
   ```

2. **Check the console output** for real-time progress and final comparison table

3. **Review the summary:**
   ```bash
   cat llm_benchmarks/benchmark_*/benchmark_summary.txt
   ```

4. **Analyze the comparison table:**
   - Open `benchmark_table.csv` in Excel/Sheets
   - Create charts comparing time, rewards, token usage

5. **Inspect individual runs:**
   - View GIF animations: `simulation.gif` in each model directory
   - Read detailed trajectory logs: `trajectory_log.json`

## Comparing Models

The benchmark uses the **same environment seed** across all models to ensure fair comparison. This means:
- Initial positions are identical
- Coin spawn patterns are identical
- Any randomness is controlled

### Key Comparison Dimensions

1. **Performance**: Which model achieves higher rewards?
2. **Efficiency**: Which model uses fewer tokens or runs faster?
3. **Behavior**: How do action distributions differ?
4. **Communication**: Which models communicate more/less?

## Tips for Effective Benchmarking

1. **Run multiple seeds:** Use different `--seed` values to test robustness
2. **Vary episode length:** Test with different `--steps` values (10, 20, 50, 100)
3. **Temperature sweep:** Try different temperatures (0.3, 0.7, 1.0) to see impact on behavior
4. **Cost analysis:** Use token counts to estimate API costs (check your provider's pricing)

## Troubleshooting

### API Key Issues
```bash
# Verify your API key is set
echo $OPENAI_API_KEY

# If using custom endpoint, set base URL
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
```

### Model Not Found
If a model fails with "model not found", either:
- The model name is incorrect (check your API provider's documentation)
- You don't have access to that model
- Comment out that model in the `--models` argument

### Memory Issues
For longer runs, reduce:
- `--steps`: Fewer simulation steps
- Visualization frequency: Edit `benchmark_llms.py` line with `if t % 5 == 0`

### Slow Execution
- Use faster models for testing: `--models gpt-5-mini`
- Reduce steps: `--steps 10`
- Increase temperature for less compute-intensive outputs: `--temperature 1.0`

## Advanced Usage

### Custom Model Lists
```bash
# Test only fast models
python benchmark_llms.py --models gpt-5-mini o4-mini

# Test a single model
python benchmark_llms.py --models gpt-5
```

### Batch Processing
```bash
# Run benchmarks for multiple seeds
for seed in 42 123 456 789; do
  python benchmark_llms.py --seed $seed --output-dir ./benchmarks_seed_$seed
done
```

### Integration with Experiments
```python
from benchmark_llms import BenchmarkRunner

runner = BenchmarkRunner(
    models=["gpt-5", "gpt-5-mini"],
    num_steps=20,
    seed=42,
    base_output_dir="./my_experiment",
    temperature=0.7
)
runner.run_benchmark()
```

## Citation

If you use this benchmark in your research, please cite:
```bibtex
@software{socialjax_llm_benchmark,
  title={LLM Benchmark for Multi-Agent Social Dilemmas},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/SocialJax}
}
```

## License

See LICENSE file in the repository root.

