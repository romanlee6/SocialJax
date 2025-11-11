# LLM Agents for Multi-Agent Social Dilemmas

This directory contains implementations and benchmarking tools for Large Language Model (LLM) agents playing multi-agent social dilemma games in the SocialJax environment.

## 📁 Directory Contents

### Core Simulation
- **`coins_llm_simulation.py`** - Main simulation framework for LLM agents in the Coins Game
  - Observation-to-language translation
  - LLM-based agent decision making
  - Communication between agents
  - Trajectory logging and visualization

### Benchmarking Tools
- **`benchmark_llms.py`** - Benchmark multiple LLMs for comparison
- **`analyze_benchmark_results.py`** - Analyze and visualize benchmark results
- **`test_benchmark_setup.py`** - Test environment setup before running benchmarks
- **`run_benchmark.sh`** - Shell script for easy benchmark execution

### Documentation
- **`QUICKSTART.md`** - ⭐ Start here! 5-minute quick start guide
- **`BENCHMARK_README.md`** - Detailed benchmark documentation
- **`LOGGING_INFO.md`** - Information about logging and debugging

### Other Files
- `coins_llm_mock.py` - Mock implementation for testing
- `example_usage.py` - Example usage patterns
- `add_debug_logging.py` - Debug utilities
- `DEBUG_TOOLS_SUMMARY.txt` - Summary of debugging tools

## 🚀 Quick Start

### 1. First Time Setup

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Test the setup (recommended)
python test_benchmark_setup.py
```

### 2. Run a Simple Simulation

```bash
# Single model, 20 steps
python coins_llm_simulation.py --model gpt-5-mini --steps 20 --seed 42
```

### 3. Run a Benchmark Comparison

```bash
# Compare multiple models
./run_benchmark.sh

# Or with Python
python benchmark_llms.py --models gpt-5 gpt-5-mini o3 --steps 20
```

### 4. Analyze Results

```bash
# Auto-analyze the most recent benchmark
python analyze_benchmark_results.py --auto
```

## 📖 Documentation

- **New to LLM benchmarking?** → Read [`QUICKSTART.md`](QUICKSTART.md)
- **Want detailed benchmark docs?** → Read [`BENCHMARK_README.md`](BENCHMARK_README.md)
- **Need to debug?** → Check [`LOGGING_INFO.md`](LOGGING_INFO.md)

## 🎯 What Can You Do?

### 1. Single Agent Simulation
Run LLM agents in the Coins Game with natural language observations and actions:
```bash
python coins_llm_simulation.py --model gpt-5 --steps 50 --seed 42
```

Outputs:
- Timestep-by-timestep logs
- Visualizations (PNG frames + GIF animation)
- Trajectory data (JSON format)
- Performance statistics

### 2. Model Comparison Benchmark
Compare multiple LLMs on the same environment:
```bash
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini o3 o4-mini gpt-oss-120b \
  --steps 20 \
  --seed 42
```

Tracks:
- ✅ Performance (rewards, episode length)
- ⏱️ Time efficiency (seconds per step)
- 🎫 Token usage (input/output tokens)
- 💬 Communication patterns
- 🎮 Action distributions

Outputs:
- Individual logs per model
- Aggregate benchmark summary
- Comparison tables (TXT, CSV, JSON)
- Visualizations

### 3. Results Analysis
Generate visualizations and insights:
```bash
python analyze_benchmark_results.py --auto
```

Creates:
- Performance comparison charts
- Efficiency analysis plots
- Trade-off analysis (reward vs time/tokens)
- Summary statistics

## 📊 Example Workflow

```bash
# Step 1: Test your setup
python test_benchmark_setup.py

# Step 2: Run a quick test (fast model, few steps)
python benchmark_llms.py --models gpt-5-mini --steps 10

# Step 3: Run full benchmark
python benchmark_llms.py --steps 20

# Step 4: Analyze results
python analyze_benchmark_results.py --auto

# Step 5: View the summary
cat llm_benchmarks/benchmark_*/benchmark_summary.txt
```

## 🏗️ Architecture

### Observation → Language
The `ObservationDescriptor` class converts grid-based observations to natural language:
- Agent position and orientation
- Visible coins (with coordinates and relative positions)
- Other agents in field of view
- Wall locations by direction

### LLM Decision Making
The `LLMAgent` class:
- Maintains belief state
- Receives observations in natural language
- Can communicate with other agents
- Generates actions via LLM reasoning
- Uses OpenAI API (or compatible endpoints)

### Communication
Agents can send messages to coordinate, negotiate, or influence behavior.

### Logging and Reproducibility
Three types of logs:
1. **Raw trajectory**: Full LLM inputs/outputs and API responses
2. **Parsed trajectory**: Structured data for ML training
3. **Human-readable**: Debug logs with interpretable information

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
```

### Command-line Options

**For single simulations** (`coins_llm_simulation.py`):
- `--steps N`: Number of timesteps (default: 20)
- `--model MODEL`: Model name (default: gpt-5-mini)
- `--temperature T`: Sampling temperature (default: 0.7)
- `--seed S`: Random seed (default: 42)
- `--output-dir DIR`: Output directory

**For benchmarks** (`benchmark_llms.py`):
- `--models M1 M2 ...`: List of models to test
- `--steps N`: Number of timesteps (default: 20)
- `--seed S`: Random seed (default: 42)
- `--temperature T`: Sampling temperature (default: 0.7)
- `--output-dir DIR`: Output directory (default: ./llm_benchmarks)

## 📈 Supported Models

The benchmark is designed to work with any model compatible with the OpenAI API format. Default models include:
- `gpt-5`
- `gpt-5-mini`
- `o3`
- `o4-mini`
- `gpt-oss-120b`

You can test any model your API endpoint supports.

## 🎓 Research Use Cases

1. **Multi-agent cooperation**: Study how LLMs coordinate in social dilemmas
2. **Communication emergence**: Analyze agent communication patterns
3. **Model comparison**: Compare reasoning capabilities across LLMs
4. **Efficiency analysis**: Evaluate performance vs computational cost trade-offs
5. **Behavior analysis**: Study decision-making in complex multi-agent scenarios
6. **Training data generation**: Create datasets for imitation learning

## 🐛 Troubleshooting

**Problem**: API key not recognized
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your-key-here"
echo $OPENAI_API_KEY  # Verify it's set
```

**Problem**: Import errors for socialjax
```bash
# Solution: Add to Python path
export PYTHONPATH=/home/huao/Research/SocialJax:$PYTHONPATH
```

**Problem**: Slow execution
```bash
# Solution: Use fewer steps or faster model
python benchmark_llms.py --models gpt-5-mini --steps 10
```

**Problem**: Out of memory
- Reduce episode length (`--steps`)
- Reduce visualization frequency (edit code)
- Use fewer models

For more troubleshooting, see [`QUICKSTART.md`](QUICKSTART.md).

## 📝 Output Files

### From Single Simulation
```
llm_simulation_output/
  model_tempT_seedS_TIMESTAMP/
    ├── trajectory_raw.json           # Full LLM I/O
    ├── trajectory_parsed.json        # Structured data
    ├── debug_human_readable.json     # Debug info
    ├── trajectory_stats.json         # Statistics
    ├── timestep_log.txt              # Human-readable log
    ├── timestep_0000.png             # Visualizations
    ├── timestep_0001.png
    └── simulation.gif                # Animation
```

### From Benchmark
```
llm_benchmarks/
  benchmark_TIMESTAMP/
    ├── benchmark_detailed.json       # Complete results
    ├── benchmark_summary.txt         # Human-readable summary
    ├── benchmark_table.csv           # Spreadsheet format
    ├── performance_comparison.png    # Analysis plots
    ├── efficiency_comparison.png
    ├── tradeoff_analysis.png
    ├── model1_seed42/
    │   └── [individual simulation outputs]
    └── model2_seed42/
        └── [individual simulation outputs]
```

## 🤝 Contributing

To add new features:
1. **New environments**: Adapt `ObservationDescriptor` for your environment
2. **New metrics**: Extend `BenchmarkLogger` with your metrics
3. **New visualizations**: Add plots to `analyze_benchmark_results.py`
4. **New models**: Just add model name to `--models` argument

## 📚 Related

- Main SocialJax repository: `../`
- Environment implementations: `../socialjax/environments/`
- RL algorithms: `../algorithms/`

## 📧 Contact

For questions or issues, please open an issue in the repository or contact the maintainers.

---

**Status**: Active development
**Last Updated**: November 2025
