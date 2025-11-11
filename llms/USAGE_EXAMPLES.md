# Usage Examples - LLM Benchmark

This file contains copy-paste ready examples for common use cases.

## Setup

```bash
# Set API key (required)
export OPENAI_API_KEY="your-api-key-here"

# For custom endpoints (optional)
export OPENAI_BASE_URL="https://your-endpoint.com/v1"

# Navigate to llms directory
cd /home/huao/Research/SocialJax/llms
```

## Example 1: Quick Setup Test

**Goal**: Verify everything is working

```bash
# Run setup test
python test_benchmark_setup.py

# If you see "ALL TESTS PASSED", you're ready!
```

**Expected output**:
```
================================================================================
TESTING BENCHMARK SETUP
================================================================================

1. Checking API key...
   ✓ PASS: API key is set (length: 48)

2. Checking API base URL...
   ℹ Using default OpenAI endpoint

3. Checking dependencies...
   ✓ jax installed
   ✓ numpy installed
   ✓ matplotlib installed
   ✓ pillow installed
   ✓ openai installed

...

✓ ALL TESTS PASSED - Environment is ready!
```

## Example 2: Simple Benchmark (2 models)

**Goal**: Quick comparison of two models

```bash
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini \
  --steps 20 \
  --seed 42
```

**What happens**:
- Runs gpt-5 for 20 steps (seed=42)
- Runs gpt-5-mini for 20 steps (same seed)
- Saves results to `llm_benchmarks/benchmark_[timestamp]/`
- Prints comparison table

**Time**: ~2-3 minutes per model (total ~5-6 minutes)

## Example 3: Full Benchmark (All Default Models)

**Goal**: Comprehensive comparison

```bash
# Using shell script (easiest)
./run_benchmark.sh

# Or using Python directly
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini o3 o4-mini gpt-oss-120b \
  --steps 20 \
  --seed 42 \
  --temperature 0.7
```

**What happens**:
- Tests all 5 default models
- 20 steps each
- Same seed for fair comparison
- Generates comprehensive reports

**Time**: ~2-3 minutes per model (total ~10-15 minutes)

**Output files**:
```
llm_benchmarks/benchmark_2025-11-10_12-30-45/
├── benchmark_detailed.json      ← Complete data
├── benchmark_summary.txt         ← Read this first
├── benchmark_table.csv           ← Open in Excel
├── gpt-5_seed42/
│   ├── trajectory_log.json
│   ├── timestep_*.png
│   └── simulation.gif
└── ... (4 more model directories)
```

## Example 4: Analyze Results

**Goal**: Generate visualizations from benchmark

```bash
# Auto-find most recent benchmark
python analyze_benchmark_results.py --auto

# Or specify directory
python analyze_benchmark_results.py llm_benchmarks/benchmark_2025-11-10_12-30-45
```

**What happens**:
- Loads benchmark results
- Creates 3 PNG visualizations
- Prints summary table

**Output**:
```
performance_comparison.png   ← Bar chart of rewards
efficiency_comparison.png    ← Time and token usage
tradeoff_analysis.png       ← Scatter plots
```

## Example 5: Fast Testing (Single Model, Few Steps)

**Goal**: Quick test with minimal time/cost

```bash
python benchmark_llms.py \
  --models gpt-5-mini \
  --steps 10 \
  --seed 42
```

**Time**: ~30-60 seconds
**Use for**: Testing changes, debugging, development

## Example 6: Longer Episodes

**Goal**: Test behavior over extended interaction

```bash
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini \
  --steps 50 \
  --seed 42
```

**Time**: ~5-8 minutes per model
**Use for**: Studying long-term strategies, emergent behavior

## Example 7: Multiple Seeds (Robustness Test)

**Goal**: Test across different random conditions

```bash
# Run benchmark with 5 different seeds
for seed in 42 123 456 789 999; do
    echo "Running seed $seed..."
    python benchmark_llms.py \
      --models gpt-5 gpt-5-mini \
      --steps 20 \
      --seed $seed \
      --output-dir ./robustness_test
done

# Results in ./robustness_test/benchmark_*/ (5 directories)
```

**Time**: ~5-6 minutes × 5 = ~25-30 minutes total
**Use for**: Testing robustness, statistical analysis

## Example 8: Temperature Sweep

**Goal**: Compare behavior at different temperatures

```bash
for temp in 0.3 0.5 0.7 1.0; do
    echo "Running temperature $temp..."
    python benchmark_llms.py \
      --models gpt-5 \
      --steps 20 \
      --seed 42 \
      --temperature $temp \
      --output-dir ./temp_sweep
done

# Results in ./temp_sweep/benchmark_*/ (4 directories)
```

**Time**: ~2-3 minutes × 4 = ~8-12 minutes total
**Use for**: Understanding temperature effects on behavior

## Example 9: Cost Estimation

**Goal**: Estimate API costs before large experiment

```bash
# Run short benchmark
python benchmark_llms.py \
  --models gpt-5 \
  --steps 5 \
  --seed 42

# View results
cat llm_benchmarks/benchmark_*/benchmark_summary.txt | grep "Token Usage"
```

**Example output**:
```
Token Usage (Total):
  Input Tokens: 3,808
  Output Tokens: 864
  Total Tokens: 4,672
```

**Calculate cost**:
```
For 5 steps: 4,672 tokens
For 20 steps: ~18,688 tokens
For 100 steps: ~93,440 tokens

Cost = tokens × your_api_price_per_token
Example: 93,440 × $0.00002 = $1.87 per 100-step run
```

## Example 10: Custom Output Directory

**Goal**: Organize experiments with descriptive names

```bash
# Experiment 1: Baseline
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini \
  --steps 20 \
  --output-dir ./experiments/baseline_20steps

# Experiment 2: Extended
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini \
  --steps 50 \
  --output-dir ./experiments/extended_50steps

# Results organized:
# ./experiments/baseline_20steps/benchmark_*/
# ./experiments/extended_50steps/benchmark_*/
```

## Example 11: Single Model Deep Dive

**Goal**: Detailed analysis of one model

```bash
# Run single model
python benchmark_llms.py \
  --models gpt-5 \
  --steps 30 \
  --seed 42 \
  --output-dir ./gpt5_analysis

# Find the output directory
RESULT_DIR=$(ls -td ./gpt5_analysis/benchmark_* | head -1)
MODEL_DIR=$(ls -td $RESULT_DIR/gpt-5_* | head -1)

# View detailed trajectory
cat $MODEL_DIR/trajectory_log.json

# View visualization
open $MODEL_DIR/simulation.gif

# Read summary
cat $RESULT_DIR/benchmark_summary.txt
```

## Example 12: Compare Specific Models

**Goal**: Test only the models you care about

```bash
# Compare OpenAI models
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini \
  --steps 20

# Compare reasoning models
python benchmark_llms.py \
  --models o3 o4-mini \
  --steps 20

# Compare open vs closed source
python benchmark_llms.py \
  --models gpt-5 gpt-oss-120b \
  --steps 20
```

## Example 13: Production Pipeline

**Goal**: Complete pipeline from test to analysis

```bash
#!/bin/bash
# production_benchmark.sh

set -e  # Exit on error

echo "Step 1: Test setup"
python test_benchmark_setup.py --skip-simulation

echo "Step 2: Quick test"
python benchmark_llms.py --models gpt-5-mini --steps 5

echo "Step 3: Full benchmark"
python benchmark_llms.py \
  --models gpt-5 gpt-5-mini o3 o4-mini \
  --steps 20 \
  --seed 42 \
  --output-dir ./production_results

echo "Step 4: Analyze results"
python analyze_benchmark_results.py \
  --base-dir ./production_results \
  --auto

echo "Step 5: Display summary"
RESULT_DIR=$(ls -td ./production_results/benchmark_* | head -1)
cat $RESULT_DIR/benchmark_summary.txt

echo "Pipeline complete!"
```

**Usage**: `bash production_benchmark.sh`

## Example 14: Batch Processing

**Goal**: Run multiple configurations automatically

```bash
#!/bin/bash
# batch_experiments.sh

# Models to test
MODELS=("gpt-5" "gpt-5-mini" "o3")

# Seeds for robustness
SEEDS=(42 123 456)

# Run all combinations
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running $model with seed $seed..."
        python benchmark_llms.py \
          --models $model \
          --steps 20 \
          --seed $seed \
          --output-dir "./batch_results/${model}_seed${seed}"
    done
done

echo "Batch processing complete!"
echo "Results in ./batch_results/"
```

## Example 15: Development/Debug Mode

**Goal**: Fast iteration during development

```bash
# Minimal test for debugging
python benchmark_llms.py \
  --models gpt-5-mini \
  --steps 3 \
  --seed 42 \
  --output-dir ./debug

# Even faster: just test setup
python test_benchmark_setup.py
```

**Time**: ~10-20 seconds
**Use for**: Testing code changes, debugging issues

## Viewing Results

### View Summary Text
```bash
# Most recent benchmark
cat llm_benchmarks/benchmark_*/benchmark_summary.txt | less

# Specific benchmark
cat llm_benchmarks/benchmark_2025-11-10_12-30-45/benchmark_summary.txt
```

### View CSV in Terminal
```bash
# Pretty print CSV
column -t -s, llm_benchmarks/benchmark_*/benchmark_table.csv | less -S
```

### Open Visualizations
```bash
# macOS
open llm_benchmarks/benchmark_*/performance_comparison.png

# Linux
xdg-open llm_benchmarks/benchmark_*/performance_comparison.png

# Or use your preferred image viewer
```

### View GIF Animations
```bash
# Find all simulation GIFs
find llm_benchmarks -name "simulation.gif"

# Open a specific one
open llm_benchmarks/benchmark_*/gpt-5_seed42/simulation.gif
```

### Parse JSON in Terminal
```bash
# Pretty print JSON
python -m json.tool llm_benchmarks/benchmark_*/benchmark_detailed.json | less

# Extract specific field
python -c "import json; data=json.load(open('llm_benchmarks/benchmark_*/benchmark_detailed.json')); print(data['models'][0]['total_tokens'])"
```

## Troubleshooting Examples

### Test API Connection
```bash
# Simple test
python -c "
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model='gpt-5-mini',
    input='Say hello'
)
print('API works!', response.output[1].content[0].text)
"
```

### Check Environment
```bash
# Verify all environment variables
echo "API Key: ${OPENAI_API_KEY:0:10}..."
echo "Base URL: $OPENAI_BASE_URL"
echo "Python Path: $PYTHONPATH"
```

### Debug Import Issues
```bash
# Test imports
python -c "
import sys
sys.path.append('/home/huao/Research/SocialJax')
import socialjax
from coins_llm_simulation import CoinGame
print('All imports successful!')
"
```

## Tips

### Save Command History
```bash
# Add to your benchmark script
echo "python benchmark_llms.py $@" >> benchmark_history.log
```

### Time Your Benchmarks
```bash
# Measure execution time
time python benchmark_llms.py --models gpt-5-mini --steps 20
```

### Run in Background
```bash
# Run benchmark in background (for long runs)
nohup python benchmark_llms.py --steps 50 > benchmark.log 2>&1 &

# Check progress
tail -f benchmark.log
```

### Compare Two Benchmarks
```bash
# Run baseline
python benchmark_llms.py --models gpt-5 --steps 20 --output-dir ./baseline

# Run experiment
python benchmark_llms.py --models gpt-5 --steps 20 --temperature 0.5 --output-dir ./experiment

# Compare summaries
diff <(cat baseline/benchmark_*/benchmark_summary.txt) \
     <(cat experiment/benchmark_*/benchmark_summary.txt)
```

## Common Patterns

### Pattern: Test → Benchmark → Analyze
```bash
python test_benchmark_setup.py && \
python benchmark_llms.py && \
python analyze_benchmark_results.py --auto
```

### Pattern: Multi-seed Average
```bash
# Run multiple seeds
for seed in 42 123 456; do
    python benchmark_llms.py --models gpt-5 --steps 20 --seed $seed
done

# Manually average results from benchmark_summary.txt files
```

### Pattern: Model Tournament
```bash
# Test each model separately for detailed logs
for model in gpt-5 gpt-5-mini o3 o4-mini; do
    python benchmark_llms.py \
      --models $model \
      --steps 20 \
      --output-dir ./tournament/$model
done
```

---

**Pro Tip**: Start with Example 1 (setup test), then Example 2 (simple benchmark), then Example 4 (analyze results) to get familiar with the workflow!

