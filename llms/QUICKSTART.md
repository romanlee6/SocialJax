# Quick Start Guide - LLM Benchmark

This guide will get you up and running with the LLM benchmark in 5 minutes.

## Step 1: Setup Environment

### 1.1 Set API Key

```bash
export OPENAI_API_KEY="your-api-key-here"

# If using a custom endpoint (optional):
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
```

### 1.2 Verify Installation

```bash
cd llms/
python test_benchmark_setup.py
```

This will:
- ✓ Check if API key is set
- ✓ Verify all dependencies are installed
- ✓ Test import of required modules
- ✓ Run a 3-step test simulation

If you see "ALL TESTS PASSED", you're ready to go! If not, follow the error messages to fix any issues.

## Step 2: Run Your First Benchmark

### Option A: Using the Shell Script (Recommended)

```bash
./run_benchmark.sh
```

This runs the benchmark with default settings:
- Models: gpt-5, gpt-5-mini, o3, o4-mini, gpt-oss-120b
- Steps: 20
- Seed: 42

### Option B: Using Python Directly

```bash
python benchmark_llms.py --models gpt-5 gpt-5-mini --steps 20 --seed 42
```

### Option C: Quick Test (Fast)

```bash
python benchmark_llms.py --models gpt-5-mini --steps 10
```

## Step 3: Monitor Progress

You'll see real-time output like:

```
================================================================================
STARTING LLM BENCHMARK
================================================================================
Models to test: gpt-5, gpt-5-mini, o3, o4-mini, gpt-oss-120b
Number of steps: 20
Environment seed: 42
================================================================================

================================================================================
Running benchmark for model: gpt-5
================================================================================
  Step 20/20
  Completed in 45.23 seconds
  Model output saved to: ./llm_benchmarks/benchmark_2025-11-10_12-30-45/gpt-5_seed42
```

## Step 4: View Results

### Summary Report (Human-Readable)

```bash
# View the most recent benchmark
cat llm_benchmarks/benchmark_*/benchmark_summary.txt
```

### Comparison Table (CSV for Excel)

```bash
# Open in Excel/Google Sheets
open llm_benchmarks/benchmark_*/benchmark_table.csv
```

### Visualizations

```bash
# Generate analysis plots
python analyze_benchmark_results.py --auto
```

This creates:
- `performance_comparison.png` - Bar chart of rewards
- `efficiency_comparison.png` - Time and token usage
- `tradeoff_analysis.png` - Performance vs efficiency scatter plots

### Watch the Simulations

```bash
# View animated GIFs for each model
ls llm_benchmarks/benchmark_*/*/simulation.gif
```

## Step 5: Customize Your Benchmark

### Test Specific Models

```bash
python benchmark_llms.py --models gpt-5 o3 --steps 20
```

### Longer Episodes

```bash
python benchmark_llms.py --steps 50
```

### Different Random Seeds

```bash
python benchmark_llms.py --seed 123
```

### Lower Temperature (More Deterministic)

```bash
python benchmark_llms.py --temperature 0.3
```

### Custom Output Directory

```bash
python benchmark_llms.py --output-dir ./my_experiments
```

## Common Use Cases

### 1. Quick Performance Test

**Goal:** Test if a model can play the game reasonably

```bash
python benchmark_llms.py --models gpt-5-mini --steps 10
```

### 2. Full Model Comparison

**Goal:** Compare all available models thoroughly

```bash
python benchmark_llms.py --steps 50 --seed 42
```

### 3. Robustness Testing

**Goal:** Test model performance across different random seeds

```bash
for seed in 42 123 456 789 999; do
    python benchmark_llms.py --models gpt-5 --steps 20 --seed $seed --output-dir ./robustness_test
done
```

### 4. Temperature Sweep

**Goal:** Understand impact of temperature on behavior

```bash
for temp in 0.3 0.5 0.7 1.0; do
    python benchmark_llms.py --models gpt-5 --steps 20 --temperature $temp --output-dir ./temp_sweep
done
```

### 5. Cost Estimation

**Goal:** Estimate API costs before large-scale experiments

```bash
# Run short benchmark
python benchmark_llms.py --models gpt-5 --steps 5

# Check token usage in summary
cat llm_benchmarks/benchmark_*/benchmark_summary.txt

# Calculate cost: tokens * (your_api_price_per_token)
```

## Understanding the Results

### What to Look For

#### 1. Performance (Rewards)
- **Higher is better**: More coins collected
- **Compare agents**: Is one agent dominating?
- **Check fairness**: Are both agents cooperating?

#### 2. Efficiency (Time)
- **Time per step**: How fast does each model respond?
- **Total time**: Important for large-scale experiments

#### 3. Efficiency (Tokens)
- **Tokens per step**: Directly impacts API costs
- **Input vs output**: Some models generate longer responses

#### 4. Behavior (Actions)
- **Action distribution**: Is the agent exploring or stuck?
- **Communications**: Is the agent trying to coordinate?

### Reading the Summary Report

```
MODEL: gpt-5
--------------------------------------------------------------------------------
Status: success

Performance:
  Episode Length: 20 steps
  Total Time: 45.23 seconds
  Time per Step: 2.262 seconds

Agent Performance:
  Agent 0:
    Total Reward: 12.00
    Average Reward: 0.6000
    Communications Sent: 8
  Agent 1:
    Total Reward: 10.00
    Average Reward: 0.5000
    Communications Sent: 6

Token Usage (Total):
  Input Tokens: 15,234
  Output Tokens: 3,456
  Total Tokens: 18,690

Token Usage (Average per Agent per Step):
  Input Tokens: 381.0
  Output Tokens: 86.4
```

**Interpretation:**
- ✓ Both agents earning positive rewards (good cooperation)
- ✓ Reasonable time per step (~2 seconds)
- ✓ Both agents communicating (8 and 6 messages)
- Token usage: ~19k tokens for 20 steps = ~950 tokens/step

## Troubleshooting

### Issue: "OPENAI_API_KEY environment variable not set"

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
# Add to ~/.bashrc or ~/.zshrc to persist
```

### Issue: "Model not found" or API error

**Solution:**
1. Check your API key is valid
2. Verify you have access to the model
3. For custom endpoints, ensure OPENAI_BASE_URL is set correctly
4. Try with a known model first: `--models gpt-5-mini`

### Issue: Benchmark is very slow

**Solutions:**
- Use fewer steps: `--steps 10`
- Test one model: `--models gpt-5-mini`
- Use faster model: gpt-5-mini is typically faster than gpt-5

### Issue: Out of memory

**Solutions:**
- Reduce number of visualization frames (edit benchmark_llms.py)
- Run with fewer steps
- Disable visualizations for very long runs

### Issue: ImportError for socialjax

**Solution:**
```bash
# Make sure you're in the project root or set PYTHONPATH
export PYTHONPATH=/home/huao/Research/SocialJax:$PYTHONPATH
cd /home/huao/Research/SocialJax/llms
python benchmark_llms.py
```

## Next Steps

1. **Run multiple seeds** to test robustness
2. **Try different temperatures** to see behavioral changes
3. **Analyze results** with `analyze_benchmark_results.py`
4. **Compare with baselines** (RL agents, random agents)
5. **Extend to other environments** (adapt the benchmark for other games)

## Getting Help

- Check `BENCHMARK_README.md` for detailed documentation
- See `coins_llm_simulation.py` for simulation details
- Review example outputs in the results directory
- Test setup with `test_benchmark_setup.py`

## Tips for Success

1. **Start small**: Test with 10 steps before running 50
2. **Use fast models first**: gpt-5-mini is good for debugging
3. **Check costs**: Monitor token usage for cost estimation
4. **Save results**: Use descriptive `--output-dir` names
5. **Document experiments**: Add notes in a separate experiment log

Happy benchmarking! 🚀

