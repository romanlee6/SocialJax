#!/bin/bash

# Quick-start script for running LLM benchmark on Coins Game
# 
# Usage:
#   ./run_benchmark.sh                    # Run with defaults
#   ./run_benchmark.sh --help             # Show help
#   ./run_benchmark.sh --steps 30         # Custom steps

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "For custom endpoints, also set:"
    echo "  export OPENAI_BASE_URL='https://your-endpoint.com/v1'"
    exit 1
fi

# Default values
MODELS="gpt-5 gpt-5-mini o3 o4-mini gpt-oss-120b"
STEPS=20
SEED=42
TEMPERATURE=0.7
OUTPUT_DIR="./llm_benchmarks"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "LLM Benchmark for Coins Game"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODELS          Models to test (default: gpt-5 gpt-5-mini o3 o4-mini gpt-oss-120b)"
            echo "  --steps STEPS            Number of steps (default: 20)"
            echo "  --seed SEED              Random seed (default: 42)"
            echo "  --temperature TEMP       Sampling temperature (default: 0.7)"
            echo "  --output-dir DIR         Output directory (default: ./llm_benchmarks)"
            echo "  --help, -h               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                           # Run with defaults"
            echo "  $0 --models \"gpt-5 gpt-5-mini\" --steps 30  # Custom models and steps"
            echo "  $0 --seed 123 --temperature 0.5              # Different seed and temperature"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "LLM Benchmark Configuration"
echo "========================================"
echo "Models:      $MODELS"
echo "Steps:       $STEPS"
echo "Seed:        $SEED"
echo "Temperature: $TEMPERATURE"
echo "Output:      $OUTPUT_DIR"
echo "========================================"
echo ""

# Run benchmark
python benchmark_llms.py \
    --models $MODELS \
    --steps $STEPS \
    --seed $SEED \
    --temperature $TEMPERATURE \
    --output-dir "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Benchmark completed successfully!"
    echo "========================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/benchmark_*/benchmark_summary.txt"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Benchmark failed with errors"
    echo "========================================"
    exit 1
fi

