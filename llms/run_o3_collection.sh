#!/bin/bash

# Quick script to collect 3 runs with o3 model using different seeds
# 
# Usage:
#   ./run_o3_collection.sh                    # Run with defaults (seeds: 42, 123, 456)
#   ./run_o3_collection.sh --steps 30        # Custom number of steps
#   ./run_o3_collection.sh --seeds 1 2 3     # Custom seeds

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
SEEDS="68 123 456"
STEPS=20
TEMPERATURE=0.7
OUTPUT_DIR="./llm_simulation_output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
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
            echo "Collect Multiple Runs with o3 Model"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seeds SEEDS          Space-separated list of seeds (default: 42 123 456)"
            echo "  --steps STEPS          Number of steps per run (default: 20)"
            echo "  --temperature TEMP     Sampling temperature (default: 0.7)"
            echo "  --output-dir DIR       Output directory (default: ./llm_simulation_output)"
            echo "  --help, -h             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --steps 30                        # Custom steps"
            echo "  $0 --seeds \"1 2 3\" --steps 50       # Custom seeds and steps"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Print configuration
echo "========================================"
echo "Collecting o3 Model Runs"
echo "========================================"
echo "Model:       o3"
echo "Seeds:       $SEEDS"
echo "Steps:       $STEPS"
echo "Temperature: $TEMPERATURE"
echo "Output:      $OUTPUT_DIR"
echo "========================================"
echo ""

# Run collection script
python "$SCRIPT_DIR/collect_o3_runs.py" \
    --model o3 \
    --seeds $SEEDS \
    --steps $STEPS \
    --temperature $TEMPERATURE \
    --output-dir "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Collection completed successfully!"
    echo "========================================"
    echo ""
    echo "View comparative summary:"
    echo "  cat $OUTPUT_DIR/collection_*/comparative_summary.txt"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Collection failed with errors"
    echo "========================================"
    exit 1
fi

