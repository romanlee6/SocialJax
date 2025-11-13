#!/bin/bash

# Script to collect 3 runs with 100 steps each using different random seeds
# Saves accumulated rewards, coins in state, and own/other color coins eaten as .npy files
# 
# Usage:
#   ./run_3runs_100steps.sh                          # Run with defaults
#   ./run_3runs_100steps.sh --model gpt-4           # Custom model
#   ./run_3runs_100steps.sh --seeds 10 20 30        # Custom seeds

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
SEEDS="42 123 456"
STEPS=100
TEMPERATURE=0.7
MODEL="o3"
OUTPUT_DIR="./llm_simulation_output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            shift
            SEEDS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SEEDS="$SEEDS $1"
                shift
            done
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Collect 3 Runs with 100 Steps Each"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name (default: o3)"
            echo "  --seeds SEEDS          Space-separated list of 3 seeds (default: 42 123 456)"
            echo "  --steps STEPS          Number of steps per run (default: 100)"
            echo "  --temperature TEMP     Sampling temperature (default: 0.7)"
            echo "  --output-dir DIR       Output directory (default: ./llm_simulation_output)"
            echo "  --help, -h             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                        # Run with defaults"
            echo "  $0 --model o3                            # Use o3 model"
            echo "  $0 --seeds 10 20 30 --steps 50          # Custom seeds and steps"
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
echo "Collecting 3 Runs with $STEPS Steps Each"
echo "========================================"
echo "Model:       $MODEL"
echo "Seeds:       $SEEDS"
echo "Steps:       $STEPS"
echo "Temperature: $TEMPERATURE"
echo "Output:      $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Additional data will be saved as .npy files:"
echo "  - accumulated_rewards.npy"
echo "  - coins_in_state.npy"
echo "  - own_color_coins_eaten.npy"
echo "  - other_color_coins_eaten.npy"
echo "========================================"
echo ""

# Run collection script
python "$SCRIPT_DIR/collect_o3_runs.py" \
    --model "$MODEL" \
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
    echo "Each run contains the following .npy files:"
    echo "  - accumulated_rewards.npy (shape: timesteps × num_agents)"
    echo "  - coins_in_state.npy (shape: timesteps × 2 [red, green])"
    echo "  - own_color_coins_eaten.npy (shape: timesteps × num_agents)"
    echo "  - other_color_coins_eaten.npy (shape: timesteps × num_agents)"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Collection failed with errors"
    echo "========================================"
    exit 1
fi

