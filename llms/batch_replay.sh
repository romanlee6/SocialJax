#!/bin/bash

# Batch replay multiple trajectory directories
#
# Usage:
#   ./batch_replay.sh [BASE_DIR]
#
# Example:
#   ./batch_replay.sh llm_simulation_output/
#   ./batch_replay.sh llm_benchmarks/benchmark_2025-11-10_*/

set -e

# Default base directory
BASE_DIR="${1:-llm_simulation_output}"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory not found: $BASE_DIR"
    exit 1
fi

echo "========================================"
echo "Batch Trajectory Replay"
echo "========================================"
echo "Base directory: $BASE_DIR"
echo ""

# Find all directories containing trajectory files
TRAJECTORY_DIRS=$(find "$BASE_DIR" -name "trajectory_parsed.json" -exec dirname {} \; | sort)

if [ -z "$TRAJECTORY_DIRS" ]; then
    echo "No trajectory files found in $BASE_DIR"
    exit 1
fi

# Count directories
NUM_DIRS=$(echo "$TRAJECTORY_DIRS" | wc -l)
echo "Found $NUM_DIRS trajectory directories"
echo ""

# Process each directory
CURRENT=0
for DIR in $TRAJECTORY_DIRS; do
    CURRENT=$((CURRENT + 1))
    echo "========================================"
    echo "Processing $CURRENT/$NUM_DIRS: $DIR"
    echo "========================================"
    
    # Run replay
    python replay_trajectory.py "$DIR" --frame-skip 5
    
    echo ""
done

echo "========================================"
echo "Batch replay complete!"
echo "========================================"
echo "Processed $NUM_DIRS trajectories"
echo ""

