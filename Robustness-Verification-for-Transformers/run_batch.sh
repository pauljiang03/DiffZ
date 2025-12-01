#!/bin/bash

# --- Configuration ---
SAMPLES=100
LAYER_IDX=0  # Pruning after layer 0 (Index 0)

# Arrays of values to loop through
EPSILONS=(0.001 0.005 0.01)
KEEP_COUNTS=(5 9 13 17)

# Create results directory if it doesn't exist
mkdir -p results

echo "========================================================"
echo "Starting Batch Benchmark"
echo "Samples per run: $SAMPLES"
echo "Total Runs: 12"
echo "========================================================"

for eps in "${EPSILONS[@]}"; do
    for k in "${KEEP_COUNTS[@]}"; do
        
        # Define a clean filename for the output
        # e.g., results/bench_eps0.005_keep9.csv
        CSV_NAME="bench_eps${eps}_keep${k}.csv"
        
        echo ""
        echo ">>> Running: Eps=$eps | Keep=$k Tokens"
        echo ">>> Output: results/$CSV_NAME"
        
        python run_full_benchmark.py \
            --samples $SAMPLES \
            --eps $eps \
            --prune_tokens \
            --tokens_to_keep $k \
            --prune_layer_idx $LAYER_IDX \
            --csv_name "$CSV_NAME"

    done
done

echo ""
echo "========================================================"
echo "Batch Benchmark Completed."
echo "All files saved in /results"
echo "========================================================"
