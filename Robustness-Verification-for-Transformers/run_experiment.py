import subprocess
import numpy as np
import re
import sys

# --- Configuration ---
# MODIFIED: These values now match the architecture of your saved 'mnist_transformer.pt' file
# and your successful command-line run.
GPU_ID = 0
K_VAL = 15
PRUNING_LAYER = 0
NUM_LABELS = 10
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_LAYERS = 1
NUM_HEADS = 4

# --- Experiment Parameters from your notes ---
EPSILON = 0.001
NUM_CASES = 100
# NOTE: Your successful command used an output_epsilon of 0.1
OUTPUT_EPSILON = 0.1

def run_single_verification(case_num):
    """
    Runs a single instance of the prune_certify.py script and captures its output.
    """
    print(f"--- Running Case {case_num + 1}/{NUM_CASES} ---")
    
    # MODIFIED: Command now uses the corrected architectural parameters
    command = [
        sys.executable,
        'prune_certify.py',
        '--verify',
        '--gpu', str(GPU_ID),
        '--num_layers', str(NUM_LAYERS),
        '--pruning_layer', str(PRUNING_LAYER),
        '--k', str(K_VAL),
        '--num_labels', str(NUM_LABELS),
        '--hidden_size', str(HIDDEN_SIZE),
        '--intermediate_size', str(INTERMEDIATE_SIZE),
        '--num_attention_heads', str(NUM_HEADS),
        '--eps', str(EPSILON),
        '--output_epsilon', str(OUTPUT_EPSILON),
        '--samples', '1',
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running prune_certify.py for case {case_num + 1}:")
        print(result.stderr)
        return None

    # MODIFIED: Updated the regex to find the correct output from your Verifier
    # It now looks for "Max Diff Bound: 0.123456"
    match = re.search(r"Max Diff Bound:\s*([0-9\.-]+)", result.stdout)
    
    if match:
        error_value = float(match.group(1))
        print(f"Success! Found Max Diff Bound: {error_value}")
        return error_value
    else:
        print(f"Warning: Could not parse 'Max Diff Bound' for case {case_num + 1}.")
        print("Full output:")
        print(result.stdout)
        return None

def main():
    """
    Main function to run the experiment and compute statistics.
    """
    print("Starting verification experiment... 🧪")
    
    errors = []
    for i in range(NUM_CASES):
        error = run_single_verification(i)
        if error is not None:
            errors.append(error)
            
    if not errors:
        print("\nExperiment finished, but no data was collected. Please check your setup.")
        return

    np_errors = np.array(errors)
    avg_error = np.mean(np_errors)
    max_error = np.max(np_errors)
    min_error = np.min(np_errors)
    median_error = np.median(np_errors)
    
    print("\n--- ✅ Experiment Complete ✅ ---")
    print(f"Total successful cases: {len(errors)}/{NUM_CASES}")
    print("\nStatistical Analysis of 'Max Diff Bound':")
    print(f"  -> Average: {avg_error:.6f}")
    print(f"  -> Maximum: {max_error:.6f}")
    print(f"  -> Minimum: {min_error:.6f}")
    print(f"  -> Median:  {median_error:.6f}")

if __name__ == "__main__":
    main()
