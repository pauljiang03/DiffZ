import subprocess
import numpy as np
import re
import sys

# --- Configuration ---
GPU_ID = 0 # Tell the script to use GPU 0. Change if you use a different one.
K_VAL = 16
PRUNING_LAYER = 1
HIDDEN_SIZE = 256
NUM_LAYERS = 12
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
OUTPUT_EPSILON = 0.1 # Target epsilon bound for ||P - P_prime||

# --- Experiment Parameters from your notes ---
EPSILON = 0.001
NUM_CASES = 100

def run_single_verification(case_num, epsilon_val):
    """
    Runs a single instance of the prune_certify.py script and captures its output.
    """
    print(f"--- Running Case {case_num + 1}/{NUM_CASES} ---")
    
    # MODIFIED: Command now includes the --gpu flag to ensure the GPU is enabled.
    command = [
        sys.executable,
        'prune_certify.py',
        '--verify',
        '--gpu', str(GPU_ID), # Explicitly tell the script which GPU to use
        '--eps', str(epsilon_val),
        '--k', str(K_VAL),
        '--pruning_layer', str(PRUNING_LAYER),
        '--output_epsilon', str(OUTPUT_EPSILON),
        '--samples', '1', 
        '--hidden_size', str(HIDDEN_SIZE),
        '--num_layers', str(NUM_LAYERS),
        '--num_attention_heads', str(NUM_HEADS),
        '--intermediate_size', str(INTERMEDIATE_SIZE),
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running prune_certify.py for case {case_num + 1}:")
        print(result.stderr)
        return None

    # This regex looks for the logit difference from your model.
    # We are looking for a line like "max_logit_diff_abs: 0.00123"
    match = re.search(r"max_logit_diff_abs:\s*([0-9\.]+)", result.stdout)
    
    if match:
        error_value = float(match.group(1))
        print(f"Success! Found Error (Logit Diff): {error_value}")
        return error_value
    else:
        print(f"Warning: Could not parse error value for case {case_num + 1}.")
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
        error = run_single_verification(i, EPSILON)
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
    print("\nStatistical Analysis of Error (Logit Difference):")
    print(f"  -> Average Error: {avg_error:.6f}")
    print(f"  -> Maximum Error: {max_error:.6f}")
    print(f"  -> Minimum Error: {min_error:.6f}")
    print(f"  -> Median Error:  {median_error:.6f}")

if __name__ == "__main__":
    main()
