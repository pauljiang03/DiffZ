import subprocess
import numpy as np
import re
import sys

# --- Configuration ---
# You can change these parameters to match your model's architecture.
# These are default values from your Parser.py[cite: 4].
K_VAL = 16
PRUNING_LAYER = 1
HIDDEN_SIZE = 256
NUM_LAYERS = 12
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
OUTPUT_EPSILON = 0.1 # Target epsilon bound for ||P - P_prime|| [cite: 4]

# --- Experiment Parameters from your notes  ---
EPSILON = 0.001
NUM_CASES = 100

def run_single_verification(case_num, epsilon_val):
    """
    Runs a single instance of the prune_certify.py script and captures its output.
    """
    print(f"--- Running Case {case_num + 1}/{NUM_CASES} ---")
    
    # These arguments are based on your provided Python files[cite: 2, 4].
    command = [
        sys.executable,
        'prune_certify.py',
        '--verify',
        '--eps', str(epsilon_val),
        '--k', str(K_VAL),
        '--pruning_layer', str(PRUNING_LAYER),
        '--output_epsilon', str(OUTPUT_EPSILON),
        '--samples', '1', # We run 1 sample per case for 100 cases
        '--hidden_size', str(HIDDEN_SIZE),
        '--num_layers', str(NUM_LAYERS),
        '--num_attention_heads', str(NUM_HEADS),
        '--intermediate_size', str(INTERMEDIATE_SIZE),
        '--device', 'cpu' # Use '--device', 'cuda' if you have a GPU
    ]
    
    # The 'JointModel' calculates the difference between the pruned and unpruned models.
    # We assume the verifier logs this difference, which we capture here.
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running prune_certify.py for case {case_num + 1}:")
        print(result.stderr)
        return None

    # This regex is designed to find the logit difference that your model calculates.
    # You may need to adjust this if your Verifier's logging format is different.
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
    print("Starting verification experiment as per your notes... 🧪")
    
    errors = []
    for i in range(NUM_CASES):
        error = run_single_verification(i, EPSILON)
        if error is not None:
            errors.append(error)
            
    if not errors:
        print("\nExperiment finished, but no data was collected. Please check your setup.")
        return

    # Using numpy to calculate the required statistics.
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
