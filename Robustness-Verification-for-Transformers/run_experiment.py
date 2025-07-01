import subprocess
import numpy as np
import re
import sys
import json

def parse_bounds_from_output(output_text):
    """Parses lower and upper bound vectors from the verifier's stdout."""
    try:
        # Regex to find the content inside tensor([[...]])
        lower_match = re.search(r"Lower Bound of Difference \(l_diff\):\s*tensor\(\[\[(.*?)\]\]", output_text)
        upper_match = re.search(r"Upper Bound of Difference \(u_diff\):\s*tensor\(\[\[(.*?)\]\]", output_text)

        if not lower_match or not upper_match:
            return None, None

        # Extract the comma-separated numbers and convert them to a numpy array
        l_str = lower_match.group(1).strip()
        u_str = upper_match.group(1).strip()

        lower_bound = np.fromstring(l_str, sep=',')
        upper_bound = np.fromstring(u_str, sep=',')
        
        return lower_bound, upper_bound
    except Exception as e:
        print(f"Error parsing bounds from output: {e}")
        return None, None

def run_single_verification(case_num, config):
    """Runs a single instance of the prune_certify.py script."""
    model_params = config['model_architecture']
    pruning_params = config['pruning_params']
    verif_params = config['verification_params']
    
    print(f"--- Running Case {case_num + 1}/{verif_params['num_cases']} (Image Index: {case_num}) ---")
    
    command = [
        sys.executable, 'prune_certify.py',
        '--verify',
        '--gpu', str(verif_params['gpu_id']),
        '--sample_index', str(case_num),
        '--num_layers', str(model_params['num_layers']),
        '--pruning_layer', str(pruning_params['pruning_layer']),
        '--k', str(pruning_params['k']),
        '--num_labels', str(model_params['num_labels']),
        '--hidden_size', str(model_params['hidden_size']),
        '--intermediate_size', str(model_params['intermediate_size']),
        '--num_attention_heads', str(model_params['num_attention_heads']),
        '--eps', str(verif_params['input_epsilon']),
        '--output_epsilon', str(verif_params['output_epsilon']),
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running prune_certify.py for case {case_num + 1}:")
        print(result.stderr)
        return None, None

    lower_bound, upper_bound = parse_bounds_from_output(result.stdout)

    if lower_bound is not None and upper_bound is not None:
        print(f"Success! Parsed Lower and Upper Bounds.")
        return lower_bound, upper_bound
    else:
        print(f"Warning: Could not parse bounds for case {case_num + 1}.")
        print("Full output:")
        print(result.stdout)
        return None, None

def main():
    """Main function to load config, run experiment, and compute statistics."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it before running.")
        return

    print("Starting verification experiment with parameters from config.json... 🧪")
    
    lower_bounds_list = []
    upper_bounds_list = []
    
    num_cases = config['verification_params']['num_cases']
    for i in range(num_cases):
        l_bound, u_bound = run_single_verification(i, config)
        if l_bound is not None and u_bound is not None:
            lower_bounds_list.append(l_bound)
            upper_bounds_list.append(u_bound)
            
    if not lower_bounds_list:
        print("\nExperiment finished, but no data was collected. Please check setup.")
        return

    # Convert lists of arrays into 2D numpy matrices
    L_bounds = np.vstack(lower_bounds_list)
    U_bounds = np.vstack(upper_bounds_list)
    
    print("\n--- Experiment Complete ---")
    print(f"Total successful cases: {len(lower_bounds_list)}/{num_cases}")
    
    # --- Statistical Analysis ---
    # We calculate statistics column-wise for each logit
    print("\nStatistical Analysis of Lower Bounds (l):")
    print(f"  -> Average Vector: {np.mean(L_bounds, axis=0)}")
    print(f"  -> Minimum Vector: {np.min(L_bounds, axis=0)}")
    print(f"  -> Maximum Vector: {np.max(L_bounds, axis=0)}")
    print(f"  -> Median Vector:  {np.median(L_bounds, axis=0)}")
    
    print("\nStatistical Analysis of Upper Bounds (u):")
    print(f"  -> Average Vector: {np.mean(U_bounds, axis=0)}")
    print(f"  -> Minimum Vector: {np.min(U_bounds, axis=0)}")
    print(f"  -> Maximum Vector: {np.max(U_bounds, axis=0)}")
    print(f"  -> Median Vector:  {np.median(U_bounds, axis=0)}")

if __name__ == "__main__":
    main()
