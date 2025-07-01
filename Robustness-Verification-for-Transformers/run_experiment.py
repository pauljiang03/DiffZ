import subprocess
import numpy as np
import re
import sys
import json

def parse_all_bounds(output_text):
    """
    Parses the bounds for P, P', and their difference from the verifier's stdout.
    Returns a dictionary containing all found bounds.
    """
    bounds = {}
    try:
        # Define headers for all bounds we want to capture
        headers = {
            "l_p": "Logit Lower Bounds (P):",
            "u_p": "Logit Upper Bounds (P):",
            "l_p_prime": "Logit Lower Bounds (P'):",
            "u_p_prime": "Logit Upper Bounds (P'):",
            "l_diff": "Lower Bound of Difference (l_diff):",
            "u_diff": "Upper Bound of Difference (u_diff):",
        }

        for key, header in headers.items():
            start_index = output_text.find(header)
            if start_index != -1:
                bracket_open = output_text.find('[[', start_index)
                bracket_close = output_text.find(']]', bracket_open)
                if bracket_open != -1 and bracket_close != -1:
                    data_str = output_text[bracket_open + 2 : bracket_close]
                    bounds[key] = np.fromstring(data_str, sep=',')
        
        return bounds

    except Exception as e:
        print(f"Error during manual parsing: {e}")
        return {}


def run_single_verification(case_num, config):
    """Runs a single instance of the prune_certify.py script."""
    # ... (This function's content remains the same as the previous version)
    model_params = config['model_architecture']
    pruning_params = config['pruning_params']
    verif_params = config['verification_params']
    error_params = config.get('error_reduction', {})
    solver_params = config.get('solver_params', {})
    
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
    
    if error_params.get('method'):
        command.extend(['--error-reduction-method', error_params['method']])
    if error_params.get('max_terms'):
        command.extend(['--max-num-error-terms', str(error_params['max_terms'])])
        
    if solver_params.get('add_softmax_sum_constraint'):
        command.append('--add-softmax-sum-constraint')

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running prune_certify.py for case {case_num + 1}:")
        print(result.stderr)
        return {}

    return parse_all_bounds(result.stdout)
'''
def main():
    """Main function to load config, run experiment, and compute statistics."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it before running.")
        return

    print("Starting verification experiment with parameters from config.json... 🧪")
    
    # Create lists to store all the results
    results_data = {
        "l_p": [], "u_p": [],
        "l_p_prime": [], "u_p_prime": [],
        "max_abs_errors": []
    }
    
    num_cases = config['verification_params']['num_cases']
    for i in range(num_cases):
        bounds = run_single_verification(i, config)
        
        # Check if we got the essential difference bounds back
        if bounds.get("l_diff") is not None and bounds.get("u_diff") is not None:
            print("Success! Parsed all bounds.")
            # Store all collected data
            for key, val in bounds.items():
                if key in results_data:
                    results_data[key].append(val)
                elif key in ["l_diff", "u_diff"]: # backwards compatibility for max_abs_errors
                    # pass
                    pass


            # Calculate and store the max absolute error for this run
            max_error_for_run = np.max(np.maximum(np.abs(bounds["l_diff"]), np.abs(bounds["u_diff"])))
            results_data["max_abs_errors"].append(max_error_for_run)
        else:
            print(f"Warning: Could not parse essential bounds for case {i + 1}. Skipping.")
            
    if not results_data["max_abs_errors"]:
        print("\nExperiment finished, but no data was collected. Please check setup.")
        return
    
    print("\n---  Experiment Complete  ---")
    print(f"Total successful cases: {len(results_data['max_abs_errors'])}/{num_cases}")
    
    # --- FINAL STATISTICAL ANALYSIS ---
    print("\n--- Overall Error (P - P') Analysis ---")
    print("Statistical Analysis of Maximum Absolute Error:")
    print(f"  -> Average Error: {np.mean(results_data['max_abs_errors']):.6f}")
    print(f"  -> Maximum Error: {np.max(results_data['max_abs_errors']):.6f}")
    print(f"  -> Minimum Error: {np.min(results_data['max_abs_errors']):.6f}")
    print(f"  -> Median Error:  {np.median(results_data['max_abs_errors']):.6f}")

    # --- Detailed Bound Analysis ---
    for key, data_list in results_data.items():
        if key == "max_abs_errors" or not data_list:
            continue
        
        title = key.replace('_', ' ').replace('p prime', "P'").replace('p', 'P').title()
        print(f"\nStatistical Analysis for {title} Bounds:")
        
        # vstack turns the list of 1D arrays into a 2D matrix for analysis
        matrix = np.vstack(data_list)
        print(f"  -> Average Vector: {np.mean(matrix, axis=0)}")
        print(f"  -> Minimum Vector: {np.min(matrix, axis=0)}")
        print(f"  -> Maximum Vector: {np.max(matrix, axis=0)}")
        print(f"  -> Median Vector:  {np.median(matrix, axis=0)}")
'''
# In run_experiment.py

def main():
    """Main function to load config, run experiment, and compute statistics."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it before running.")
        return

    print("Starting verification experiment with parameters from config.json... 🧪")
    
    max_abs_errors = []
    
    num_cases = config['verification_params']['num_cases']
    for i in range(num_cases):
        l_bound, u_bound = run_single_verification(i, config)
        
        if l_bound is not None and u_bound is not None:
            max_error_for_run = np.max(np.maximum(np.abs(l_bound), np.abs(u_bound)))
            max_abs_errors.append(max_error_for_run)
            print(f"Success! Max absolute error for this run: {max_error_for_run:.6f}")
        else:
            print(f"Warning: Could not parse bounds for case {i + 1}. Skipping.")
            
    if not max_abs_errors:
        print("\nExperiment finished, but no data was collected. Please check setup.")
        return
    
    print("\n--- ✅ Experiment Complete ✅ ---")
    print(f"Total successful cases: {len(max_abs_errors)}/{num_cases}")
    
    # --- FINAL STATISTICAL ANALYSIS ---
    # This section is updated to include Standard Deviation and Quartiles
    print("\nStatistical Analysis of Maximum Absolute Error:")
    print(f"  -> Average Error: {np.mean(max_abs_errors):.6f}")
    print(f"  -> Standard Deviation: {np.std(max_abs_errors):.6f}")
    print(f"  -> Minimum Error: {np.min(max_abs_errors):.6f}")
    print(f"  -> 25th Percentile (Q1): {np.percentile(max_abs_errors, 25):.6f}")
    print(f"  -> Median Error (Q2):  {np.median(max_abs_errors):.6f}")
    print(f"  -> 75th Percentile (Q3): {np.percentile(max_abs_errors, 75):.6f}")
    print(f"  -> Maximum Error: {np.max(max_abs_errors):.6f}")
    print(f"  -> Interquartile Range (IQR): {np.percentile(max_abs_errors, 75) - np.percentile(max_abs_errors, 25):.6f}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
