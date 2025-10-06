# File: run_analysis.py
# This script imports and uses your verifier to run experiments and compute metrics.

import os
import sys
import psutil
import torch
import numpy as np
from typing import List, Dict, Any

# --- Assumed Local Imports ---
# These should be accessible from your main project directory.
try:
    from Parser import Parser, update_arguments
    from fake_logger import FakeLogger
    from mnist import mnist_test_dataloader, normalizer
    from vit import ViT
    from data_utils import set_seeds
except ImportError as e:
    print(f"Error: A required local module could not be imported: {e}")
    print("Please ensure Parser.py, fake_logger.py, mnist.py, vit.py, and data_utils.py are accessible.")
    sys.exit(1)

# --- Import your verifier from the subdirectory ---
# This assumes you have the verifier code saved at this location.
try:
    from Verifiers.IntervalBoundVerifier import IntervalBoundDiffVerViT
except ImportError:
    print("Error: Could not import the verifier.")
    print("Please ensure your verifier class is saved at 'Verifiers/IntervalBoundVerifier.py'")
    sys.exit(1)


# =====================================================================================
# ANALYSIS FUNCTIONS
# =====================================================================================

def analyze_individual_results(results: List[Dict[str, Any]], model_name: str, args):
    """
    Analyzes certified accuracy for a single model (P or P').
    Counts samples where the lower bound of the correct class is greater
    than the upper bound of all other classes.
    """
    if not results:
        print(f"No results to analyze for model '{model_name}'.")
        return

    num_samples = len(results)
    certified_correct = 0

    for res in results:
        label = res['label']
        lower_bounds = res['lower_bounds']
        upper_bounds = res['upper_bounds']
        
        l_c = lower_bounds[label]
        
        max_u_other = -np.inf
        for i in range(len(lower_bounds)):
            if i != label:
                max_u_other = max(max_u_other, upper_bounds[i])
        
        if l_c > max_u_other:
            certified_correct += 1

    accuracy = (certified_correct / num_samples) * 100 if num_samples > 0 else 0
    
    print("\n" + "="*60)
    print(f"INDIVIDUAL ANALYSIS: '{model_name}'")
    print(f"Epsilon (ε): {args.eps}")
    print(f"Certified Correct: {certified_correct} / {num_samples} ({accuracy:.2f}%)")
    print("="*60)

def analyze_differential_results(results: List[Dict[str, Any]], args):
    """
    Analyzes the change from the unpruned (P) to the pruned (P') model.
    Computes metrics on the bounds of the *difference* (P - P').
    """
    if not results:
        print("No results to analyze for differential mode.")
        return

    num_samples = len(results)
    margin_failures = 0
    non_monotonic_failures = 0

    for res in results:
        label = res['label']
        lb_diff = res['lower_bounds']
        ub_diff = res['upper_bounds']
        
        l_diff_c = lb_diff[label]

        # Check for non-monotonic behavior
        if l_diff_c <= 0:
            non_monotonic_failures += 1

        # Check for margin preservation failure
        max_u_diff_other = -np.inf
        for i in range(len(lb_diff)):
            if i != label:
                max_u_diff_other = max(max_u_diff_other, ub_diff[i])
        
        if l_diff_c <= max_u_diff_other:
            margin_failures += 1

    margin_fail_rate = (margin_failures / num_samples) * 100 if num_samples > 0 else 0
    non_monotonic_rate = (non_monotonic_failures / num_samples) * 100 if num_samples > 0 else 0

    print("\n" + "="*60)
    print(f"DIFFERENTIAL ANALYSIS (P - P')")
    print(f"Epsilon (ε): {args.eps}")
    print(f"Non-Monotonic Risk: {non_monotonic_failures} / {num_samples} ({non_monotonic_rate:.2f}%)")
    print(f"Margin Preservation Failures: {margin_failures} / {num_samples} ({margin_fail_rate:.2f}%)")
    print("="*60)


# =====================================================================================
# MAIN SCRIPT EXECUTION
# =====================================================================================
if __name__ == "__main__":
    parser = Parser.get_parser()

    # Add arguments for pruning and mode selection
    parser.add_argument('--prune_tokens', action='store_true', help='Enable token pruning in the P\' model.')
    parser.add_argument('--prune_layer_idx', type=int, default=0, help='Transformer layer index AFTER which to apply token pruning.')
    parser.add_argument('--tokens_to_keep', type=int, default=9, help='Number of tokens to keep after pruning.')
    parser.add_argument('--mode', type=str, default='individual', choices=['individual', 'differential'],
                        help="Analysis mode: 'individual' for separate P/P' bounds, 'differential' for P-P' bounds.")
    
    args, _ = parser.parse_known_args(sys.argv[1:])

    # --- Setup and Configuration ---
    args.samples = 100
    args = update_arguments(args) # Assuming this function sets other necessary defaults
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    set_seeds(args.seed)

    # --- Load Data and Model ---
    test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
    model.eval()
    
    data_normalized = [{"label": y, "image": x} for i, (x, y) in enumerate(test_data) if i < args.samples]
    logger = FakeLogger()

    print("--- Verification Setup ---")
    print(f"Mode: {args.mode}")
    print(f"Target Samples: {args.samples}")
    print(f"Pruning Active: {args.prune_tokens}")
    print(f"Epsilon (ε): {args.eps}")
    print("--------------------------")

    if not hasattr(args, 'eps') or args.eps <= 0:
        print("\nERROR: Argument --eps must be set to a positive value for verification.")
    else:
        # --- Instantiate and Run Verifier ---
        verifier = IntervalBoundDiffVerViT(args, model, logger, num_classes=10, normalizer=normalizer)
        
        if args.mode == 'individual':
            results_p, results_p_prime = verifier.run(data_normalized, mode='individual')
            analyze_individual_results(results_p, model_name='P (Original)', args=args)
            analyze_individual_results(results_p_prime, model_name="P' (Pruned)", args=args)
        else: # args.mode == 'differential'
            diff_results = verifier.run(data_normalized, mode='differential')
            analyze_differential_results(diff_results, args=args)
