import os
import sys
import torch
import numpy as np
import time
from typing import List, Dict, Any

# --- Import Environment Modules ---
from Parser import Parser, update_arguments
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from data_utils import set_seeds
from Verifiers.VerifierTopKPrune import VerifierTopKPrune

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================
EPSILONS = [0.001, 0.005, 0.01]
TOKENS_TO_KEEP_LIST = [9, 12, 15, 18, 25]  # Varying tokens pruned
PRUNE_LAYERS_LIST = [0]         # Typically fixed for this specific metric check
SAMPLES_PER_CONFIG = 5
GPU_ID = 0
# ==============================================================================

def calculate_metrics(results_diff):
    """
    Calculates only the average differential lower and upper bounds.
    """
    valid_samples = len(results_diff)
    if valid_samples == 0:
        return None

    total_low_diff = 0.0
    total_up_diff = 0.0

    for i in range(valid_samples):
        lbl = results_diff[i]['label']
        
        # Diff Bounds
        total_low_diff += results_diff[i]['lower_bounds'][lbl]
        total_up_diff += results_diff[i]['upper_bounds'][lbl]

    return {
        "avg_diff_lb": total_low_diff / valid_samples,
        "avg_diff_ub": total_up_diff / valid_samples,
    }

def main():
    # --- 1. Setup Environment ---
    argv = sys.argv[1:]
    parser = Parser.get_parser()
    
    parser.add_argument('--prune_tokens', action='store_true')
    parser.add_argument('--prune_layer_idx', type=int, default=0)
    parser.add_argument('--tokens_to_keep', type=int, default=9)

    args, _ = parser.parse_known_args(argv)
    
    args.samples = SAMPLES_PER_CONFIG
    args.verbose = False
    args.debug = False
    args.prune_tokens = True
    args.gpu = GPU_ID
    
    args = update_arguments(args)
    args.error_reduction_method = 'box'
    args.max_num_error_terms = 30000
    args.with_lirpa_transformer = False
    args.all_words = True
    args.concretize_special_norm_error_together = True
    args.num_input_error_terms = 28 * 28

    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seeds(args.seed)
    
    # --- 2. Load Model & Data ---
    # Suppress loading noise
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
    model.eval()

    test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
    data_normalized = []
    for i, (x, y) in enumerate(test_data):
        data_normalized.append({"label": y.to(device), "image": x.to(device)})
        if i == args.samples - 1:
            break
            
    logger = FakeLogger()
    
    # Restore stdout
    sys.stdout = original_stdout
    
    print("\n" + "="*60)
    print(f"Starting Top-K Pruning Experiment Suite")
    print("="*60)

    # --- 3. Experiment Loop ---
    for eps in EPSILONS:
        print(f"\n--- Epsilon: {eps} ---")
        print(f"{'Tokens Kept':<12} | {'Avg Diff Lower Bound':<20} | {'Avg Diff Upper Bound':<20}")
        print("-" * 60)
        
        for tokens in TOKENS_TO_KEEP_LIST:
            # Update dynamic arguments
            args.eps = eps
            args.prune_layer_idx = PRUNE_LAYERS_LIST[0] # Using first defined layer
            args.tokens_to_keep = tokens
            
            # Instantiate Verifier
            # We suppress print output from the verifier itself to keep terminal clean
            sys.stdout = open(os.devnull, 'w')
            verifier = VerifierTopKPrune(args, model, logger, num_classes=10, normalizer=normalizer)
            results_diff, results_p, results_p_prime = verifier.run(data_normalized)
            sys.stdout = original_stdout
            
            # Calculate Metrics
            metrics = calculate_metrics(results_diff)
            
            if metrics:
                print(f"{tokens:<12} | {metrics['avg_diff_lb']:<20.5f} | {metrics['avg_diff_ub']:<20.5f}")
            else:
                print(f"{tokens:<12} | {'No valid results':<20} | {'-':<20}")

    print("\n" + "="*60)
    print("Experiment Suite Completed.")
    print("="*60)

if __name__ == "__main__":
    main()
