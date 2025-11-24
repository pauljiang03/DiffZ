import os
import sys
import torch
import numpy as np
import csv
import time
from datetime import datetime
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
# Modify these lists to define your experiment search space
# ==============================================================================
EPSILONS = [0.001, 0.0025, 0.005, 0.01]
TOKENS_TO_KEEP_LIST = [9, 16, 25]  # e.g. 9 = [CLS] + 8 patches
PRUNE_LAYERS_LIST = [0, 1]         # 0 = After Layer 1, 1 = After Layer 2
SAMPLES_PER_CONFIG = 5             # Keep small for testing, increase for final results
GPU_ID = 0
# ==============================================================================

def calculate_metrics(results_diff, results_p, results_p_prime):
    """
    Aggregates the raw results into summary statistics for the CSV.
    """
    valid_samples = len(results_diff)
    if valid_samples == 0:
        return None

    # 1. Differential Bounds (P - P')
    total_low_diff = 0.0
    total_up_diff = 0.0
    
    # 2. Robustness Counters
    robust_p_count = 0
    robust_pp_count = 0

    for i in range(valid_samples):
        lbl = results_diff[i]['label']
        
        # Diff Bounds
        total_low_diff += results_diff[i]['lower_bounds'][lbl]
        total_up_diff += results_diff[i]['upper_bounds'][lbl]
        
        # Robustness P (Unpruned)
        low_p = results_p[i]['lower_bounds'][lbl]
        # Get max upper bound of OTHER classes
        up_p_others = [results_p[i]['upper_bounds'][c] for c in range(10) if c != lbl]
        if low_p > max(up_p_others):
            robust_p_count += 1
            
        # Robustness P' (Pruned)
        low_pp = results_p_prime[i]['lower_bounds'][lbl]
        up_pp_others = [results_p_prime[i]['upper_bounds'][c] for c in range(10) if c != lbl]
        if low_pp > max(up_pp_others):
            robust_pp_count += 1

    return {
        "avg_diff_lb": total_low_diff / valid_samples,
        "avg_diff_ub": total_up_diff / valid_samples,
        "robustness_p": robust_p_count / valid_samples,
        "robustness_pp": robust_pp_count / valid_samples,
        "valid_samples": valid_samples
    }

def main():
    # --- 1. Setup Environment ---
    # We reuse the standard Parser logic to ensure model compatibility
    argv = sys.argv[1:]
    parser = Parser.get_parser()
    
    # Add pruning args explicitly if not present in Parser yet
    # (Since we are manually injecting them into 'args', this is just for safety)
    parser.add_argument('--prune_tokens', action='store_true')
    parser.add_argument('--prune_layer_idx', type=int, default=0)
    parser.add_argument('--tokens_to_keep', type=int, default=9)

    args, _ = parser.parse_known_args(argv)
    
    # Force specific settings for the experiment
    args.samples = SAMPLES_PER_CONFIG
    args.verbose = False
    args.debug = False
    args.prune_tokens = True # Always enable differential mode
    args.gpu = GPU_ID
    
    # Standard updates
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
    print("Loading Model and Data...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    
    # Initialize ViT
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
    model.eval()

    # Load Data
    test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
    data_normalized = []
    for i, (x, y) in enumerate(test_data):
        data_normalized.append({"label": y.to(device), "image": x.to(device)})
        if i == args.samples - 1:
            break
            
    logger = FakeLogger()
    
    # --- 3. Experiment Loop ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_csv = f"experiment_results_{timestamp}.csv"
    
    print(f"\nStarting Experiment Suite...")
    print(f"Output File: {output_csv}")
    print("=" * 60)

    # Initialize CSV
    fieldnames = ["epsilon", "prune_layer", "tokens_kept", "avg_diff_lb", "avg_diff_ub", "robustness_p", "robustness_pp", "time_elapsed"]
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through configurations
        total_configs = len(EPSILONS) * len(PRUNE_LAYERS_LIST) * len(TOKENS_TO_KEEP_LIST)
        current_config = 0

        for eps in EPSILONS:
            for layer in PRUNE_LAYERS_LIST:
                for tokens in TOKENS_TO_KEEP_LIST:
                    current_config += 1
                    print(f"\n[{current_config}/{total_configs}] Running Config: Eps={eps}, Layer={layer}, Tokens={tokens}")
                    
                    start_time = time.time()
                    
                    # Update dynamic arguments
                    args.eps = eps
                    args.prune_layer_idx = layer
                    args.tokens_to_keep = tokens
                    
                    # Instantiate Verifier
                    verifier = VerifierTopKPrune(args, model, logger, num_classes=10, normalizer=normalizer)
                    
                    # Run Verification
                    # Note: We suppress stdout inside run() if desired, but VerifierTopKPrune prints proof logs.
                    results_diff, results_p, results_p_prime = verifier.run(data_normalized)
                    
                    # Calculate Metrics
                    metrics = calculate_metrics(results_diff, results_p, results_p_prime)
                    
                    elapsed = time.time() - start_time
                    
                    if metrics:
                        row = {
                            "epsilon": eps,
                            "prune_layer": layer,
                            "tokens_kept": tokens,
                            "avg_diff_lb": f"{metrics['avg_diff_lb']:.5f}",
                            "avg_diff_ub": f"{metrics['avg_diff_ub']:.5f}",
                            "robustness_p": f"{metrics['robustness_p']:.2f}",
                            "robustness_pp": f"{metrics['robustness_pp']:.2f}",
                            "time_elapsed": f"{elapsed:.2f}"
                        }
                        writer.writerow(row)
                        csv_file.flush() # Update file immediately
                        
                        print(f"   -> Diff Bounds: [{row['avg_diff_lb']}, {row['avg_diff_ub']}]")
                        print(f"   -> Robustness: P={row['robustness_p']}, P'={row['robustness_pp']}")
                    else:
                        print("   -> No valid results collected.")

    print("\n" + "="*60)
    print(f"Experiment Suite Completed. Data saved to {output_csv}")
    print("="*60)

if __name__ == "__main__":
    main()
