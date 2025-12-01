import os
import sys
import psutil
import torch
import numpy as np
from typing import List, Dict, Any

# --- Assumed Environmental Imports ---
# Ensure these files are in your python path
from Parser import Parser, update_arguments
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from vit_attack import pgd_attack
from data_utils import set_seeds

# --- Import the Updated Verifier ---
# Ensure the file 'Verifiers/VerifierTopKPrune.py' exists with the new class
from Verifiers.VerifierTopKPrune import VerifierTopKPrune


# ==============================================================================
#  ANALYSIS HELPER FUNCTIONS
# ==============================================================================

def analyze_robustness(results: List[Dict[str, Any]], model_name: str):
    """
    Analyzes robustness for a single model's results (P or P').
    Checks if LowerBound(TrueClass) > UpperBound(OtherClasses).
    """
    num_robustness_failures = 0
    valid_samples = len(results)

    if valid_samples == 0:
        return 

    for result in results:
        label = result['label']
        lower = result['lower_bounds']
        upper = result['upper_bounds']
        num_classes = len(lower)

        L_real = lower[label]

        # Check for overlap with any other class
        is_failure = False
        for j in range(num_classes):
            if j == label:
                continue
            U_other = upper[j]
            # Failure condition: The worst-case score for the real class 
            # is lower than the best-case score for an incorrect class.
            if L_real <= U_other:
                is_failure = True
                break 
        
        if is_failure:
            num_robustness_failures += 1

    print(f"\nRobustness Verification ({model_name}):")
    print(f"   Samples where L[real_class] <= U[other_class] for at least one other class:")
    print(f"   Failures: {num_robustness_failures} / {valid_samples} ({num_robustness_failures/valid_samples*100:.2f}%)")
    print(f"   Verified Robust: {valid_samples - num_robustness_failures} / {valid_samples}")


def analyze_all_results(results_diff: List[Dict[str, Any]],
                        results_p: List[Dict[str, Any]],
                        results_p_prime: List[Dict[str, Any]],
                        args):
    """
    Analyzes differential bounds, prints PER-SAMPLE per-class bounds, and then averages.
    """
    valid_samples = len(results_diff)
    if valid_samples == 0:
        print("No valid samples were processed to analyze.")
        return

    # --- 1. Real Class Differential Metrics ---
    total_lower_bound_real_class = 0.0
    total_upper_bound_real_class = 0.0
    
    # --- 2. Per-Class Accumulators for Final Average ---
    num_classes = 10
    avg_lower_p = np.zeros(num_classes)
    avg_upper_p = np.zeros(num_classes)
    avg_lower_p_prime = np.zeros(num_classes)
    avg_upper_p_prime = np.zeros(num_classes)
    avg_lower_diff = np.zeros(num_classes)
    avg_upper_diff = np.zeros(num_classes)

    print("\n" + "="*100)
    print(f"DETAILED SAMPLE ANALYSIS ({valid_samples} SAMPLES)")
    
    prune_mode = f"Bottom-{args.tokens_to_prune}" if args.tokens_to_prune > 0 else f"Top-{args.tokens_to_keep}"
    print(f"P: Unpruned | P': Pruned (Layer {args.prune_layer_idx}, Strategy: {prune_mode})")
    print("="*100)

    for i in range(valid_samples):
        res_diff = results_diff[i]
        res_p = results_p[i]
        res_pp = results_p_prime[i]
        
        label = res_diff['label']
        total_lower_bound_real_class += res_diff['lower_bounds'][label]
        total_upper_bound_real_class += res_diff['upper_bounds'][label]
        
        print(f"\n--- Sample {i} (True Label: {label}) ---")
        print(f"   {'Class':<5} | {'P Low':<10} | {'P Up':<10} | {'P\' Low':<10} | {'P\' Up':<10} | {'Diff Low':<10} | {'Diff Up':<10}")
        print("-" * 95)
        
        for c in range(num_classes):
            # Current Sample Values
            lp = res_p['lower_bounds'][c]
            up = res_p['upper_bounds'][c]
            lpp = res_pp['lower_bounds'][c]
            upp = res_pp['upper_bounds'][c]
            ldiff = res_diff['lower_bounds'][c]
            udiff = res_diff['upper_bounds'][c]
            
            marker = "<<" if c == label else ""
            print(f"   {c:<5} | {lp:<10.4f} | {up:<10.4f} | {lpp:<10.4f} | {upp:<10.4f} | {ldiff:<10.4f} | {udiff:<10.4f} {marker}")

            # Accumulate for Averages
            avg_lower_p[c] += lp
            avg_upper_p[c] += up
            avg_lower_p_prime[c] += lpp
            avg_upper_p_prime[c] += upp
            avg_lower_diff[c] += ldiff
            avg_upper_diff[c] += udiff

    # Compute Averages
    avg_lower_p /= valid_samples
    avg_upper_p /= valid_samples
    avg_lower_p_prime /= valid_samples
    avg_upper_p_prime /= valid_samples
    avg_lower_diff /= valid_samples
    avg_upper_diff /= valid_samples
    
    avg_L_real = total_lower_bound_real_class / valid_samples
    avg_U_real = total_upper_bound_real_class / valid_samples
    
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    
    print("1. Average Differential Bounds (Real Class):")
    print(f"   Lower Bound (Avg L_P - U_P'): {avg_L_real:.5f}")
    print(f"   Upper Bound (Avg U_P - L_P'): {avg_U_real:.5f}")
    
    print("\n2. Average Bounds per Class (All Samples):")
    print(f"   {'Class':<5} | {'P Low':<10} | {'P Up':<10} | {'P\' Low':<10} | {'P\' Up':<10} | {'Diff Low':<10} | {'Diff Up':<10}")
    print("-" * 95)
    for c in range(num_classes):
        print(f"   {c:<5} | {avg_lower_p[c]:<10.4f} | {avg_upper_p[c]:<10.4f} | {avg_lower_p_prime[c]:<10.4f} | {avg_upper_p_prime[c]:<10.4f} | {avg_lower_diff[c]:<10.4f} | {avg_upper_diff[c]:<10.4f}")

    # --- Part 3: Individual Robustness Metrics ---
    analyze_robustness(results_p, "P (Unpruned)")
    analyze_robustness(results_p_prime, "P' (Pruned)")
    print("="*80)


# ==============================================================================
#  MAIN EXECUTION SETUP
# ==============================================================================

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = Parser.get_parser()

    # --- New Pruning Arguments ---
    parser.add_argument('--prune_tokens', action='store_true',
                        help='Enable token pruning in the P\' model for differential verification.')
    parser.add_argument('--prune_layer_idx', type=int, default=0,
                        help='Transformer layer index AFTER which to apply token pruning in P\'.')
    parser.add_argument('--tokens_to_keep', type=int, default=9,
                        help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')
    parser.add_argument('--tokens_to_prune', type=int, default=0,
                        help='Number of tokens to prune (Bottom-X). If > 0, overrides tokens_to_keep.')

    # Parse and Update
    args, _ = parser.parse_known_args(argv)

    # --- Manual Overrides for Testing ---
    # You can comment these out to use pure command line args
    args.samples = 5
    args.verbose = False
    args.debug = False
    args.log_error_terms_and_time = False

    args = update_arguments(args)
    
    # --- Zonotope Specific Settings ---
    args.error_reduction_method = 'box'
    args.max_num_error_terms = 30000
    args.with_lirpa_transformer = False
    args.all_words = True
    args.concretize_special_norm_error_together = True
    args.num_input_error_terms = 28 * 28  # MNIST specific

    # --- Hardware Setup ---
    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if psutil.cpu_count() > 4 and args.cpu_range != "Default":
        try:
            start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
            os.sched_setaffinity(0, {i for i in range(start, end + 1)})
        except:
            print("[Warning] Could not set CPU affinity.")

    set_seeds(args.seed)

    # --- Data Loading ---
    # Load raw dataloader
    test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
    # Reset seeds again to ensure consistency after dataloader init
    set_seeds(args.seed)

    # Move to list of dicts for Verifier compatibility
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    
    # Pre-fetch data into memory and normalize structure
    data_normalized = []
    print("Loading data...")
    for i, (x, y) in enumerate(test_data):
        data_normalized.append({
            "label": y.to(device),
            "image": x.to(device)
        })
        # Optimization: Only load as many as we need to sample from
        # (Since sample_correct_samples picks randomly, we load a buffer)
        if i >= 200: 
            break

    # --- Model Setup ---
    # Initialize tiny MNIST ViT
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

    # Load Weights
    weights_path = "mnist_transformer.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"[WARNING] {weights_path} not found. Running with random weights.")
    
    model.eval()

    # --- Run Verification ---
    print(f"\n--- Verification Setup ---")
    print(f"Target Samples: {args.samples}")
    if args.prune_tokens:
        if args.tokens_to_prune > 0:
            print(f"Pruning Strategy: Bottom-{args.tokens_to_prune} (Sound)")
        else:
            print(f"Pruning Strategy: Top-{args.tokens_to_keep} (Sound)")
        print(f"Pruning Layer Index: {args.prune_layer_idx}")
    else:
        print(f"Pruning Active: False")
    print(f"Epsilon: {args.eps}")
    print("--------------------------")

    logger = FakeLogger()

    run_pgd = args.pgd if hasattr(args, 'pgd') else False
    if run_pgd:
        print("PGD attack execution skipped. Focused on verification.")
    else:
        if not hasattr(args, 'eps') or args.eps <= 0:
            print("Argument --eps must be set to a positive value for verification.")
        else:
            # Instantiate the Sound Top-K Verifier
            # Note: We pass the 'normalizer' imported from mnist
            verifier = VerifierTopKPrune(args, model, logger, num_classes=10, normalizer=normalizer)
            
            # Run verification
            results_diff, results_p, results_p_prime = verifier.run(data_normalized)
            
            # Analyze and print using the detailed reporter
            analyze_all_results(results_diff, results_p, results_p_prime, args)
