import os
import sys
import psutil
import torch
import numpy as np
from typing import List, Dict, Any

# Assuming these are available in your environment
from Parser import Parser, update_arguments
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from vit_attack import pgd_attack

# --- Import the updated verifier ---
# Ensure IntervalBoundVerifier.py is in the same directory
from Verifiers.IntervalBoundVerifier import IntervalBoundDiffVerViT, sample_correct_samples


def analyze_results(results: List[Dict[str, Any]], args):
    """
    Analyzes verification results to compare unpruned (P) and pruned (P') models.

    This function assumes the 'results' list contains dictionaries with the following structure
    for each sample, which must be returned by your verifier:
    {
        'label': int,
        'lower_bounds_unpruned': np.ndarray,
        'upper_bounds_unpruned': np.ndarray,
        'lower_bounds_pruned': np.ndarray,
        'upper_bounds_pruned': np.ndarray,
        'lower_bounds_diff': np.ndarray,  // Bounds on (P - P')
        'upper_bounds_diff': np.ndarray,  // Bounds on (P - P')
    }
    """
    if not results:
        print("No valid samples were processed to analyze.")
        return

    # --- Metric Accumulators ---
    num_samples = len(results)
    
    # For Unpruned Model (P)
    certified_correct_unpruned = 0
    
    # For Pruned Model (P')
    certified_correct_pruned = 0
    
    # For Differential Analysis (P - P')
    diff_safety_failures = 0
    diff_margin_failures = 0
    total_lower_bound_diff_real_class = 0.0

    for res in results:
        label = res['label']
        num_classes = len(res['lower_bounds_unpruned'])
        other_indices = [i for i in range(num_classes) if i != label]

        # --- 1. Unpruned Model (P) Analysis ---
        L_unpruned_real = res['lower_bounds_unpruned'][label]
        U_unpruned_others = res['upper_bounds_unpruned'][other_indices]
        if L_unpruned_real > np.max(U_unpruned_others):
            certified_correct_unpruned += 1

        # --- 2. Pruned Model (P') Analysis ---
        L_pruned_real = res['lower_bounds_pruned'][label]
        U_pruned_others = res['upper_bounds_pruned'][other_indices]
        if L_pruned_real > np.max(U_pruned_others):
            certified_correct_pruned += 1
            
        # --- 3. Differential (P vs P') Analysis ---
        L_diff_real = res['lower_bounds_diff'][label]
        U_diff_others = res['upper_bounds_diff'][other_indices]
        
        total_lower_bound_diff_real_class += L_diff_real
        
        # Safety Failure: Is it possible for the pruned model's confidence in the
        # correct class to be higher? (i.e., is P_real - P'_real <= 0 possible?)
        if L_diff_real <= 0:
            diff_safety_failures += 1
            
        # Margin Failure: Can the drop in confidence for the real class be worse than
        # the drop for another class? (i.e., is (L1-U2)_real <= (U1-L2)_other possible?)
        if L_diff_real <= np.max(U_diff_others):
            diff_margin_failures += 1

    # --- Calculate Final Metrics ---
    # Unpruned Model
    cert_acc_unpruned = (certified_correct_unpruned / num_samples) * 100
    
    # Pruned Model
    cert_acc_pruned = (certified_correct_pruned / num_samples) * 100
    # This directly answers your question: the number of failures is (num_samples - certified_correct_pruned)
    robustness_failures_pruned = num_samples - certified_correct_pruned

    # Differential
    avg_L_diff_real = total_lower_bound_diff_real_class / num_samples
    diff_safety_fail_rate = (diff_safety_failures / num_samples) * 100
    diff_margin_fail_rate = (diff_margin_failures / num_samples) * 100

    # --- Print Results Table ---
    print("\n" + "="*80)
    print(f"VERIFICATION ANALYSIS ({num_samples} SAMPLES, ε={args.eps})")
    print(f"Unpruned (P) vs. Pruned (P') at Layer {args.prune_layer_idx}, Keeping {args.tokens_to_keep} Tokens")
    print("="*80)

    print("\n## Model Robustness Metrics\n")
    print(f"**Certified Accuracy (Unpruned):** {cert_acc_unpruned:.2f}% ({certified_correct_unpruned}/{num_samples})")
    print("  - Interpretation: Model is provably correct for this many samples.")
    print(f"**Certified Accuracy (Pruned):** {cert_acc_pruned:.2f}% ({certified_correct_pruned}/{num_samples})")
    print("  - Interpretation: Model is provably correct after being pruned.")
    
    print(f"\n**Robustness Failures (Pruned Model):** {robustness_failures_pruned} / {num_samples} samples")
    print("  - This is the count you asked for: samples where the pruned model's lower bound for the\n"
          "    correct class can be surpassed by the upper bound of another class.")
    print("-" * 80)
    
    print("\n## Differential Analysis Metrics (P - P')\n")
    print(f"**Guaranteed Confidence Drop (Avg):** {avg_L_diff_real:.5f}")
    print("  - Interpretation: The average guaranteed drop in the correct class's logit value after pruning.\n"
          "    A negative value means confidence is guaranteed to decrease on average.")
    
    print(f"\n**Potential Confidence Loss:** {diff_safety_failures} / {num_samples} ({diff_safety_fail_rate:.2f}%)")
    print("  - Interpretation: Samples where the pruned model is NOT guaranteed to have lower confidence\n"
          "    in the correct class than the unpruned model (i.e., L_diff[real] <= 0).")

    print(f"\n**Guaranteed Margin Preservation Failure:** {diff_margin_failures} / {num_samples} ({diff_margin_fail_rate:.2f}%)")
    print("  - Interpretation: Samples where the confidence drop for the correct class is NOT guaranteed\n"
          "    to be smaller than the confidence drop for all other classes.")

    print("="*80)


# --- Main Execution Setup ---
if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = Parser.get_parser()

    parser.add_argument('--prune_tokens', action='store_true',
                        help='Enable First-K token pruning in the P\' model for differential verification.')
    parser.add_argument('--prune_layer_idx', type=int, default=0,
                        help='Transformer layer index AFTER which to apply token pruning in P\'.')
    parser.add_argument('--tokens_to_keep', type=int, default=9,
                        help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')

    args, _ = parser.parse_known_args(argv)

    # --- Configuration for 100 Samples and Quiet Output ---
    args.samples = 100
    args.verbose = False
    args.debug = False
    args.log_error_terms_and_time = False
    # --------------------------------------------------------

    args = update_arguments(args)
    args.error_reduction_method = 'box'
    args.max_num_error_terms = 30000

    # Set other necessary arguments based on your environment
    args.with_lirpa_transformer = False
    args.all_words = True
    args.concretize_special_norm_error_together = True
    args.num_input_error_terms = 28 * 28

    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if psutil.cpu_count() > 4 and args.cpu_range != "Default":
        start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
        os.sched_setaffinity(0, {i for i in range(start, end + 1)})

    from data_utils import set_seeds
    set_seeds(args.seed)

    test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
    model.eval()

    print(f"--- Differential Verification Setup ---")
    print(f"Target Samples: {args.samples}")
    print(f"Pruning Active: {args.prune_tokens} (Layer {args.prune_layer_idx}, Keep {args.tokens_to_keep})")
    print(f"Epsilon: {args.eps}")
    print("---------------------------------------")

    logger = FakeLogger()

    data_normalized = []
    for i, (x, y) in enumerate(test_data):
        data_normalized.append({
            "label": y.to(device),
            "image": x.to(device)
        })
        if i == args.samples - 1:
            break

    run_pgd = args.pgd if hasattr(args, 'pgd') else False
    if run_pgd:
        print("PGD attack execution skipped. Focused on differential verification.")
    else:
        if not hasattr(args, 'eps') or args.eps <= 0:
            print("Argument --eps must be set to a positive value for differential verification.")
        else:
            verifier = IntervalBoundDiffVerViT(args, model, logger, num_classes=10, normalizer=normalizer)
            
            # Run verification and collect the results list
            # NOTE: Your verifier must be updated to return the new data structure!
            aggregated_results = verifier.run(data_normalized)
            
            # Analyze the collected data and print the metrics
            analyze_results(aggregated_results, args)
