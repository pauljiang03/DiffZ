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
    Analyzes the collected differential bounds (P - P') to evaluate the impact of pruning.
    This function uses ONLY the differential bounds provided by the verifier.
    """
    if not results:
        print("No valid samples were processed to analyze.")
        return

    # --- Metric Accumulators ---
    num_samples = len(results)
    
    total_lower_bound_diff_real = 0.0
    non_monotonic_confidence_failures = 0
    margin_preservation_failures = 0
    
    # For calculating the severity of failures
    total_overlapping_classes_in_failures = 0

    for res in results:
        # NOTE: 'lower_bounds' and 'upper_bounds' from your verifier are the
        # bounds on the DIFFERENCE between the unpruned and pruned models (P - P').
        label = res['label']
        lower_diff = res['lower_bounds']
        upper_diff = res['upper_bounds']

        num_classes = len(lower_diff)
        other_indices = [i for i in range(num_classes) if i != label]

        # Get the guaranteed bounds for the difference of the correct class
        L_diff_real = lower_diff[label]
        
        # --- Accumulate data for metrics ---
        total_lower_bound_diff_real += L_diff_real
        
        # Metric 2: Non-Monotonic Confidence Risk
        # Does the lower bound of the difference dip below zero?
        # If L_diff_real <= 0, it's possible that P_real <= P'_real, meaning the
        # pruned model could be MORE confident than the original, which is non-monotonic.
        if L_diff_real <= 0:
            non_monotonic_confidence_failures += 1
            
        # Metric 3: Robustness Margin Preservation Failure
        # Is the guaranteed drop for the real class potentially worse than the
        # drop for some other class? This happens if L_diff[real] <= max(U_diff[other]).
        max_U_diff_others = np.max(upper_diff[other_indices])
        is_failure = L_diff_real <= max_U_diff_others
        
        if is_failure:
            margin_preservation_failures += 1
            
            # Metric 4: Severity of Failure
            # If it's a failure, count how many other classes contribute to it.
            overlapping_classes_count = np.sum(upper_diff[other_indices] >= L_diff_real)
            total_overlapping_classes_in_failures += overlapping_classes_count

    # --- Calculate Final Metrics ---
    # Metric 1
    avg_L_diff_real = total_lower_bound_diff_real / num_samples
    
    # Metric 2
    non_monotonic_fail_rate = (non_monotonic_confidence_failures / num_samples) * 100
    
    # Metric 3
    margin_fail_rate = (margin_preservation_failures / num_samples) * 100
    
    # Metric 4
    avg_severity = 0.0
    if margin_preservation_failures > 0:
        avg_severity = total_overlapping_classes_in_failures / margin_preservation_failures

    # --- Print Results Table ---
    print("\n" + "="*80)
    print(f"DIFFERENTIAL VERIFICATION METRICS ({num_samples} SAMPLES, ε={args.eps})")
    print(f"Analysis of the change from Unpruned (P) to Pruned (P')")
    print("="*80)

    print("\n## Metric 1: Guaranteed Confidence Change (Correct Class)\n")
    print(f"**Average Lower Bound on Difference (P_real - P'_real):** {avg_L_diff_real:.5f}")
    print("  - **Interpretation:** The average guaranteed change in the correct class's logit.\n"
          "    A negative value means confidence is guaranteed to drop after pruning.")
    print("-" * 80)

    print("\n## Metric 2: Non-Monotonic Confidence Risk\n")
    print(f"**Risk Count:** {non_monotonic_confidence_failures} / {num_samples} ({non_monotonic_fail_rate:.2f}%)")
    print("  - **Interpretation:** Samples where the pruned model is NOT guaranteed to be less\n"
          "    confident than the original. Pruning should ideally always reduce confidence.")
    print("-" * 80)

    print("\n## Metric 3: Margin Preservation Failure\n")
    print(f"**Failure Count:** {margin_preservation_failures} / {num_samples} ({margin_fail_rate:.2f}%)")
    print("  - **Interpretation:** Samples where the prediction margin is NOT guaranteed to be preserved.\n"
          "    The confidence drop for the correct class could be worse than for another class.")
    print("-" * 80)

    print("\n## Metric 4: Average Failure Severity\n")
    print(f"**Avg. Overlapping Classes (in failure cases):** {avg_severity:.2f}")
    print("  - **Interpretation:** When the margin fails (Metric 3), this is the average number of\n"
          "    other classes that pose a risk. Higher is worse.")
    print("="*80)
    print("\n**Note:** These metrics analyze the *change* due to pruning. To get the final\n"
          "Certified Accuracy of the pruned model, the verifier must be modified.")

# --- Main Execution Setup ---
# (The rest of your script from if __name__ == "__main__": onwards stays the same)
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
            aggregated_results = verifier.run(data_normalized)
            
            # Analyze the collected data and print the metrics
            analyze_results(aggregated_results, args)
