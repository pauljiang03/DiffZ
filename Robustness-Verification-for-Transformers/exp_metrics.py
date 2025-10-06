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


def analyze_robustness(results: List[Dict[str, Any]], model_name: str):
    """
    Analyzes robustness for a single model's results (P or P').
    Counts samples where the lower bound of the correct class is not strictly
    greater than the upper bound of all other classes.
    """
    num_robustness_failures = 0
    valid_samples = len(results)

    if valid_samples == 0:
        return  # Nothing to do

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
            if L_real <= U_other:
                is_failure = True
                break  # Found one overlap, no need to check others for this sample
        
        if is_failure:
            num_robustness_failures += 1

    print(f"\n4. Robustness Verification ({model_name}):")
    print(f"   Samples where L[real_class] <= U[other_class] for at least one other class:")
    print(f"   Count: {num_robustness_failures} / {valid_samples} ({num_robustness_failures/valid_samples*100:.2f}%)")
    print("   Interpretation: This indicates samples that are not provably robust against the given perturbation.")


def analyze_all_results(results_diff: List[Dict[str, Any]],
                        results_p: List[Dict[str, Any]],
                        results_p_prime: List[Dict[str, Any]],
                        args):
    """
    Analyzes differential bounds and individual model robustness metrics.
    """
    # --- Part 1: Differential Metrics ---
    total_lower_bound_real_class = 0.0
    total_upper_bound_real_class = 0.0
    max_upper_bound_real_class = -float('inf')
    num_verification_failures = 0
    valid_samples = len(results_diff)

    if valid_samples == 0:
        print("No valid samples were processed to analyze.")
        return

    for result in results_diff:
        label = result['label']
        lower = result['lower_bounds']
        upper = result['upper_bounds']

        L_real = lower[label]
        U_real = upper[label]
        
        total_lower_bound_real_class += L_real
        total_upper_bound_real_class += U_real
        max_upper_bound_real_class = max(max_upper_bound_real_class, U_real)

        if lower[label] <= 0:
            num_verification_failures += 1

    avg_L_real = total_lower_bound_real_class / valid_samples
    avg_U_real = total_upper_bound_real_class / valid_samples
    max_U_real = max_upper_bound_real_class
    
    print("\n" + "="*80)
    print(f"VERIFICATION METRICS ({valid_samples} SAMPLES)")
    print(f"P: Unpruned Model | P': Pruned Model (Layer {args.prune_layer_idx}, Keep {args.tokens_to_keep})")
    print("="*80)
    
    print("1. Average Differential Bounds (Real Class):")
    print(f"   Lower Bound (Avg L_P - U_P'): {avg_L_real:.5f}")
    print(f"   Upper Bound (Avg U_P - L_P'): {avg_U_real:.5f}")
    
    print("\n2. Highest Differential Upper Bound (Real Class):")
    print(f"   Max (U_P - L_P'): {max_U_real:.5f}")
    
    print("\n3. Pruning Efficacy/Safety Metric:")
    print(f"   Samples where Differential Lower Bound (L_P - U_P') for REAL class <= 0:")
    print(f"   Count: {num_verification_failures} / {valid_samples} ({num_verification_failures/valid_samples*100:.2f}%)")
    print("   Interpretation: This indicates samples where the difference P_real - P'_real might be non-positive.")
    
    # --- Part 2: Individual Robustness Metrics ---
    analyze_robustness(results_p, "P (Unpruned)")
    analyze_robustness(results_p_prime, "P' (Pruned)")
    print("="*80)


# --- Main Execution Setup ---
argv = sys.argv[1:]
parser = Parser.get_parser()

parser.add_argument('--prune_tokens', action='store_true',
                    help='Enable First-K token pruning in the P\' model for differential verification.')
parser.add_argument('--prune_layer_idx', type=int, default=0,
                    help='Transformer layer index AFTER which to apply token pruning in P\'.')
parser.add_argument('--tokens_to_keep', type=int, default=9,
                    help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')

args, _ = parser.parse_known_args(argv)

args.samples = 5
args.verbose = False
args.debug = False
args.log_error_terms_and_time = False

args = update_arguments(args)
args.error_reduction_method = 'box'
args.max_num_error_terms = 30000

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

print(f"--- Verification Setup ---")
print(f"Target Samples: {args.samples}")
print(f"Pruning Active: {args.prune_tokens} (Layer {args.prune_layer_idx}, Keep {args.tokens_to_keep})")
print(f"Epsilon: {args.eps}")
print("--------------------------")

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
    print("PGD attack execution skipped. Focused on verification.")
else:
    if not hasattr(args, 'eps') or args.eps <= 0:
        print("Argument --eps must be set to a positive value for verification.")
    else:
        verifier = IntervalBoundDiffVerViT(args, model, logger, num_classes=10, normalizer=normalizer)
        
        # Run verification to get all three sets of results
        results_diff, results_p, results_p_prime = verifier.run(data_normalized)
        
        # Analyze and print all metrics
        analyze_all_results(results_diff, results_p, results_p_prime, args)
