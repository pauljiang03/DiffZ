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
from IntervalBoundVerifier import IntervalBoundDiffVerViT, sample_correct_samples 


def analyze_results(results: List[Dict[str, Any]], args):
    """
    Analyzes the collected differential bounds for the three required metrics.
    """
    total_lower_bound_real_class = 0.0
    total_upper_bound_real_class = 0.0
    max_upper_bound_real_class = -float('inf')
    num_verification_failures = 0
    valid_samples = len(results)

    if valid_samples == 0:
        print("No valid samples were processed to analyze.")
        return

    for result in results:
        label = result['label']
        lower = result['lower_bounds']
        upper = result['upper_bounds']

        # 1. & 2. Metrics for the Real Class (Label)
        L_real = lower[label]
        U_real = upper[label]
        
        total_lower_bound_real_class += L_real
        total_upper_bound_real_class += U_real
        max_upper_bound_real_class = max(max_upper_bound_real_class, U_real)

        # 3. Pruning Efficacy/Safety Metric (Proxy):
        # We check if the lower bound of the difference (P_real - P'_real) is non-positive.
        # This means P_real <= P'_real is possible, which is a potential safety failure 
        # (pruning P' has shifted the score enough to violate the original prediction ordering).
        L_real_diff = lower[label] 
        
        if L_real_diff <= 0:
             num_verification_failures += 1
             

    # 1. Average Bounds
    avg_L_real = total_lower_bound_real_class / valid_samples
    avg_U_real = total_upper_bound_real_class / valid_samples
    
    # 2. Highest Upper Bound
    max_U_real = max_upper_bound_real_class
    
    print("\n" + "="*70)
    print(f"DIFFERENTIAL VERIFICATION METRICS ({valid_samples} SAMPLES)")
    print(f"P: Unpruned Model | P': Pruned Model (Layer {args.prune_layer_idx}, Keep {args.tokens_to_keep})")
    print("="*70)
    
    print("1. Average Differential Bounds (Real Class):")
    print(f"   Lower Bound (Avg L1 - U2): {avg_L_real:.5f}")
    print(f"   Upper Bound (Avg U1 - L2): {avg_U_real:.5f}")
    
    print("\n2. Highest Differential Upper Bound (Real Class):")
    print(f"   Max (U1 - L2): {max_U_real:.5f}")
    
    print("\n3. Pruning Efficacy/Safety Metric (L_diff[real_class] <= 0):")
    print(f"   Samples where Differential Lower Bound (L1-U2) for REAL class <= 0:")
    print(f"   Count: {num_verification_failures} / {valid_samples} ({num_verification_failures/valid_samples*100:.2f}%)")
    print("   Interpretation: This indicates the number of samples where the difference P_real - P'_real might be non-positive.")
    print("="*70)



# --- Main Execution Setup ---
argv = sys.argv[1:]
parser = Parser.get_parser()

parser.add_argument('--prune_tokens', action='store_true',
                    help='Enable First-K token pruning in the P\' model for differential verification.')
parser.add_argument('--prune_layer_idx', type=int, default=0,
                    help='Transformer layer index AFTER which to apply token pruning in P\'.')
parser.add_argument('--tokens_to_keep', type=int, default=9,
                    help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')

# --- CONFLICTING ARGUMENT DEFINITIONS REMOVED ---
# We rely on the base Parser to define --debug, --verbose, and --log_error_terms_and_time.
# --------------------------------------------------

args, _ = parser.parse_known_args(argv)

# --- Configuration for 100 Samples and Quiet Output ---
# We keep these lines to enforce a quiet run and set the sample count.
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
    # Note: args.gpu is supplied via command line and handled here
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if psutil.cpu_count() > 4 and args.cpu_range != "Default":
    start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
    os.sched_setaffinity(0, {i for i in range(start, end + 1)})

# Assuming data_utils is available for set_seeds
from data_utils import set_seeds
set_seeds(args.seed)

test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
set_seeds(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
args.device = device 
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

# Ensure 'mnist_transformer.pt' is available
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
    # Ensure we only grab up to 'args.samples' amount of data
    if i == args.samples - 1:
        break


run_pgd = args.pgd if hasattr(args, 'pgd') else False
if run_pgd:
    # PGD attack logic (omitted for brevity, as we focus on verification)
    print("PGD attack execution skipped. Focused on differential verification.")
else:
    if not hasattr(args, 'eps') or args.eps <= 0:
        # Note: 'eps' should now be accessible since it was passed via command line
        print("Argument --eps must be set to a positive value for differential verification.")
    else:
        verifier = IntervalBoundDiffVerViT(args, model, logger, num_classes=10, normalizer=normalizer)
        
        # Run verification and collect the results list
        aggregated_results = verifier.run(data_normalized)
        
        # Analyze the collected data and print the metrics
        analyze_results(aggregated_results, args)
