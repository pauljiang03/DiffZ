import os
import sys
import psutil
import torch
import numpy as np
from typing import List, Dict, Any

# Assuming these are available in your environment based on your previous files
from Parser import Parser, update_arguments
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from vit_attack import pgd_attack

# --- Import the VerifierTopKPrune ---
from Verifiers.VerifierTopKPrune import VerifierTopKPrune, sample_correct_samples


def print_detailed_analysis(results_diff: List[Dict[str, Any]],
                            results_p: List[Dict[str, Any]],
                            results_p_prime: List[Dict[str, Any]],
                            args):
    """
    Generates a detailed statistical report on robustness and differential stability.
    """
    valid_samples = len(results_diff)
    if valid_samples == 0:
        print("No valid samples were processed to analyze.")
        return

    # --- Helper to extract core robustness stats ---
    def get_model_stats(res_list):
        # Margin = Lower_bound(correct_class) - Max(Upper_bound(other_classes))
        # If Margin > 0, the sample is PROVABLY ROBUST.
        margins = []
        widths = []
        verified_count = 0
        
        for r in res_list:
            lbl = r['label']
            low = r['lower_bounds']
            up = r['upper_bounds']
            
            # 1. Interval Width (Uncertainty) of the correct class
            # Tighter bounds (smaller width) -> More precise verification
            widths.append(up[lbl] - low[lbl])
            
            # 2. Robustness Margin
            # Find the best "challenger" class (highest upper bound among incorrect classes)
            other_classes = [i for i in range(len(up)) if i != lbl]
            max_u_other = np.max([up[i] for i in other_classes])
            
            # The gap between our guaranteed lower bound and their guaranteed upper bound
            margin = low[lbl] - max_u_other
            margins.append(margin)
            
            if margin > 0:
                verified_count += 1
                
        return np.array(margins), np.array(widths), verified_count

    # 1. Analyze P (Unpruned)
    margins_p, widths_p, verified_p = get_model_stats(results_p)
    
    # 2. Analyze P' (Pruned)
    margins_p_prime, widths_p_prime, verified_p_prime = get_model_stats(results_p_prime)
    
    # 3. Analyze Differential (P - P')
    # These bounds represent the range of possible divergence: [Min(P-P'), Max(P-P')]
    diff_lowers = []
    diff_uppers = []
    diff_widths = [] # How "loose" is our differential bound?
    
    for r in results_diff:
        lbl = r['label']
        L_diff = r['lower_bounds'][lbl]
        U_diff = r['upper_bounds'][lbl]
        
        diff_lowers.append(L_diff)
        diff_uppers.append(U_diff)
        diff_widths.append(U_diff - L_diff)
    
    diff_lowers = np.array(diff_lowers)
    diff_uppers = np.array(diff_uppers)
    diff_widths = np.array(diff_widths)
    
    # 4. Timing
    times = np.array([r['time'] for r in results_diff])
    
    # --- PRINT REPORT ---
    print("\n" + "="*80)
    print(f"DETAILED VERIFICATION REPORT ({valid_samples} SAMPLES)")
    print(f"Noise (eps): {args.eps} | Pruning: Keep {args.tokens_to_keep} tokens after Layer {args.prune_layer_idx}")
    print("="*80)
    
    print(f"\n--- 1. BASELINE MODEL P (Unpruned) ---")
    print(f"   Verified Robustness: {verified_p}/{valid_samples} ({verified_p/valid_samples*100:.1f}%)")
    print(f"   Avg Safety Margin:   {np.mean(margins_p):.5f}  (Higher is better, >0 is safe)")
    print(f"   Avg Interval Width:  {np.mean(widths_p):.5f}   (Lower is tighter)")
    print(f"   Worst Safety Margin: {np.min(margins_p):.5f}")

    print(f"\n--- 2. PRUNED MODEL P' (Top-{args.tokens_to_keep}) ---")
    print(f"   Verified Robustness: {verified_p_prime}/{valid_samples} ({verified_p_prime/valid_samples*100:.1f}%)")
    print(f"   Avg Safety Margin:   {np.mean(margins_p_prime):.5f}")
    print(f"   Avg Interval Width:  {np.mean(widths_p_prime):.5f}")
    print(f"   Margin Shift vs P:   {np.mean(margins_p_prime - margins_p):.5f} (Negative means pruning hurt robustness)")

    print(f"\n--- 3. DIFFERENTIAL STABILITY (P vs P') ---")
    print(f"   Measures the output divergence of the correct class.")
    print(f"   Range [L_diff, U_diff] contains the true difference (P - P').")
    print(f"   Avg Differential Range:  [{np.mean(diff_lowers):.5f}, {np.mean(diff_uppers):.5f}]")
    print(f"   Max Possible Divergence: {np.max(np.abs(np.concatenate((diff_lowers, diff_uppers)))):.5f}")
    print(f"   Avg Diff Uncertainty:    {np.mean(diff_widths):.5f} (Size of the differential gap)")

    print(f"\n--- 4. PERFORMANCE ---")
    print(f"   Avg Time per Sample: {np.mean(times):.4f}s")
    print(f"   Total Time:          {np.sum(times):.4f}s")
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
        verifier = VerifierTopKPrune(args, model, logger, num_classes=10, normalizer=normalizer)
        
        # Run verification
        results_diff, results_p, results_p_prime = verifier.run(data_normalized)
        
        # Print the new detailed report
        print_detailed_analysis(results_diff, results_p, results_p_prime, args)
