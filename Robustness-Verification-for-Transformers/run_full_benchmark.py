import os
import sys
import psutil
import torch
import numpy as np
import time
import csv
from typing import List, Dict, Any

# --- Imports ---
from Parser import Parser, update_arguments
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from data_utils import set_seeds
from Verifiers.VerifierTopKPrune import VerifierTopKPrune

# --- Metrics Helper ---
def check_robustness(res: Dict[str, Any]) -> bool:
    """Returns True if the sample is certified robust (LowerBound_True > UpperBound_Others)."""
    label = res['label']
    lb = res['lower_bounds']
    ub = res['upper_bounds']
    
    # Threshold: The lower bound of the true class
    true_lb = lb[label]
    
    # Check against all other classes
    # Robust if True_LB > Max(Other_UB)
    other_classes = [i for i in range(len(ub)) if i != label]
    max_other_ub = np.max(ub[other_classes])
    
    return true_lb > max_other_ub

def safe_add_argument(parser, arg_name, **kwargs):
    """Helper to add arguments only if they don't already exist."""
    # Check if any of the option strings are already present
    existing_opts = parser._option_string_actions.keys()
    if arg_name not in existing_opts:
        parser.add_argument(arg_name, **kwargs)

# --- Main Script ---

if __name__ == "__main__":
    # 1. Parse Arguments
    argv = sys.argv[1:]
    parser = Parser.get_parser()

    # Pruning Args (Safely add them)
    safe_add_argument(parser, '--prune_tokens', action='store_true', help='Enable token pruning in P\'.')
    safe_add_argument(parser, '--prune_layer_idx', type=int, default=0, help='Layer index to start pruning.')
    safe_add_argument(parser, '--tokens_to_keep', type=int, default=9, help='Top-K tokens to keep.')
    safe_add_argument(parser, '--tokens_to_prune', type=int, default=0, help='Bottom-X tokens to prune.')
    
    # Benchmark Args
    # Note: 'samples' is usually already in Parser.py, so we skip adding it.
    # If it is NOT in Parser.py, uncomment the line below:
    # safe_add_argument(parser, '--samples', type=int, default=100, help='Number of samples to evaluate.')
    
    safe_add_argument(parser, '--csv_name', type=str, default=None, help='Custom name for output CSV.')

    args, _ = parser.parse_known_args(argv)
    args = update_arguments(args)

    # Force "Quiet" Mode for bulk processing
    args.verbose = False
    args.debug = False
    
    # Zonotope Settings
    args.error_reduction_method = 'box'
    args.max_num_error_terms = 30000
    args.num_input_error_terms = 28 * 28

    # 2. Setup Device & Seeds
    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.device = device

    # 3. Load Data
    # Load enough data for the requested samples
    print(f"Loading {args.samples} samples from MNIST (Please wait, loading imports)...")
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    
    data_normalized = []
    # Only load as many as we need (plus a buffer) to speed up startup
    target_count = args.samples
    
    for i, (x, y) in enumerate(test_loader):
        if i >= target_count: break
        data_normalized.append({"label": y.to(device), "image": x.to(device)})

    # 4. Load Model
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    
    # Try to load weights
    weights = "mnist_transformer.pt"
    if os.path.exists(weights):
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)
        # print(f"Loaded weights: {weights}")
    else:
        print("WARNING: Using random weights (File not found).")
    model.eval()

    # 5. Initialize Verifier
    verifier = VerifierTopKPrune(args, model, FakeLogger(), 10, normalizer)

    # 6. Run Experiment
    print("="*60)
    print(f"STARTING BENCHMARK")
    print(f"Samples: {len(data_normalized)}")
    print(f"Epsilon: {args.eps}")
    if args.prune_tokens:
        strat = f"Bottom-{args.tokens_to_prune}" if args.tokens_to_prune > 0 else f"Top-{args.tokens_to_keep}"
        print(f"Pruning: YES ({strat} after Layer {args.prune_layer_idx})")
    else:
        print(f"Pruning: NO")
    print("="*60)

    start_time = time.time()
    
    # The verifier runs the loop internally
    results_diff, results_p, results_p_prime = verifier.run(data_normalized)
    
    total_time = time.time() - start_time

    # 7. Calculate Metrics & Save CSV
    
    # Determine Output Filename
    if args.csv_name:
        out_file = args.csv_name
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        p_str = "pruned" if args.prune_tokens else "baseline"
        out_file = f"benchmark_{p_str}_eps{args.eps}_{timestamp}.csv"
    
    out_path = os.path.join("results", out_file)
    if not os.path.exists("results"): os.makedirs("results")

    certified_p = 0
    certified_p_prime = 0
    valid_count = len(results_diff)

    print(f"\nAnalysis complete. Saving to {out_path}...")

    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "sample_idx", "label", "time", 
            "robust_unpruned", "robust_pruned", 
            "L_real_unpruned", "L_real_pruned", 
            "U_real_unpruned", "U_real_pruned"
        ])

        for i in range(valid_count):
            res_p = results_p[i]
            res_pp = results_p_prime[i]
            label = res_p['label']

            # Robustness Check
            is_rob_p = check_robustness(res_p)
            is_rob_pp = check_robustness(res_pp)
            
            if is_rob_p: certified_p += 1
            if is_rob_pp: certified_p_prime += 1

            # Get Bounds for the True Class
            l_p = res_p['lower_bounds'][label]
            u_p = res_p['upper_bounds'][label]
            l_pp = res_pp['lower_bounds'][label]
            u_pp = res_pp['upper_bounds'][label]

            writer.writerow([
                res_p['index'], label, f"{res_p['time']:.4f}",
                is_rob_p, is_rob_pp,
                f"{l_p:.4f}", f"{l_pp:.4f}",
                f"{u_p:.4f}", f"{u_pp:.4f}"
            ])

    # 8. Print Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Samples Processed: {valid_count}")
    print(f"Total Runtime:           {total_time:.2f}s ({total_time/valid_count:.2f}s/sample)")
    print("-" * 30)
    print(f"Certified Robustness (Unpruned P):  {certified_p}/{valid_count} ({certified_p/valid_count*100:.2f}%)")
    print(f"Certified Robustness (Pruned P'):   {certified_p_prime}/{valid_count} ({certified_p_prime/valid_count*100:.2f}%)")
    print("="*60)
