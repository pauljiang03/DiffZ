import torch
import os
import sys
from typing import List, Dict, Any

# --- Import Project Modules ---
from vit import ViT
from mnist import mnist_test_dataloader, normalizer
from Verifiers.VerifierTopKPrune import VerifierTopKPrune
from fake_logger import FakeLogger

# --- Configuration Helper ---
class MockArgs:
    """Helper to create an arguments object required by the Verifier."""
    def __init__(self, eps, p_tokens, tokens_keep, tokens_prune, layer_idx):
        # Experiment Settings
        self.eps = eps
        self.p = float('inf')
        self.prune_tokens = p_tokens
        self.tokens_to_keep = tokens_keep
        self.tokens_to_prune = tokens_prune
        self.prune_layer_idx = layer_idx
        
        # Standard System Settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.samples = 1  # We run one by one manually
        self.verbose = False
        self.debug = False
        self.results_directory = "results_detailed" # Dump logs here
        if not os.path.exists(self.results_directory): os.makedirs(self.results_directory)
        
        # Model Specs
        self.res = 28
        self.method = "zonotope"
        self.hidden_act = "relu"
        
        # Zonotope Specs (Critical for shapes)
        self.error_reduction_method = 'box'
        self.max_num_error_terms = 30000
        self.num_input_error_terms = 28 * 28
        self.num_fast_dot_product_layers_due_to_switch = 100
        self.add_softmax_sum_constraint = False
        self.keep_intermediate_zonotopes = False
        self.with_lirpa_transformer = False
        self.all_words = True
        self.concretize_special_norm_error_together = True

def main():
    # 1. Setup Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running detailed analysis on {device}...")
    
    # 2. Load Model
    print("Loading Model...")
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    
    weights = "mnist_transformer.pt"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        print("[WARNING] Weights file not found! Using random weights.")
    model.eval()

    # 3. Select 3 Specific Valid Samples
    # We scan the dataset and pick the first 3 that the model classifies correctly.
    print("Selecting 3 valid samples for consistent comparison...")
    loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    selected_samples = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=-1)
        
        if pred.item() == y.item():
            # Store in the format expected by the Verifier (list of dicts)
            selected_samples.append({"image": x, "label": y})
        
        if len(selected_samples) == 3:
            break
            
    print(f"Selected samples with True Labels: {[s['label'].item() for s in selected_samples]}\n")

    # 4. Run the Grid Search
    epsilons = [0.001, 0.005, 0.01]
    tokens_kept = [5, 9, 13, 17]

    for eps in epsilons:
        for k in tokens_kept:
            print("="*100)
            print(f"SETTING: Epsilon = {eps} | Keep Tokens = {k} (Layer 0)")
            print("="*100)
            
            # Initialize Verifier for this specific setting
            args = MockArgs(eps=eps, p_tokens=True, tokens_keep=k, tokens_prune=0, layer_idx=0)
            verifier = VerifierTopKPrune(args, model, FakeLogger(), 10, normalizer)
            
            # Run on each sample individually
            for i, sample in enumerate(selected_samples):
                # We wrap [sample] in a list because run() expects a dataset list
                _, res_p_list, res_pp_list = verifier.run([sample])
                
                # Extract the single result
                res_p = res_p_list[0]   # Unpruned P
                res_pp = res_pp_list[0] # Pruned P'
                label = res_p['label']
                
                print(f"\n>>> Sample {i} (True Label: {label})")
                print(f"{'Class':<5} | {'P Low':<10} | {'P Up':<10} || {'P\' Low':<10} | {'P\' Up':<10} | {'Diff Low (P - P\')'}")
                print("-" * 80)
                
                # Print bounds for all 10 classes
                for c in range(10):
                    marker = "<<" if c == label else ""
                    
                    lp = res_p['lower_bounds'][c]
                    up = res_p['upper_bounds'][c]
                    lpp = res_pp['lower_bounds'][c]
                    upp = res_pp['upper_bounds'][c]
                    
                    diff = lp - lpp
                    
                    print(f"{c:<5} | {lp:<10.4f} | {up:<10.4f} || {lpp:<10.4f} | {upp:<10.4f} | {diff:<10.4f} {marker}")
            print("\n")

if __name__ == "__main__":
    main()
