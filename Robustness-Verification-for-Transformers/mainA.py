import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import sys

# --- Imports from your project structure ---
# Ensure these files/folders are in your python path
from vit import ViT  
from VerifierTopKPruneA import VerifierTopKPruneA

# --- Simple Normalizer Wrapper ---
class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean).to(device).view(1, -1, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, -1, 1, 1)

def parse_args():
    parser = argparse.ArgumentParser(description='Symbolic Differential Verification (MainA)')

    # --- Verification Parameters ---
    parser.add_argument('--eps', type=float, default=0.0039, help='Perturbation epsilon (e.g. 1/255)')
    parser.add_argument('--p', type=float, default=float('inf'), help='Lp norm (inf or 2)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to verify')
    
    # --- Pruning Parameters ---
    parser.add_argument('--prune_tokens', action='store_true', help='Enable token pruning for the P-prime run')
    parser.add_argument('--prune_layer_idx', type=int, default=0, help='Layer index to start pruning')
    parser.add_argument('--tokens_to_keep', type=int, default=100, help='Number of tokens to keep (Top-K)')
    parser.add_argument('--tokens_to_prune', type=int, default=0, help='Number of tokens to prune (Bottom-X). Overrides keep.')

    # --- Model & Data ---
    parser.add_argument('--model_path', type=str, default='models/vit_cifar.pth', help='Path to saved model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    # --- Output ---
    parser.add_argument('--results_directory', type=str, default='logs/')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    # --- Technical Verifier Settings ---
    parser.add_argument('--method', type=str, default='zonotope')
    parser.add_argument('--error_reduction_method', type=str, default='box', choices=['box', 'none'])
    parser.add_argument('--max_num_error_terms', type=int, default=10000, help='Max error terms before reduction')
    parser.add_argument('--add_softmax_sum_constraint', action='store_true', help='Constraint softmax sum to 1')
    parser.add_argument('--keep_intermediate_zonotopes', action='store_false', help='Keep intermediate layers in memory')
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--res', action='store_true', help='Use residual connections')
    parser.add_argument('--num_fast_dot_product_layers_due_to_switch', type=int, default=-1)

    return parser.parse_args()

def load_data(args):
    """Loads dataset and returns a list of dictionaries used by the verifier."""
    print(f"Loading {args.dataset}...")
    
    if args.dataset == 'cifar10':
        # Standard CIFAR10 Normalization
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_dim = (3, 32, 32)
        
    elif args.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_dim = (1, 28, 28)
        
    else:
        raise NotImplementedError("Only CIFAR10/MNIST supported in this main script.")

    # Convert to list of dicts as expected by VerifierTopKPrune run()
    data_list = []
    # Only load enough to cover the sample request (plus buffer for incorrect predictions)
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    
    print("Formatting data...")
    for idx in indices[:args.samples * 2]: # Load slightly more than needed
        img, label = test_set[idx]
        data_list.append({
            "image": img.unsqueeze(0), # Add batch dim
            "label": torch.tensor(label),
            "index": idx
        })

    normalizer = Normalizer(mean, std, args.device)
    return data_list, num_classes, normalizer, input_dim

def load_model(args, num_classes, input_dim):
    """Initializes ViT and loads checkpoint."""
    print(f"Loading Model from {args.model_path}...")
    
    # Example ViT instantiation - adjust parameters to match your specific architecture
    # You might need to adjust these args based on how your 'ViT' class is defined
    model = ViT(
        image_size=input_dim[1],
        patch_size=4,
        num_classes=num_classes,
        dim=128,          # Adjust based on your trained model
        depth=6,          # Adjust based on your trained model
        heads=4,          # Adjust based on your trained model
        mlp_dim=256,      # Adjust based on your trained model
        dropout=0.0,
        emb_dropout=0.0
    ).to(args.device)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        # Handle state dict loading (sometimes wrapped in 'state_dict' key)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model path {args.model_path} not found. Initializing with random weights for testing.")

    model.eval()
    return model

def main():
    args = parse_args()
    
    # Set Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup
    data_list, num_classes, normalizer, input_dim = load_data(args)
    model = load_model(args, num_classes, input_dim)

    # Initialize Symbolic Verifier
    # We pass a simple print function as logger if you don't have a specific logger object
    class SimpleLogger:
        def log(self, msg): print(msg)
    
    logger = SimpleLogger()

    print("\n--- Initializing VerifierTopKPruneA (Symbolic) ---")
    verifier = VerifierTopKPruneA(
        args=args,
        target=model,
        logger=logger,
        num_classes=num_classes,
        normalizer=normalizer
    )

    # Run Verification
    # Returns (diff_results, p_results, p_prime_results)
    # Note: p_results and p_prime_results will be empty in the A-variant as we only track diff
    diff_results, _, _ = verifier.run(data_list)
    
    print("\n--- Summary ---")
    if len(diff_results) > 0:
        avg_time = np.mean([r['time'] for r in diff_results])
        print(f"Processed {len(diff_results)} samples.")
        print(f"Average Time per Sample: {avg_time:.4f}s")
        print(f"Results saved to: {verifier.results_directory}")
    else:
        print("No samples were successfully verified (check data loading or model accuracy).")

if __name__ == "__main__":
    main()
