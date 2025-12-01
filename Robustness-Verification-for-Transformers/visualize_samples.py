import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --- Import your environment ---
from vit import ViT
from mnist import mnist_test_dataloader
from data_utils import set_seeds

def save_images(args):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # CRITICAL: Set the seed to match your previous run
    # If you didn't specify one before, it was likely 0 or determined by Parser.py defaults
    set_seeds(args.seed)
    
    # 2. Load Data
    print("Loading MNIST...")
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    data_list = []
    for x, y in test_loader:
        data_list.append({"image": x.to(device), "label": y.to(device)})
    
    # 3. Load Model (Depth=3)
    print("Loading Model...")
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    
    weights = "mnist_transformer.pt"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        print("WARNING: Weights not found, using random init (Selection might differ if model is untrained)")
    model.eval()

    # 4. Reproduce the Random Sampling Logic
    # This logic matches 'sample_correct_samples' from VerifierTopKPrune.py
    print(f"Selecting {args.count} samples (Seed={args.seed})...")
    
    examples = []
    attempts = 0
    
    # We must use the exact same loop structure to hit the same random states
    while len(examples) < args.count and attempts < args.count * 20:
        attempts += 1
        idx = random.randint(0, len(data_list) - 1)
        example = data_list[idx]
        
        # Check correctness
        with torch.no_grad():
            logits = model(example["image"])
            pred = torch.argmax(logits, dim=-1)

        if pred.item() == example["label"].item():
            examples.append(example)

    # 5. Plot and Save
    print(f"\nFound {len(examples)} samples.")
    print("Labels found: ", [ex['label'].item() for ex in examples])
    
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 3))
    if len(examples) == 1: axes = [axes] # Handle single sample case

    for i, ex in enumerate(examples):
        img_tensor = ex['image'].cpu().squeeze() # [1, 28, 28] -> [28, 28]
        label = ex['label'].item()
        
        axes[i].imshow(img_tensor, cmap='gray')
        axes[i].set_title(f"Sample {i}\nTrue: {label}")
        axes[i].axis('off')
    
    out_file = f"samples_seed{args.seed}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\n[Image of MNIST samples] Saved visualization to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed (try 0, 100, or 42)')
    parser.add_argument('--count', type=int, default=5, help='Number of samples to retrieve')
    args = parser.parse_args()
    
    save_images(args)
