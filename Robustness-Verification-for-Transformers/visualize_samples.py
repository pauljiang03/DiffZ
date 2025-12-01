import torch
import random
import sys
import os
import argparse
import matplotlib.pyplot as plt

# --- Import your environment ---
try:
    from vit import ViT
    from mnist import mnist_test_dataloader
    from data_utils import set_seeds
except ImportError:
    print("Error: Could not import project modules. Make sure you are in the root directory.")
    sys.exit(1)

def main(args):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(args.seed)
    
    # 2. Load Data & Model
    print("Loading Data and Model...")
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    
    weights = "mnist_transformer.pt"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # 3. Select Samples (Preserving main.py logic)
    # CRITICAL: Limit the data pool to match main.py (default 200)
    print(f"Loading first {args.pool_size} images into memory (Pool)...")
    data_list = []
    for i, (x, y) in enumerate(test_loader):
        data_list.append((x, y))
        if i >= args.pool_size: 
            break

    print(f"Selecting {args.count} samples from pool of {len(data_list)} (Seed={args.seed})...\n")

    found_samples = []
    attempts = 0
    
    while len(found_samples) < args.count and attempts < args.count * 50:
        attempts += 1
        idx = random.randint(0, len(data_list) - 1)
        image, label = data_list[idx]
        image = image.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            out = model(image)
            pred = out.argmax(dim=-1)
            
        if pred.item() == label.item():
            # Store the CPU tensor and the label
            found_samples.append({
                "image": image.cpu().squeeze(),
                "label": label.item(),
                "pred": pred.item(),
                "idx": idx
            })
            print(f"Found Sample {len(found_samples)}: True {label.item()} (Index {idx})")

    # 4. Plot and Save
    if not found_samples:
        print("No samples found matching criteria.")
        return

    print("\nGenerating PNG...")
    fig, axes = plt.subplots(1, len(found_samples), figsize=(3 * len(found_samples), 3.5))
    
    # Handle single sample case (axes is not a list)
    if len(found_samples) == 1:
        axes = [axes]

    for i, sample in enumerate(found_samples):
        ax = axes[i]
        ax.imshow(sample["image"], cmap='gray')
        ax.set_title(f"Sample {i}\nTrue: {sample['label']}", fontsize=14)
        ax.axis('off')
    
    out_file = f"samples_seed{args.seed}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved visualization to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed used in previous run')
    parser.add_argument('--count', type=int, default=6, help='Number of samples to show')
    parser.add_argument('--pool_size', type=int, default=200, help='Size of dataset pool. Use 200 for main.py, 100 for benchmark.')
    args = parser.parse_args()
    
    main(args)
