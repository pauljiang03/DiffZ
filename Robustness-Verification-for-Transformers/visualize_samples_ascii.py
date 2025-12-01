import torch
import random
import sys
import os
import argparse

# --- Import your environment ---
try:
    from vit import ViT
    from mnist import mnist_test_dataloader
    from data_utils import set_seeds
except ImportError:
    print("Error: Could not import project modules. Make sure you are in the root directory.")
    sys.exit(1)

def draw_ascii(image_tensor):
    """Converts a 28x28 tensor into ASCII art for terminal viewing."""
    # MNIST images are normalized, so values might be negative or positive.
    # We assume standard scaling roughly between -1 and 1, or 0 and 1.
    # Map pixels to density characters.
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    
    # Unnormalize if necessary (approximation based on standard MNIST mean/std)
    # If your data is 0-1, this line isn't strictly needed, but it helps contrast.
    img = image_tensor.cpu().squeeze()
    
    # Simple min-max normalization for visualization
    img = (img - img.min()) / (img.max() - img.min())
    
    width, height = img.shape
    
    # Print Top Border
    print("+" + "-" * width + "+")
    
    for y in range(height):
        row = "|"
        for x in range(width):
            val = img[y, x].item()
            char_idx = int(val * (len(chars) - 1))
            row += chars[char_idx]
        row += "|"
        print(row)
        
    # Print Bottom Border
    print("+" + "-" * width + "+")

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

    # 3. Select Samples
    # CRITICAL FIX: Limit the data pool to match main.py (default 200)
    # main.py stops loading after 200 items. Random sampling depends on this list size.
    print(f"Loading first {args.pool_size} images into memory (Pool)...")
    data_list = []
    for i, (x, y) in enumerate(test_loader):
        data_list.append((x, y))
        if i >= args.pool_size: # Match the break condition in main.py
            break

    print(f"Selecting {args.count} samples from pool of {len(data_list)} (Seed={args.seed})...\n")

    found = 0
    attempts = 0
    
    while found < args.count and attempts < args.count * 50:
        attempts += 1
        idx = random.randint(0, len(data_list) - 1)
        image, label = data_list[idx]
        image = image.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            out = model(image)
            pred = out.argmax(dim=-1)
            
        if pred.item() == label.item():
            print(f"Sample {found+1}: True Label = {label.item()} | Predicted = {pred.item()}")
            draw_ascii(image)
            print("\n")
            found += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed used in previous run')
    parser.add_argument('--count', type=int, default=5, help='Number of samples to show')
    parser.add_argument('--pool_size', type=int, default=200, help='Size of dataset pool. Use 200 for main.py, 100 for benchmark.')
    args = parser.parse_args()
    
    main(args)
