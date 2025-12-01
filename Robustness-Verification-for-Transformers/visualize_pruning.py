import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
import sys
from einops import repeat  # <--- Added import

# --- Import Environment ---
try:
    from vit import ViT
    from mnist import mnist_test_dataloader
    from data_utils import set_seeds
except ImportError:
    print("Error: Run this from the root folder (where vit.py is).")
    sys.exit(1)

def get_attention_scores(model, image):
    """
    Manually runs the first layer of the ViT to get CLS attention scores.
    """
    # 1. Patch Embedding
    x = model.to_patch_embedding(image)
    b, n, _ = x.shape
    
    # 2. Add CLS Token (Fix: Was missing in previous version)
    cls_tokens = repeat(model.cls_token, '() n d -> b n d', b=b)
    x = torch.cat((cls_tokens, x), dim=1)
    
    # 3. Add Positional Embedding
    # n is num_patches (16), so n+1 is total tokens (17)
    x += model.pos_embedding[:, :(n + 1)]
    
    # 4. Layer 0 Input Norm
    layer_0 = model.transformer.layers[0]
    attn_fn = layer_0[0].fn # The Attention object
    x_norm = attn_fn.norm(x)
    
    # 5. Compute Q, K (Inner Attention Logic)
    inner_attn = attn_fn.fn
    q = inner_attn.to_q(x_norm)
    k = inner_attn.to_k(x_norm)
    
    # Split heads
    h = inner_attn.heads
    q = q.view(b, n + 1, h, -1).permute(0, 2, 1, 3) # [1, h, n+1, d]
    k = k.view(b, n + 1, h, -1).permute(0, 2, 1, 3)
    
    # 6. Dot Product (Attention Scores)
    # Shape: [1, Heads, N+1, N+1]
    dots = torch.matmul(q, k.transpose(-1, -2)) * inner_attn.scale
    
    # 7. Extract CLS Attention (Token 0 attending to all others)
    # We average across heads to get the "general" importance
    # Shape: [N+1] (17 tokens: CLS + 16 Patches)
    cls_attn = dots[0, :, 0, :].mean(dim=0)
    
    return cls_attn

def visualize(args):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(args.seed)
    
    # 2. Load Model
    print("Loading Model...")
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    
    weights = "mnist_transformer.pt"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # 3. Get Samples
    print(f"Selecting {args.count} samples (Seed={args.seed})...")
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    
    # Fast forward to random selection logic (Pool 200)
    pool = []
    for i, pair in enumerate(test_loader):
        if i >= 200: break
        pool.append(pair)
        
    samples = []
    attempts = 0
    while len(samples) < args.count and attempts < 1000:
        attempts += 1
        idx = random.randint(0, len(pool)-1)
        x, y = pool[idx]
        x, y = x.to(device), y.to(device)
        
        # Verify prediction
        if model(x).argmax() == y:
            samples.append((x, y))

    # 4. Plotting
    print("Generating visualization...")
    fig, axes = plt.subplots(args.count, 3, figsize=(12, 4 * args.count))
    if args.count == 1: axes = [axes]
    
    patch_size = 7
    grid_dim = 28 // patch_size # 4
    
    for row_idx, (img_tensor, label) in enumerate(samples):
        # A. Get Scores
        scores = get_attention_scores(model, img_tensor)
        
        # Scores[0] is CLS-to-CLS (ignore). Scores[1:] are patches.
        patch_scores = scores[1:] 
        
        # Identify Top-K Indices (Indices 0-15 corresponding to patches)
        # Note: args.keep includes CLS, so we keep (args.keep - 1) patches
        k_patches = max(1, args.keep - 1)
        top_k_indices = torch.topk(patch_scores, k_patches).indices.tolist()
        
        # B. Prepare Image
        img_np = img_tensor.cpu().squeeze().numpy()
        
        # --- Column 1: Original with Grid ---
        ax1 = axes[row_idx][0]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f"True: {label.item()} | Keep: {args.keep}")
        ax1.axis('off')
        # Overlay Grid
        for i in range(1, grid_dim):
            ax1.axhline(i * patch_size - 0.5, color='red', linewidth=0.5, alpha=0.5)
            ax1.axvline(i * patch_size - 0.5, color='red', linewidth=0.5, alpha=0.5)

        # --- Column 2: Attention Heatmap ---
        # Remap 1D scores (16) back to 2D grid (4x4)
        heatmap = patch_scores.view(grid_dim, grid_dim).detach().cpu().numpy()
        ax2 = axes[row_idx][1]
        im2 = ax2.imshow(heatmap, cmap='plasma')
        ax2.set_title("CLS Attention Scores")
        ax2.axis('off')
        
        # --- Column 3: The Pruned View ---
        # Create a mask image
        pruned_img = img_np.copy()
        
        ax3 = axes[row_idx][2]
        ax3.imshow(img_np, cmap='gray')
        
        # Darken the pruned patches
        for r in range(grid_dim):
            for c in range(grid_dim):
                patch_idx = r * grid_dim + c
                
                # Draw the box
                rect_x = c * patch_size
                rect_y = r * patch_size
                
                if patch_idx in top_k_indices:
                    # KEPT: Draw Green Border
                    rect = patches.Rectangle((rect_x, rect_y), patch_size, patch_size, 
                                           linewidth=2, edgecolor='#00FF00', facecolor='none')
                    ax3.add_patch(rect)
                else:
                    # PRUNED: Darken / Red Cross
                    rect = patches.Rectangle((rect_x, rect_y), patch_size, patch_size, 
                                           linewidth=0, facecolor='black', alpha=0.85)
                    ax3.add_patch(rect)
                    
        ax3.set_title(f"Pruned (Top-{k_patches} Patches)")
        ax3.axis('off')

    out_file = f"pruning_viz_keep{args.keep}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep', type=int, default=5, help='Total tokens to keep (including CLS)')
    parser.add_argument('--count', type=int, default=3, help='Rows to generate')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    visualize(args)
