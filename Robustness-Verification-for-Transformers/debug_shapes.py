import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from vit import ViT
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args
from mnist import normalizer

# --- Mock Args Class ---
class Args:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_num_error_terms = 1000
        self.num_input_error_terms = 28*28
        self.verbose = True
        self.debug = True
        self.error_reduction_method = 'box'
        self.dtype = torch.float32

args = Args()

# --- Setup Model ---
print(f"1. Initializing ViT (Depth=3)...")
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(args.device)

# Load weights (Optional for shape check, but good practice)
try:
    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=args.device))
    print("   Weights loaded successfully.")
except:
    print("   WARNING: Using random weights (File not found or mismatch).")

model.eval()

# --- Debugging Function ---
def debug_bound_input(image, eps):
    print("\n2. Debugging _bound_input...")
    
    # A. Rearrange
    patch_size = model.patch_size
    rearrange = Rearrange('1 c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    image = rearrange(image)
    print(f"   Image Rearranged: {image.shape} (Expected: [16, 49])")

    # B. Zonotope Init
    eps_scaled = eps / normalizer.std[0]
    bounds = Zonotope(args, p=float('inf'), eps=eps_scaled,
                      perturbed_word_index=None, value=image,
                      start_perturbation=0, end_perturbation=image.shape[0])
    print(f"   Zonotope W Initial: {bounds.zonotope_w.shape}")
    
    # C. Dense Projection
    bounds = bounds.dense(model.to_patch_embedding[1])
    print(f"   Zonotope After Dense: {bounds.zonotope_w.shape}")

    e, n, _ = bounds.zonotope_w.shape
    print(f"   Extracted n (num_patches): {n}")

    # D. CLS Token
    cls_tokens = repeat(model.cls_token, '() n d -> n d')
    cls_tokens_value_w = cls_tokens.unsqueeze(0)
    cls_tokens_errors_w = torch.zeros_like(cls_tokens).unsqueeze(0).repeat(e - 1, 1, 1)
    cls_tokens_zonotope_w = torch.cat([cls_tokens_value_w, cls_tokens_errors_w], dim=0)
    
    # E. Concatenate
    full_zonotope_w = torch.cat((cls_tokens_zonotope_w, bounds.zonotope_w), dim=1)
    print(f"   Full Zonotope W (Patches + CLS): {full_zonotope_w.shape} (Expected: [E, 17, 64])")

    # F. Create New Zonotope
    bounds = make_zonotope_new_weights_same_args(full_zonotope_w, bounds, clone=False)
    
    # G. Pos Embedding Add (THE SUSPECT)
    pos_shape = model.pos_embedding.shape
    print(f"   Pos Embedding Shape: {pos_shape} (Expected: [1, 17, 64])")
    
    try:
        sliced_pos = model.pos_embedding[:, :(n + 1)]
        print(f"   Sliced Pos Embedding: {sliced_pos.shape}")
        bounds = bounds.add(sliced_pos)
        print("   >>> SUCCESS: _bound_input finished without error.")
    except Exception as e:
        print(f"   >>> FAIL: Error during Pos Embedding Add: {e}")
        return None

    return bounds

# --- Run Debug ---
input_image = torch.randn(1, 1, 28, 28).to(args.device)
bounds = debug_bound_input(input_image, eps=0.001)

if bounds is not None:
    print("\n3. Debugging Layer 0...")
    # Try to run one layer to see if Attention crashes
    attn, ff = model.transformer.layers[0]
    
    try:
        # We mimic the logic from _bound_layer / _bound_attention
        print("   Running Layer Norm 1...")
        ln_fn = attn.fn.norm
        bounds_ln = bounds.layer_norm(ln_fn, "no_var")
        
        print("   Running Q/K Dense...")
        attn_layer = attn.fn.fn
        query = bounds_ln.dense(attn_layer.to_q)
        key = bounds_ln.dense(attn_layer.to_k)
        
        print(f"   Query Shape: {query.zonotope_w.shape}")
        
        print("   Adding Heads...")
        query = query.add_attention_heads_dim(attn_layer.heads)
        key = key.add_attention_heads_dim(attn_layer.heads)
        
        print(f"   Query Reshaped: {query.zonotope_w.shape} (Expected [E, 17, 4, 16])")
        
        print("   Running Dot Product...")
        # This is where the IndexError likely lives
        scores = query.dot_product(key)
        print("   >>> SUCCESS: Dot Product finished.")
        
    except Exception as e:
        print(f"   >>> FAIL: Error during Layer 0: {e}")
