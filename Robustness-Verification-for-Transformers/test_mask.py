import torch
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args
from argparse import Namespace

# --- Minimal args stub to satisfy Zonotope constructor ---
args = Namespace(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    p=2,
    perturbed_words=1,
    attack_type="synonym",
    all_words=True,
    num_input_error_terms=1,
    concretize_special_norm_error_together=True,
    cpu=False
)

# --- Helper to create a simple zonotope with fixed logits ---
def make_constant_zonotope(values):
    # Create a single "layer" zonotope with deterministic center (no error terms)
    z_w = torch.zeros((1, len(values), 1), device=args.device)
    z_w[0, :, 0] = torch.tensor(values, device=args.device)
    return Zonotope(args, p=2, eps=0.0, perturbed_word_index=0, zonotope_w=z_w)

# --- Step 1: Construct a zonotope of 5 logits ---
logits = make_constant_zonotope([2.0, 1.0, 0.0, -1.0, -2.0])

# --- Step 2: Create a binary mask (keep first 3 tokens) ---
mask_tensor = torch.tensor([[1, 1, 1, 0, 0]], device=args.device).unsqueeze(-1)
mask_z = logits.new_from_constant(mask_tensor)

# --- Step 3: Apply masked softmax ---
masked_softmax = logits.mask_softmax(mask_z, verbose=True)

# --- Step 4: Concretize the masked softmax bounds ---
lower, upper = masked_softmax.concretize()

print("\n=== Masked Softmax Sanity Check ===")
print("Logits (center):", logits.zonotope_w[0, :, 0].cpu().numpy())
print("Mask applied:   ", mask_tensor.squeeze(-1).cpu().numpy())
print("Lower bounds:   ", lower[0, :, 0].cpu().numpy())
print("Upper bounds:   ", upper[0, :, 0].cpu().numpy())

# Expected behavior:
# - For indices 0–2 (kept): probabilities > 0, sum ≈ 1 over them
# - For indices 3–4 (pruned): both lower and upper ≈ 0
