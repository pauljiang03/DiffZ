from argparse import Namespace
import torch
from Verifiers.Zonotope import Zonotope

# ==========================================================
# 1. Minimal fake args (consistent with verifier config)
# ==========================================================
args = Namespace(
    device='cpu',
    perturbed_words=1,
    attack_type='synonym',
    all_words=False,                 # avoids negative dimension in constructor
    num_input_error_terms=0,
    use_dot_product_variant3=False,
    use_other_dot_product_ordering=False,
    batch_softmax_computation=False,
    add_softmax_sum_constraint=False,
    keep_intermediate_zonotopes=False,
    p=11,
)

# ==========================================================
# 2. Create a base Zonotope with embedding dim = 5
# ==========================================================
dummy_value = torch.randn(1, 5)
z_base = Zonotope(args=args, p=args.p, eps=1e-6, perturbed_word_index=0, value=dummy_value)

# ==========================================================
# 3. Build logits and mask tensors with (1, 5)
# ==========================================================
logits_tensor = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
mask_tensor   = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])

logits = z_base.new_from_constant(logits_tensor)
mask_z = logits.new_from_constant(mask_tensor)

# ==========================================================
# 4. Patch for attention-style softmax shape
# ==========================================================
# Your softmax expects (E, A, N, D). Add dummy attention head dim A=1.
logits.zonotope_w = logits.zonotope_w.unsqueeze(1)  # (E, A=1, N, D)
mask_z.zonotope_w = mask_z.zonotope_w.unsqueeze(1)  # (E, A=1, N, D)

# ==========================================================
# 5. Apply masked softmax
# ==========================================================
masked_softmax = logits.mask_softmax(mask_z, verbose=False)

# ==========================================================
# 6. Concretize and print results
# ==========================================================
lower, upper = masked_softmax.concretize()

print("\n=== Masked Softmax Sanity Check ===")
print("Logits (center):", logits_tensor.tolist())
print("Mask applied:   ", mask_tensor.tolist())
print("Lower bounds:   ", lower[0, 0, :, 0].tolist())  # note: extra head dim
print("Upper bounds:   ", upper[0, 0, :, 0].tolist())
print("Sum kept (~1):  ", float(upper[0, 0, :3, 0].sum()))
print("Sum masked (~0):", float(upper[0, 0, 3:, 0].sum()))
