from argparse import Namespace
import torch
from Verifiers.Zonotope import Zonotope

# ==========================================================
# 1. Minimal fake args (consistent with your verifier config)
# ==========================================================
args = Namespace(
    device='cpu',
    perturbed_words=1,
    attack_type='synonym',
    all_words=True,
    num_input_error_terms=0,
    use_dot_product_variant3=False,
    use_other_dot_product_ordering=False,
    batch_softmax_computation=False,
    add_softmax_sum_constraint=False,   # used in attention sometimes
    keep_intermediate_zonotopes=False,  # safe default
    p=11,
)


# ==========================================================
# 2. Create a base Zonotope (5 tokens × 1-dim embedding)
# ==========================================================
num_tokens = 5
dummy_value = torch.randn(num_tokens, 1)
z_base = Zonotope(args=args, p=args.p, eps=1e-6, perturbed_word_index=0, value=dummy_value)

# ==========================================================
# 3. Build constant logits & mask zonotopes from base
# ==========================================================
# Logits: 5 tokens, simple decreasing sequence
logits_tensor = torch.tensor([[2.0], [1.0], [0.0], [-1.0], [-2.0]])
logits = z_base.new_from_constant(logits_tensor)

# Mask: keep first 3 tokens (CLS + 2 others), prune rest
mask_tensor = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0]])
mask_z = logits.new_from_constant(mask_tensor)

# ==========================================================
# 4. Apply masked softmax
# ==========================================================
masked_softmax = logits.mask_softmax(mask_z, verbose=False)

# ==========================================================
# 5. Concretize and display results
# ==========================================================
lower, upper = masked_softmax.concretize()

print("\n=== Masked Softmax Sanity Check ===")
print("Logits (center):", logits_tensor.squeeze().tolist())
print("Mask applied:   ", mask_tensor.squeeze().tolist())
print("Lower bounds:   ", lower[0, :, 0].tolist())
print("Upper bounds:   ", upper[0, :, 0].tolist())
print("Sum of kept tokens (~1):", float(upper[0, :3, 0].sum()))
print("Sum of masked tokens (~0):", float(upper[0, 3:, 0].sum()))
