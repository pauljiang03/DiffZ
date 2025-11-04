from argparse import Namespace
import torch
from Verifiers.Zonotope import Zonotope

# ---- 1) Minimal args your Zonotope expects ----
args = Namespace(
    device='cpu',
    perturbed_words=1,
    attack_type='synonym',
    all_words=False,                 # <— IMPORTANT for num_words=1 case
    num_input_error_terms=0,
    use_dot_product_variant3=False,
    use_other_dot_product_ordering=False,
    batch_softmax_computation=False,
    add_softmax_sum_constraint=False,
    keep_intermediate_zonotopes=False,
    p=11,                            # use "infinity-like" norm mode your impl expects
)

# ---- 2) Base zonotope with shape (length=1, embedding=5) ----
# softmax will operate across embedding=5
dummy_value = torch.randn(1, 5)      # (num_words = 1, embedding = 5)
z_base = Zonotope(args=args, p=args.p, eps=1e-6, perturbed_word_index=0, value=dummy_value)

# ---- 3) Build constant logits & mask with SAME shape as z_base center ----
logits_tensor = torch.tensor([[ 2.0,  1.0,  0.0, -1.0, -2.0]])   # shape (1,5)
mask_tensor   = torch.tensor([[ 1.0,  1.0,  1.0,  0.0,  0.0]])   # shape (1,5)

logits = z_base.new_from_constant(logits_tensor)  # inherits args/p/eps/error dims
mask_z = logits.new_from_constant(mask_tensor)    # same dims as logits

# ---- 4) Apply masked softmax (your new method) ----
masked_softmax = logits.mask_softmax(mask_z, verbose=False)

# ---- 5) Concretize & print ----
lower, upper = masked_softmax.concretize()        # shapes (length=1, embedding=5)

print("\n=== Masked Softmax Sanity Check ===")
print("Logits (center):", logits_tensor.tolist())
print("Mask applied:   ", mask_tensor.tolist())
print("Lower bounds:   ", lower[0, :].tolist())   # (1,5) -> index row 0
print("Upper bounds:   ", upper[0, :].tolist())
print("Sum kept (~1):  ", float(upper[0, :3].sum()))  # first 3 kept
print("Sum masked (~0):", float(upper[0, 3:].sum()))

