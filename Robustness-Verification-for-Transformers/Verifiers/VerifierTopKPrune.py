import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime

import torch
import numpy as np
from einops.einops import repeat
from einops.layers.torch import Rearrange

# Imports from your existing environment
from Verifiers.Verifier import Verifier
from Verifiers.Zonotope import (
    Zonotope, 
    make_zonotope_new_weights_same_args, 
    cleanup_memory
)
from vit import ViT

# --- Helper Functions ---

def get_layernorm(x):
    return x.fn

def get_inner(x):
    return x.fn.fn

def sample_correct_samples(args, data, target):
    """
    Samples correctly classified examples from the dataset.
    """
    examples = []
    # Safety check for requested samples vs dataset size
    num_samples = min(args.samples, len(data))
    
    attempts = 0
    # Try to find correct samples, but don't loop forever
    while len(examples) < num_samples and attempts < num_samples * 10:
        attempts += 1
        idx = random.randint(0, len(data) - 1)
        example = data[idx]
        
        # Quick inference check
        with torch.no_grad():
            logits = target(example["image"].to(args.device))
            prediction = torch.argmax(logits, dim=-1)

        if prediction == example["label"]:
            examples.append(example)

    return examples

# --- Local Softmax Implementation (Robust Version) ---

def softmax_with_mask(zonotope: Zonotope, 
                      mask: Optional[torch.Tensor] = None, 
                      verbose=False,
                      no_constraints=True,
                      add_value_positivity_constraint=False,
                      use_new_reciprocal=True) -> Zonotope:
    """
    A standalone, STABLE implementation of Zonotope Softmax.
    Includes robust shape matching for the mask to prevent broadcasting errors.
    """
    
    num_values = zonotope.zonotope_w.size(-1)

    # 1. Compute Exponential: e^x
    # minimal_area=False prevents instability with negative differences
    zonotope_exp = zonotope.exp(minimal_area=False)

    # 2. Apply Mask: m * e^x
    if mask is not None:
        # --- ROBUST SHAPE ALIGNMENT ---
        w_shape = zonotope_exp.zonotope_w.shape
        m_shape = mask.shape
        
        # If the Zonotope has more dimensions than the mask (e.g. Error Terms),
        # we assume the mask corresponds to the LAST dimensions (Batch, Heads, Q, K).
        # We prepend 1s to the mask to align it.
        if len(w_shape) > len(m_shape):
            diff = len(w_shape) - len(m_shape)
            # Create a view: e.g. Mask (1, 4, 17, 17) -> (1, 1, 1, 4, 17, 17)
            new_view_shape = (1,) * diff + m_shape
            mask_aligned = mask.view(new_view_shape)
            
            # Debugging print (Optional: remove after confirming fix)
            # print(f"   [Debug] Auto-Aligning Mask: {list(m_shape)} -> {list(new_view_shape)} to match W {list(w_shape)}")
        else:
            mask_aligned = mask

        # Multiply
        try:
            zonotope_exp = zonotope_exp.multiply(mask_aligned)
        except RuntimeError as e:
            print(f"   [CRITICAL] Mask Multiplication Failed!")
            print(f"   Zonotope W Shape: {w_shape}")
            print(f"   Original Mask Shape: {m_shape}")
            print(f"   Aligned Mask Shape: {mask_aligned.shape}")
            raise e

    # 3. Compute Denominator: Sum(m * e^x)
    # We sum across the last dimension (dim=-1) to get the normalization factor.
    zonotope_sum_w = zonotope_exp.zonotope_w.sum(dim=-1, keepdim=True).repeat(1, 1, 1, num_values)
    zonotope_sum = make_zonotope_new_weights_same_args(zonotope_sum_w, zonotope, clone=False)

    # 4. Compute Softmax: (m * e^x) / Sum(...)
    zonotope_softmax = zonotope_exp.divide(
        zonotope_sum, 
        use_original_reciprocal=not use_new_reciprocal, 
        y_positive_constraint=add_value_positivity_constraint
    )

    if no_constraints:
        return zonotope_softmax

    # 5. (Optional) Add Equality Constraints
    u, l = zonotope_softmax.concretize()
    if (l.sum(dim=-1) - 1).abs().max().item() < 1e-6 and (u.sum(dim=-1) - 1).abs().max().item() < 1e-6:
        del u, l
        cleanup_memory()
        return zonotope_softmax

    return zonotope_softmax.add_equality_constraint_on_softmax()


# --- Main Class Definition ---

class VerifierTopKPrune(Verifier):
    """
    Differential ViT Verifier implementing Sound Top-K / Bottom-X Pruning.
    """
    def __init__(self, args, target: ViT, logger, num_classes: int, normalizer):
        self.args = args
        self.device = args.device
        self.target = target
        self.logger = logger
        self.res = args.res
        self.results_directory = args.results_directory

        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps
        self.debug = args.debug 
        self.verbose = args.verbose 
        self
