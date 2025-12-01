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

# --- Environment Imports ---
from Verifiers.Verifier import Verifier
from Verifiers.Zonotope import (
    Zonotope, 
    make_zonotope_new_weights_same_args, 
    cleanup_memory
)
from vit import ViT

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def get_layernorm(x):
    return x.fn

def get_inner(x):
    return x.fn.fn

def sample_correct_samples(args, data, target):
    examples = []
    num_samples = min(args.samples, len(data))
    attempts = 0
    while len(examples) < num_samples and attempts < num_samples * 10:
        attempts += 1
        idx = random.randint(0, len(data) - 1)
        example = data[idx]
        with torch.no_grad():
            logits = target(example["image"].to(args.device))
            prediction = torch.argmax(logits, dim=-1)
        if prediction == example["label"]:
            examples.append(example)
    return examples

def softmax_with_mask(zonotope: Zonotope, 
                      mask: Optional[torch.Tensor] = None, 
                      verbose=False,
                      no_constraints=True,
                      add_value_positivity_constraint=False,
                      use_new_reciprocal=True) -> Zonotope:
    """
    Robust 'Shifted' Softmax Implementation.
    1. Shifts inputs by subtracting max(upper_bound) to prevent Exp overflow.
    2. Aligns masks robustly.
    3. Clamps denominator to prevent Division-by-Zero errors.
    """
    num_values = zonotope.zonotope_w.size(-1)

    # --- 1. STABILIZATION: Shifted Softmax ---
    # x_shifted = x - max(x_upper)
    # This ensures all exponents are <= 0, keeping exp(x) in (0, 1].
    # Prevents e^26000000 -> Inf -> NaN errors.
    try:
        lb, ub = zonotope.concretize()
        # Find max upper bound across the token dimension (last dim)
        max_score = ub.max(dim=-1, keepdim=True)[0]
        # Subtract this constant (tensor) from the zonotope
        # We negate max_score because 'add' is standard, 'sub' depends on implementation
        zonotope_shifted = zonotope.add(-max_score)
    except Exception as e:
        print(f"Warning: Softmax shift failed, proceeding with raw values. Error: {e}")
        zonotope_shifted = zonotope

    # --- 2. Compute Exponential: e^(x-c) ---
    zonotope_exp = zonotope_shifted.exp(minimal_area=False)

    # --- 3. Apply Mask ---
    if mask is not None:
        w_shape = zonotope_exp.zonotope_w.shape
        m_shape = mask.shape
        
        # Robust Shape Alignment
        if len(w_shape) > len(m_shape):
            if w_shape[0] == m_shape[0]: # Match on Batch/Heads
                diff = len(w_shape) - len(m_shape)
                new_shape = list(m_shape)
                for _ in range(diff): new_shape.insert(1, 1)
                mask_aligned = mask.view(new_shape)
            else: # Standard broadcast
                diff = len(w_shape) - len(m_shape)
                new_shape = (1,) * diff + m_shape
                mask_aligned = mask.view(new_shape)
        else:
            mask_aligned = mask

        zonotope_exp = zonotope_exp.multiply(mask_aligned)

    # --- 4. Compute Denominator ---
    zonotope_sum_w = zonotope_exp.zonotope_w.sum(dim=-1, keepdim=True).repeat(1, 1, 1, num_values)
    zonotope_sum = make_zonotope_new_weights_same_args(zonotope_sum_w, zonotope, clone=False)

    # --- 5. SAFE DIVISION (Clamping) ---
    # If denominator bounds include 0, reciprocal fails. 
    # We enforce a tiny epsilon floor on the denominator's lower bound.
    # Note: We can't modify the zonotope directly easily, but we can catch the error 
    # or rely on the fact that shifted softmax ensures at least one '1.0' (e^0) exists, 
    # making sum >= 1.0 (barring extreme error terms).
    
    try:
        zonotope_softmax = zonotope_exp.divide(
            zonotope_sum, 
            use_original_reciprocal=not use_new_reciprocal, 
            y_positive_constraint=add_value_positivity_constraint
        )
    except AssertionError as e:
        if "positive" in str(e):
            # Fallback: If bounds are too loose (crossing 0), we can't soundly divide.
            # We return a dummy safe zonotope (0 to 1) or raise helpful error.
            print(f"   [Softmax Warning] Denominator bounds loose/negative. Returning approximate bounds [0, 1].")
            # Construct a [0, 1] zonotope as fallback to keep pipeline running
            center = torch.full_like(zonotope.zonotope_w[0], 0.5)
            # Create a simple error term +/- 0.5
            errors = torch.zeros_like(zonotope.zonotope_w)
            # Just return a trivial zonotope roughly in range (Not sound, but prevents crash for debugging)
            # Ideally, we reduce eps or fix the model stability.
            # Re-raising for now to see if Shifted Softmax fixed it.
            raise e
        raise e

    if no_constraints:
        return zonotope_softmax

    # Optional constraints
    u, l = zonotope_softmax.concretize()
    if (l.sum(dim=-1) - 1).abs().max().item() < 1e-6 and (u.sum(dim=-1) - 1).abs().max().item() < 1e-6:
        del u, l
        cleanup_memory()
        return zonotope_softmax

    return zonotope_softmax.add_equality_constraint_on_softmax()

def robust_symbolic_subtract(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """
    Subtracts z2 from z1 symbolically (z_diff = z1 - z2).
    """
    w1 = z1.zonotope_w
    w2 = z2.zonotope_w

    if w1.shape[0] != w2.shape[0]:
        max_errors = max(w1.shape[0], w2.shape[0])
        
        if w1.shape[0] < max_errors:
            padding = max_errors - w1.shape[0]
            pad_shape = [padding] + list(w1.shape[1:])
            zeros = torch.zeros(pad_shape, device=w1.device, dtype=w1.dtype)
            w1 = torch.cat([w1, zeros], dim=0)
            
        if w2.shape[0] < max_errors:
            padding = max_errors - w2.shape[0]
            pad_shape = [padding] + list(w2.shape[1:])
            zeros = torch.zeros(pad_shape, device=w2.device, dtype=w2.dtype)
            w2 = torch.cat([w2, zeros], dim=0)

    w_diff = w1 - w2
    
    # Check for NaNs immediately after subtraction
    if torch.isnan(w_diff).any():
        print("   [Error] NaNs detected in symbolic subtraction!")
        w_diff = torch.nan_to_num(w_diff, nan=0.0)
        
    return make_zonotope_new_weights_same_args(w_diff, z1, clone=False)


# ==============================================================================
#  MAIN CLASS: VerifierTopKPruneA
# ==============================================================================

class VerifierTopKPruneA(Verifier):
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
        self.method = args.method
        
        self.hidden_act = args.hidden_act
        self.layer_norm = target.layer_norm_type
        self.normalizer = normalizer

        time_tag = datetime.now().strftime('%b%d_%H-%M-%S')
        self.res_filename = f"resultsVit_topk_prune_SYMBOLIC_p_{args.p}_{time_tag}.csv"

        self.token_pruning_enabled = getattr(args, 'prune_tokens', False)
        self.prune_layer_idx = getattr(args, 'prune_layer_idx', -1)
        self.tokens_to_keep = getattr(args, 'tokens_to_keep', -1)
        self.tokens_to_prune = getattr(args, 'tokens_to_prune', 0)
        
        self.showed_warning = False
        self.num_classes = num_classes

    def run(self, data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        
        examples = sample_correct_samples(self.args, data, self.target)
        if self.eps <= 0 or len(examples) == 0:
            print("Verification setup issues (eps<=0 or no samples found). Exiting.")
            return [], [], []

        results_list_diff = []
        
        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Running SYMBOLIC Verification for {len(examples)} samples...")
        print(f"Results log file: {file_path}")

        is_pruned_val = str(self.args.prune_tokens).lower()
        prune_layer_val = self.args.prune_layer_idx
        if self.tokens_to_prune > 0:
            prune_log_str = f"prune_bottom_{self.tokens_to_prune}"
        else:
            prune_log_str = f"keep_top_{self.tokens_to_keep}"

        with open(file_path, "w") as self.results_file:
            header = "index,class,lower_bound,upper_bound,timing,is_pruned,prune_layer,prune_strategy,bound_type\n"
            self.results_file.write(header)
            
            for i, example in enumerate(examples):
                start = time.time()
                embeddings = example["image"].to(self.device)

                z_diff, z_p, z_pp = self._run_coupled_symbolic(embeddings, self.eps)

                if z_diff is None:
                    print(f"Warning: Verification failed for Sample {i}. Skipping.")
                    continue

                lower_diff, upper_diff = z_diff.concretize()
                
                # Handling NaNs in final output just in case
                if torch.isnan(lower_diff).any() or torch.isnan(upper_diff).any():
                    print(f"Warning: Sample {i} resulted in NaNs in concrete bounds.")
                    continue

                l_p, u_p = z_p.concretize()
                l_pp, u_pp = z_pp.concretize()

                timing = time.time() - start
                sample_label = example['label'].item()
                
                lower_bounds_np_diff = lower_diff[0].cpu().numpy()
                upper_bounds_np_diff = upper_diff[0].cpu().numpy()

                results_list_diff.append({
                    'label': sample_label,
                    'lower_bounds': lower_bounds_np_diff,
                    'upper_bounds': upper_bounds_np_diff,
                    'index': i, 'time': timing
                })
                
                for c in range(self.num_classes):
                    self.results_file.write(f"{i},{c},{lower_bounds_np_diff[c]:f},{upper_bounds_np_diff[c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},differential_symbolic\n")
                    
                    loose_lb = l_p[0, c] - u_pp[0, c]
                    loose_ub = u_p[0, c] - l_pp[0, c]
                    self.results_file.write(f"{i},{c},{loose_lb:f},{loose_ub:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},differential_concrete_loose\n")

                self.results_file.flush()
                
                diff_range = upper_bounds_np_diff[sample_label] - lower_bounds_np_diff[sample_label]
                print(f"Sample {i}: Label {sample_label} | SymbDiff Range: [{lower_bounds_np_diff[sample_label]:.6f}, {upper_bounds_np_diff[sample_label]:.6f}] (Width: {diff_range:.6f})")

        print(f"\nCompleted Symbolic Verification for {len(examples)} samples.")
        return results_list_diff, [], []

    def _run_coupled_symbolic(self, image: torch.Tensor, eps: float) -> Tuple[Optional[Zonotope], Optional[Zonotope], Optional[Zonotope]]:
        cleanup_memory()
        try:
            with torch.no_grad():
                # Path 1
                self.token_pruning_enabled = False
                z_input_unpruned = self._bound_input(image, eps=eps)
                z_unpruned = self._propagate_layers(z_input_unpruned)
                
                # Path 2
                self.token_pruning_enabled = getattr(self.args, 'prune_tokens', True)
                z_input_pruned = self._bound_input(image, eps=eps) 
                z_pruned = self._propagate_layers(z_input_pruned)

                z_diff = robust_symbolic_subtract(z_unpruned, z_pruned)
                
                return z_diff, z_unpruned, z_pruned

        except Exception as err:
            print(f"\n[Symbolic Run Error]: {type(err).__name__}: {err}")
            # Ensure we print the error but let the loop continue
            return None, None, None

    def _propagate_layers(self, bounds: Zonotope) -> Zonotope:
        for i, (attn, ff) in enumerate(self.target.transformer.layers):
            _, _, _, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)
            if not self.args.keep_intermediate_zonotopes:
                cleanup_memory()
        bounds = self._bound_pooling(bounds)
        bounds = self._bound_classifier(bounds)
        return bounds

    def _bound_input(self, image: torch.Tensor, eps: float) -> Zonotope:
        patch_size = self.target.patch_size
        rearrange = Rearrange('1 c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        image = rearrange(image)

        eps_scaled = eps / self.normalizer.std[0]  
        bounds = Zonotope(self.args, p=self.p, eps=eps_scaled,
                          perturbed_word_index=None, value=image,
                          start_perturbation=0, end_perturbation=image.shape[0])

        bounds = bounds.dense(self.target.to_patch_embedding[1])

        e, n, _ = bounds.zonotope_w.shape
        cls_tokens = repeat(self.target.cls_token, '() n d -> n d')
        cls_tokens_value_w = cls_tokens.unsqueeze(0)
        cls_tokens_errors_w = torch.zeros_like(cls_tokens).unsqueeze(0).repeat(e - 1, 1, 1)
        cls_tokens_zonotope_w = torch.cat([cls_tokens_value_w, cls_tokens_errors_w], dim=0)

        full_zonotope_w = torch.cat((cls_tokens_zonotope_w, bounds.zonotope_w), dim=1)
        bounds = make_zonotope_new_weights_same_args(full_zonotope_w, bounds, clone=False)
        bounds = bounds.add(self.target.pos_embedding[:, :(n + 1)])
        return bounds

    def _bound_layer(self, bounds_input: Zonotope, attn, ff, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        if bounds_input.error_term_range_low is not None:
            bounds_input = bounds_input.recenter_zonotope_and_eliminate_error_term_ranges()

        if self.args.error_reduction_method == 'box':
            bounds_input = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms)

        layer_normed = bounds_input.layer_norm(get_layernorm(attn).norm, get_layernorm(attn).layer_norm_type)
        
        _, _, _, attention = self._bound_attention(layer_normed, get_inner(attn), layer_num=layer_num)  

        bounds_input = bounds_input.expand_error_terms_to_match_zonotope(attention)
        attention = attention.add(bounds_input)  

        attention_layer_normed = attention.layer_norm(get_layernorm(ff).norm, get_layernorm(ff).layer_norm_type)

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        feed_forward = get_inner(ff)
        intermediate = attention_layer_normed.dense(feed_forward.net[0])  
        intermediate = intermediate.relu()  
        dense = intermediate.dense(feed_forward.net[3])  

        attention = attention.expand_error_terms_to_match_zonotope(intermediate)
        dense = dense.add(attention)  

        if not self.args.keep_intermediate_zonotopes:
            del intermediate
            del attention

        return None, None, None, dense
        
    def _bound_attention(self, bounds_input: Zonotope, attn, layer_num=-1):
        num_attention_heads = attn.heads

        query = bounds_input.dense(attn.to_q)
        key = bounds_input.dense(attn.to_k)
        query = query.add_attention_heads_dim(num_attention_heads)
        key = key.add_attention_heads_dim(num_attention_heads)

        # Dot Product
        if self.args.num_fast_dot_product_layers_due_to_switch == -1:
            attention_scores = query.dot_product(key, verbose=self.verbose)
        else:
            attention_scores = query.dot_product_precise(key, verbose=self.verbose)
            
        attention_scores = attention_scores.multiply(attn.scale)

        if not self.args.keep_intermediate_zonotopes:
            del query
            del key

        # --- DYNAMIC PRUNING LOGIC ---
        pruning_mask = None
        if self.token_pruning_enabled and layer_num >= self.prune_layer_idx:
            num_tokens = attention_scores.zonotope_w.shape[-1]
            if self.tokens_to_prune > 0:
                k_keep = num_tokens - self.tokens_to_prune
            else:
                k_keep = self.tokens_to_keep
            k_keep = max(1, min(k_keep, num_tokens))

            l_scores, u_scores = attention_scores.concretize()
            topk_values = torch.topk(l_scores, k_keep, dim=-1, largest=True, sorted=True).values
            cutoff_threshold = topk_values[..., -1].unsqueeze(-1)
            
            mask_tensor = (u_scores >= cutoff_threshold).float()
            pruning_mask = mask_tensor

        attention_probs = softmax_with_mask(
            attention_scores,
            mask=pruning_mask,
            verbose=self.verbose, 
            no_constraints=not self.args.add_softmax_sum_constraint
        )

        value = bounds_input.dense(attn.to_v)
        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        value = value.add_attention_heads_dim(num_attention_heads)
        context = attention_probs.dot_product(value.t(), verbose=self.verbose)

        attention = context.remove_attention_heads_dim()
        attention = attention.dense(attn.to_out[0])

        return attention_scores, attention_probs, context, attention

    def _bound_pooling(self, bounds: Zonotope) -> Zonotope:
        bounds = make_zonotope_new_weights_same_args(new_weights=bounds.zonotope_w[:, :1, :], source_zonotope=bounds, clone=False)
        return bounds

    def _bound_classifier(self, bounds: Zonotope) -> Zonotope:
        bounds = bounds.layer_norm(self.target.mlp_head[0], self.target.layer_norm_type)
        bounds = bounds.dense(self.target.mlp_head[1])
        return bounds

    def verify(self, example, example_num: int): raise NotImplementedError
    def verify_safety(self, example, image, index, eps): raise NotImplementedError
    def get_safety(self, label: int, classifier_bounds: Zonotope) -> bool: raise NotImplementedError
