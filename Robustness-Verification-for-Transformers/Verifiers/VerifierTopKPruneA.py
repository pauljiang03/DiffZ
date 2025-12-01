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

# --- Local Softmax Implementation (Robust Version) ---

def softmax_with_mask(zonotope: Zonotope, 
                      mask: Optional[torch.Tensor] = None, 
                      verbose=False,
                      no_constraints=True,
                      add_value_positivity_constraint=False,
                      use_new_reciprocal=True) -> Zonotope:
    num_values = zonotope.zonotope_w.size(-1)
    zonotope_exp = zonotope.exp(minimal_area=False)

    if mask is not None:
        w_shape = zonotope_exp.zonotope_w.shape
        m_shape = mask.shape
        
        # Robust Shape Alignment
        if len(w_shape) > len(m_shape):
            if w_shape[0] == m_shape[0]:
                diff = len(w_shape) - len(m_shape)
                new_shape = list(m_shape)
                for _ in range(diff): new_shape.insert(1, 1)
                mask_aligned = mask.view(new_shape)
            else:
                diff = len(w_shape) - len(m_shape)
                new_shape = (1,) * diff + m_shape
                mask_aligned = mask.view(new_shape)
        else:
            mask_aligned = mask

        try:
            zonotope_exp = zonotope_exp.multiply(mask_aligned)
        except RuntimeError as e:
            print(f"   [CRITICAL] Mask Multiplication Failed! W: {w_shape}, M: {m_shape}")
            raise e

    zonotope_sum_w = zonotope_exp.zonotope_w.sum(dim=-1, keepdim=True).repeat(1, 1, 1, num_values)
    zonotope_sum = make_zonotope_new_weights_same_args(zonotope_sum_w, zonotope, clone=False)

    zonotope_softmax = zonotope_exp.divide(
        zonotope_sum, 
        use_original_reciprocal=not use_new_reciprocal, 
        y_positive_constraint=add_value_positivity_constraint
    )

    if no_constraints:
        return zonotope_softmax

    u, l = zonotope_softmax.concretize()
    if (l.sum(dim=-1) - 1).abs().max().item() < 1e-6 and (u.sum(dim=-1) - 1).abs().max().item() < 1e-6:
        del u, l
        cleanup_memory()
        return zonotope_softmax

    return zonotope_softmax.add_equality_constraint_on_softmax()


# --- Main Class Definition ---

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
        self.res_filename = f"resultsVit_topk_prune_p_{args.p}_{time_tag}.csv"

        self.token_pruning_enabled = getattr(args, 'prune_tokens', False)
        self.prune_layer_idx = getattr(args, 'prune_layer_idx', -1)
        self.tokens_to_keep = getattr(args, 'tokens_to_keep', -1)
        self.tokens_to_prune = getattr(args, 'tokens_to_prune', 0)
        self.num_classes = num_classes
        self.showed_warning = False

    def run(self, data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        examples = sample_correct_samples(self.args, data, self.target)
        if self.eps <= 0 or len(examples) == 0:
            print("Verification setup issues (eps<=0 or no samples found). Exiting.")
            return [], [], []

        results_list_diff = []
        results_list_p = []
        results_list_p_prime = []
        
        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Running Coupled Verification for {len(examples)} samples...")
        
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

                # --- NEW: COUPLED EXECUTION ---
                # We run both models in the same 'session' to share error terms
                res_P, res_P_prime, res_Diff = self._get_coupled_bounds(embeddings, self.eps)

                if res_P is None:
                    print(f"Warning: Verification failed for Sample {i}. Skipping.")
                    continue

                # Concretize Results
                l_P, u_P = res_P.concretize()
                l_Pp, u_Pp = res_P_prime.concretize()
                l_Diff, u_Diff = res_Diff.concretize()
                
                timing = time.time() - start
                sample_label = example['label'].item()
                
                # Convert to numpy
                l_P_np, u_P_np = l_P[0].cpu().numpy(), u_P[0].cpu().numpy()
                l_Pp_np, u_Pp_np = l_Pp[0].cpu().numpy(), u_Pp[0].cpu().numpy()
                l_Diff_np, u_Diff_np = l_Diff[0].cpu().numpy(), u_Diff[0].cpu().numpy()

                results_list_diff.append({
                    'label': sample_label,
                    'lower_bounds': l_Diff_np, # Now holds SYMBOLIC diff
                    'upper_bounds': u_Diff_np,
                    'index': i, 'time': timing
                })
                
                results_list_p.append({
                    'label': sample_label,
                    'lower_bounds': l_P_np,
                    'upper_bounds': u_P_np,
                    'index': i, 'time': timing
                })

                results_list_p_prime.append({
                    'label': sample_label,
                    'lower_bounds': l_Pp_np,
                    'upper_bounds': u_Pp_np,
                    'index': i, 'time': timing
                })

                for c in range(self.num_classes):
                    # Note: We now write the TIGHT symbolic diff here
                    self.results_file.write(f"{i},{c},{l_Diff_np[c]:f},{u_Diff_np[c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},differential\n")
                    self.results_file.write(f"{i},{c},{l_P_np[c]:f},{u_P_np[c]:f},{timing:.4f},False,-1,-1,individual_P\n")
                    self.results_file.write(f"{i},{c},{l_Pp_np[c]:f},{u_Pp_np[c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},individual_P_prime\n")

                self.results_file.flush()
                if (i + 1) % 1 == 0:
                    print(f"Completed {i+1}/{len(examples)} samples...")

        print(f"\nCompleted verification for {len(examples)} samples.")
        return results_list_diff, results_list_p, results_list_p_prime

    def _get_coupled_bounds(self, image, eps) -> Tuple[Optional[Zonotope], Optional[Zonotope], Optional[Zonotope]]:
        """
        Runs Unpruned and Pruned models in a single session to allow symbolic subtraction.
        Returns: (Bounds_P, Bounds_P_prime, Bounds_Diff)
        """
        cleanup_memory() # Clear old errors ONLY ONCE at the start
        
        try:
            with torch.no_grad():
                # 1. Generate Input Zonotope (Shared Source of Truth)
                # Error terms created here are e0...ek
                z_root = self._bound_input(image, eps=eps)
                
                # 2. Run Branch 1: Unpruned P
                # We clone z_root so P doesn't consume it
                self.token_pruning_enabled = False
                z_P = self._propagate_model(z_root.clone())
                
                # 3. Run Branch 2: Pruned P'
                # The verifier continues allocating NEW error terms (ek+1...em) 
                # effectively stacking them. This is what we want.
                self.token_pruning_enabled = getattr(self.args, 'prune_tokens', True)
                z_P_prime = self._propagate_model(z_root.clone())
                
                # 4. Symbolic Subtraction (Z_P - Z_P_prime)
                # e0...ek (Input noise) will cancel out perfectly!
                # Remaining terms represent the divergence due to pruning.
                
                # Use add(x * -1) because standard zonotopes usually support affine add
                z_neg_prime = z_P_prime.multiply(-1.0)
                z_Diff = z_P.add(z_neg_prime)
                
                return z_P, z_P_prime, z_Diff
                
        except Exception as err:
            print(f"\n[Coupled Run Error]: {err}")
            if self.debug: 
                import traceback
                traceback.print_exc()
            return None, None, None

    def _propagate_model(self, bounds: Zonotope) -> Zonotope:
        """
        Runs the Zonotope through the full ViT (Transformer + Head).
        Does NOT call cleanup_memory(), to preserve error terms.
        """
        # Transformer Layers
        for i, (attn, ff) in enumerate(self.target.transformer.layers):
            # Pass layer_num to enable pruning check inside _bound_layer
            _, _, _, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)
            
            # Optional: Intermediate cleanup (careful with this!)
            if not self.args.keep_intermediate_zonotopes:
                # We can't do full cleanup_memory() here or we lose the error registry.
                # Just rely on Python GC for intermediate objects.
                pass

        # Pooling & Head
        bounds = self._bound_pooling(bounds)
        bounds = self._bound_classifier(bounds)
        return bounds

    # --- Standard Bound Functions (Unchanged logic, just used by _propagate_model) ---

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

    def is_in_fast_layer(self, layer_num: int):
        return True

    def _bound_layer(self, bounds_input: Zonotope, attn, ff, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        if bounds_input.error_term_range_low is not None:
            bounds_input = bounds_input.recenter_zonotope_and_eliminate_error_term_ranges()

        if self.args.error_reduction_method == 'box':
            bounds_input = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms)

        layer_normed = bounds_input.layer_norm(get_layernorm(attn).norm, get_layernorm(attn).layer_norm_type)

        attention_scores, attention_probs, context, attention = self._bound_attention(
            layer_normed, get_inner(attn), layer_num=layer_num
        )  

        bounds_input = bounds_input.expand_error_terms_to_match_zonotope(attention)
        attention = attention.add(bounds_input)  

        attention_layer_normed = attention.layer_norm(get_layernorm(ff).norm, get_layernorm(ff).layer_norm_type)
        feed_forward = get_inner(ff)

        intermediate = attention_layer_normed.dense(feed_forward.net[0])  
        intermediate = intermediate.relu()  
        dense = intermediate.dense(feed_forward.net[3])  

        attention = attention.expand_error_terms_to_match_zonotope(intermediate)
        dense = dense.add(attention)  
        return attention_scores, attention_probs, context, dense
        
    def do_dot_product(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int):
        if self.args.num_fast_dot_product_layers_due_to_switch == -1:
            return left_z.dot_product(right_z, verbose=self.verbose)
        if self.is_in_fast_layer(layer_num=current_layer_num):
            return left_z.dot_product_fast(right_z, verbose=self.verbose)
        else:
            return left_z.dot_product_precise(right_z, verbose=self.verbose)
        
    def do_context(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int):
        return self.do_dot_product(left_z, right_z.t(), current_layer_num)

    def _bound_attention(self, bounds_input: Zonotope, attn, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        num_attention_heads = attn.heads
        query = bounds_input.dense(attn.to_q)
        key = bounds_input.dense(attn.to_k)
        query = query.add_attention_heads_dim(num_attention_heads)
        key = key.add_attention_heads_dim(num_attention_heads)

        attention_scores = self.do_dot_product(query, key, layer_num)
        attention_scores = attention_scores.multiply(attn.scale)

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
            attention_scores, mask=pruning_mask, verbose=self.verbose, 
            no_constraints=not self.args.add_softmax_sum_constraint
        )

        value = bounds_input.dense(attn.to_v)
        value = value.add_attention_heads_dim(num_attention_heads)
        context = self.do_context(attention_probs, value, layer_num)
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

    # ... Stub methods for abstract base class ...
    def verify(self, example, example_num: int): raise NotImplementedError()
    def verify_safety(self, example, image, index, eps): raise NotImplementedError()
    def get_safety(self, label: int, classifier_bounds: Zonotope) -> bool: raise NotImplementedError()
