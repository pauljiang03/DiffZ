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
        
        # If lengths differ, we must align
        if len(w_shape) > len(m_shape):
            # CASE A: The first dimensions match (e.g. Heads=4 match Heads=4)
            # This implies the missing "Error Term" dimension is INSIDE (at index 1)
            # Zono: [4, 3093, 17, 17] vs Mask: [4, 17, 17] -> Needs [4, 1, 17, 17]
            if w_shape[0] == m_shape[0]:
                diff = len(w_shape) - len(m_shape)
                # Insert '1's starting at index 1
                new_shape = list(m_shape)
                for _ in range(diff):
                    new_shape.insert(1, 1)
                mask_aligned = mask.view(new_shape)
                
            # CASE B: The first dimensions DO NOT match
            # This implies the missing dimensions are at the START (standard broadcasting)
            # Zono: [3093, 4, 17, 17] vs Mask: [4, 17, 17] -> Needs [1, 4, 17, 17]
            else:
                diff = len(w_shape) - len(m_shape)
                new_shape = (1,) * diff + m_shape
                mask_aligned = mask.view(new_shape)
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
        
        self.showed_warning = False
        self.num_classes = num_classes
        
    def _run_single_model_bounds(self, image: torch.Tensor, eps: float, is_pruned: bool) -> Optional[Zonotope]:
        original_prune_enabled = self.token_pruning_enabled
        self.token_pruning_enabled = is_pruned
        
        try:
            bounds = self.get_bounds_difference_in_scores(image, eps)
        except Exception as e:
            if self.debug: 
                import traceback
                traceback.print_exc()
            print(f"Error in single model run (Pruned={is_pruned}): {e}")
            bounds = None
        finally:
            self.token_pruning_enabled = original_prune_enabled
            
        return bounds

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
        
        print(f"Running Verification for {len(examples)} samples using Sound Pruning...")
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

                bounds_P = self._run_single_model_bounds(embeddings, self.eps, is_pruned=False)
                bounds_P_prime = self._run_single_model_bounds(embeddings, self.eps, is_pruned=self.args.prune_tokens)

                if bounds_P is None or bounds_P_prime is None:
                    print(f"Warning: Verification failed for Sample {i}. Skipping.")
                    continue

                lower_P, upper_P = bounds_P.concretize()
                lower_P_prime, upper_P_prime = bounds_P_prime.concretize()
                
                timing = time.time() - start
                sample_label = example['label'].item()
                
                lower_bound_diff = lower_P - upper_P_prime
                upper_bound_diff = upper_P - lower_P_prime
                
                lower_bounds_np_diff = lower_bound_diff[0].cpu().numpy()
                upper_bounds_np_diff = upper_bound_diff[0].cpu().numpy()

                results_list_diff.append({
                    'label': sample_label,
                    'lower_bounds': lower_bounds_np_diff,
                    'upper_bounds': upper_bounds_np_diff,
                    'index': i, 'time': timing
                })
                
                results_list_p.append({
                    'label': sample_label,
                    'lower_bounds': lower_P[0].cpu().numpy(),
                    'upper_bounds': upper_P[0].cpu().numpy(),
                    'index': i, 'time': timing
                })

                results_list_p_prime.append({
                    'label': sample_label,
                    'lower_bounds': lower_P_prime[0].cpu().numpy(),
                    'upper_bounds': upper_P_prime[0].cpu().numpy(),
                    'index': i, 'time': timing
                })

                for c in range(self.num_classes):
                    self.results_file.write(f"{i},{c},{lower_bounds_np_diff[c]:f},{upper_bounds_np_diff[c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},differential\n")
                    self.results_file.write(f"{i},{c},{lower_P[0,c]:f},{upper_P[0,c]:f},{timing:.4f},False,-1,-1,individual_P\n")
                    self.results_file.write(f"{i},{c},{lower_P_prime[0,c]:f},{upper_P_prime[0,c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{prune_log_str},individual_P_prime\n")

                self.results_file.flush()
                
                if (i + 1) % 1 == 0:
                    print(f"Completed {i+1}/{len(examples)} samples...")

        print(f"\nCompleted verification for {len(examples)} samples.")
        return results_list_diff, results_list_p, results_list_p_prime

    def get_bounds_difference_in_scores(self, image: torch.Tensor, eps: float) -> Optional[Zonotope]:
        if self.args.error_reduction_method == "None" and not self.showed_warning:
            print("Warning: No error reduction method for Zonotope verifier!")
            self.showed_warning = True

        cleanup_memory()
        
        try:
            with torch.no_grad():
                bounds = self._bound_input(image, eps=eps)

                for i, (attn, ff) in enumerate(self.target.transformer.layers):
                    attention_scores, attention_probs, self_attention_output, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)

                    if not self.args.keep_intermediate_zonotopes:
                        del attention_scores, attention_probs, self_attention_output
                        cleanup_memory()

                bounds = self._bound_pooling(bounds)
                bounds = self._bound_classifier(bounds)
                return bounds
        except Exception as err:
            print("\n=========================================================================")
            print(f"VERIFICATION FAILED: {type(err).__name__} occurred during bounding.")
            print(f"Current Pruning State (P'): {self.token_pruning_enabled}")
            print(f"Error: {err}")
            print("=========================================================================")
            if self.debug:
                 import traceback
                 traceback.print_exc()
            raise err 

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

        output = dense

        return attention_scores, attention_probs, context, output
        
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

        if not self.args.keep_intermediate_zonotopes:
            del query
            del key

        # =========================================================
        #  SOUND DYNAMIC PRUNING LOGIC (Bottom-X / Top-K)
        # =========================================================
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
            
            # Mask Shape: [Batch, Heads, Q, K] (or similar)
            mask_tensor = (u_scores >= cutoff_threshold).float()
            pruning_mask = mask_tensor

        # =========================================================

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

    def verify(self, example, example_num: int):
        raise NotImplementedError("Max-eps search disabled.")
    
    def verify_safety(self, example, image, index, eps):
        raise NotImplementedError("Method not used.")

    def get_safety(self, label: int, classifier_bounds: Zonotope) -> bool:
        raise NotImplementedError("Method not used.")
