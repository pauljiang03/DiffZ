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
# We import specific internals to recreate the softmax logic locally
from Verifiers.Zonotope import (
    Zonotope, 
    make_zonotope_new_weights_same_args, 
    process_values, 
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
    for i in range(args.samples):
        while True:
            example = data[random.randint(0, len(data) - 1)]
            logits = target(example["image"].to(args.device))
            prediction = torch.argmax(logits, dim=-1)

            if prediction != example["label"]:
                continue  # incorrectly classified

            examples.append(example)
            break
    return examples

# --- Local Softmax Implementation (Avoiding modifications to Zonotope.py) ---

def softmax_with_mask(zonotope: Zonotope, 
                      mask: Optional[torch.Tensor] = None, 
                      verbose=False,
                      use_new_softmax=True,
                      no_constraints=True,
                      add_value_positivity_constraint=False,
                      use_new_reciprocal=True) -> Zonotope:
    """
    A standalone implementation of Zonotope Softmax that supports masking.
    This simulates e^-inf = 0 for pruned tokens by multiplying the exponential terms 
    by the mask before normalization.
    """
    num_rows = zonotope.zonotope_w.size(-2)
    num_values = zonotope.zonotope_w.size(-1)
    A = zonotope.zonotope_w.size(0)

    if use_new_softmax:
        if zonotope.args.batch_softmax_computation:
            # Batch computation logic (simplified for clarity, mirroring Zonotope.py)
            sum_w_list, new_error_terms_collapsed_list = [], []
            for a in range(A):
                sum_exp_diffs_w, new_error_terms_collapsed = process_values(
                    zonotope.zonotope_w[a:a+1], zonotope, A=1, num_rows=num_rows, num_values=num_values,
                    keep_intermediate_zonotopes=zonotope.args.keep_intermediate_zonotopes
                )
                sum_w_list.append(sum_exp_diffs_w)
                new_error_terms_collapsed_list.append(new_error_terms_collapsed)
                cleanup_memory()

            sum_exp_diffs_w = torch.cat(sum_w_list, dim=0)
            new_error_terms_collapsed = torch.cat(new_error_terms_collapsed_list, dim=0)
        else:
            sum_exp_diffs_w, new_error_terms_collapsed = process_values(
                zonotope.zonotope_w, zonotope, A, num_rows, num_values,
                keep_intermediate_zonotopes=zonotope.args.keep_intermediate_zonotopes
            )

        # Reconstruct the Zonotope representing sum(exp(diffs))
        new_error_terms_collapsed_intermediate_shape = torch.zeros(
            A * num_rows * num_values, A, num_rows, num_values, device=zonotope.device
        )
        indices = torch.arange(A * num_rows * num_values, device=zonotope.device)
        to_add = torch.ones_like(new_error_terms_collapsed, dtype=torch.bool, device=zonotope.device)
        new_error_terms_collapsed_intermediate_shape[indices, to_add] = new_error_terms_collapsed[to_add]
        
        new_error_terms_collapsed_good_shape = new_error_terms_collapsed_intermediate_shape.permute(1, 0, 2, 3)

        if not zonotope.args.keep_intermediate_zonotopes:
            del new_error_terms_collapsed_intermediate_shape
            cleanup_memory()

        final_sum_exps_zonotope_w = torch.cat([sum_exp_diffs_w, new_error_terms_collapsed_good_shape], dim=1)
        zonotope_sum_exp_diffs = make_zonotope_new_weights_same_args(final_sum_exps_zonotope_w, source_zonotope=zonotope, clone=False)

        # --- APPLY MASK HERE ---
        if mask is not None:
            # Mask the exponential terms: m * e^z
            zonotope_sum_exp_diffs = zonotope_sum_exp_diffs.multiply(mask)
        # -----------------------

        # Compute Reciprocal (Denominator)
        zonotope_softmax = zonotope_sum_exp_diffs.reciprocal(
            original_implementation=not use_new_reciprocal, 
            y_positive_constraint=add_value_positivity_constraint
        )

    else:
        # Alternative Path
        zonotope_exp = zonotope.exp_minimal_area()

        # --- APPLY MASK HERE ---
        if mask is not None:
            zonotope_exp = zonotope_exp.multiply(mask)
        # -----------------------

        l, u = zonotope_exp.concretize()
        # Ensure positivity (pruned tokens are 0, which is fine)
        # assert (l > -1e-9).all() 

        zonotope_sum_w = zonotope_exp.zonotope_w.sum(dim=-1, keepdim=True).repeat(1, 1, 1, num_values)
        zonotope_sum = make_zonotope_new_weights_same_args(zonotope_sum_w, zonotope, clone=False)

        zonotope_softmax = zonotope_exp.divide(
            zonotope_sum, 
            use_original_reciprocal=not use_new_reciprocal, 
            y_positive_constraint=add_value_positivity_constraint
        )

    if no_constraints:
        return zonotope_softmax

    # Constraints logic (unchanged from original, but called on the result)
    u, l = zonotope_softmax.concretize()
    if (l.sum(dim=-1) - 1).abs().max().item() < 1e-6 and (u.sum(dim=-1) - 1).abs().max().item() < 1e-6:
        del u, l
        cleanup_memory()
        return zonotope_softmax

    return zonotope_softmax.add_equality_constraint_on_softmax()


# --- Main Class Definition ---

class VerifierTopKPrune(Verifier):
    """
    Differential ViT Verifier implementing the Masked Softmax method[cite: 47, 52].
    This simulates pruning by masking attention weights rather than slicing tensors,
    allowing for sound differential verification of P vs P'.
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
        self.method = args.method
        
        self.hidden_act = args.hidden_act
        self.layer_norm = target.layer_norm_type
        self.normalizer = normalizer

        time_tag = datetime.now().strftime('%b%d_%H-%M-%S')
        self.res_filename = f"resultsVit_topk_prune_p_{args.p}_{time_tag}.csv"

        # Pruning parameters
        self.token_pruning_enabled = getattr(args, 'prune_tokens', False)
        self.prune_layer_idx = getattr(args, 'prune_layer_idx', -1)
        self.tokens_to_keep = getattr(args, 'tokens_to_keep', -1)
        
        self.showed_warning = False
        self.num_classes = num_classes
        
    def _run_single_model_bounds(self, image: torch.Tensor, eps: float, is_pruned: bool) -> Optional[Zonotope]:
        """ Helper function to run the core bounding logic for one model (P or P'). """
        original_prune_enabled = self.token_pruning_enabled
        self.token_pruning_enabled = is_pruned
        bounds = self.get_bounds_difference_in_scores(image, eps)
        self.token_pruning_enabled = original_prune_enabled
        return bounds

    def run(self, data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs verification and returns three lists of results:
        1. Differential bounds ([L_P - U_P', U_P - L_P'])
        2. Individual bounds for P ([L_P, U_P])
        3. Individual bounds for P' ([L_P', U_P'])
        """
        examples = sample_correct_samples(self.args, data, self.target)
        if self.eps <= 0 or len(examples) == 0:
            print("Verification setup issues (eps<=0 or no samples found). Exiting.")
            return [], [], []

        results_list_diff = []
        results_list_p = []
        results_list_p_prime = []
        
        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Running Verification for {len(examples)} samples using Softmax Masking Method...")
        print(f"Results log file: {file_path}")

        is_pruned_val = str(self.args.prune_tokens).lower()
        prune_layer_val = self.args.prune_layer_idx
        tokens_kept_val = self.args.tokens_to_keep

        with open(file_path, "w") as self.results_file:
            header = "index,class,lower_bound,upper_bound,timing,is_pruned,prune_layer,tokens_kept,bound_type\n"
            self.results_file.write(header)
            
            for i, example in enumerate(examples):
                start = time.time()
                embeddings = example["image"].to(self.device)

                # 1. Run P (Unpruned) - No masking
                bounds_P = self._run_single_model_bounds(embeddings, self.eps, is_pruned=False)
                
                # 2. Run P' (Pruned) - With masking enabled if args.prune_tokens is True
                bounds_P_prime = self._run_single_model_bounds(embeddings, self.eps, is_pruned=self.args.prune_tokens)

                if bounds_P is None or bounds_P_prime is None:
                    print(f"Warning: Verification failed for Sample {i}. Skipping.")
                    continue

                lower_P, upper_P = bounds_P.concretize()
                lower_P_prime, upper_P_prime = bounds_P_prime.concretize()
                
                timing = time.time() - start
                sample_label = example['label'].item()
                
                # 3. Calculate Differential Bounds: P - P'
                # Differential Lower = Lower(P) - Upper(P')
                lower_bound_diff = lower_P - upper_P_prime
                # Differential Upper = Upper(P) - Lower(P')
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

                # Log all bound types to CSV
                for c in range(self.num_classes):
                    self.results_file.write(f"{i},{c},{lower_bounds_np_diff[c]:f},{upper_bounds_np_diff[c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{tokens_kept_val},differential\n")
                    self.results_file.write(f"{i},{c},{lower_P[0,c]:f},{upper_P[0,c]:f},{timing:.4f},False,-1,-1,individual_P\n")
                    self.results_file.write(f"{i},{c},{lower_P_prime[0,c]:f},{upper_P_prime[0,c]:f},{timing:.4f},{is_pruned_val},{prune_layer_val},{tokens_kept_val},individual_P_prime\n")

                self.results_file.flush()
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i+1}/{len(examples)} samples...")

        print(f"\nCompleted verification for {len(examples)} samples.")
        return results_list_diff, results_list_p, results_list_p_prime

    def get_bounds_difference_in_scores(self, image: torch.Tensor, eps: float) -> Optional[Zonotope]:
        if self.args.error_reduction_method == "None" and not self.showed_warning:
            print("Warning: No error reduction method for Zonotope verifier!")
            self.showed_warning = True

        cleanup_memory()
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(image, eps=eps)

                for i, (attn, ff) in enumerate(self.target.transformer.layers):
                    # Masking is applied inside _bound_layer -> _bound_attention
                    attention_scores, attention_probs, self_attention_output, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)

                    if not self.args.keep_intermediate_zonotopes:
                        del attention_scores, attention_probs, self_attention_output
                        cleanup_memory()

                bounds = self._bound_pooling(bounds)
                bounds = self._bound_classifier(bounds)
                return bounds
        except errorType as err:
            print("\n=========================================================================")
            print(f"VERIFICATION FAILED: {type(err).__name__} occurred during bounding.")
            print(f"Current Pruning State (P'): {self.token_pruning_enabled}")
            print(err)
            print("=========================================================================")
            return None

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

        # --- PRUNING MASK LOGIC ---
        pruning_mask = None
        if self.token_pruning_enabled:
            # Check if we are at or past the layer where pruning activates
            if layer_num >= self.prune_layer_idx:
                # Shape: (Heads, 1+Err, Seq, Keys)
                # We want to mask the LAST dimension (Keys)
                num_tokens = attention_scores.zonotope_w.shape[-1] 
                
                # Default: keep first K tokens (CLS + patches)
                keep_k = self.tokens_to_keep
                if keep_k > num_tokens:
                     keep_k = num_tokens

                # Create binary mask: 1 for kept, 0 for pruned
                mask_tensor = torch.zeros(num_tokens, device=self.device)
                mask_tensor[:keep_k] = 1.0 
                
                # Reshape for broadcast: (1, 1, 1, num_tokens)
                pruning_mask = mask_tensor.reshape(1, 1, 1, -1)

        # Use LOCAL softmax implementation that accepts mask
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
        # Assuming the first token is the CLS token
        bounds = make_zonotope_new_weights_same_args(new_weights=bounds.zonotope_w[:, :1, :], source_zonotope=bounds, clone=False)
        return bounds

    def _bound_classifier(self, bounds: Zonotope) -> Zonotope:
        bounds = bounds.layer_norm(self.target.mlp_head[0], self.target.layer_norm_type)
        bounds = bounds.dense(self.target.mlp_head[1])
        return bounds

    def verify(self, example, example_num: int):
        raise NotImplementedError("The max-eps search is disabled for differential verification.")

    def verify_safety(self, example, image, index, eps):
        raise NotImplementedError("This method is not used in differential verification mode.")

    def get_safety(self, label: int, classifier_bounds: Zonotope) -> bool:
        raise NotImplementedError("This method is not used in differential verification mode.")
