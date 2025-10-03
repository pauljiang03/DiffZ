import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional

from datetime import datetime
import torch
from einops.einops import repeat
from einops.layers.torch import Rearrange

from Verifiers.Verifier import Verifier
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args, cleanup_memory
from vit import ViT
        


def get_layernorm(x):
    return x.fn


def get_inner(x):
    return x.fn.fn


def sample_correct_samples(args, data, target):
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


class DiffVerZonotopeViT(Verifier):
    def __init__(self, args, target: ViT, logger, num_classes: int, normalizer):
        self.args = args
        self.device = args.device
        self.target = target
        self.logger = logger
        self.res = args.res
        self.results_directory = args.results_directory

        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps # Used as fixed epsilon for differential check
        self.debug = args.debug
        self.verbose = args.debug or args.verbose
        self.method = args.method
        self.num_verify_iters = args.num_verify_iters
        self.max_eps = args.max_eps
        self.debug_pos = args.debug_pos
        self.perturbed_words = args.perturbed_words
        self.warmed = False

        self.hidden_act = args.hidden_act
        self.layer_norm = target.layer_norm_type
        self.normalizer = normalizer

        time_tag = datetime.now().strftime('%b%d_%H-%M-%S')
        self.res_filename = f"resultsVit_diff_p_{args.p}_{time_tag}.csv"

        self.ibp = args.method == "ibp"

        ### Pruning
        self.token_pruning_enabled = getattr(args, 'prune_tokens', False)
        self.prune_layer_idx = getattr(args, 'prune_layer_idx', -1)
        self.tokens_to_keep = getattr(args, 'tokens_to_keep', -1)
        ###

        self.showed_warning = False
        self.target: ViT
        self.num_classes = num_classes
        
    def _run_single_model_bounds(self, image: torch.Tensor, eps: float, is_pruned: bool) -> Optional[Zonotope]:
        """
        Helper function to run the core bounding logic for one model (P or P').
        It temporarily overrides the pruning setting and restores it.
        """
        original_prune_enabled = self.token_pruning_enabled
        
        self.token_pruning_enabled = is_pruned
        
        bounds = self.get_bounds_difference_in_scores(image, eps)
        
        self.token_pruning_enabled = original_prune_enabled
        
        return bounds

    def _check_num_error_terms(self, zonotope, name: str, layer_num: int):
        if hasattr(zonotope, 'error_terms') and zonotope.error_terms is not None:
            num_error_terms = zonotope.error_terms.size(0)
            pruning_limit = getattr(self, 'pruning_n', -1) 
            if pruning_limit > 0 and num_error_terms > pruning_limit:
                print(f"[DiffVer Layer {layer_num}] WARNING: {name} has {num_error_terms} error terms, exceeding the pruning limit of {pruning_limit}.")
            print(f"[DiffVer Layer {layer_num}] DEBUG: {name} completed with {num_error_terms} error terms (Zonotope center size: {zonotope.center.shape}).")
        elif hasattr(zonotope, 'center'):
             print(f"[DiffVer Layer {layer_num}] DEBUG: {name} is not a full Zonotope object or is missing error terms.")


    def run(self, data):
        examples = sample_correct_samples(self.args, data, self.target)

        if not self.args.prune_tokens:
             print("Warning: Pruning (--prune_tokens) is not enabled in args. The output will show P-P (difference should be zero).")
        
        if self.eps <= 0:
            print("Differential verification requires a positive perturbation epsilon (--eps). Exiting.")
            return

        if len(examples) == 0:
            print("No correctly classified samples found. Exiting.")
            return

        sum_avg_L, sum_avg_U = 0, 0
        
        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as self.results_file:
            self.results_file.write("index,class,lower_bound,upper_bound,timing,is_pruned,prune_layer,tokens_kept\n")
            
            print(f"Running Differential Verification (P - P') for fixed eps={self.eps}...")
            
            is_pruned_val = str(self.args.prune_tokens).lower()
            prune_layer_val = self.args.prune_layer_idx
            tokens_kept_val = self.args.tokens_to_keep

            for i, example in enumerate(examples):
                self.logger.write("Sample", i)
                start = time.time()
                
                embeddings = example["image"]
                embeddings = embeddings if self.args.cpu else embeddings.cuda()

                # --- 1. Compute bounds for P (Unpruned) and P' (Pruned) ---
                bounds_P = self._run_single_model_bounds(embeddings, self.eps, is_pruned=False)
                bounds_P_prime = self._run_single_model_bounds(embeddings, self.eps, is_pruned=self.args.prune_tokens)

                if bounds_P is None or bounds_P_prime is None:
                    print(f"Warning: Verification failed for Sample {i}. Skipping differential verification.")
                    continue

                # --- 2. Concretize P and P' bounds ---
                lower_P, upper_P = bounds_P.concretize()
                lower_P_prime, upper_P_prime = bounds_P_prime.concretize()

                # --- 3. Print P Bounds (Unpruned) ---
                print(f"\n--- Sample {i} (Label: {example['label'].item()}) Bounds P (Unpruned) ---")
                # Time is roughly the time to compute P bounds
                print(f"Time: {time.time() - start:.4f}s | Eps: {self.eps} | Pruning Active: False") 
                print("Class | Lower Bound | Upper Bound")
                print("---------------------------------")
                for c in range(self.num_classes):
                    L = lower_P[0, c].item()
                    U = upper_P[0, c].item()
                    print(f"{c:<5} | {L:11.5f} | {U:11.5f}")

                # --- 4. Print P' Bounds (Pruned) ---
                print(f"\n--- Sample {i} (Label: {example['label'].item()}) Bounds P' (Pruned) ---")
                # Time is roughly the total time up to computing P' bounds
                print(f"Time: {time.time() - start:.4f}s | Eps: {self.eps} | Pruning Active: True") 
                print(f"[Pruning] Layer {prune_layer_val}: Keeping {tokens_kept_val} tokens.")
                print("Class | Lower Bound | Upper Bound")
                print("---------------------------------")
                for c in range(self.num_classes):
                    L = lower_P_prime[0, c].item()
                    U = upper_P_prime[0, c].item()
                    print(f"{c:<5} | {L:11.5f} | {U:11.5f}")

                # --- 5. Calculate and Print Differential Bounds (P - P') ---
                try:
                    zonotope_difference = bounds_P - bounds_P_prime
                except Exception as e:
                    print(f"Error during Zonotope subtraction for Sample {i}: {e}. Skipping sample.")
                    continue

                end = time.time()
                timing = end - start
                
                lower_bound, upper_bound = zonotope_difference.concretize() # This calculates the P - P' bounds

                print(f"\n--- Sample {i} (Label: {example['label'].item()}) Differential Bounds P - P' ---")
                print(f"Time: {timing:.4f}s | Eps: {self.eps} | Pruning Active: {self.args.prune_tokens}")
                print("Class | Lower Bound | Upper Bound")
                print("---------------------------------")
                
                for c in range(self.num_classes):
                    L = lower_bound[0, c].item()
                    U = upper_bound[0, c].item()
                    
                    sum_avg_L += L
                    sum_avg_U += U
                    
                    print(f"{c:<5} | {L:11.5f} | {U:11.5f}")

                    self.results_file.write(
                        f"{i},{c},{L:f},{U:f},{timing:f},{is_pruned_val},{prune_layer_val},{tokens_kept_val}\n"
                    )
                self.results_file.flush()

        num_samples = len(examples)
        if num_samples > 0:
            avg_L = sum_avg_L / (num_samples * self.num_classes)
            avg_U = sum_avg_U / (num_samples * self.num_classes)
            self.logger.write(f"\nTotal Samples: {num_samples}")
            self.logger.write(f"Average Lower Bound (All Classes): {avg_L:.5f}")
            self.logger.write(f"Average Upper Bound (All Classes): {avg_U:.5f}")
        
        self.logger.write(f"Differential results saved to: {file_path}")
        
        return 0, 0  # Not used in differential mode


    def get_bounds_difference_in_scores(self, image: torch.Tensor, eps: float) -> Optional[Zonotope]:
        if self.args.error_reduction_method == "None" and not self.showed_warning:
            START_WARNING = '\033[93m'
            END_WARNING = '\033[0m'
            print(START_WARNING + "Warning: No error reduction method for Zonotope verifier!" + END_WARNING)
            self.showed_warning = True

        cleanup_memory()
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(image, eps=eps)

                for i, (attn, ff) in enumerate(self.target.transformer.layers):
                    if self.args.log_error_terms_and_time:
                        print(f"\nLayer {i} (P' is_pruned={self.token_pruning_enabled})")

                    start = time.time()
                    attention_scores, attention_probs, self_attention_output, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)
                    end = time.time()

                    if self.args.log_error_terms_and_time:
                        print("Time to do attention layer %d is %.3f seconds" % (i, end - start))

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
            print("Error Details:")
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
        if self.args.log_error_terms_and_time:
            print("Bound_input before error reduction has %d error terms" % bounds_input.num_error_terms)

        if bounds_input.error_term_range_low is not None:
            bounds_input = bounds_input.recenter_zonotope_and_eliminate_error_term_ranges()

        if self.args.error_reduction_method == 'box':
            bounds_input_reduced_box = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms)
            bounds_input = bounds_input_reduced_box

        layer_normed = bounds_input.layer_norm(get_layernorm(attn).norm, get_layernorm(attn).layer_norm_type)  # Layer norm 1

        attention_scores, attention_probs, context, attention = self._bound_attention(
            layer_normed, get_inner(attn), layer_num=layer_num
        )  

        bounds_input = bounds_input.expand_error_terms_to_match_zonotope(attention)
        attention = attention.add(bounds_input)  

        if self.token_pruning_enabled and layer_num == self.prune_layer_idx:
            print(f"[Pruning] Layer {layer_num}: Keeping {self.tokens_to_keep} tokens.")
            # Note: This slicing assumes the tokens to keep are the first self.tokens_to_keep after the CLS token, 
            # and that the CLS token is handled correctly upstream (it is, since attention is the input to this block).
            print(f"[VERIFY PRUNE] Shape BEFORE Pruning: {attention.zonotope_w.shape}")
            print(f"[Pruning] Layer {layer_num}: Keeping {self.tokens_to_keep} tokens.")
            
            pruned_zonotope_w = attention.zonotope_w[:, :self.tokens_to_keep + 1, :]
            attention = make_zonotope_new_weights_same_args(pruned_zonotope_w, attention)

            print(f"[VERIFY PRUNE] Shape AFTER Pruning: {attention.zonotope_w.shape}")


        attention_layer_normed = attention.layer_norm(get_layernorm(ff).norm, get_layernorm(ff).layer_norm_type)  # prenorm 2

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        feed_forward = get_inner(ff)

        intermediate = attention_layer_normed.dense(feed_forward.net[0])  
        self._check_num_error_terms(intermediate, "MLP Intermediate (Pre-ReLU)", layer_num)
        intermediate = intermediate.relu()  
        self._check_num_error_terms(intermediate, "\033[91mMLP Intermediate (Post-ReLU)\033[0m", layer_num)
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

        attention_probs = attention_scores.softmax(verbose=self.verbose, no_constraints=not self.args.add_softmax_sum_constraint)

        value = bounds_input.dense(attn.to_v)

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        value = value.add_attention_heads_dim(num_attention_heads)
        context = self.do_context(attention_probs, value, layer_num)

        if self.verbose:
            value.print("value")
            context.print("context")

        context = context.remove_attention_heads_dim()

        attention = context.dense(attn.to_out[0])

        return attention_scores, attention_probs, context, attention

    def _bound_pooling(self, bounds: Zonotope) -> Zonotope:
        # Note: assuming the first token is the CLS token
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
