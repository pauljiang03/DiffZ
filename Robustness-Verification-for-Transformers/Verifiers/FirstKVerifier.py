import os
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
from argparse import Namespace
from einops import rearrange, repeat 

from Verifiers.Verifier import Verifier
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args, cleanup_memory
from model import JointModel 
from mnist import normalizer 
from fake_logger import FakeLogger 

def sample_correct_samples(args, data, target):
    examples = []
    for i in range(args.samples):
        while True:
            example = data[random.randint(0, len(data) - 1)]
            logits = target(example["image"])
            prediction = torch.argmax(logits, dim=-1)

            if prediction != example["label"]:
                continue  

            examples.append(example)
            break

    return examples



class FirstKVerifier(Verifier):

    def __init__(self, args: Namespace, target: JointModel, logger, num_classes: int, normalizer, output_epsilon: float):
        self.args = args
        self.device = args.device
        self.target_unified: JointModel = target.to(args.device)
        self.target_unified.eval()
        self.logger = logger
        self.num_classes = num_classes
        self.normalizer = normalizer
        self.results_directory = args.results_directory

        self.p = args.p if args.p < 10 else float("inf")
        self.input_eps = args.eps
        self.output_epsilon = output_epsilon

        self.verbose = args.debug or args.verbose

        time_tag = time.strftime('%b%d_%H-%M-%S')
        self.res_filename = f"resultsPruningCert_k{target.k}_layer{target.pruning_layer}_outEps{output_epsilon}_{time_tag}.csv"

    def run(self, data):
        examples = data

        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        results_file = open(file_path, "w")
        results_file.write("index,is_safe,max_abs_diff_bound,timing\n")

        safe_count = 0
        total_count = len(examples)
        if total_count == 0:
             print("No examples to verify.")
             results_file.close()
             return

        print(f"Verifying {total_count} examples")
        for i, example in enumerate(examples):
            self.logger.write(f"--- Verifying Sample {i+1}/{total_count} ---")
            is_safe, max_abs_diff_bound, timing = self.check_pruning_bound(example, self.input_eps, self.output_epsilon)

            if is_safe is None:
                self.logger.write("  Result: ERROR")
                results_file.write(f"{i},ERROR,{max_abs_diff_bound},{timing}\n")
            else:
                if is_safe:
                    safe_count += 1
                    self.logger.write(f"  Result: SAFE (Max Diff Bound: {max_abs_diff_bound:.6f}, Epsilon: {self.output_epsilon:.6f})")
                else:
                    self.logger.write(f"  Result: UNSAFE (Max Diff Bound: {max_abs_diff_bound:.6f}, Epsilon: {self.output_epsilon:.6f})")
                results_file.write(f"{i},{int(is_safe)},{max_abs_diff_bound},{timing}\n")

            results_file.flush()
            cleanup_memory()

        self.logger.write(f"--- Verification Summary ---")
        self.logger.write(f"Total Verified: {total_count}")
        self.logger.write(f"Verified Safe: {safe_count}")
        if total_count > 0:
            self.logger.write(f"Safety Rate: {100.0 * safe_count / total_count:.2f}%")
        results_file.close()


    def check_pruning_bound(self, example, input_eps: float, output_epsilon: float) -> Tuple[Optional[bool], float, float]:
        image = example["image"].to(self.device)
        start_time = time.time()
        max_abs_diff_bound = float('inf')
        is_within_bounds = None

        try:
            Z_diff = self.get_output_difference_bounds(image, input_eps)

            if Z_diff is None:
                print("Verification failed: Propagation returned None.")
                is_within_bounds = None
            else:
                l, u = Z_diff.concretize()
                if l is None or u is None: 
                    print("Verification failed: Concretization returned None.")
                    is_within_bounds = None
                else:
                    print("\n--- Difference Zonotope (P - P') Bounds ---")
                    print("Lower Bound of Difference (l_diff):")
                    print(l)
                    print("\nUpper Bound of Difference (u_diff):")
                    print(u)
                    print("-" * 45)
                    
                    max_abs_val = torch.max(torch.abs(l), torch.abs(u))
                    max_abs_diff_bound = max_abs_val.max().item()

                    tolerance = 1e-6
                    is_within_bounds = (max_abs_diff_bound < output_epsilon - tolerance)

                    max_abs_val = torch.max(torch.abs(l), torch.abs(u))
                    max_abs_diff_bound = max_abs_val.max().item()

                    tolerance = 1e-6
                    is_within_bounds = (max_abs_diff_bound < output_epsilon - tolerance)

        except Exception as e:
            print(f"ERROR during verification for sample: {e}")
            import traceback
            traceback.print_exc()
            is_within_bounds = None

        end_time = time.time()
        timing = end_time - start_time
        if not isinstance(max_abs_diff_bound, float):
             max_abs_diff_bound = float('inf')
        return is_within_bounds, max_abs_diff_bound, timing


    def get_output_difference_bounds(self, image: torch.Tensor, input_eps: float) -> Optional[Zonotope]:
        cleanup_memory()
        try:
            with torch.no_grad():
                Z_input = self._bound_input_unified(image, input_eps)
                if Z_input is None: raise ValueError("Input bounding failed")

                Z_logits_P = self._propagate_path(Z_input.clone(), self.target_unified.unpruned_blocks, apply_pruning=False)
                if Z_logits_P is None: raise ValueError("Unpruned path propagation failed")
                cleanup_memory()
                #print(f"Shape of Z_logits_P before concretize: {Z_logits_P.zonotope_w.shape if hasattr(Z_logits_P, 'zonotope_w') else 'No zonotope_w'}")


                #print("  Concretizing unpruned path bounds")
                l_P, u_P = Z_logits_P.concretize()
                if l_P is not None and u_P is not None:
                    max_abs_P = torch.max(torch.abs(l_P), torch.abs(u_P)).max().item()
                    print(f"  Max abs bound (Unpruned Path P): {max_abs_P:.6f}")
                else:
                    print("  Concretization failed for Unpruned Path P.")
                #print(f"Shape of l_P: {l_P.shape if l_P is not None else 'None'}, Shape of u_P: {u_P.shape if u_P is not None else 'None'}")



                Z_logits_P_prime = self._propagate_path(Z_input.clone(), self.target_unified.pruned_blocks, apply_pruning=True)
                if Z_logits_P_prime is None: raise ValueError("Pruned path propagation failed")
                cleanup_memory()
                #print(f"Shape of Z_logits_P_prime before concretize: {Z_logits_P_prime.zonotope_w.shape if hasattr(Z_logits_P_prime, 'zonotope_w') else 'No zonotope_w'}")



                #print("  Concretizing pruned path bounds")
                l_P_prime, u_P_prime = Z_logits_P_prime.concretize()
                if l_P_prime is not None and u_P_prime is not None:
                    max_abs_P_prime = torch.max(torch.abs(l_P_prime), torch.abs(u_P_prime)).max().item()
                    #print(f"  Max abs bound (Pruned Path P'): {max_abs_P_prime:.6f}")
                else:
                    print("  Concretization failed for Pruned Path P'.")
                #print(f"Shape of l_P_prime: {l_P_prime.shape if l_P_prime is not None else 'None'}, Shape of u_P_prime: {u_P_prime.shape if u_P_prime is not None else 'None'}")


                print("\n" + "="*20 + " LOGIT ANALYSIS " + "="*20)
                l_P, u_P = Z_logits_P.concretize()
                l_P_prime, u_P_prime = Z_logits_P_prime.concretize()
                print("\n--- Unpruned Path (P) Logits ---")
                if l_P is not None:
                    print("Center Logits (P):")
                    print(Z_logits_P.zonotope_w[0])
                    print("\nLogit Lower Bounds (P):")
                    print(l_P)
                    print("\nLogit Upper Bounds (P):")
                    print(u_P)
                else:
                    print("Could not concretize unpruned path logits.")
                
                print("\n--- Pruned Path (P') Logits ---")
                if l_P_prime is not None:
                    print("Center Logits (P'):")
                    print(Z_logits_P_prime.zonotope_w[0])
                    print("\nLogit Lower Bounds (P'):")
                    print(l_P_prime)
                    print("\nLogit Upper Bounds (P'):")
                    print(u_P_prime)
                else:
                    print("Could not concretize pruned path logits.")

                print("="*58 + "\n") # Separator



                Z_diff = Z_logits_P.subtract(Z_logits_P_prime)
                #print(f"Shape of Z_diff before concretize: {Z_diff.zonotope_w.shape if hasattr(Z_diff, 'zonotope_w') else 'No zonotope_w'}")
                l_diff, u_diff = Z_diff.concretize()
                #print(f"Shape of l_diff: {l_diff.shape if l_diff is not None else 'None'}, Shape of u_diff: {u_diff.shape if u_diff is not None else 'None'}")
                return Z_diff

        except Exception as err:
            print(f"ERROR during abstract interpretation: {err}")
            import traceback
            traceback.print_exc()
            return None


    def _bound_input_unified(self, image: torch.Tensor, eps: float) -> Zonotope:
        rearrange_layer = self.target_unified.patch_embedder_rearrange
        patch_size = self.target_unified.patch_size
        in_chans = image.shape[1]
        img_size = image.shape[2]
        patch_dim = in_chans * patch_size ** 2
        num_patches = (img_size // patch_size) ** 2
        PREFIX_TOKEN_COUNT = self.target_unified.prefix_tokens.shape[1]

        image_r = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        image_patched = image_r.squeeze(0)

        #print(f"\n--- Inside _bound_input_unified ---")
        #print(f"Input eps (received): {eps}")

        eps_scaled = eps / self.normalizer.std[0]
        #print(f"Scaled eps: {eps_scaled}")
        num_input_errors = image_patched.nelement() if eps > 0 else 0 
        self.args.num_input_error_terms = num_input_errors 
        bounds = Zonotope(self.args, p=self.p, eps=eps_scaled,
                          perturbed_word_index=None,
                          value=image_patched,
        
                          start_perturbation=0,
                          end_perturbation=num_input_errors
                         )
        #print(f"Shape of initial bounds.zonotope_w: {bounds.zonotope_w.shape if hasattr(bounds, 'zonotope_w') else 'No zonotope_w'}")
        #print(f"Number of error terms in initial bounds: {bounds.get_num_error_terms() if hasattr(bounds, 'get_num_error_terms') else 'No get_num_error_terms'}")


        bounds = bounds.dense(self.target_unified.patch_embedder_linear)
        #print(f"Shape of bounds.zonotope_w after patch embedder: {bounds.zonotope_w.shape if hasattr(bounds, 'zonotope_w') else 'No zonotope_w'}")
        #print(f"Number of error terms after patch embedder: {bounds.get_num_error_terms() if hasattr(bounds, 'get_num_error_terms') else 'No get_num_error_terms'}")

        if bounds.zonotope_w.ndim == 3: 
            e, n, d = bounds.zonotope_w.shape
        else: 
            raise ValueError("Expected 3D Zonotope after patch embedding linear layer")

        cls_token_param = self.target_unified.prefix_tokens

        cls_value_w = cls_token_param.squeeze(0) 
        cls_zeros_w = torch.zeros(e - 1, cls_value_w.shape[0], cls_value_w.shape[1], device=bounds.device)
        cls_zonotope_w = torch.cat([cls_value_w.unsqueeze(0), cls_zeros_w], dim=0) 

        full_zonotope_w = torch.cat((cls_zonotope_w, bounds.zonotope_w), dim=1) 
        bounds = make_zonotope_new_weights_same_args(full_zonotope_w, bounds, clone=False)
        bounds.num_words = full_zonotope_w.shape[1] 

        pos_embed = self.target_unified.pos_embed[:, :bounds.num_words, :]
        bounds = bounds.add(pos_embed)

        #print(f"Shape of final bounds.zonotope_w in _bound_input_unified: {bounds.zonotope_w.shape if hasattr(bounds, 'zonotope_w') else 'No zonotope_w'}")
        #print(f"Number of error terms in final bounds of _bound_input_unified: {bounds.get_num_error_terms() if hasattr(bounds, 'get_num_error_terms') else 'No get_num_error_terms'}")

        return bounds


    def _propagate_path(self, Z_current: Zonotope, blocks: nn.ModuleList, apply_pruning: bool) -> Optional[Zonotope]:
        target_model = self.target_unified
        k_for_pruning = target_model.k
        pruning_layer_index = target_model.pruning_layer

        
        #print(f"\n--- Starting _propagate_path (apply_pruning: {apply_pruning}) ---")
        #print(f"Shape of Z_current.zonotope_w at start: {Z_current.zonotope_w.shape if hasattr(Z_current, 'zonotope_w') else 'No zonotope_w'}")
        #print(f"Number of error terms in Z_current at start: {Z_current.get_num_error_terms() if hasattr(Z_current, 'get_num_error_terms') else 'No get_num_error_terms'}")

        for i, block in enumerate(blocks):
            Z_res_attn = Z_current.clone()
            norm_layer_attn = block.attn.fn.norm
            Z_normed_attn = Z_current.layer_norm(norm_layer_attn, target_model.layer_norm_type)
            #print(f"Shape after attn layer norm: {Z_normed_attn.zonotope_w.shape if hasattr(Z_normed_attn, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after attn layer norm: {Z_normed_attn.get_num_error_terms() if hasattr(Z_normed_attn, 'get_num_error_terms') else 'No get_num_error_terms'}")
            attn_module = block.attn.fn.fn
            Z_attn_output = self._bound_attention_custom(Z_normed_attn, attn_module, layer_num=i)
            #print(f"Shape after attention: {Z_attn_output.zonotope_w.shape if hasattr(Z_attn_output, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after attention: {Z_attn_output.get_num_error_terms() if hasattr(Z_attn_output, 'get_num_error_terms') else 'No get_num_error_terms'}")


            Z_res_attn = Z_res_attn.expand_error_terms_to_match_zonotope(Z_attn_output)
            Z_attn_output = Z_attn_output.expand_error_terms_to_match_zonotope(Z_res_attn)
            Z_current = Z_attn_output.add(Z_res_attn)
            #print(f"Shape after residual + attn: {Z_current.zonotope_w.shape if hasattr(Z_current, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after residual + attn: {Z_current.get_num_error_terms() if hasattr(Z_current, 'get_num_error_terms') else 'No get_num_error_terms'}")


            Z_res_ff = Z_current.clone()
            norm_layer_ff = block.ff.fn.norm
            Z_normed_ff = Z_current.layer_norm(norm_layer_ff, target_model.layer_norm_type)
            #print(f"Shape after ff layer norm: {Z_normed_ff.zonotope_w.shape if hasattr(Z_normed_ff, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after ff layer norm: {Z_normed_ff.get_num_error_terms() if hasattr(Z_normed_ff, 'get_num_error_terms') else 'No get_num_error_terms'}")

            ff_module = block.ff.fn.fn
            Z_ff_output = self._bound_feed_forward_custom(Z_normed_ff, ff_module)
            #print(f"Shape after feed forward: {Z_ff_output.zonotope_w.shape if hasattr(Z_ff_output, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after feed forward: {Z_ff_output.get_num_error_terms() if hasattr(Z_ff_output, 'get_num_error_terms') else 'No get_num_error_terms'}")

            Z_res_ff = Z_res_ff.expand_error_terms_to_match_zonotope(Z_ff_output)
            Z_ff_output = Z_ff_output.expand_error_terms_to_match_zonotope(Z_res_ff)
            Z_current = Z_ff_output.add(Z_res_ff)
            #print(f"Shape after residual + ff: {Z_current.zonotope_w.shape if hasattr(Z_current, 'zonotope_w') else 'No zonotope_w'}")
            #print(f"Error terms after residual + ff: {Z_current.get_num_error_terms() if hasattr(Z_current, 'get_num_error_terms') else 'No get_num_error_terms'}")


            if apply_pruning and i == pruning_layer_index:
                 Z_current = Z_current.first_k_prune(k_for_pruning)

            if hasattr(self.args, 'error_reduction_method') and self.args.error_reduction_method == 'box' and hasattr(self.args, 'max_num_error_terms') and Z_current.get_num_error_terms() > self.args.max_num_error_terms:
                 if Z_current.error_term_range_low is not None: 
                      Z_current = Z_current.recenter_zonotope_and_eliminate_error_term_ranges()
                 Z_current = Z_current.reduce_num_error_terms_box(self.args.max_num_error_terms)

            cleanup_memory()

        Z_pooled = self._bound_pooling_unified(Z_current)
        Z_logits = self._bound_classifier_unified(Z_pooled)

        return Z_logits


    def _bound_attention_custom(self, bounds_input: Zonotope, attn: 'CustomAttention', layer_num=-1) -> Zonotope:
         num_attention_heads = attn.heads

         query = bounds_input.dense(attn.to_q)
         key = bounds_input.dense(attn.to_k)
         value = bounds_input.dense(attn.to_v)

         query = query.add_attention_heads_dim(num_attention_heads)
         key = key.add_attention_heads_dim(num_attention_heads)
         value = value.add_attention_heads_dim(num_attention_heads)

         attention_scores = self.do_dot_product(query, key, layer_num)
         attention_scores = attention_scores.multiply(attn.scale)

         with torch.no_grad():
             l_scores, u_scores = attention_scores.concretize()
             if u_scores is not None:
                 max_u_for_softmax = u_scores.max(dim=-1, keepdim=True)[0] # Shape: (A, N_q, 1)
                 if attention_scores.zonotope_w.ndim == 4: # Check it's 4D as expected
                     current_centers = attention_scores.zonotope_w[:, 0, :, :] # Shape: (A, N_q, N_k_dim)
                     attention_scores.zonotope_w[:, 0, :, :] = current_centers - max_u_for_softmax
                 elif attention_scores.zonotope_w.ndim == 3: # Fallback if it's still 3D for some reason
                     print(f"WARNING: attention_scores.zonotope_w is 3D. Shape: {attention_scores.zonotope_w.shape}")
                     print(f"Shape of max_u_for_softmax: {max_u_for_softmax.shape}")
                     attention_scores.zonotope_w[0, :, :] = attention_scores.zonotope_w[0, :, :] - max_u_for_softmax.squeeze(0) # Attempt to make it work if A=1
                 else:
                     print(f"ERROR: Unexpected zonotope_w ndim: {attention_scores.zonotope_w.ndim}")
             else:
                 print("WARNING: Could not apply softmax stability trick because upper bounds were None.")

         add_constraint = hasattr(self.args, 'add_softmax_sum_constraint') and self.args.add_softmax_sum_constraint
         print("\n--- Debug: attention_scores before softmax ---")
         l_as, u_as = attention_scores.concretize()
         if l_as is not None:
            print(f"attention_scores Lower - Min: {l_as.min().item():.4e}, Max: {l_as.max().item():.4e}, Mean: {l_as.mean().item():.4e}, NaN_count: {torch.isnan(l_as).sum().item()}")
         if u_as is not None:
            print(f"attention_scores Upper - Min: {u_as.min().item():.4e}, Max: {u_as.max().item():.4e}, Mean: {u_as.mean().item():.4e}, NaN_count: {torch.isnan(u_as).sum().item()}")
         attention_probs = attention_scores.softmax(verbose=self.verbose, no_constraints=not add_constraint)

         value = value.expand_error_terms_to_match_zonotope(attention_probs)
         context = self.do_context(attention_probs, value, layer_num)

         context = context.remove_attention_heads_dim()

         if isinstance(attn.to_out, nn.Sequential):
              attention_output = context.dense(attn.to_out[0])
         elif isinstance(attn.to_out, nn.Identity):
              attention_output = context
         else:
              raise TypeError(f"Unexpected type for attn.to_out: {type(attn.to_out)}")


         return attention_output


    def _bound_feed_forward_custom(self, bounds_input: Zonotope, ff: 'FeedForward') -> Zonotope:
         x = bounds_input.dense(ff.net[0])
         x = x.relu()
         x = x.dense(ff.net[3])
         return x


    def _bound_pooling_unified(self, bounds: Zonotope) -> Zonotope:
         if bounds.zonotope_w.ndim == 4:
             bounds = bounds.remove_attention_heads_dim()

         seq_dim = 1 
         if bounds.zonotope_w.shape[seq_dim] == 1:
              pooled_weights = bounds.zonotope_w 
         else:
              slices = [slice(None)] * bounds.zonotope_w.ndim
              slices[seq_dim] = slice(0, 1) 
              pooled_weights = bounds.zonotope_w[tuple(slices)]

         if pooled_weights.shape[seq_dim] != 1:
              raise ValueError(f"Pooling resulted in unexpected sequence length: {pooled_weights.shape[seq_dim]}")

         pooled_z = make_zonotope_new_weights_same_args(pooled_weights, source_zonotope=bounds, clone=False)
         pooled_z.num_words = 1
         return pooled_z


    def _bound_classifier_unified(self, bounds: Zonotope) -> Zonotope:
         if not (bounds.zonotope_w.ndim == 3 and bounds.zonotope_w.shape[1] == 1):
              raise ValueError(f"Unexpected shape entering classifier: {bounds.zonotope_w.shape}")

         norm_layer = self.target_unified.final_norm
         linear_layer = self.target_unified.classification_head


         bounds = bounds.layer_norm(norm_layer, self.target_unified.layer_norm_type)


         bounds = bounds.dense(linear_layer)
         return bounds


    def do_dot_product(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int) -> Zonotope:

         use_precise = False
         if hasattr(self.args, 'num_fast_dot_product_layers_due_to_switch'):
             if self.args.num_fast_dot_product_layers_due_to_switch != -1:

                 is_fast = current_layer_num < self.args.num_fast_dot_product_layers_due_to_switch 
                 use_precise = not is_fast
         elif hasattr(self.args, 'zonotope_slow') and self.args.zonotope_slow:
             use_precise = True

         if use_precise:
             return left_z.dot_product_precise(right_z, verbose=self.verbose)
         else:
             return left_z.dot_product_fast(right_z, verbose=self.verbose)


    def do_context(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int) -> Zonotope:
         return self.do_dot_product(left_z, right_z.t(), current_layer_num)
