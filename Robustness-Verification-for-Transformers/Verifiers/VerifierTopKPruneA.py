import time
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from Verifiers.Zonotope import (
    Zonotope, 
    make_zonotope_new_weights_same_args, 
    cleanup_memory
)
# Import the base class from your original file (assuming it is named VerifierTopKPrune.py or similar)
# Adjust the import below to match your actual file name
from VerifierTopKPrune import VerifierTopKPrune, sample_correct_samples

# --- Helper: Symbolic Alignment & Subtraction ---

def robust_symbolic_subtract(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """
    Subtracts z2 from z1 symbolically (z_diff = z1 - z2).
    
    CRITICAL: This handles the [Coupled Run Error] where z1 and z2 have 
    different numbers of error terms (dim 0 of weights). It pads the 
    smaller zonotope with zeros to match the larger one before subtraction.
    """
    w1 = z1.zonotope_w
    w2 = z2.zonotope_w

    # Shape is usually [ErrorTerms, Batch, Dim, ...]
    # We need to align Dimension 0 (ErrorTerms)
    if w1.shape[0] != w2.shape[0]:
        max_errors = max(w1.shape[0], w2.shape[0])
        
        # Pad w1 if necessary
        if w1.shape[0] < max_errors:
            padding = max_errors - w1.shape[0]
            pad_shape = [padding] + list(w1.shape[1:])
            zeros = torch.zeros(pad_shape, device=w1.device, dtype=w1.dtype)
            w1 = torch.cat([w1, zeros], dim=0)
            
        # Pad w2 if necessary
        if w2.shape[0] < max_errors:
            padding = max_errors - w2.shape[0]
            pad_shape = [padding] + list(w2.shape[1:])
            zeros = torch.zeros(pad_shape, device=w2.device, dtype=w2.dtype)
            w2 = torch.cat([w2, zeros], dim=0)

    # Calculate Symbolic Difference
    # This preserves the correlation of the input noise terms!
    w_diff = w1 - w2
    
    # Return new Zonotope with the differenced weights
    return make_zonotope_new_weights_same_args(w_diff, z1, clone=False)


# --- Main Class Definition (Symbolic) ---

class VerifierTopKPruneA(VerifierTopKPrune):
    """
    'A' variant: Performs Symbolic Abstract Subtraction.
    Instead of concretizing runs separately, it subtracts the Zonotopes 
    tensor-wise to preserve input error term cancellation.
    """

    def __init__(self, args, target, logger, num_classes, normalizer):
        super().__init__(args, target, logger, num_classes, normalizer)
        self.res_filename = self.res_filename.replace(".csv", "_SYMBOLIC.csv")

    def _run_coupled_symbolic(self, image: torch.Tensor, eps: float) -> Optional[Zonotope]:
        """
        Runs the model twice (Pruned vs Unpruned) using the same input constraints,
        then subtracts the resulting Zonotopes symbolically.
        """
        cleanup_memory()
        
        try:
            with torch.no_grad():
                # 1. Generate the shared Input Zonotope
                # We generate it once to ensure the 'Input' noise terms (0 to N) are identical.
                z_input = self._bound_input(image, eps=eps)

                # 2. Run Path A: Standard (Pruning Disabled)
                # We assume _propagate_model is a helper that runs the layer loop
                # If _propagate_model doesn't exist in base, we implement the loop here.
                
                # We need to clone z_input or ensure the first run doesn't mutate it destructively
                # Zonotope ops usually return new objects, but let's be safe.
                # (Assuming make_zonotope... clones or creates new tensors)
                
                # --- Path 1: Unpruned ---
                original_prune_setting = self.token_pruning_enabled
                self.token_pruning_enabled = False
                z_unpruned = self._propagate_layers(z_input)
                
                # --- Path 2: Pruned ---
                self.token_pruning_enabled = getattr(self.args, 'prune_tokens', True)
                # We must regenerate input or rely on z_input being immutable. 
                # To be absolutely safe and independent:
                z_input_2 = self._bound_input(image, eps=eps) 
                z_pruned = self._propagate_layers(z_input_2)
                
                # Restore state
                self.token_pruning_enabled = original_prune_setting

                # 3. Symbolic Subtraction with Alignment
                # z_diff = z_unpruned - z_pruned
                z_diff = robust_symbolic_subtract(z_unpruned, z_pruned)
                
                return z_diff, z_unpruned, z_pruned

        except Exception as err:
            print(f"\n[Symbolic Run Error]: {err}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, None, None

    def _propagate_layers(self, bounds: Zonotope) -> Zonotope:
        """
        Helper to run the forward pass loop on a Zonotope.
        """
        for i, (attn, ff) in enumerate(self.target.transformer.layers):
            _, _, _, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)
        
        bounds = self._bound_pooling(bounds)
        bounds = self._bound_classifier(bounds)
        return bounds

    def run(self, data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        
        examples = sample_correct_samples(self.args, data, self.target)
        
        results_list_diff = []
        
        print(f"Running SYMBOLIC Verification for {len(examples)} samples...")

        for i, example in enumerate(examples):
            start = time.time()
            embeddings = example["image"].to(self.device)

            # --- The Core Change ---
            # Instead of two separate concretizations, we get the diff Zonotope
            z_diff, z_unpruned, z_pruned = self._run_coupled_symbolic(embeddings, self.eps)

            if z_diff is None:
                continue

            # Concretize the DIFFERENCE Zonotope
            # This yields [min(P - P'), max(P - P')] directly
            lower_diff, upper_diff = z_diff.concretize()
            
            # For logging purposes, we can also concretize the individuals
            l_p, u_p = z_unpruned.concretize()
            l_pp, u_pp = z_pruned.concretize()

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

            # Print concise log
            print(f"Sample {i}: Diff Range [{lower_bounds_np_diff[sample_label]:.4f}, {upper_bounds_np_diff[sample_label]:.4f}]")

        print(f"\nCompleted Symbolic Verification.")
        return results_list_diff, [], []
