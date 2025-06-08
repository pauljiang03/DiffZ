import torch
import torch.nn as nn
from typing import Union, Dict, Tuple, Optional, List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from termcolor import colored

PREFIX_TOKEN_COUNT = 1

def FirstKPrune(x: torch.Tensor, k: int) -> torch.Tensor:
    num_prefix_tokens = PREFIX_TOKEN_COUNT
    total_seq_len = x.size(1)
    max_seq_tokens_can_keep = total_seq_len - num_prefix_tokens

    if k < 0:
        k = 0
    k = min(k, max_seq_tokens_can_keep)

    num_tokens_to_keep = num_prefix_tokens + k

    if num_tokens_to_keep >= total_seq_len:
        return x

    pruned_x = x[:, 0:num_tokens_to_keep, :]
    return pruned_x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, use_no_var_impl=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.use_no_var_impl = use_no_var_impl

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        if self.use_no_var_impl:
             x = x - u
        else:
             s = (x - u).pow(2).mean(-1, keepdim=True)
             x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layer_norm_type='no_var'):
        super().__init__()
        self.norm = LayerNorm(dim, use_no_var_impl=(layer_norm_type == 'no_var'))
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q,k,v])
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class ViTBlock(nn.Module):
     def __init__(self, dim, heads, dim_head, mlp_dim, layer_norm_type='no_var', dropout=0.):
         super().__init__()
         self.attn = Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), layer_norm_type))
         self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), layer_norm_type))
     def forward(self, x):
         x = self.attn(x)
         x = self.ff(x)
         return x

class JointModel(nn.Module):
    def __init__(self, k: int = 100, img_size: int = 28, embed_dim: int = 64, num_classes: int = 10,
                 in_chans: int = 1, depth: int = 1, heads: int = 4, mlp_dim: int = 128,
                 pruning_layer: int = 0, layer_norm_type: str = 'no_var', dropout: float = 0.0,
                 patch_size: int = 7, pool: str = 'cls', dim_head: int = 64):
        super().__init__()

        self.k = k
        self.depth = depth
        self.pruning_layer = pruning_layer
        self.num_heads = heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.layer_norm_type = layer_norm_type
        self.dropout = dropout
        self.patch_size = patch_size
        self.pool = pool
        self.num_classes = num_classes
        self.dim_head = dim_head

        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size ** 2
        self.num_patches = num_patches
        #dim_head = embed_dim // heads
       
        self.patch_embedder_rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_embedder_linear = nn.Linear(patch_dim, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + PREFIX_TOKEN_COUNT, embed_dim) * .02)
        self.prefix_tokens = nn.Parameter(torch.zeros(1, PREFIX_TOKEN_COUNT, embed_dim))
        self.input_dropout = nn.Dropout(dropout)

        self.final_norm = LayerNorm(embed_dim, use_no_var_impl=(layer_norm_type == 'no_var'))
        self.classification_head = nn.Linear(embed_dim, num_classes)

        self.unpruned_blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, layer_norm_type=layer_norm_type, dropout=dropout)
            for _ in range(depth)
        ])
        self.pruned_blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, layer_norm_type=layer_norm_type, dropout=dropout)
            for _ in range(depth)
        ])

        self._initialize_with_same_weights()

    def load_from_original_vit(self, path: str):
        print("Loading weights from original ViT checkpoint...")
        pretrained_state_dict = torch.load(path)
        new_state_dict = {}
        for key, value in pretrained_state_dict.items():
            new_key = key
            if key.startswith('to_patch_embedding.1.'):
                new_key = key.replace('to_patch_embedding.1.', 'patch_embedder_linear.')
            elif key.startswith('mlp_head.0.'):
                new_key = key.replace('mlp_head.0.', 'final_norm.')
            elif key.startswith('mlp_head.1.'):
                new_key = key.replace('mlp_head.1.', 'classification_head.')
            elif key.startswith('transformer.layers.'):
                parts = key.split('.')
                layer_idx = parts[2]
                block_type_idx = parts[3]
                if block_type_idx == '0': 
                    rest_of_key = '.'.join(parts[4:])
                    new_key = f"unpruned_blocks.{layer_idx}.attn.{rest_of_key}"
                elif block_type_idx == '1': 
                    rest_of_key = '.'.join(parts[4:])
                    new_key = f"unpruned_blocks.{layer_idx}.ff.{rest_of_key}"
            
            elif key == 'pos_embedding':
                new_key = 'pos_embed'
            elif key == 'cls_token':
                new_key = 'prefix_tokens'
            
            new_state_dict[new_key] = value
    
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(colored("Warning: The following keys were missing from the new state_dict:", "yellow"))
            for k in missing_keys: print(f"  {k}")
        if unexpected_keys:
            print(colored("Warning: The following keys in the new state_dict were not used by the model:", "yellow"))
            for k in unexpected_keys: print(f"  {k}")
            
        self._initialize_with_same_weights()
        print("Successfully loaded and mapped weights.")

    def _initialize_with_same_weights(self):
        self.pruned_blocks.load_state_dict(self.unpruned_blocks.state_dict())

    def _pruned_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._common_preprocessing(x)
        for i, block in enumerate(self.unpruned_blocks):
            x = block(x)
            if i == self.pruning_layer:
                x = FirstKPrune(x, self.k)
                
        x = self._apply_pooling_and_head(x)
        return x

    def _common_preprocessing(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder_rearrange(x)
        x = self.patch_embedder_linear(x)
        prefix = self.prefix_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((prefix, x), dim=1)
        current_seq_len = x.shape[1]
        x = x + self.pos_embed[:, :current_seq_len, :]
        x = self.input_dropout(x)
        return x

    def _apply_pooling_and_head(self, x: torch.Tensor) -> torch.Tensor:
         if self.pool == 'mean':
             x = x[:, PREFIX_TOKEN_COUNT:, :].mean(dim=1) 
         else: 
             x = x[:, 0]
         x = self.final_norm(x)
         x = self.classification_head(x)
         return x

    def _unpruned_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._common_preprocessing(x)
        for block in self.unpruned_blocks:
            x = block(x)
        x = self._apply_pooling_and_head(x)
        return x

    def _pruned_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._common_preprocessing(x)
        for i, block in enumerate(self.pruned_blocks):
            x = block(x)
            if i == self.pruning_layer:
                x = FirstKPrune(x, self.k)
        x = self._apply_pooling_and_head(x)
        return x

    def forward(self, x: torch.Tensor, return_full_info: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        unpruned_logits = self._unpruned_forward(x)
        pruned_logits = self._pruned_forward(x.clone())
        logits_diff = unpruned_logits - pruned_logits

        if return_full_info:
            max_logit_diff_abs = torch.max(torch.abs(logits_diff), dim=-1)[0]
            unpruned_probs = torch.softmax(unpruned_logits, dim=-1)
            pruned_probs = torch.softmax(pruned_logits, dim=-1)
            probs_diff = unpruned_probs - pruned_probs
            max_prob_diff_abs = torch.max(torch.abs(probs_diff), dim=-1)[0]
            return {
                "unpruned_logits": unpruned_logits,
                "pruned_logits": pruned_logits,
                "logits_diff": logits_diff,
                "max_logit_diff_abs": max_logit_diff_abs,
                "unpruned_probs": unpruned_probs,
                "pruned_probs": pruned_probs,
                "probs_diff": probs_diff,
                "max_prob_diff_abs": max_prob_diff_abs
            }
        else:
             max_logit_diff_abs = torch.max(torch.abs(logits_diff))
             return max_logit_diff_abs

if __name__ == "__main__":
    img_size = 28
    patch_size = 7
    in_chans = 1
    num_classes = 10
    embed_dim = 64
    depth = 3
    heads = 4
    mlp_dim = 128
    k = 16
    pruning_layer = 1

    x_input = torch.randn(1, in_chans, img_size, img_size)

    model = JointModel(
        k=k,
        img_size=img_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        in_chans=in_chans,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pruning_layer=pruning_layer,
        layer_norm_type='no_var',
        patch_size=patch_size,
        pool='cls'
    )
    model.eval()

    with torch.no_grad():
        result = model(x_input, return_full_info=True)

    print("Model:")
    print(f"  k (tokens to keep): {k}")
    print(f"  Pruning Layer Index: {pruning_layer}")
    print(f"  Depth: {depth}")
    print(f"  Embed Dim: {embed_dim}")
    print(f"  Heads: {heads}")
    print(f"  MLP Dim: {mlp_dim}")
    print(f"  LayerNorm Type: {model.layer_norm_type}")
    print(f"  Patch Size: {patch_size}")
    print( )
    print("Shapes:")
    print(f"  Input Shape: {x_input.shape}")
    print(f"  Unpruned Logits Shape: {result['unpruned_logits'].shape}")
    print(f"  Pruned Logits Shape: {result['pruned_logits'].shape}")
    print(f"  Logits Diff Shape: {result['logits_diff'].shape}")
    print( )
    print("Difference Metrics (Max over classes per batch element):")
    print(f"  Max Absolute Logit Difference (|P-P'|): {result['max_logit_diff_abs'].tolist()}")
    if 'max_prob_diff_abs' in result:
         print(f"  Max Absolute Probability Difference: {result['max_prob_diff_abs'].tolist()}")
    print( )

    original_tokens = model.num_patches + PREFIX_TOKEN_COUNT
    expected_pruned_tokens = original_tokens
    if model.pruning_layer < model.depth:
        expected_pruned_tokens = min(original_tokens, PREFIX_TOKEN_COUNT + model.k)

    print("Token Count:")
    print(f"  Original token count (incl. prefix): {original_tokens}")
    print(f"  Expected token count after pruning (layer {model.pruning_layer}): {expected_pruned_tokens} (kept {model.k} sequence tokens)")
    if original_tokens > 0:
        tokens_removed = original_tokens - expected_pruned_tokens
        percent_removed = (tokens_removed / original_tokens) * 100 if original_tokens > PREFIX_TOKEN_COUNT else 0
        print(f"  Tokens removed by pruning: {tokens_removed} ({percent_removed:.1f}%)")
