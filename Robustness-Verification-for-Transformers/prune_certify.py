import os
import sys
import torch

from Parser import Parser, update_arguments
from Verifiers.VerifierZonotopeViT import VerifierZonotopeViT, sample_correct_samples
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from data_utils import set_seeds

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = Parser.get_parser()

    # --- Arguments to control token pruning ---
    parser.add_argument('--prune_tokens', action='store_true',
                        help='Enable First-K token pruning during verification.')
    parser.add_argument('--prune_layer_idx', type=int, default=0,
                        help='Transformer layer index AFTER which to apply token pruning.')
    parser.add_argument('--tokens_to_keep', type=int, default=9,
                        help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')

    args, _ = parser.parse_known_args(argv)
    args = update_arguments(args)

    # --- Basic Setup ---
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu !=-1 else "cpu")
    args.device = device  # Pass device through args

    # --- Data Loading ---
    test_data = mnist_test_dataloader(batch_size=1, shuffle=True)
    data_normalized = []
    for i, (x, y) in enumerate(test_data):
        if i >= args.samples:
            break
        data_normalized.append({
            "label": y.to(device),
            "image": x.to(device)
        })
    print(f"Loaded {len(data_normalized)} samples for verification.")

    # --- Model Loading ---
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
    model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
    model.eval()

    if args.prune_tokens:
        print("--- Running in TOKEN PRUNING mode ---")
    else:
        print("--- Running in BASELINE (no pruning) mode ---")

    # --- Run Verification ---
    logger = FakeLogger()
    verifier = VerifierZonotopeViT(args, model, logger, num_classes=10, normalizer=normalizer)

    # Select correctly classified samples for verification
    correct_samples = sample_correct_samples(args, data_normalized, model)

    if not correct_samples:
        print("\nCould not find any correctly classified samples to verify. Exiting.")
        sys.exit(0)

    print(f"\nStarting verification on {len(correct_samples)} correctly classified samples...")
    verifier.run(correct_samples)

    print("\nVerification complete.")
