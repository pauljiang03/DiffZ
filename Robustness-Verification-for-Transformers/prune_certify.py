import os
import sys
import psutil
import torch

from Parser import Parser, update_arguments
from Verifiers.VerifierZonotopeViT import VerifierZonotopeViT, sample_correct_samples
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from vit_attack import pgd_attack

# --- Argument Parsing (Original + Pruning) ---
argv = sys.argv[1:]
parser = Parser.get_parser()

# --- ADDED: Pruning Arguments ---
parser.add_argument('--prune_tokens', action='store_true',
                    help='Enable First-K token pruning during verification.')
parser.add_argument('--prune_layer_idx', type=int, default=0,
                    help='Transformer layer index AFTER which to apply token pruning.')
parser.add_argument('--tokens_to_keep', type=int, default=9,
                    help='Number of tokens to keep after pruning (e.g., 9 = [CLS] + 8 patches).')

args, _ = parser.parse_known_args(argv)
args = update_arguments(args)

# --- KEPT: Hardcoded settings from original script ---
args.with_lirpa_transformer = False
args.all_words = True
args.concretize_special_norm_error_together = True
args.num_input_error_terms = 28 * 28

# --- KEPT: GPU/CPU Setup from original script ---
if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if psutil.cpu_count() > 4 and args.cpu_range != "Default":
    start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
    os.sched_setaffinity(0, {i for i in range(start, end + 1)})

# --- KEPT: Seeding and Data Loading from original script ---
from data_utils import set_seeds
set_seeds(args.seed)

test_data = mnist_test_dataloader(batch_size=1, shuffle=False)
set_seeds(args.seed)

# --- KEPT: Model Definition from original script (using 3 layers) ---
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
args.device = device # Pass device through args

model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

model.load_state_dict(torch.load("mnist_transformer.pt", map_location=device))
model.eval()

print(f"Using device: {args.device}")
print(args)
print(f"Test Dataset size: {len(test_data)}")

logger = FakeLogger()
print("\n")

# --- KEPT: Data normalization loop from original script ---
data_normalized = []
for i, (x, y) in enumerate(test_data):
    data_normalized.append({
        "label": y.to(device),
        "image": x.to(device)
    })
    # Load a fixed pool of 101 images to sample from, matching original script
    if i == 100:
        break

# --- ADDED: Print current running mode ---
if args.prune_tokens:
    print("--- Running in TOKEN PRUNING mode ---")
else:
    print("--- Running in BASELINE (no pruning) mode ---")

# --- KEPT: Main PGD / Verification logic from original script ---
run_pgd = args.pgd
if run_pgd:
    args.num_pgd_starts = 10
    args.pgd_iterations = 50
    args.max_eps = 2.0
    # NOTE: The original script's sample_correct_samples call is kept inside the verifier's run method
    examples = sample_correct_samples(args, data_normalized, model)
    pgd_attack(model, args, examples, normalizer)
else:
    # IMPROVEMENT: We now use the --samples argument from the command line
    # instead of hardcoding it to 100.
    verifier = VerifierZonotopeViT(args, model, logger, num_classes=10, normalizer=normalizer)
    verifier.run(data_normalized)
