import os
import sys
import psutil
import torch
from argparse import Namespace 


from Parser import Parser, update_arguments
from Verifiers.FirstKVerifier import FirstKVerifier
from model import JointModel
from fake_logger import FakeLogger 
from mnist import mnist_test_dataloader, normalizer 
from data_utils import set_seeds

def main():
    argv = sys.argv[1:]
    parser = Parser.get_parser()

    args, _ = parser.parse_known_args(argv)
    args = update_arguments(args) 

    args.with_lirpa_transformer = False 

    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if psutil.cpu_count() > 4 and args.cpu_range != "Default":
        start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
        os.sched_setaffinity(0, {i for i in range(start, end + 1)})

    set_seeds(args.seed)

    test_data_loader = mnist_test_dataloader(batch_size=1, shuffle=False) 

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu !=-1 else "cpu")
    args.device = device 
    model = JointModel(
        k=args.k,
        pruning_layer=args.pruning_layer,
        img_size=28, 
        patch_size=7, 
        num_classes=args.num_labels, 
        in_chans=1, 
        embed_dim=args.hidden_size, 
        depth=args.num_layers, 
        heads=args.num_attention_heads, 
        mlp_dim=args.intermediate_size, 
        layer_norm_type=args.layer_norm, 
        dropout=args.dropout if hasattr(args, 'dropout') else 0.0,
        pool = 'mean'
    ).to(device)

    model.load_from_original_vit("mnist_transformer.pt")

    model.eval()

    print("Arguments:", args)
    print(f"Device: {device}")
    print(f"Test Dataset size: {len(test_data_loader.dataset)}") 

    logger = FakeLogger() 

    print("\nPreparing data...")
    data_normalized = []
    num_samples_to_prepare = getattr(args, 'samples', 100) 
    for i, (x, y) in enumerate(test_data_loader):

        data_normalized.append({
            "label": y.to(device), 
            "image": x.to(device)
        })
        if i + 1 >= num_samples_to_prepare:
            break
    print(f"Prepared {len(data_normalized)} samples.")

    verifier = FirstKVerifier(
        args=args,
        target=model,
        logger=logger,
        num_classes=model.num_classes, 
        normalizer=normalizer, 
        output_epsilon=args.output_epsilon 
    )

    print("\nStarting verification...")
    verifier.run(data_normalized)
    print("\nVerification finished.")

if __name__ == "__main__":
    main()
