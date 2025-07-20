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
from model import JointModel, FirstKPrune

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
        #pool = 'mean'
    ).to(device)

    model.load_from_original_vit("mnist_transformer.pt")

    model.eval()
    print("\n=== SIMPLE PRUNING TEST ===")
    test_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Test 1: FirstKPrune function directly
    print("Testing FirstKPrune function:")
    x = torch.randn(1, 17, 64)
    print(f"Before: {x.shape}")
    x_pruned = FirstKPrune(x, 1)
    print(f"After k=1: {x_pruned.shape}")
    
    # Test 2: Model with different k values
    print("\nTesting model with different k values:")
    with torch.no_grad():
        original_k = model.k
        
        # k=16 (should be similar to unpruned)
        model.k = 16
        logits_k16 = model._pruned_forward(test_input)
        
        # k=1 (should be very different)
        model.k = 1
        logits_k1 = model._pruned_forward(test_input)
        
        # k=0 (most aggressive)
        model.k = 0
        logits_k0 = model._pruned_forward(test_input)
        
        # Unpruned
        logits_unpruned = model._unpruned_forward(test_input)
        
        model.k = original_k  # Restore
        
        print(f"Unpruned vs k=16 diff: {(logits_unpruned - logits_k16).abs().max().item():.8f}")
        print(f"Unpruned vs k=1 diff:  {(logits_unpruned - logits_k1).abs().max().item():.8f}")
        print(f"Unpruned vs k=0 diff:  {(logits_unpruned - logits_k0).abs().max().item():.8f}")

    #print("Arguments:", args)
    #print(f"Device: {device}")
    #print(f"Test Dataset size: {len(test_data_loader.dataset)}") 

    logger = FakeLogger() 

    #print("\nPreparing data...")
    '''
    data_normalized = []
    num_samples_to_prepare = getattr(args, 'samples', 100) 
    for i, (x, y) in enumerate(test_data_loader):

        data_normalized.append({
            "label": y.to(device), 
            "image": x.to(device)
        })
        if i + 1 >= num_samples_to_prepare:
            break
    '''
    data_normalized = []
    
    # --- REPLACE YOUR OLD LOOP WITH THIS LOGIC ---
    test_dataset = test_data_loader.dataset
    if args.sample_index != -1:
        print(f"Loading specific sample index: {args.sample_index}")
        x, y = test_dataset[args.sample_index]
        data_normalized.append({
            "label": torch.tensor([y]).to(device),
            "image": x.unsqueeze(0).to(device)
        })
    else:
        # This is the original behavior, in case you don't provide an index
        print("Loading first 'n' samples from the dataset...")
        # The default number of samples is 10
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
