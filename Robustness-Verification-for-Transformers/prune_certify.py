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
    print("\n=== DETAILED PRUNING DEBUG ===")
    test_input = torch.randn(1, 1, 28, 28).to(device)
    
    with torch.no_grad():
        print(f"Model config: depth={model.depth}, pruning_layer={model.pruning_layer}, k={model.k}")
        
        # Step-by-step debugging
        original_k = model.k
        model.k = 1  # Very aggressive pruning
        
        # Trace through preprocessing
        x_prep = model._common_preprocessing(test_input)
        print(f"After preprocessing: {x_prep.shape}")
        
        # Trace through unpruned forward
        x_unpruned = x_prep.clone()
        for i, block in enumerate(model.unpruned_blocks):
            print(f"  Unpruned - Before block {i}: {x_unpruned.shape}")
            x_unpruned = block(x_unpruned)
            print(f"  Unpruned - After block {i}: {x_unpruned.shape}")
        
        # Trace through pruned forward
        x_pruned = x_prep.clone()
        for i, block in enumerate(model.unpruned_blocks):
            print(f"  Pruned - Before block {i}: {x_pruned.shape}")
            x_pruned = block(x_pruned)
            print(f"  Pruned - After block {i}: {x_pruned.shape}")
            
            # Check if pruning should happen here
            if i == model.pruning_layer:
                print(f"  *** APPLYING PRUNING at layer {i} with k={model.k} ***")
                x_before_prune = x_pruned.clone()
                x_pruned = FirstKPrune(x_pruned, model.k)
                print(f"  Before pruning: {x_before_prune.shape}")
                print(f"  After pruning: {x_pruned.shape}")
                
                # Check if content actually changed
                if x_before_prune.shape == x_pruned.shape:
                    content_changed = not torch.allclose(x_before_prune, x_pruned)
                    print(f"  Content changed: {content_changed}")
                else:
                    print(f"  Shape changed - pruning worked!")
        
        # Final outputs
        unpruned_final = model._apply_pooling_and_head(x_unpruned)
        pruned_final = model._apply_pooling_and_head(x_pruned)
        
        print(f"Final unpruned logits: {unpruned_final}")
        print(f"Final pruned logits: {pruned_final}")
        print(f"Max difference: {(unpruned_final - pruned_final).abs().max().item():.8f}")
        
        model.k = original_k  # Restore

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
