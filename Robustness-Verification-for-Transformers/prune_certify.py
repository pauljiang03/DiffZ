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
        pool = 'mean'
    ).to(device)

    print(f"Model pool setting: {model.pool}")
    print(f"Model pooling method in _apply_pooling_and_head:")
    import inspect
    print(inspect.getsource(model._apply_pooling_and_head))

    model.load_from_original_vit("mnist_transformer.pt")

    model.eval()
    print(f"DEBUG: Model parameters from args:")
    print(f"  k={args.k}")
    print(f"  pruning_layer={args.pruning_layer}")
    print(f"  depth={args.num_layers}")
    print(f"  hidden_size={args.hidden_size}")
    print(f"  num_attention_heads={args.num_attention_heads}")
    print(f"  intermediate_size={args.intermediate_size}")
    print("\n=== REPRODUCING STANDALONE TEST IN VERIFICATION ===")

    # Test 1: Reproduce your exact standalone setup
    print("1. Testing with EXACT standalone parameters:")
    standalone_model = JointModel(
        k=2,  # Your standalone uses k=2
        pruning_layer=0,
        pool='mean',  # Explicit mean pooling
        img_size=28,
        patch_size=7,
        num_classes=10,
        in_chans=1,
        embed_dim=64,
        depth=1,
        heads=4,
        mlp_dim=128,
        layer_norm_type='no_var'
    ).to(device)
    
    standalone_model.load_from_original_vit("mnist_transformer.pt")
    standalone_model.eval()
    
    # Test on a batch like your standalone code
    test_loader = mnist_test_dataloader(batch_size=10, shuffle=False)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    print(f"Testing batch shape: {images.shape}")
    
    with torch.no_grad():
        # Test exactly like your standalone
        unpruned_outputs = standalone_model._unpruned_forward(images)
        pruned_outputs = standalone_model._pruned_forward(images)
        
        unpruned_preds = torch.max(unpruned_outputs.data, 1)[1]
        pruned_preds = torch.max(pruned_outputs.data, 1)[1]
        
        print(f"Unpruned predictions: {unpruned_preds}")
        print(f"Pruned predictions:   {pruned_preds}")
        print(f"Predictions match: {torch.equal(unpruned_preds, pruned_preds)}")
        
        # Check logit differences
        logit_diffs = (unpruned_outputs - pruned_outputs).abs().max(dim=1)[0]
        print(f"Max logit differences per sample: {logit_diffs}")
        print(f"Overall max logit difference: {logit_diffs.max().item():.8f}")
    
    # Test 2: Compare single sample processing
    print("\n2. Testing single sample (like verification):")
    single_image = images[0:1]  # Take first sample
    single_label = labels[0:1]
    
    with torch.no_grad():
        single_unpruned = standalone_model._unpruned_forward(single_image)
        single_pruned = standalone_model._pruned_forward(single_image)
        
        print(f"Single sample unpruned: {single_unpruned}")
        print(f"Single sample pruned:   {single_pruned}")
        print(f"Single sample difference: {(single_unpruned - single_pruned).abs().max().item():.8f}")
    
    # Test 3: Step by step debugging of a single forward pass
    print("\n3. Step-by-step debugging:")
    with torch.no_grad():
        x = single_image
        print(f"Input shape: {x.shape}")
        
        # Common preprocessing
        x_prep = standalone_model._common_preprocessing(x)
        print(f"After preprocessing: {x_prep.shape}, mean: {x_prep.mean().item():.6f}")
        
        # Unpruned path
        x_unpruned = x_prep.clone()
        for i, block in enumerate(standalone_model.unpruned_blocks):
            x_unpruned = block(x_unpruned)
            print(f"Unpruned after block {i}: shape={x_unpruned.shape}, mean={x_unpruned.mean().item():.6f}")
        
        # Pruned path  
        x_pruned = x_prep.clone()
        for i, block in enumerate(standalone_model.unpruned_blocks):
            x_pruned = block(x_pruned)
            print(f"Pruned after block {i}: shape={x_pruned.shape}, mean={x_pruned.mean().item():.6f}")
            if i == standalone_model.pruning_layer:
                print(f"  BEFORE pruning: shape={x_pruned.shape}")
                x_pruned = FirstKPrune(x_pruned, standalone_model.k)
                print(f"  AFTER pruning: shape={x_pruned.shape}")
        
        # Apply pooling and head
        print(f"\nBefore pooling - Unpruned: shape={x_unpruned.shape}, mean={x_unpruned.mean().item():.6f}")
        print(f"Before pooling - Pruned: shape={x_pruned.shape}, mean={x_pruned.mean().item():.6f}")
        
        final_unpruned = standalone_model._apply_pooling_and_head(x_unpruned)
        final_pruned = standalone_model._apply_pooling_and_head(x_pruned)
        
        print(f"Final unpruned: {final_unpruned}")
        print(f"Final pruned:   {final_pruned}")
        print(f"Final difference: {(final_unpruned - final_pruned).abs().max().item():.8f}")
    
    # Test 4: Check if the issue is in _initialize_with_same_weights
    print("\n4. Checking if block weights are truly identical:")
    for i, (unpruned_block, pruned_block) in enumerate(zip(standalone_model.unpruned_blocks, standalone_model.pruned_blocks)):
        weights_identical = True
        for name, param1 in unpruned_block.named_parameters():
            param2 = dict(pruned_block.named_parameters())[name]
            if not torch.equal(param1, param2):
                weights_identical = False
                print(f"  Block {i}, {name}: weights differ by {(param1-param2).abs().max().item():.8f}")
        if weights_identical:
            print(f"  Block {i}: All weights identical ✓")
        else:
            print(f"  Block {i}: Some weights differ ✗")

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
