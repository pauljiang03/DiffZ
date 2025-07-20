import torch
import numpy as np
from model import JointModel, FirstKPrune
from mnist import mnist_test_dataloader

def comprehensive_model_test():
    """Comprehensive test to verify pruning behavior"""
    
    # Configuration
    K_VALUES = [2, 4, 8, 16]  # Test multiple k values
    PRUNING_LAYER = 0
    POOL = 'mean'
    DEPTH = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    model = JointModel(
        k=10, # This will be changed dynamically
        pruning_layer=PRUNING_LAYER,
        pool=POOL,
        img_size=28,
        patch_size=7,
        num_classes=10,
        in_chans=1,
        embed_dim=64,
        depth=DEPTH,
        heads=4,
        mlp_dim=128,
        layer_norm_type='no_var'
    ).to(device)
    
    model.load_from_original_vit("mnist_transformer.pt")
    model.eval()
    
    # Test 1: Verify pruning actually happens
    print("=" * 60)
    print("TEST 1: Verify pruning mechanics")
    print("=" * 60)
    
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    test_image, test_label = next(iter(test_loader))
    test_image = test_image.to(device)
    
    with torch.no_grad():
        # Test pruning at different stages
        x = model._common_preprocessing(test_image)
        print(f"After preprocessing: {x.shape}")
        
        # Manually apply blocks and pruning
        for k_test in [1, 2, 4, 8]:
            x_test = x.clone()
            for i, block in enumerate(model.unpruned_blocks):
                x_test = block(x_test)
                if i == PRUNING_LAYER:
                    print(f"\nK={k_test}:")
                    print(f"  Before pruning: {x_test.shape}")
                    x_test = FirstKPrune(x_test, k_test)
                    print(f"  After pruning: {x_test.shape}")
    
    # Test 2: Compare logits with different k values
    print("\n" + "=" * 60)
    print("TEST 2: Logit changes with different k values")
    print("=" * 60)
    
    logits_by_k = {}
    for k in K_VALUES:
        model.k = k  # Change k dynamically
        with torch.no_grad():
            logits = model._pruned_forward(test_image)
            logits_by_k[k] = logits.cpu().numpy()
            pred = logits.argmax(dim=1).item()
            print(f"k={k}: prediction={pred}, max_logit={logits.max().item():.3f}")
    
    # Test 3: Full accuracy evaluation
    print("\n" + "=" * 60)
    print("TEST 3: Full test set accuracy")
    print("=" * 60)
    
    # Test on full dataset
    test_loader_full = mnist_test_dataloader(batch_size=128, shuffle=False)
    
    results = {}
    for k in K_VALUES:
        model.k = k
        correct_unpruned = 0
        correct_pruned = 0
        total = 0
        
        predictions_unpruned = []
        predictions_pruned = []
        
        with torch.no_grad():
            for images, labels in test_loader_full:
                images, labels = images.to(device), labels.to(device)
                
                # Unpruned
                outputs_unpruned = model._unpruned_forward(images)
                _, predicted_unpruned = torch.max(outputs_unpruned, 1)
                correct_unpruned += (predicted_unpruned == labels).sum().item()
                predictions_unpruned.extend(predicted_unpruned.cpu().numpy())
                
                # Pruned
                outputs_pruned = model._pruned_forward(images)
                _, predicted_pruned = torch.max(outputs_pruned, 1)
                correct_pruned += (predicted_pruned == labels).sum().item()
                predictions_pruned.extend(predicted_pruned.cpu().numpy())
                
                total += labels.size(0)
                
                # Stop after 1000 samples for quick test
                if total >= 1000:
                    break
        
        acc_unpruned = 100 * correct_unpruned / total
        acc_pruned = 100 * correct_pruned / total
        
        results[k] = {
            'acc_unpruned': acc_unpruned,
            'acc_pruned': acc_pruned,
            'pred_dist_unpruned': np.bincount(predictions_unpruned, minlength=10),
            'pred_dist_pruned': np.bincount(predictions_pruned, minlength=10)
        }
        
        print(f"\nk={k} (keeping {k} patches + CLS token):")
        print(f"  Unpruned accuracy: {acc_unpruned:.2f}%")
        print(f"  Pruned accuracy: {acc_pruned:.2f}%")
        print(f"  Accuracy drop: {acc_unpruned - acc_pruned:.2f}%")
    
    # Test 4: Analyze prediction distributions
    print("\n" + "=" * 60)
    print("TEST 4: Prediction distribution analysis")
    print("=" * 60)
    
    for k, res in results.items():
        print(f"\nk={k}:")
        print("  Unpruned predictions per class:", res['pred_dist_unpruned'])
        print("  Pruned predictions per class:  ", res['pred_dist_pruned'])
        
        # Check if model is biased toward certain classes
        unpruned_entropy = -np.sum((res['pred_dist_unpruned'] / res['pred_dist_unpruned'].sum()) * 
                                   np.log(res['pred_dist_unpruned'] / res['pred_dist_unpruned'].sum() + 1e-10))
        pruned_entropy = -np.sum((res['pred_dist_pruned'] / res['pred_dist_pruned'].sum()) * 
                                 np.log(res['pred_dist_pruned'] / res['pred_dist_pruned'].sum() + 1e-10))
        
        print(f"  Unpruned entropy: {unpruned_entropy:.3f}")
        print(f"  Pruned entropy: {pruned_entropy:.3f}")
    
    # Test 5: Verify forward pass implementations
    print("\n" + "=" * 60)
    print("TEST 5: Verify forward pass consistency")
    print("=" * 60)
    
    model.k = 2  # Reset to k=2
    test_batch, _ = next(iter(mnist_test_dataloader(batch_size=5, shuffle=False)))
    test_batch = test_batch.to(device)
    
    with torch.no_grad():
        # Method 1: Using internal methods
        unpruned1 = model._unpruned_forward(test_batch)
        pruned1 = model._pruned_forward(test_batch)
        
        # Method 2: Using main forward (should match)
        result = model(test_batch, return_full_info=True)
        unpruned2 = result['unpruned_logits']
        pruned2 = result['pruned_logits']
        
        print("Unpruned logits match:", torch.allclose(unpruned1, unpruned2, rtol=1e-5))
        print("Pruned logits match:", torch.allclose(pruned1, pruned2, rtol=1e-5))
        
        if not torch.allclose(unpruned1, unpruned2, rtol=1e-5):
            print("WARNING: Unpruned forward methods don't match!")
            print(f"Max difference: {(unpruned1 - unpruned2).abs().max().item()}")
        
        if not torch.allclose(pruned1, pruned2, rtol=1e-5):
            print("WARNING: Pruned forward methods don't match!")
            print(f"Max difference: {(pruned1 - pruned2).abs().max().item()}")

if __name__ == "__main__":
    comprehensive_model_test()
