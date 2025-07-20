import torch
import sys
from model import JointModel
from mnist import mnist_test_dataloader

def diagnose_difference():
    """Compare exact setup between working and non-working tests"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Test 1: Exact reproduction of train-test.py
    print("TEST 1: Exact reproduction of train-test.py")
    print("-" * 60)
    
    model1 = JointModel(
        k=10,
        pruning_layer=0,
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
    
    print("Loading weights...")
    model1.load_from_original_vit("mnist_transformer.pt")
    model1.eval()
    
    # Test on first 100 samples
    test_loader = mnist_test_dataloader(batch_size=128, shuffle=False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model1._unpruned_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= 100:
                break
    
    print(f"Model1 (k=10) unpruned accuracy on 100 samples: {100 * correct / total:.2f}%")
    
    # Test 2: Exact reproduction of test_pruning.py
    print("\n" + "=" * 60)
    print("TEST 2: Exact reproduction of test_pruning.py")
    print("-" * 60)
    
    model2 = JointModel(
        k=2,  # Note: using k=2 here
        pruning_layer=0,
        pool='mean',
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
    
    print("Loading weights...")
    model2.load_from_original_vit("mnist_transformer.pt")
    model2.eval()
    
    # Test pruned forward
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model2._pruned_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= 100:
                break
    
    print(f"Model2 (k=2) pruned accuracy on 100 samples: {100 * correct / total:.2f}%")
    
    # Test 3: Check if pool parameter matters
    print("\n" + "=" * 60)
    print("TEST 3: Check pooling parameter")
    print("-" * 60)
    
    # Without explicit pool parameter (like train-test.py)
    model3a = JointModel(
        k=10,
        pruning_layer=0,
        img_size=28,
        patch_size=7,
        num_classes=10,
        in_chans=1,
        embed_dim=64,
        depth=1,
        heads=4,
        mlp_dim=128,
        layer_norm_type='no_var'
        # NO pool parameter - uses default
    ).to(device)
    
    print(f"Model3a pool setting: {model3a.pool}")
    
    # With explicit pool='mean' (like test_pruning.py)
    model3b = JointModel(
        k=10,
        pruning_layer=0,
        pool='mean',  # Explicit pool parameter
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
    
    print(f"Model3b pool setting: {model3b.pool}")
    
    # Test 4: Check model state after loading
    print("\n" + "=" * 60)
    print("TEST 4: Model state comparison")
    print("-" * 60)
    
    # Get a single test image
    test_image, test_label = next(iter(mnist_test_dataloader(batch_size=1, shuffle=False)))
    test_image = test_image.to(device)
    
    # Compare outputs from all models
    with torch.no_grad():
        out1 = model1._unpruned_forward(test_image)
        out2_unpruned = model2._unpruned_forward(test_image)
        out3a = model3a._unpruned_forward(test_image)
        out3b = model3b._unpruned_forward(test_image)
        
        print("Single image test:")
        print(f"Model1 (k=10) prediction: {out1.argmax().item()}")
        print(f"Model2 (k=2) unpruned prediction: {out2_unpruned.argmax().item()}")
        print(f"Model3a (no pool param) prediction: {out3a.argmax().item()}")
        print(f"Model3b (pool='mean') prediction: {out3b.argmax().item()}")
        
        print("\nLogits comparison:")
        print(f"Model1 vs Model2: max diff = {(out1 - out2_unpruned).abs().max().item():.6f}")
        print(f"Model1 vs Model3a: max diff = {(out1 - out3a).abs().max().item():.6f}")
        print(f"Model1 vs Model3b: max diff = {(out1 - out3b).abs().max().item():.6f}")
    
    # Test 5: Check if model weights are identical
    print("\n" + "=" * 60)
    print("TEST 5: Weight comparison")
    print("-" * 60)
    
    # Compare classification head weights
    w1 = model1.classification_head.weight
    w2 = model2.classification_head.weight
    
    print(f"Classification head weights equal: {torch.allclose(w1, w2)}")
    if not torch.allclose(w1, w2):
        print(f"Max weight difference: {(w1 - w2).abs().max().item():.6f}")
    
    # Check specific weight statistics
    print(f"\nModel1 classification head stats:")
    print(f"  Mean: {w1.mean().item():.6f}, Std: {w1.std().item():.6f}")
    print(f"  Class 7 weights mean: {w1[7].mean().item():.6f}")
    
    print(f"\nModel2 classification head stats:")
    print(f"  Mean: {w2.mean().item():.6f}, Std: {w2.std().item():.6f}")
    print(f"  Class 7 weights mean: {w2[7].mean().item():.6f}")

if __name__ == "__main__":
    diagnose_difference()
