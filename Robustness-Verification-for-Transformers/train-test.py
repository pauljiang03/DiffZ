import torch
from model import JointModel  # Your model definition
from mnist import mnist_test_dataloader
def evaluate_model():
    """Evaluates the base model's accuracy on the MNIST test set."""
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load your model structure
    model = JointModel(
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

    # Load the trained weights
    print("Loading weights from mnist_transformer.pt...")
    model.load_from_original_vit("mnist_transformer.pt")
    model.eval()

    # --- Evaluation ---
    # Use a batch size of 1 because the model is not batch-compatible
    test_loader = mnist_test_dataloader(batch_size=1, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # The dimension is now 1 because the batch dimension was squeezed
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n--- Results ---")
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Model Accuracy: {accuracy:.2f}%")
if __name__ == "__main__":
    evaluate_model()
