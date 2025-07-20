import torch
from model import JointModel
from mnist import mnist_test_dataloader

def evaluate_pruned_model_accuracy():
    """Evaluates the pruned model's accuracy on the MNIST test set."""

    # --- CONFIGURE YOUR PRUNING STRATEGY HERE ---
    # These parameters MUST match the pruning you want to test.
    K_TOKENS_TO_KEEP = 2
    PRUNING_LAYER_INDEX = 0
    POOLING_METHOD = 'mean'  # 'mean' or 'cls'
    MODEL_DEPTH = 1 # The --num_layers you used
    # --------------------------------------------

    print("--- Evaluating PRUNED model ---")
    print(f"Tokens to keep (k): {K_TOKENS_TO_KEEP}")
    print(f"Pruning Layer: {PRUNING_LAYER_INDEX}")
    print(f"Pooling Method: {POOLING_METHOD}\n")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your model structure with the correct parameters
    model = JointModel(
        k=K_TOKENS_TO_KEEP,
        pruning_layer=PRUNING_LAYER_INDEX,
        pool=POOLING_METHOD,
        img_size=28,
        patch_size=7,
        num_classes=10,
        in_chans=1,
        embed_dim=64,
        depth=MODEL_DEPTH,
        heads=4,
        mlp_dim=128,
        layer_norm_type='no_var'
    ).to(device)

    # Load the trained weights
    print("Loading weights from mnist_transformer.pt...")
    model.load_from_original_vit("mnist_transformer.pt")
    model.eval()

    # --- Evaluation ---
    test_loader = mnist_test_dataloader(batch_size=128, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Get model predictions by calling the PRUNED forward pass
            outputs = model._pruned_forward(images)

            # Find the class with the highest logit
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n--- Results ---")
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Pruned Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_pruned_model_accuracy()
