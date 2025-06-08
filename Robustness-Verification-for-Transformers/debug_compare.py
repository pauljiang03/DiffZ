import torch
from model import JointModel  # Your JointModel class
from vit import ViT            # The original ViT class
from mnist import mnist_test_dataloader

def main():
    """
    This script performs a direct comparison of the concrete forward pass
    of the original ViT model and your JointModel to isolate any differences.
    """
    # 1. Load the exact same single data point
    # IMPORTANT: Ensure your mnist_test_dataloader is set to shuffle=False
    print("--- Loading Data ---")
    try:
        dataloader = mnist_test_dataloader(batch_size=1, shuffle=False)
        image, label = next(iter(dataloader))
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Failed to load data. Make sure mnist.py is accessible and correct. Error: {e}")
        return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image = image.to(device)

    # 2. Initialize and load the reference ViT model
    print("\n--- Loading Reference ViT Model ---")
    try:
        vit_model = ViT(
            image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var"
        ).to(device)
        vit_model.load_state_dict(torch.load("mnist_transformer.pt"))
        vit_model.eval()
        print("Reference ViT model loaded.")
    except Exception as e:
        print(f"Failed to load reference ViT. Make sure vit.py is accessible and correct. Error: {e}")
        return

    # 3. Initialize and load your JointModel
    print("\n--- Loading Your JointModel ---")
    try:
        joint_model = JointModel(
            depth=1, num_classes=10, embed_dim=64, mlp_dim=128,
            heads=4, dim_head=64, layer_norm_type='no_var', pool='cls',
            k=15, pruning_layer=0 # Add any other necessary default args
        ).to(device)
        joint_model.load_from_original_vit("mnist_transformer.pt")
        joint_model.eval()
        print("Your JointModel loaded.")
    except Exception as e:
        print(f"Failed to load your JointModel. Make sure model.py is accessible and correct. Error: {e}")
        return

    # 4. Perform a concrete forward pass and compare the results
    print("\n" + "="*20 + " COMPARING CONCRETE FORWARD PASS " + "="*20)
    with torch.no_grad():
        vit_output = vit_model(image)
        # We use the internal _unpruned_forward for a direct comparison
        joint_model_output = joint_model._unpruned_forward(image)

    print("\nReference ViT Output Logits:")
    print(vit_output)

    print("\nYour JointModel (Unpruned) Output Logits:")
    print(joint_model_output)

    # 5. Check for equality with a small tolerance for floating point differences
    are_equal = torch.allclose(vit_output, joint_model_output, atol=1e-6)
    
    print("\n" + "="*58)
    print(f"Are the outputs functionally identical? {are_equal}")
    print("="*58 + "\n")

    if not are_equal:
        print("!!! The models are NOT functionally identical. !!!")
        print("This confirms a subtle difference in the model class definitions or weight loading logic.")
        diff = torch.abs(vit_output - joint_model_output)
        print("Max absolute difference between logits:", diff.max().item())

if __name__ == "__main__":
    main()
