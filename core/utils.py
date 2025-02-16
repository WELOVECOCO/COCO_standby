"""
This module provides a collection of utility functions that is needed for future phases of COCO.

It includes:
- NN Layers Grad Check.
- More to be implemented functions in the future.
"""
import numpy as np
from core import nn
from core import optim
import matplotlib.pyplot as plt
def visualize_feature_maps(model, image):
        """
        Visualize feature maps for convolutional layers.
        """
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]
        elif image.ndim == 4 and image.shape[0] != 1:
            raise ValueError("Only single image batches are supported.")

        featuremaps = []
        model.forward(image, test=True, visualize=True)
        cnn_layers = [layer for layer in model.layers.values() if isinstance(layer, nn.Conv2D)]
        for layer in cnn_layers:
            featuremaps.append(layer.output.data)
        num_conv_layers = len(model.featuremaps)
        if num_conv_layers == 0:
            print("No convolutional layers found.")
            return

        # Create a separate figure for each convolutional layer
        for layer_idx, fm in enumerate(model.featuremaps):
            fm = fm[0]  # Remove batch dimension -> (C, H, W)
            num_channels = fm.shape[0]

            # Create a new figure for this layer
            plt.figure(figsize=(16, 8))
            plt.suptitle(f"Layer {layer_idx+1} Feature Maps", fontsize=14, y=1.02)

            # Calculate grid dimensions
            cols = 8  # Max 8 filters per row
            rows = int(np.ceil(num_channels / cols))

            # Plot each channel
            for channel_idx in range(num_channels):
                plt.subplot(rows, cols, channel_idx + 1)
                channel_data = fm[channel_idx]

                # Normalize to [0, 1] for better contrast
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)

                plt.imshow(channel_data, cmap='gray')
                plt.axis('off')
                plt.title(f'Ch{channel_idx+1}', fontsize=8)

            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)  # Increase spacing
            plt.show()  # Show layer-specific figure (will create multiple windows)


def one_hot_encode(labels, num_classes=None):
    """
    Convert integer labels to one-hot encoded vectors.
    """
    if labels.ndim == 2:
        # Check if all rows have exactly one `1` and the rest `0`
        is_one_hot = np.all(np.isin(labels, [0, 1])) and np.all(labels.sum(axis=1) == 1)
        if num_classes is not None:
            is_one_hot = is_one_hot and (labels.shape[1] == num_classes)
        if is_one_hot:
            return labels.astype(int)  # Ensure integer type

    # Proceed to encode if not one-hot
    if num_classes is None:
        num_classes = np.max(labels) + 1  # Infer from integer labels
    return np.eye(num_classes, dtype=int)[labels]
def gradient_check(model, X, Y, loss_fn, epsilon=1e-7, threshold=1e-5):
    """
    Performs gradient checking for a given model.
    """
    # Forward pass to compute initial loss
    model(X)
    loss, grad_output = loss_fn(Y, model.last_out, axis=1)

    # Backward pass to get analytical gradients (without optimizer step)
    error_grad = grad_output
    for _, layer in reversed(list(model.layers.items())):
        error_grad = layer.backward(error_grad)

    for _,layer in list(model.layers.items()):
        if not hasattr(layer, 'weights') or layer.weights is None:
            continue

        W, grad_W = layer.weights, layer.grad_w

        # ------ Numerical Gradient for Weights (element-wise) ------
        numerical_grad_W = np.zeros_like(W)
        original_weights = W.copy()

        for idx in np.ndindex(W.shape):
            # Perturb weight at idx by +epsilon
            W[idx] += epsilon
            model(X)
            loss_plus = loss_fn(Y, model.last_out, axis=1)[0]

            # Perturb weight at idx by -epsilon
            W[idx] = original_weights[idx] - epsilon
            model(X)
            loss_minus = loss_fn(Y, model.last_out, axis=1)[0]

            # Compute numerical gradient for this weight
            numerical_grad_W[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            # Restore original weight
            W[idx] = original_weights[idx]

        # Compare analytical and numerical gradients
        diff_W = np.linalg.norm(grad_W - numerical_grad_W) / (np.linalg.norm(grad_W) + np.linalg.norm(numerical_grad_W) + 1e-15)
        print(f"Layer {layer.__class__.__name__} (Weights): Gradient Check {'PASSED' if diff_W < threshold else 'FAILED'} (diff: {diff_W:.8e})")

        # ------ Numerical Gradient for Biases (if present) ------
        if hasattr(layer, 'bias') and layer.bias is not None:
            b, grad_b = layer.bias, layer.grad_b
            numerical_grad_b = np.zeros_like(b)
            original_bias = b.copy()

            for idx in np.ndindex(b.shape):
                b[idx] += epsilon
                model(X)
                loss_plus = loss_fn(Y, model.last_out, axis=1)[0]

                b[idx] = original_bias[idx] - epsilon
                model(X)
                loss_minus = loss_fn(Y, model.last_out, axis=1)[0]

                numerical_grad_b[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                b[idx] = original_bias[idx]

            diff_b = np.linalg.norm(grad_b - numerical_grad_b) / (np.linalg.norm(grad_b) + np.linalg.norm(numerical_grad_b) + 1e-15)
            print(f"Layer {layer.__class__.__name__} (Biases): Gradient Check {'PASSED' if diff_b < threshold else 'FAILED'} (diff: {diff_b:.8e})\n")

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model