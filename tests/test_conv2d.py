
from core.nn import Conv2d
from core.tensor import Tensor  
import numpy as np
import torch
import torch.nn.functional as F

# Example tensor shape
B, C_in, H, W = 2, 3, 12, 12
C_out = 4
tolerance = 1e-4  # acceptable difference threshold

# Random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Test for various kernel sizes
for kernel_size in (3, 5, 7):
    print(f"Testing kernel size {kernel_size}...")

    # Use padding to maintain "same" size
    padding =0

    # Create input
    x_np = np.random.randn(B, C_in, H, W).astype(np.float32)
    x_torch = torch.tensor(x_np, dtype=torch.float32)

    # Create weights and bias
    weight_np = np.random.randn(C_out, C_in, kernel_size, kernel_size).astype(np.float32)
    bias_np = np.random.randn(C_out).astype(np.float32)

    # ----- PyTorch -----
    conv_torch = torch.nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=True)
    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(weight_np))
        conv_torch.bias.copy_(torch.tensor(bias_np))
    out_torch = conv_torch(x_torch)

    # ----- CoCo (your implementation) -----
    conv_coco = Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=True)
    conv_coco.weights.data = weight_np.copy()
    conv_coco.bias.data = bias_np.copy()

    x_tensor = Tensor(x_np, requires_grad=True)
    out_coco = conv_coco(x_tensor).data

    # Compare forward
    out_torch_np = out_torch.detach().numpy()
    assert out_coco.shape == out_torch_np.shape, f"Shape mismatch: {out_coco.shape} vs {out_torch_np.shape}"

    diff_fwd = np.max(np.abs(out_coco - out_torch_np))
    print(f"  Forward max abs diff: {diff_fwd:.2e}")
    assert diff_fwd < tolerance, f"Forward mismatch for kernel {kernel_size}"

print("all test cases passed ✅ ✅ ✅ ")