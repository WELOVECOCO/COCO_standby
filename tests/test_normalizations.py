import torch
import torch.nn as nn
import numpy as np
from core.new_tensor import Tensor
from core.nn import ConvBatchNorm2D, InstanceNorm2D, LayerNorm2D, GroupNorm, rmsnorm

# Disable test mode to use training-time statistics
import core.config as Config
Config.TEST = False

np.random.seed(42)
torch.manual_seed(42)

THRESHOLD = 1e-5  # Tolerance threshold


def compare_outputs(name, out_custom, out_torch):
    diff = np.abs(out_custom - out_torch).max()
    if diff < THRESHOLD:
        print(f"[PASS] {name} Output diff: {diff:.2e}")
    else:
        print(f"[FAIL] {name} Output diff: {diff:.2e}")


def compare_gradients(name, grad_custom, grad_torch):
    diff = np.abs(grad_custom - grad_torch).max()
    if diff < THRESHOLD:
        print(f"[PASS] {name} Grad diff: {diff:.2e}")
    else:
        print(f"[FAIL] {name} Grad diff: {diff:.2e}")


def test_batchnorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    bn = nn.BatchNorm2d(8, affine=True, track_running_stats=False)
    bn.weight.data.fill_(1.0)
    bn.bias.data.fill_(0.0)
    out_torch = bn(x_torch)
    loss_torch = out_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    custom_x = Tensor(x_np, requires_grad=True)
    custom_bn = ConvBatchNorm2D(8)
    out_custom = custom_bn(custom_x)
    loss_custom = out_custom.mean()
    loss_custom.backward()
    grad_custom = custom_x.grad

    compare_outputs("BatchNorm2D", out_custom.data, out_torch.detach().numpy())
    compare_gradients("BatchNorm2D", grad_custom, grad_torch)


def test_instancenorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    inst = nn.InstanceNorm2d(8, affine=True, track_running_stats=False)
    inst.weight.data.fill_(1.0)
    inst.bias.data.fill_(0.0)
    out_torch = inst(x_torch)
    loss_torch = out_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    custom_x = Tensor(x_np, requires_grad=True)
    custom_inst = InstanceNorm2D(8)
    out_custom = custom_inst(custom_x)
    loss_custom = out_custom.mean()
    loss_custom.backward()
    grad_custom = custom_x.grad

    compare_outputs("InstanceNorm2D", out_custom.data, out_torch.detach().numpy())
    compare_gradients("InstanceNorm2D", grad_custom, grad_torch)


def test_layernorm2d():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    ln = nn.LayerNorm([8, 16, 16], elementwise_affine=True)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)
    out_torch = ln(x_torch)
    loss_torch = out_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    custom_x = Tensor(x_np, requires_grad=True)
    custom_ln = LayerNorm2D(8)
    out_custom = custom_ln(custom_x)
    loss_custom = out_custom.mean()
    loss_custom.backward()
    grad_custom = custom_x.grad

    compare_outputs("LayerNorm2D", out_custom.data, out_torch.detach().numpy())
    compare_gradients("LayerNorm2D", grad_custom, grad_torch)


def test_groupnorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    gn = nn.GroupNorm(4, 8, affine=True)
    gn.weight.data.fill_(1.0)
    gn.bias.data.fill_(0.0)
    out_torch = gn(x_torch)
    loss_torch = out_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    custom_x = Tensor(x_np, requires_grad=True)
    custom_gn = GroupNorm(8, 4)
    out_custom = custom_gn(custom_x)
    loss_custom = out_custom.mean()
    loss_custom.backward()
    grad_custom = custom_x.grad

    compare_outputs("GroupNorm", out_custom.data, out_torch.detach().numpy())
    compare_gradients("GroupNorm", grad_custom, grad_torch)


def test_rmsnorm():
    x_np = np.random.randn(4, 128, 64).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    class TorchRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.gamma * (x / rms)

    torch_rms = TorchRMSNorm(64)
    torch_rms.gamma.data.fill_(1.0)
    out_torch = torch_rms(x_torch)
    loss_torch = out_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    custom_x = Tensor(x_np, requires_grad=True)
    custom_rms = rmsnorm(64)
    out_custom = custom_rms(custom_x)
    loss_custom = out_custom.mean()
    loss_custom.backward()
    grad_custom = custom_x.grad

    compare_outputs("RMSNorm", out_custom.data, out_torch.detach().numpy())
    compare_gradients("RMSNorm", grad_custom, grad_torch)


if __name__ == "__main__":
    test_batchnorm()
    test_instancenorm()
    test_layernorm2d()
    test_groupnorm()
    test_rmsnorm()
