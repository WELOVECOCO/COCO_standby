import torch
import torch.nn as nn
import numpy as np
from core.tensor import Tensor
from core.nn import ConvBatchNorm2D, InstanceNorm2D, LayerNorm2D, GroupNorm, rmsnorm

np.random.seed(42)
torch.manual_seed(42)

THRESHOLD = 1e-5  # Set your threshold for passing

def compare_outputs(name, out_custom, out_torch):
    diff = np.abs(out_custom - out_torch).max()
    if diff < THRESHOLD:
        print(f"[PASS] {name} diff: {diff:.2e}")
    else:
        print(f"[FAIL] {name} diff: {diff:.2e}")

def test_batchnorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np)

    bn = nn.BatchNorm2d(8, affine=True, track_running_stats=False)
    bn.weight.data.fill_(1.0)
    bn.bias.data.fill_(0.0)

    out_torch = bn(x_torch)

    custom_bn = ConvBatchNorm2D(8)
    out_custom = custom_bn(Tensor(x_np)).data

    compare_outputs("BatchNorm2D", out_custom, out_torch.detach().numpy())

def test_instancenorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np)

    inst = nn.InstanceNorm2d(8, affine=True, track_running_stats=False)
    inst.weight.data.fill_(1.0)
    inst.bias.data.fill_(0.0)

    out_torch = inst(x_torch)

    custom_inst = InstanceNorm2D(8)
    out_custom = custom_inst(Tensor(x_np)).data

    compare_outputs("InstanceNorm2D", out_custom, out_torch.detach().numpy())

def test_layernorm2d():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np)

    ln = nn.LayerNorm([8, 16, 16], elementwise_affine=True)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)

    out_torch = ln(x_torch)

    custom_ln = LayerNorm2D(8)
    out_custom = custom_ln(Tensor(x_np)).data

    compare_outputs("LayerNorm2D", out_custom, out_torch.detach().numpy())

def test_groupnorm():
    x_np = np.random.randn(4, 8, 16, 16).astype(np.float32)
    x_torch = torch.tensor(x_np)

    gn = nn.GroupNorm(4, 8, affine=True)
    gn.weight.data.fill_(1.0)
    gn.bias.data.fill_(0.0)

    out_torch = gn(x_torch)

    custom_gn = GroupNorm(8, 4)
    out_custom = custom_gn(Tensor(x_np)).data

    compare_outputs("GroupNorm", out_custom, out_torch.detach().numpy())

def test_rmsnorm():
    x_np = np.random.randn(4, 128, 64).astype(np.float32)
    x_torch = torch.tensor(x_np)

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

    custom_rms = rmsnorm(64)
    out_custom = custom_rms(Tensor(x_np)).data
    out_torch = torch_rms(x_torch).detach().numpy()

    compare_outputs("RMSNorm", out_custom, out_torch)

if __name__ == "__main__":
    test_batchnorm()
    test_instancenorm()
    test_layernorm2d()
    test_groupnorm()
    test_rmsnorm()
