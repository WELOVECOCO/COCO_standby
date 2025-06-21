import torch
import numpy as np
from core.tensor import Tensor
from core.nn import MultiHeadAttention  

np.random.seed(42)
torch.manual_seed(42)
B, T, D = 2, 10, 16
H = 2  # Number of heads

x_np = np.random.randn(B, T, D).astype(np.float32)
x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
x_coco = Tensor(x_np, requires_grad=True)

# === PyTorch MHA ===
mha_pt = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=False, batch_first=True)
y_pt, _ = mha_pt(x_pt, x_pt, x_pt)
y_pt_np = y_pt.detach().numpy()

# === Coco MHA ===
mha_coco = MultiHeadAttention(dmodel=D, n_heads=H)

# === Copy weights ===
qkv_weight = mha_pt.in_proj_weight.detach().numpy()  # (3D, D)
out_proj_weight = mha_pt.out_proj.weight.detach().numpy()  # (D, D)
mha_coco.qkv_proj.weights.data[:] = qkv_weight.T
mha_coco.out_proj.weights.data[:] = out_proj_weight.T

# === Forward (Coco) ===
y_coco = mha_coco(x_coco)
y_coco_np = y_coco.data

# === Compare forward ===
res_forward = np.allclose(y_pt_np, y_coco_np, atol=1e-7)

# === Backward ===
dy = np.ones_like(y_pt_np, dtype=np.float32)

y_pt.backward(torch.tensor(dy))
y_coco.backward(dy)

grad_pt = x_pt.grad.detach().numpy()
grad_coco = x_coco.grad

res_grad = np.allclose(grad_pt, grad_coco, atol=1e-6)

# === Results ===
print("FORWARD MATCH: yaaaaaaaaaaaaaaaaaaaaaaaaaaay", res_forward)
print("GRADIENT MATCH:  yaaaaaaaaaaaaaaaaaaaaaaaaaaay", res_grad)

if not res_forward:
    print("FAILED (forward): max diff =", np.abs(y_pt_np - y_coco_np).max())

if not res_grad:
    print("FAILED (grad): max diff =", np.abs(grad_pt - grad_coco).max())
