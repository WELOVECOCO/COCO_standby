import numpy as np
import weakref
from core.new_tensor import Tensor
from core.new_backward_ops import BCE, SCE, MSE, NopNode

EPSILON = 1e-12

def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_pred = y_pred.clip(EPSILON, 1 - EPSILON)
    out_data = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean().data
    out = Tensor(out_data, requires_grad=y_pred.requires_grad)

    if out.requires_grad:
        grad_fn = BCE()
        grad_fn.saved_data["y_true"] = y_true.data
        grad_fn.saved_data["y_pred"] = y_pred.data
        grad_fn.next_functions = [y_pred.get_grad_fn if y_pred.requires_grad else NopNode()]
        grad_fn.tensor = weakref.ref(out)
        out._grad_fn = grad_fn

    return out


def sparse_categorical_cross_entropy(y_true: Tensor, y_pred: Tensor, axis=1) -> Tensor:
    if (y_pred.sum(axis=axis).data.round(3) != 1.0).any():
        y_pred = y_pred.softmax(axis=axis)
    
    y_pred = y_pred.clip(EPSILON, 1 - EPSILON)
    out_data = -(y_true * y_pred.log()).sum(axis=axis).mean().data
    out = Tensor(out_data, requires_grad=y_pred.requires_grad)

    if out.requires_grad:
        grad_fn = SCE()
        grad_fn.saved_data["y_true"] = y_true.data
        grad_fn.saved_data["y_pred"] = y_pred.data
        grad_fn.next_functions = [y_pred.get_grad_fn if y_pred.requires_grad else NopNode()]
        grad_fn.tensor = weakref.ref(out)
        out._grad_fn = grad_fn

    return out


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    out_data = ((y_true - y_pred) ** 2).mean().data
    out = Tensor(out_data, requires_grad=y_pred.requires_grad)

    if out.requires_grad:
        grad_fn = MSE()
        grad_fn.saved_data["y_true"] = y_true.data
        grad_fn.saved_data["y_pred"] = y_pred.data
        grad_fn.next_functions = [y_pred.get_grad_fn if y_pred.requires_grad else NopNode()]
        grad_fn.tensor = weakref.ref(out)
        out._grad_fn = grad_fn

    return out



@staticmethod
def get_loss_fn(name):
    if name == "bce":
        return binary_cross_entropy
    elif name == "cat_cross_entropy":
        return sparse_categorical_cross_entropy
    elif name == "mse":
        return mean_squared_error
    else:
        raise ValueError(f"Unknown loss function: {name}")
