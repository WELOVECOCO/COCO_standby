import numpy as np
import weakref
from core.new_tensor import Tensor
from core.new_backward_ops import TANH, SIGMOID, RELU, LEAKY_RELU, SOFTMAX,NopNode

class Tanh:
    def __call__(self, x: Tensor) -> Tensor:
        out_data = np.tanh(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        if out.requires_grad:
            grad_fn = TANH()
            grad_fn.saved_data["out"] = out_data
            grad_fn.next_functions = [x.get_grad_fn if x.requires_grad else NopNode()]
            grad_fn.tensor = weakref.ref(out)
            out._grad_fn = grad_fn

        return out


class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        out_data = 1 / (1 + np.exp(-x.data))
        out = Tensor(out_data, requires_grad=x.requires_grad)

        if out.requires_grad:
            grad_fn = SIGMOID()
            grad_fn.saved_data["out"] = out_data
            grad_fn.next_functions = [x.get_grad_fn if x.requires_grad else NopNode()]
            grad_fn.tensor = weakref.ref(out)
            out._grad_fn = grad_fn

        return out


class Relu:
    def __call__(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        if out.requires_grad:
            grad_fn = RELU()
            grad_fn.saved_data["x"] = x.data
            grad_fn.next_functions = [x.get_grad_fn if x.requires_grad else NopNode()]
            grad_fn.tensor = weakref.ref(out)
            out._grad_fn = grad_fn

        return out


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: Tensor) -> Tensor:
        out_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        if out.requires_grad:
            grad_fn = LEAKY_RELU()
            grad_fn.saved_data["x"] = x.data
            grad_fn.saved_data["alpha"] = self.alpha
            grad_fn.next_functions = [x.get_grad_fn if x.requires_grad else NopNode()]
            grad_fn.tensor = weakref.ref(out)
            out._grad_fn = grad_fn

        return out


class Softmax:
    def __call__(self, x: Tensor, axis=1) -> Tensor:
        shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
        e_x = np.exp(shifted)
        s = e_x / np.sum(e_x, axis=axis, keepdims=True)

        out = Tensor(s, requires_grad=x.requires_grad)

        if out.requires_grad:
            grad_fn = SOFTMAX()
            grad_fn.saved_data["s"] = s
            grad_fn.saved_data["axis"] = axis
            grad_fn.next_functions = [x.get_grad_fn if x.requires_grad else NopNode()]
            grad_fn.tensor = weakref.ref(out)
            out._grad_fn = grad_fn

        return out
