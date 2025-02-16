import numpy as np

class AddBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad)
        if self.b.requires_grad:
            self.b.assign_grad(grad)

class MulBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad * self.b.data)
        if self.b.requires_grad:
            self.b.assign_grad(grad * self.a.data)

class DivBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad / self.b.data)
        if self.b.requires_grad:
            self.b.assign_grad(-grad * self.a.data / (self.b.data**2))

class PowBackward:
    def __init__(self, a, power):
        self.a = a
        self.power = power
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad * self.power * (self.a.data ** (self.power - 1)))

class NegBackward:
    def __init__(self, a):
        self.a = a
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(-grad)

class ReshapeBackward:
    def __init__(self, a, original_shape):
        self.a = a
        self.original_shape = original_shape
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad.reshape(self.original_shape))

class TransposeBackward:
    def __init__(self, a, axes):
        self.a = a
        self.axes = axes
    def __call__(self, grad):
        if self.a.requires_grad:
            # If axes was None, .T is a simple transpose.
            if self.axes is None:
                self.a.assign_grad(grad.T)
            else:
                inverse_axes = np.argsort(self.axes)
                self.a.assign_grad(grad.transpose(*inverse_axes))

class MatMulBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad @ self.b.data.T)
        if self.b.requires_grad:
            self.b.assign_grad(self.a.data.T @ grad)

class LogBackward:
    def __init__(self, a):
        self.a = a
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad / self.a.data)

class MeanBackward:
    def __init__(self, a, axis, keepdims):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self, grad):
        if self.a.requires_grad:
            # The gradient of mean is simply 1/N distributed to every element.
            # Compute the number of elements reduced.
            if self.axis is None:
                n = self.a.data.size
            else:
                # When axis is not None, we need to get the size along the reduced axis.
                n = np.prod(np.array(self.a.data.shape)[self.axis])
            # Expand grad to the shape of the original data.
            grad_expanded = grad
            if not self.keepdims and self.axis is not None:
                grad_expanded = np.expand_dims(grad, self.axis)
            self.a.assign_grad(np.ones_like(self.a.data) * (grad_expanded / n))
