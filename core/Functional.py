import numpy as np

class ADD_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad)
        if self.b.requires_grad:
            self.b.assign_grad(grad)


class MUL_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if grad is None:
            raise ValueError("Gradient is None")
        if self.a.requires_grad:
            self.a.assign_grad(grad * self.b.data)
        if self.b.requires_grad:
            self.b.assign_grad(grad * self.a.data)


class MatMulBackward:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __call__(self, grad):
        if self.A.requires_grad:
            self.A.assign_grad(grad @ self.B.data.T)
        if self.B.requires_grad:
            self.B.assign_grad(self.A.data.T @ grad)


class SUB_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad)
        if self.b.requires_grad:
            self.b.assign_grad(-grad)


class NEG_BACKWARD:
    def __init__(self, a):
        self.a = a

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(-grad)


class DIV_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad / self.b.data)
        if self.b.requires_grad:
            self.b.assign_grad(-grad * self.a.data / (self.b.data ** 2))


class POW_BACKWARD:
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def __call__(self, grad):
        if self.base.requires_grad:
            self.base.assign_grad(grad * self.exponent * (self.base.data ** (self.exponent - 1)))


class SUM_BACKWARD:
    def __init__(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, grad):
        if self.a.requires_grad:
            grad_expanded = grad if self.keepdims else np.expand_dims(grad, axis)
            self.a.assign_grad(np.ones_like(self.a.data) * grad_expanded)


class RESHAPE_BACKWARD:
    def __init__(self, a):
        self.a = a

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad.reshape(self.a.data.shape))


class TRANSPOSE_BACKWARD:
    def __init__(self, a):
        self.a = a

    def __call__(self, grad, axes):
        if self.a.requires_grad:
            self.a.assign_grad(grad.transpose(*axes))