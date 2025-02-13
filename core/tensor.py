import numpy as np
from core.Functional import *
from collections import deque


class Tensor:
    def __init__(self, data, requires_grad=True):
        """Initialize a Tensor with data, gradient tracking, and computational graph details."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None
        self.parents = []

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def shape(self):
        return self.data.shape

    def assign_grad(self, grad):
        """Safely assign or accumulate gradients."""
        if not self.requires_grad:
            return
        if grad is None:
            raise ValueError("Gradient assigned to tensor is None")
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None

    def _topological_sort(self):
        """Compute topological order via DFS (post-order traversal)."""
        visited = set()
        order = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    visit(parent)
                order.append(node)

        visit(self)
        return order[::-1]

    def backward(self, grad=None, retain_graph=False):
        """Perform backpropagation through the computational graph."""
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data) if self.data.shape != () else np.array(1.0)

        self.assign_grad(grad)
        topo_order = self._topological_sort()

        for node in topo_order:
            if node._grad_fn is not None:
                node._grad_fn(node.grad)

        if not retain_graph:
            for node in topo_order:
                node._grad_fn = None

    # Operator Overloading
    def __add__(self, other):
        """Element-wise addition of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = ADD_BACKWARD(self, other)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """Element-wise multiplication of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = MUL_BACKWARD(self, other)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """Matrix multiplication of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = MatMulBackward(self, other)
        return out

    def __sub__(self, other):
        """Element-wise subtraction of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data - other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = SUB_BACKWARD(self, other)
        return out

    def __rsub__(self, other):
        return other - self

    def __neg__(self):
        """Negation of a tensor."""
        out = Tensor(-self.data, self.requires_grad)
        out.parents = [self]
        out._grad_fn = NEG_BACKWARD(self)
        return out

    def __truediv__(self, other):
        """Element-wise division of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data / other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = DIV_BACKWARD(self, other)
        return out

    def pow(self, power):
        """Raise tensor elements to a power."""
        out = Tensor(self.data ** power, self.requires_grad)
        out.parents = [self]
        out._grad_fn = POW_BACKWARD(self, power)
        return out

    def sum(self, axis=None, keepdims=False):
        """Compute sum over a specified axis."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        out._grad_fn = SUM_BACKWARD(self, axis, keepdims)
        return out
