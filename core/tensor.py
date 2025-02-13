import numpy as np
from core.Functional import *
from collections import deque


class Tensor:
    def __init__(self, data, requires_grad=True):
        """
        Initializes a Tensor object with data, gradient tracking, and computational graph metadata.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None  # Function to compute gradient for this op.
        self.parents = []  # Tensors that contributed to this Tensor.

    def __repr__(self):
        """
        Returns a string representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def shape(self):
        """
        Returns the shape of the tensor data.
        """
        return self.data.shape

    def assign_grad(self, grad):
        """
        Safely assigns or accumulates gradients to the tensor during backpropagation.
        """
        if not self.requires_grad:
            return
        if grad is None:
            raise ValueError("Gradient assigned to tensor is None")
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def zero_grad(self):
        """
        Resets the gradient of the tensor to None.
        """
        self.grad = None

    def _topological_sort(self):
        """
        Computes the topological order of the computation graph using DFS.
        """
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
        """
        Performs backpropagation through the computational graph.
        """
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
        """
        Performs element-wise addition of tensors and records the operation for autograd.
        """
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
        """
        Performs element-wise multiplication of tensors and records the operation for autograd.
        """
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
        """
        Performs matrix multiplication between tensors and records the operation for autograd.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad)
        out.parents = [self, other]
        out._grad_fn = MatMulBackward(self, other)
        return out

    def __sub__(self, other):
        """
        Performs element-wise subtraction of tensors and records the operation for autograd.
        """
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        """
        Performs element-wise negation of a tensor and records the operation for autograd.
        """
        return self * -1

    def __truediv__(self, other):
        """
        Performs element-wise division of tensors using reciprocal multiplication and records the operation for autograd.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return self * other.pow(-1)

    def pow(self, power):
        """
        Raises the tensor to a specified power and records the operation for autograd.
        """
        out = Tensor(self.data ** power, self.requires_grad)
        out.parents = [self]
        return out

    def sum(self, axis=None, keepdims=False):
        """
        Computes the sum of all elements in the tensor along a specified axis and records the operation for autograd.
        """
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        return out
