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
            # Ensure that the gradient is reshaped back to the original shape
            self.a.assign_grad(grad.reshape(self.original_shape))


class TransposeBackward:
    def __init__(self, a, axes):
        self.a = a
        self.axis = axes  # The axes that were used for the transpose

    def __call__(self, grad):
        if self.a.requires_grad:
            # Reverse the axes of the gradient to properly propagate it
            
            # Check if the number of axes of grad matches self.a's data
            if grad.ndim == self.a.data.ndim:
                self.a.assign_grad(grad.transpose(*self.axis))
            else:
                # If grad does not have the same number of dimensions as self.a, handle broadcasting
                # This is a more advanced scenario, but typically we expect them to match
                raise ValueError("Grad shape does not match the expected shape for transpose.", grad.shape, self.a.data.shape)


class MatMulBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(np.matmul(grad, np.swapaxes(self.b.data, -2, -1)))
        if self.b.requires_grad:
            self.b.assign_grad(np.matmul(np.swapaxes(self.a.data, -2, -1), grad))



class SplitBackward:
    def __init__(self, tensor, indices_or_sections, axis, num_splits):
        self.tensor = tensor
        self.indices_or_sections = indices_or_sections
        self.axis = axis
        self.num_splits = num_splits

    def __call__(self, grad):
        # Check that grad's shape is compatible with the split tensor's shape
        if grad.shape[self.axis] != self.tensor.data.shape[self.axis]:
            raise ValueError(f"Grad shape along axis {self.axis} is incompatible with the original tensor shape.", grad.shape, self.tensor.data.shape)

        # Initialize an empty array for the gradient that matches the original tensor's shape
        split_grad = np.zeros_like(self.tensor.data)

        # Split the gradient along the given axis and assign it back to the original tensor
        grad_parts = np.split(grad, self.num_splits, axis=self.axis)
        
        # Ensure the grad_parts is aligned with the correct axis
        if len(grad_parts) != self.num_splits:
            raise ValueError(f"Unable to split grad into {self.num_splits} parts. Ensure the grad shape is compatible.")
        
        for i, grad_part in enumerate(grad_parts):
            # Place each gradient part back into the original tensor's grad
            if self.axis == 0:  # Split along batch axis
                split_grad[i, :] = grad_part
            elif self.axis == 1:  # Split along time axis
                split_grad[:, i] = grad_part
            else:
                split_grad[i] = grad_part  # For other axes, handle accordingly
        
        # Now assign the computed gradient back to the original tensor
        self.tensor.assign_grad(split_grad)


       




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


class SumBackward:
    def __init__(self, a, axis, keepdims):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self, grad):
        if self.a.requires_grad:
            # The gradient of sum is simply 1/N distributed to every element.
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