import numpy as np

class AddBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _reduce_grad(self, grad, target_shape):
        # Align shapes by adding leading 1s to target_shape if needed
        ndim_diff = len(grad.shape) - len(target_shape)
        target_shape_extended = (1,) * ndim_diff + target_shape

        # Find axes where target_shape is 1 but grad.shape > 1 → broadcasted axes
        reduce_axes = tuple(
            i for i, (t_dim, g_dim) in enumerate(zip(target_shape_extended, grad.shape))
            if t_dim == 1 and g_dim > 1
        )

        # Sum over broadcasted axes
        if reduce_axes:
            grad = grad.sum(axis=reduce_axes, keepdims=True)

        # Remove added leading dims to match target_shape exactly
        grad = grad.reshape(target_shape)

        return grad

    def __call__(self, grad):
        if self.a.requires_grad:
            grad_a = grad
            if self.a.data.shape != grad.shape:
                grad_a = self._reduce_grad(grad, self.a.data.shape)
            self.a.assign_grad(grad_a)

        if self.b.requires_grad:
            grad_b = grad
            if self.b.data.shape != grad.shape:
                grad_b = self._reduce_grad(grad, self.b.data.shape)
            self.b.assign_grad(grad_b)


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
            self.a.assign_grad(grad * self.power * (np.power(self.a.data , (self.power - 1))))

class SubBackward:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad)
        if self.b.requires_grad:
            self.b.assign_grad(-grad)

class NegBackward:
    def __init__(self, a):
        self.a = a
    def __call__(self, grad):
        if self.a.requires_grad:
            grad = -grad
            self.a.assign_grad(grad)

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
        self.axes = axes  # fix typo: use self.axes

    def __call__(self, grad):
        if not self.a.requires_grad:
            return

        if grad.ndim != self.a.data.ndim:
            raise ValueError("Gradient shape mismatch in transpose.", grad.shape, self.a.data.shape)

        if self.axes is None:
            # Default transpose: reverse all axes
            reversed_axes = list(range(grad.ndim))[::-1]
            self.a.assign_grad(grad.transpose(reversed_axes))
        else:
            # Compute inverse permutation
            inverse_axes = np.argsort(self.axes)
            self.a.assign_grad(grad.transpose(inverse_axes))



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
    def __init__(self, tensor, num_splits, axis):
        self.tensor = tensor
        self.axis = axis
        self.collected_grads = [None] * num_splits
        self.count = 0

    def __call__(self, grad, index):
        self.collected_grads[index] = grad
        self.count += 1

        if self.count == len(self.collected_grads):
            # All grads collected
            
            full_grad = np.concatenate(self.collected_grads, axis=self.axis)
            self.tensor.assign_grad(full_grad)



       




class LogBackward:
    def __init__(self, a):
        self.a = a
    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad / self.a.data)

class StdBackward:
    def __init__(self, a, axis, keepdims):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        self.mean = np.mean(a.data, axis=axis, keepdims=True)  # Compute mean
        self.std = np.std(a.data, axis=axis, keepdims=True)  # Compute std
    
    def __call__(self, grad):
        if self.a.requires_grad:
            
            # Ensure std is not zero to avoid division by zero
            std_safe = self.std + 1e-10
            
            # Gradient formula: (X - mean) / (N * std)
            n = np.prod(np.array(self.a.data.shape)[self.axis]) if self.axis is not None else self.a.data.size
            grad_expanded = np.expand_dims(grad, self.axis) if (not self.keepdims and self.axis is not None) else grad
            grad_input = (self.a.data - self.mean) / (n * std_safe) * grad_expanded
            
            self.a.assign_grad(grad_input)

class MeanBackward:
    def __init__(self, a, axis, keepdims):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self, grad):
        if self.a.requires_grad:
           
            # print(f"grad: {grad}")
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

            # print(f"a type: {type(self.a.data)}, grad type: {type(grad_expanded)}")
            # print(f"grad_expanded: {grad_expanded}, n: {n}")
            self.a.assign_grad(np.ones_like(self.a.data) * (grad_expanded / n))


class SumBackward:
    def __init__(self, a, axis, keepdims):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, grad):
        if self.a.requires_grad:
            # Expand grad back to original shape for broadcasting if needed
            grad_expanded = grad
            if not self.keepdims and self.axis is not None:
                grad_expanded = np.expand_dims(grad, axis=self.axis)
            
            # Broadcast gradient to match input shape
            grad_result = np.ones_like(self.a.data) * grad_expanded
            self.a.assign_grad(grad_result)
