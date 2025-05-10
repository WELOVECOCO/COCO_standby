import numpy as np
from core.Functional import (AddBackward, MulBackward, DivBackward, PowBackward,
                             NegBackward, ReshapeBackward, TransposeBackward, MatMulBackward,
                             LogBackward, MeanBackward,SumBackward,SplitBackward,StdBackward,SubBackward)

#TODO: what the fuck is this doing in memory?
class Tensor:
    def __init__(self, data, requires_grad=True,grad_fn=None,parents=None):
        # Ensure data is a numpy array of type float32 for numerical stability.
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        else:
            data = data.astype(np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = grad_fn  # backward function (or node)
        self.parents = [parents] if parents is not None else []     # list of Tensors used to compute this Tensor

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # === Helper Methods for Gradients ===

    def assign_grad(self, grad):
        """Accumulate gradient, creating it if necessary."""
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def zero_grad(self):
        self.grad = None

    def _topological_sort(self):
        """Topologically sort nodes of the computation graph."""
        visited = set()
        order = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    visit(parent)
                order.append(node)
        visit(self)
        return order[::-1]  # reverse topo order

    def backward(self, grad=None, retain_graph=False):
        """Backward pass: compute gradients through the computation graph."""
        if not self.requires_grad:
            return

        if grad is None:
            # if scalar loss, the default grad is 1.
            grad = np.ones_like(self.parents[0].data)

        self.assign_grad(grad)
        # Get nodes in topological order so that we call each node’s backward function only after its dependents.
        topo_order = self._topological_sort()
        for node in topo_order:
            if node._grad_fn is not None and node.requires_grad:
                # print("func:",node._grad_fn)
                node._grad_fn(node.grad)
                # print(f"grad of function {node._grad_fn}: {node.grad}")


    # === Overloaded Operators ===

    def __add__(self, other):
        if not isinstance(other, Tensor):
            print("ADD: other is not a Tensor, converting to Tensor")
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = AddBackward(self, other)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = MulBackward(self, other)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            
            # a - b = a + (-b)
            out._grad_fn = SubBackward(self, other)
        return out

    def __rsub__(self, other):
        # other - self
        
        return Tensor(other, requires_grad=True).__sub__(self)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # Negation is equivalent to multiplying by -1.	
            out._grad_fn = NegBackward(self)
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = DivBackward(self, other)
        return out

    def pow(self, power):
        out = Tensor(self.data ** power, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = PowBackward(self, power)
        return out

    def __pow__(self, power):
        return self.pow(power)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = MatMulBackward(self, other)
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # === Additional Operations Needed for Losses ===

    def log(self):
        """Element-wise natural logarithm."""
        out = Tensor(np.log(self.data), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = LogBackward(self)
        return out

    def clip(self, min_value, max_value):
        """Clip the data for numerical stability. For simplicity, we pass gradients through unchanged."""
        out = Tensor(np.clip(self.data, min_value, max_value), self.requires_grad)
        out.parents = [self]
        # We can choose to ignore the discontinuities in the gradient at the boundaries.
        if out.requires_grad:
            # Identity backward (i.e. d(clip)/dx = 1 almost everywhere)
            out._grad_fn = lambda grad: self.assign_grad(grad)
        return out

    def mean(self, axis=None, keepdims=False):
        """Compute the mean and record the backward operation."""
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = MeanBackward(self, axis, keepdims)
        return out

    def std(self, axis=None, keepdims=False):
        """Compute the standard deviation and record the backward operation."""
        std_value = np.std(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(std_value, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = StdBackward(self, axis, keepdims)
        return out
    
    def sum(self, axis=None, keepdims=False):
        """Compute the sum and record the backward operation."""
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = SumBackward(self, axis, keepdims)
        return out
    

    def transpose(self, *axes):
        """Transpose."""
        out = Tensor(self.data.transpose(axes), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # We use an axes permutation that swaps dimensions.
            out._grad_fn = TransposeBackward(self, axes)
        return out

    def std(self, axis=None, keepdims=False):
        """Compute the standard deviation along the specified axis."""
        out = Tensor(np.std(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out.parents = [self]
        if self.requires_grad:
            out._grad_fn = StdBackward(self, axis, keepdims)
        return out
    def __getitem__(self, key):
        """
        Enable NumPy-like indexing and slicing, e.g., tensor[:2, :, :, :-1].
        
        Args:
            key: Index or slice (int, slice, tuple of slices/ints, etc.).
        
        Returns:
            Tensor: A new Tensor instance with the sliced data.
        """
        # Apply the slice/index to the underlying data
        sliced_data = self.data[key]

        # If this tensor doesn't require gradients, return a simple Tensor
        if not self.requires_grad:
            return Tensor(sliced_data, requires_grad=False)

        # Define a backward function to propagate gradients to the original tensor
        def backward_sliced(grad):
            # Create a zero gradient array matching the original shape
            full_grad = np.zeros_like(self.data)
            # Place the incoming gradient into the appropriate slice
            full_grad[key] = grad
            # Propagate the gradient to this tensor
            self.assign_grad(full_grad)

        # Return a new Tensor with the sliced data and gradient tracking
        return Tensor(
            data=sliced_data,
            requires_grad=True,
            parents=[self],
            grad_fn=backward_sliced
        )
    

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
    def reshape(self, *shape):
        """Reshape."""
        out = Tensor(self.data.reshape(shape), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = ReshapeBackward(self, self.data.shape)
        return out
    # === Other Utilities ===

    @staticmethod
    def triu(matrix, k=0):
        return Tensor(np.triu(matrix, k=k), requires_grad=False)
    
    @staticmethod
    def tril(matrix, k=0):
        return Tensor(np.tril(matrix, k=k), requires_grad=False)
    
    @staticmethod
    def expand_dims(tensor, axis):
        return Tensor(np.expand_dims(tensor.data, axis), requires_grad=False)
    
    @staticmethod
    def squeeze(tensor, axis):
        return Tensor(np.squeeze(tensor, axis), requires_grad=False)
    
    @staticmethod
    def stack(tensors, axis=0):
        return Tensor(np.stack(tensors, axis=axis), requires_grad=False)
    
    @staticmethod
    def hstack(tensors):
        return Tensor(np.hstack(tensors), requires_grad=False)
    
    @staticmethod
    def vstack(tensors):
        return Tensor(np.vstack(tensors), requires_grad=False)
    
    @staticmethod
    def concatenate(tensors, axis=0):
        return Tensor(np.concatenate(tensors, axis=axis), requires_grad=False)
    
    @staticmethod
    def split(tensor, indices_or_sections, axis):
        # Get the split tensors from np.split
        out = np.split(tensor.data, indices_or_sections, axis=axis)
        
        # Ensure that the split tensors have the same requires_grad as the original tensor
        split_tensors = []
        num_splits = len(out)  # Number of splits
        grad_fn = SplitBackward(tensor, indices_or_sections, axis, num_splits)  # Get the original gradient function
        for i, split_tensor in enumerate(out):
            split_tensor = Tensor(split_tensor, requires_grad=tensor.requires_grad)
            split_tensor._grad_fn = grad_fn
            split_tensor.parents = [tensor]  # Set parent to the original tensor
            split_tensors.append(split_tensor)
        
        return tuple(split_tensors)

    
    @staticmethod
    def tile(tensor, reps):
        return Tensor(np.tile(tensor, reps), requires_grad=False)
    
    @staticmethod
    def repeat(tensor, repeats, axis=None):
        return Tensor(np.repeat(tensor, repeats, axis=axis), requires_grad=False)
    
    @staticmethod
    def softmax(x, axis=1):
        e_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
        return Tensor(e_x / e_x.sum(axis=axis, keepdims=True), requires_grad=False)



    @property
    def T(self):
        """Transpose."""
        out = Tensor(self.data.T, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # We use an axes permutation that swaps dimensions.
            out._grad_fn = TransposeBackward(self, None)
        return out

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def view_graph(self, filename="computational_graph", format="png", view=False):
       
        from graphviz import Digraph
        dot = Digraph(format=format, graph_attr={'rankdir': 'LR'})
        visited = set()

        def add_tensor(tensor):
            tensor_id = f"tensor_{id(tensor)}"
            if tensor_id in visited:
                return
            visited.add(tensor_id)
            # Create label for the tensor node.
            label = "Tensor"
            if hasattr(tensor, "name"):
                label = f"{tensor.name}\n{label}"
            label += f"\nshape: {tensor.data.shape}\nrequires_grad: {tensor.requires_grad}"
            dot.node(tensor_id, label, shape="oval")

            # If this tensor was produced by an operation (_grad_fn), create a separate function node.
            if tensor._grad_fn is not None:
                func_id = f"func_{id(tensor._grad_fn)}"
                # Determine the function node label.
                if hasattr(tensor._grad_fn, 'layer_type'):
                    fn_name = f"{tensor._grad_fn.layer_type}.backward"
                elif hasattr(tensor._grad_fn, '__self__'):
                    # It's a bound method. Use the type name of its owner (the layer).
                    fn_name = type(tensor._grad_fn.__self__).__name__.lower() + ".backward"
                else:
                    grad_fn_class = type(tensor._grad_fn).__name__
                    if grad_fn_class.endswith("Backward"):
                        fn_name = grad_fn_class[:-len("Backward")].lower() + ".backward"
                    else:
                        fn_name = grad_fn_class
                dot.node(func_id, fn_name, shape="box", style="filled", fillcolor="lightblue")
                # Connect the function node to the current tensor node.
                dot.edge(func_id, tensor_id)
                # Recursively add all parent tensors and connect them to the function node.
                for parent in tensor.parents:
                    add_tensor(parent)
                    parent_id = f"tensor_{id(parent)}"
                    dot.edge(parent_id, func_id)
            else:
                # For tensors without an associated _grad_fn (like inputs), simply add their parent's edges.
                for parent in tensor.parents:
                    add_tensor(parent)
                    parent_id = f"tensor_{id(parent)}"
                    dot.edge(parent_id, tensor_id)

        add_tensor(self)
        dot.render(filename, view=view)
        return dot