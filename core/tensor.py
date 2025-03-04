import numpy as np
from core.Functional import (AddBackward, MulBackward, DivBackward, PowBackward,
                             NegBackward, ReshapeBackward, TransposeBackward, MatMulBackward,
                             LogBackward, MeanBackward,SumBackward)

class Tensor:
    def __init__(self, data, requires_grad=True):
        # Ensure data is a numpy array of type float32 for numerical stability.
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        else:
            data = data.astype(np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None  # backward function (or node)
        self.parents = []     # list of Tensors used to compute this Tensor

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
        return order[::-1]  # reverse postorder

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
            if node._grad_fn is not None:
                # print("function now :",node._grad_fn)
                node._grad_fn(node.grad)
                # print("grad:",node.grad)

        if not retain_graph:
            for node in topo_order:
                node._grad_fn = None

    # === Overloaded Operators ===

    def __add__(self, other):
        if not isinstance(other, Tensor):
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
            out._grad_fn = AddBackward(self, -other)
        return out

    def __rsub__(self, other):
        # other - self
        return Tensor(other, requires_grad=False).__sub__(self)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
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


    def sum(self, axis=None, keepdims=False):
        """Compute the sum and record the backward operation."""
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = SumBackward(self, axis, keepdims)
        return out
    

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


    # === Other Utilities ===

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
