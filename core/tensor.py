import numpy as np
from numpy.lib.stride_tricks import as_strided
from core.Functional import (AddBackward, MulBackward, DivBackward, PowBackward,
                             NegBackward, ReshapeBackward, TransposeBackward, MatMulBackward,
                             LogBackward, MeanBackward,SumBackward,SplitBackward,StdBackward,SubBackward)

#TODO: what the fuck is this doing in memory?
class Tensor:
    def __init__(self, data, requires_grad=True, grad_fn=None, parents=None, name=None):
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
        self.name = name  # Add name attribute for better visualization
        self.op_name = None

    def __repr__(self):
        name_str = f", name='{self.name}'" if self.name else ""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}{name_str})"

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
            grad = np.ones_like(self.data)

        self.assign_grad(grad)
        # Get nodes in topological order so that we call each node's backward function only after its dependents.
        topo_order = self._topological_sort()
        for node in topo_order:
            if node._grad_fn is not None and node.requires_grad:
                # print("func:",node._grad_fn)
                node._grad_fn(node.grad)
                # print(f"grad of function {node._grad_fn}: {node.grad}")

    # === Overloaded Operators ===

    def __add__(self, other):
        if not isinstance(other, Tensor):
            # print("ADD: other is not a Tensor, converting to Tensor")
            other = Tensor(other, requires_grad=False)
        
        
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = AddBackward(self, other)
            out.op_name = "add"
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Create name for the result

        
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = MulBackward(self, other)
            out.op_name = "mul"
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Create name for the result

        
        out = Tensor(self.data - other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            # a - b = a + (-b)
            out._grad_fn = SubBackward(self, other)
            out.op_name = "sub"
        return out

    def __rsub__(self, other):
        # other - self
        return Tensor(other, requires_grad=True).__sub__(self)

    def __neg__(self):
        self_name = self.name or "tensor"
        result_name = f"-{self_name}"
        
        out = Tensor(-self.data, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # Negation is equivalent to multiplying by -1.	
            out._grad_fn = NegBackward(self)
            out.op_name = "neg"
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Create name for the result


        
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = DivBackward(self, other)
            out.op_name = "div"
        return out


    def pad(self, pad_width, mode='constant', constant_values=0):
        padded_data = np.pad(self.data, pad_width, mode=mode, constant_values=constant_values)

        if not self.requires_grad:
            return Tensor(padded_data, requires_grad=False)

        def backward(grad_output):
            grad_input = np.zeros_like(self.data)
            slices = tuple(slice(p[0], grad_output.shape[i] - p[1]) for i, p in enumerate(pad_width))
            grad_input += grad_output[slices]
            self.assign_grad(grad_input)

        out = Tensor(padded_data, requires_grad=True)
        out.parents = [self]
        out._grad_fn = backward
        out.op_name = "pad"
        return out


    def as_strided(self, shape, strides):
        data = as_strided(self.data, shape=shape, strides=strides)
        
        if not self.requires_grad:
            return Tensor(data, requires_grad=False)

        # Backward function
        def backward(grad_output):
            grad_input = np.zeros_like(self.data)
            # This is critical: accumulate using the same view!
            grad_view = as_strided(grad_input, shape=shape, strides=strides)
            np.add.at(grad_view, tuple(np.indices(shape)), grad_output)
            self.assign_grad(grad_input)

        
        
        out = Tensor(
            data,
            requires_grad=True
        )
        out.parents = [self]
        out._grad_fn = backward
        out.op_name = "as_strided"
        return out

    def pow(self, power):
        self_name = self.name or "tensor"
        result_name = f"{self_name}^{power}"
        
        out = Tensor(np.power(self.data, power), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = PowBackward(self, power)
            out.op_name = "pow"
        return out

    def __pow__(self, power):
        return self.pow(power)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Create name for the result


        
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        out.parents = [self, other]
        if out.requires_grad:
            out._grad_fn = MatMulBackward(self, other)
            out.op_name = "matmul"
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # === Additional Operations Needed for Losses ===

    def log(self):
        """Element-wise natural logarithm."""
        self_name = self.name or "tensor"
        result_name = f"log({self_name})"
        
        out = Tensor(np.log(self.data), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = LogBackward(self)
            out.op_name = "log"
        return out

    def clip(self, min_value, max_value):
        """Clip the data for numerical stability. For simplicity, we pass gradients through unchanged."""
        self_name = self.name or "tensor"
        result_name = f"clip({self_name})"
        
        out = Tensor(np.clip(self.data, min_value, max_value), self.requires_grad)
        out.parents = [self]
        # We can choose to ignore the discontinuities in the gradient at the boundaries.
        if out.requires_grad:
            # Identity backward (i.e. d(clip)/dx = 1 almost everywhere)
            def clip_backward(grad):
                self.assign_grad(grad)
            clip_backward.op_name = "clip"
            out._grad_fn = clip_backward
        return out
    

    def mean(self, axis=None, keepdims=False):
        """Compute the mean and record the backward operation."""
        self_name = self.name or "tensor"
        result_name = f"mean({self_name})"
        
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]

        if out.requires_grad:
            out._grad_fn = MeanBackward(self, axis, keepdims)
            out.op_name = "mean"
        return out


    
    def sqrt(self):
        """Compute the square root and record the backward operation."""
        return self.pow(0.5)

    def softmax(self, axis=1):
        """Compute softmax along specified axis."""
        self_name = self.name or "tensor"
        
        
        # Subtract max for numerical stability
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        e_x = np.exp(shifted)
        s = e_x / np.sum(e_x, axis=axis, keepdims=True)

        out = Tensor(s, requires_grad=self.requires_grad, name=f"softmax({self_name})")
        out.parents = [self]

        if self.requires_grad:
            def softmax_backward(grad):
                grad_input = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))
                self.assign_grad(grad_input)
            out._grad_fn = softmax_backward
            out.op_name = "softmax"

        return out



    def var(self, axis=None, keepdims=False):
        """Compute the variance and record the backward operation."""
        self_name = self.name or "tensor"
        result_name = f"var({self_name})"
        
        var_value = np.var(self.data, axis=axis, keepdims=keepdims)
        def var_backward(grad):
            grad = np.array(grad)
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            grad = grad / np.sqrt(var_value)
            self.assign_grad(grad)
        out = Tensor(var_value, self.requires_grad)
        out.parents = [self]
        out._grad_fn = var_backward
        out.op_name = "var"
        return out

    def std(self, axis=None, keepdims=False):
        """Compute the standard deviation and record the backward operation."""
        self_name = self.name or "tensor"
        result_name = f"std({self_name})"
        
        std_value = np.std(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(std_value, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = StdBackward(self, axis, keepdims)
            out.op_name = "std"
        return out
    
    def sum(self, axis=None, keepdims=False):
        """Compute the sum and record the backward operation."""
        self_name = self.name or "tensor"
        result_name = f"sum({self_name})"
        
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = SumBackward(self, axis, keepdims)
            out.op_name = "sum"
        return out

    def transpose(self, *axes):
        """Transpose."""
        self_name = self.name or "tensor"
        result_name = f"transpose({self_name})"
        
        out = Tensor(self.data.transpose(axes), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # We use an axes permutation that swaps dimensions.
            out._grad_fn = TransposeBackward(self, axes)
            out.op_name = "transpose"
        return out



    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data  # Extract NumPy array
        self.data[key] = value

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
        
        self_name = self.name or "tensor"
        result_name = f"{self_name}[{key}]"

        # If this tensor doesn't require gradients, return a simple Tensor


        # Define a backward function to propagate gradients to the original tensor
        def backward_sliced(grad):
            # Create a zero gradient array matching the original shape
            full_grad = np.zeros_like(self.data)
            # Place the incoming gradient into the appropriate slice
            full_grad[key] = grad
            # Propagate the gradient to this tensor
            self.assign_grad(full_grad)
        
        
        out = Tensor(sliced_data, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = backward_sliced
            out.op_name = "getitem"

        # Return a new Tensor with the sliced data and gradient tracking
        return out


    def max(self, axis=None, keepdims=False):
        """
        Compute the maximum and record the backward operation.
        Supports arbitrary axes and keeps track for gradient propagation.
        """

        max_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_data, requires_grad=self.requires_grad)
        out.parents = [self]
        out.op_name = "max"

        def max_backward(grad):
            # Broadcast grad to input shape
            grad = np.array(grad)
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)

            # Build a mask where data equals the max
            max_mask = (self.data == max_data) if axis is None else (self.data == np.max(self.data, axis=axis, keepdims=True))

            # If multiple max values, split gradient among them
            num_max = np.sum(max_mask, axis=axis, keepdims=True)
            grad_input = max_mask * grad / num_max

            self.assign_grad(grad_input)


        out._grad_fn = max_backward
        return out



    def reshape(self, *shape):
        """Reshape."""
        self_name = self.name or "tensor"
        result_name = f"reshape({self_name})"
        
        out = Tensor(self.data.reshape(shape), self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            out._grad_fn = ReshapeBackward(self, self.data.shape)
            out.op_name = "reshape"
        return out

    # === Other Utilities ===

    @staticmethod
    def triu(matrix, k=0):
        return Tensor(np.triu(matrix, k=k), requires_grad=False, name="triu")
    
    @staticmethod
    def tril(matrix, k=0):
        return Tensor(np.tril(matrix, k=k), requires_grad=False, name="tril")
    
    @staticmethod
    def expand_dims(tensor, axis):
        tensor_name = tensor.name or "tensor"
        result_name = f"expand_dims({tensor_name})"
        return Tensor(np.expand_dims(tensor.data, axis), requires_grad=False)
    
    @staticmethod
    def squeeze(tensor, axis):
        tensor_name = tensor.name or "tensor"
        result_name = f"squeeze({tensor_name})"
        return Tensor(np.squeeze(tensor, axis), requires_grad=False)
    
    @staticmethod
    def stack(tensors, axis=0):
        return Tensor(np.stack(tensors, axis=axis), requires_grad=False, name="stack")
    
    @staticmethod
    def hstack(tensors):
        return Tensor(np.hstack(tensors), requires_grad=False, name="hstack")
    
    @staticmethod
    def vstack(tensors):
        return Tensor(np.vstack(tensors), requires_grad=False, name="vstack")
    
    @staticmethod
    def concatenate(tensors, axis=0):
        return Tensor(np.concatenate(tensors, axis=axis), requires_grad=False, name="concatenate")
    
    

    def split(self, indices_or_sections, axis):
        
        splits = np.split(self.data, indices_or_sections, axis=axis)
        num_splits = len(splits)
        grad_fn = SplitBackward(self, num_splits, axis)

        result = []
        for i, s in enumerate(splits):
            t = Tensor(s, requires_grad=self.requires_grad)
            if t.requires_grad:
                def make_backward(i):
                    return lambda grad: grad_fn(grad, i)
                t._grad_fn = make_backward(i)
                t.parents = [self]
                t.op_name = f"split[{i}]"
            result.append(t)

        return tuple(result)

    
    @staticmethod
    def tile(tensor, reps):
        tensor_name = tensor.name or "tensor"
        result_name = f"tile({tensor_name})"
        return Tensor(np.tile(tensor, reps), requires_grad=False)
    
    @staticmethod
    def repeat(tensor, repeats, axis=None):
        tensor_name = tensor.name or "tensor"
        result_name = f"repeat({tensor_name})"
        return Tensor(np.repeat(tensor, repeats, axis=axis), requires_grad=False)
    



    @property
    def T(self):
        """Transpose."""
        
        
        out = Tensor(self.data.T, self.requires_grad)
        out.parents = [self]
        if out.requires_grad:
            # We use an axes permutation that swaps dimensions.
            out._grad_fn = TransposeBackward(self, None)
            out.op_name = "transpose"
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

    def view_graph(self, filename="computational_graph", view=True,save=True):
        """
        Visualize the computational graph using PyVis.
        
        Args:
            filename (str): Name of the HTML file to save
            view (bool): Whether to open the file in browser automatically
        
        Returns:
            Network: PyVis network object
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis is required for graph visualization. Install with: pip install pyvis")
        
        # Create a PyVis network
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            directed=True
        )
        
        # Set physics options for better layout with improved spacing for parallel branches
        net.set_options("""
        var options = {
        "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
            "centralGravity": 0.0,
            "springLength": 150,
            "springConstant": 0.005,
            "nodeDistance": 200,
            "damping": 0.12
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 100}
        },
        "layout": {
            "hierarchical": {
            "enabled": true,
            "levelSeparation": 200,
            "nodeSpacing": 150,
            "treeSpacing": 250,
            "blockShifting": true,
            "edgeMinimization": false,
            "parentCentralization": false,
            "direction": "LR",
            "sortMethod": "directed"
            }
        }
        }
        """)

        visited = set()

        def add_tensor(tensor):
            tensor_id = id(tensor)
            if tensor_id in visited:
                return
            visited.add(tensor_id)
            
            # Create label for the tensor node
            tensor_name = tensor.name if tensor.name else "tensor"
            label = f"{tensor_name} \n shape: {tensor.data.shape} \n grad: {tensor.requires_grad}"
            
            # Add tensor node (oval/ellipse shape)
            net.add_node(
                tensor_id, 
                label=label,
                shape="ellipse",
                color={"background": "#e1f5fe", "border": "#0277bd"},
                size=25,
                font={"size": 12}
            )

            # If this tensor was produced by an operation (_grad_fn), create a separate function node
            if tensor._grad_fn is not None:
                func_id = f"func_{id(tensor._grad_fn)}"
                
                # Determine the function node label
                if hasattr(tensor, 'op_name'):
                    fn_name = tensor.op_name
                else:
                    fn_name = str(tensor._grad_fn)
                
                # Add function node (box shape)
                net.add_node(
                    func_id,
                    label=fn_name,
                    shape="box",
                    color={"background": "#fff3e0", "border": "#f57c00"},
                    size=20,
                    font={"size": 10}
                )
                
                # Connect the function node to the current tensor node
                net.add_edge(func_id, tensor_id, arrows="to")
                
                # Recursively add all parent tensors and connect them to the function node
                for parent in tensor.parents:
                    add_tensor(parent)
                    parent_id = id(parent)
                    net.add_edge(parent_id, func_id, arrows="to")
            else:
                # For tensors without an associated _grad_fn (like inputs), simply add their parent's edges
                for parent in tensor.parents:
                    add_tensor(parent)
                    parent_id = id(parent)
                    net.add_edge(parent_id, tensor_id, arrows="to")

        # Build the graph starting from this tensor
        add_tensor(self)
        
        # Save the graph
        html_filename = f"{filename}.html"
        net.save_graph(html_filename)
        
        if view:
            import webbrowser
            import os
            webbrowser.open(f"file://{os.path.abspath(html_filename)}")
        
        print(f"Graph saved as {html_filename}")
        return net