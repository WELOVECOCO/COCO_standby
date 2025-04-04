import numpy as np
from core.tensor import Tensor
from core.config import Config
def get_activation(activation_name, dropout=None):
    """
    Retrieve an activation function instance based on its name.

    Args:
        activation_name (str): Name of the activation function.
        dropout (float, optional): Dropout rate.

    Returns:
        An instance of the corresponding activation function.
    """
    if activation_name is None or activation_name.lower() == "none":
        return None
    elif activation_name.lower() == "relu":
        return Relu(dropout)
    elif activation_name.lower() == "sigmoid":
        return Sigmoid(dropout)
    elif activation_name.lower() == "tanh":
        return Tanh(dropout)
    elif activation_name.lower() == "softmax":
        return Softmax(dropout)
    elif activation_name.lower() == "leaky_relu":
        return LeakyReLU(dropout)
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


class Activation:
    """
    Base class for activation functions.
    """

    def __init__(self, dropout=None):
        """
        Initialize the activation function with optional dropout.

        Args:
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        self.fused = None
        self.dropout = dropout

    def apply_dropout(self, input):
        """
        Apply dropout to the output tensor if dropout is set and in training mode.
        This is done by composing the dropout operations as Tensor operations,
        so that the graph (and thus parents/grad_fn) is maintained.

        Args:
            tensor (Tensor): The output Tensor of an activation.

        Returns:
            Tensor: The dropout-modified Tensor.
        """
        if self.dropout != None :
            # Create a dropout mask as a Tensor (no gradient required)
            self.mask = (np.random.rand(*input.data.shape) < (1 - self.dropout)).astype(np.float32)

            # Multiply by mask and rescale. Note: the returned Tensor is a new node.
            input = (input * self.mask) / (1 - self.dropout)
        return input


class Tanh(Activation):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **kwargs):
        # If the input is a Tensor (autograd mode)
        if isinstance(x, Tensor):
            self.fused = False
            out_data = np.tanh(x.data)
            self.predropout = out_data.copy()  # store the pre-dropout values
            out_data = Tensor(out_data, requires_grad=x.requires_grad)
            out_data.parents = [x]
            out_data._grad_fn = self.grad_fn
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            self.output = out_data  # store for use in grad_fn
            return out_data
        else:
            self.fused = True
            # Fused mode (x is a numpy array)
            out_data = np.tanh(x)
            self.predropout = out_data.copy()
            # Apply dropout if specified.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            # In fused mode, we simply return the result.
            return out_data
            
    def grad_fn(self, grad):
        # Compute derivative of tanh: 1 - tanh(x)^2, using pre-dropout values.
        dx = (1 - self.predropout ** 2)
        if self.dropout is not None:
            # Multiply by the dropout mask and account for scaling.
            d_dropout = self.mask / (1 - self.dropout)
            dx *= d_dropout
        
        if self.fused:
            return dx * grad
        else:
            self.output.parents[0].assign_grad(dx * grad)



class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **kwargs):

        if isinstance(x, Tensor):
            self.fused = False
            out_data = 1 / (1 + np.exp(-x.data))
            self.predropout = out_data.copy()  # store the pre-dropout values
            out_data = Tensor(out_data, requires_grad=x.requires_grad)
            out_data.parents = [x]
            out_data._grad_fn = self.grad_fn
            # Apply dropout if specified. Make sure apply_dropout returns a Tensor.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            self.output = out_data  # store for use in grad_fn
            return out_data
        else:
            self.fused = True
            # Fused mode (x is a numpy array)
            out_data = 1 / (1 + np.exp(-x))
            self.predropout = out_data.copy()
            # Apply dropout if specified.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            # In fused mode, we simply return the result.
            return out_data
            
    def grad_fn(self, grad):
        # Compute derivative of tanh: 1 - tanh(x)^2, using pre-dropout values.
        dx = self.predropout * (1 - self.predropout)
        if self.dropout is not None:
            # Multiply by the dropout mask and account for scaling.
            d_dropout = self.mask / (1 - self.dropout)
            dx *= d_dropout
        # Propagate the gradient through the graph.
        if self.fused:
            return dx * grad
        else:
            self.output.parents[0].assign_grad(dx * grad)


class Softmax(Activation):
    """
    Softmax activation function.
    """

    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **kwargs):
        """
        Forward pass using softmax along axis 1.

        Args:
            x (Tensor): Input Tensor.

        Returns:
            Tensor: Output Tensor with softmax applied.
        """
        # Subtract max for numerical stability
        axs = kwargs.get('axis', 1)
        if isinstance(x, Tensor):
            self.fused = False
            max_vals = np.max(x.data, axis=axs, keepdims=True)
            exp_x = np.exp(x.data - max_vals)
            sum_exp = np.sum(exp_x, axis=axs, keepdims=True)
            out_data = exp_x / sum_exp
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)

            self.output = Tensor(out_data, requires_grad=x.requires_grad)
            self.output.parents = [x]
            self.output._grad_fn = self.grad_fn
            return self.output
        else:
            self.fused = True
            # Fused mode (x is a numpy array)
            max_vals = np.max(x, axis=axs, keepdims=True)
            exp_x = np.exp(x - max_vals)
            sum_exp = np.sum(exp_x, axis=axs, keepdims=True)
            out_data = exp_x / sum_exp
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            
            return out_data

    def grad_fn(self,grad):
        if self.fused:
            return grad
        else:
            self.output.parents[0].assign_grad(grad)



class Relu(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **kwargs):
        """
        Forward pass using ReLU.

        Args:
            x (Tensor): Input Tensor.

        Returns:
            Tensor: Output Tensor with ReLU applied.
        """
            
        # If the input is a Tensor (autograd mode)
        if isinstance(x, Tensor):
            self.fused = False
            out_data = np.maximum(0, x.data)
            self.predropout = out_data.copy()  # store the pre-dropout values
            out_data = Tensor(out_data, requires_grad=x.requires_grad)
            out_data.parents = [x]
            out_data._grad_fn = self.grad_fn
            # Apply dropout if specified. Make sure apply_dropout returns a Tensor.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            self.output = out_data  # store for use in grad_fn
            return out_data
        else:
            self.fused = True
            # Fused mode (x is a numpy array)
            out_data = np.maximum(0, x)
            self.predropout = out_data.copy()
            # Apply dropout if specified.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            # In fused mode, we simply return the result.
            return out_data


    def grad_fn(self, grad):
        # Compute derivative of tanh: 1 - tanh(x)^2, using pre-dropout values.
        dx = np.ones_like(self.predropout)
        dx[self.predropout <= 0] = 0
        if self.dropout is not None:
            # Multiply by the dropout mask and account for scaling.
            d_dropout = self.mask / (1 - self.dropout)
            dx *= d_dropout
        # Propagate the gradient through the graph.
        if self.fused:
            return dx * grad
        else:
            self.output.parents[0].assign_grad(dx * grad)




class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    """

    def __init__(self, dropout=None,alpha=0.01):
        super().__init__(dropout)
        self.alpha = alpha

    def __call__(self, x, **kwargs):
        """
        Forward pass using Leaky ReLU.

        Args:
            x (Tensor): Input Tensor.
            alpha (float, optional): Slope for negative values. Defaults to 0.01.

        Returns:
            Tensor: Output Tensor with Leaky ReLU applied.
        """
        if isinstance(x, Tensor):
            self.fused = False
            out_data = np.where(x.data > 0, x.data, self.alpha * x.data)
            self.predropout = out_data.copy()  # store the pre-dropout values
            out_data = Tensor(out_data, requires_grad=x.requires_grad)
            out_data.parents = [x]
            out_data._grad_fn = self.grad_fn
            # Apply dropout if specified. Make sure apply_dropout returns a Tensor.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            self.output = out_data  # store for use in grad_fn
            return out_data
        else:
            self.fused = True
            # Fused mode (x is a numpy array)
            out_data = np.where(x > 0, x, self.alpha * x)
            self.predropout = out_data.copy()
            # Apply dropout if specified.
            if Config.TEST == False:
                out_data = self.apply_dropout(out_data)
            # In fused mode, we simply return the result.
            return out_data


    def grad_fn(self, grad):
        # Compute derivative of tanh: 1 - tanh(x)^2, using pre-dropout values.
        dx = np.ones_like(self.predropout)
        dx[self.predropout <= 0] = self.alpha
        if self.dropout is not None:
            # Multiply by the dropout mask and account for scaling.
            d_dropout = self.mask / (1 - self.dropout)
            dx *= d_dropout
        # Propagate the gradient through the graph.
        if self.fused:
            return dx * grad
        else:
            self.output.parents[0].assign_grad(dx * grad)
