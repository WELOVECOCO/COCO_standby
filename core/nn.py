
import numpy as np
from core.operations import FastConvolver
from numpy.lib.stride_tricks import sliding_window_view
from core.tensor import Tensor
from core.Function import *
class Module:
    def __init__(self):
        pass

class Layer(Module):
    def __init__(self):
        """
        
        """
        self.initialize_type = None
        self.weights=None
        self.bias=None
        self.input=None
        self.output=None
        self.name = None
        self.testt = False


    def test(self):
        self.testt = True    
    def parameters(self):
        return [self.weights,self.bias]
    
    def __call__(self, input,**kwargs):
        pass

    def backward(self,grad):
        pass

    def initialize_weights(self, shape,initialize_type):
        pass

    

class Linear(Layer):
    """
    A fully connected linear layer that performs an affine transformation: 
    output = input * weights + bias. Supports different weight initialization methods
    and optional dropout for regularization.
    """
    
    def __init__(self,input_dim, output_dim, initialize_type="random",activation="none",dropout=None):
        """
        Initializes a linear layer with specified weight initialization and optional dropout.
        
        Args:
            shape (tuple): A tuple of (input_dim, output_dim).
            initialize_type (str, optional): Weight initialization method. Defaults to "random".
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__()
        self.activation = get_activation(activation, dropout) if activation != "none" else None
        
        self.weights, self.bias = self.initialize_weights(input_dim, output_dim, initialize_type)

    def parameters(self):

        return [self.weights,self.bias]
    

    def set_parameters(self,weights,bias):
        self.weights = weights
        self.bias = bias
    def initialize_weights(self,input_dim, output_dim, initialize_type="xavier"):   
        """
        Initializes weights and biases based on the specified initialization method.
        
        Args:
            shape (tuple): (input_dim, output_dim) specifying layer dimensions.
            initialize_type (str): Initialization method (zero, random, xavier, he, lecun).
            
        Returns:
            tuple: Initialized weight matrix and bias vector.
        """
        
        
        if initialize_type == 'zero':
            w = np.zeros((input_dim, output_dim))

        elif initialize_type == 'random':
            w = np.random.randn((input_dim, output_dim)) * 0.01

        elif initialize_type == 'xavier':  # Also known as Glorot
            fan_in, fan_out = input_dim, output_dim
            bound = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-bound, bound, size=(input_dim, output_dim))

        elif initialize_type == 'he':  # He (Kaiming) initialization for ReLU
            std = np.sqrt(2 / input_dim)
            w = np.random.normal(0, std, size=(input_dim, output_dim))

        elif initialize_type == 'lecun':  # LeCun initialization for tanh/sigmoid
            std = np.sqrt(1 / input_dim)
            w = np.random.normal(0, std, size=(input_dim, output_dim))

        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        b = np.zeros((output_dim))  # Bias is always initialized to zeros
        return Tensor(w, requires_grad=True), Tensor(b, requires_grad=True)  # Return Tensor objects for weights and biases

    def __call__(self, input,**kwargs):
        """
        Performs the forward pass of the linear layer.
        
        Args:
            input (np.ndarray): Input tensor.
            test (bool, optional): Whether the model is in test mode (disables dropout). Defaults to False.
            
        Returns:
            np.ndarray: Output of the affine transformation.
        """
        self.input = input
        if self.input.data.ndim == 4:  # If input is 4D (e.g., [B, C, H, W])
            B, C, H, W = self.input.data.shape
            parents = self.input.parents
            grad_fn = self.input._grad_fn
            self.input = Tensor(self.input.data.reshape(B, -1))  # Flatten to [B, C*H*W]
            self.input.parents = parents
            self.input._grad_fn = grad_fn
        elif self.input.ndim != 2:  # If input is not 2D or 4D, raise an error
            raise ValueError(f"Linear layer received input of unsupported shape {self.input.shape}")
        
        out = (self.input.data @ self.weights.data) + self.bias.data  
        if self.activation is not None:
            out = self.activation(out)
        self.output = Tensor(out, requires_grad=True)    
        self.output._grad_fn = self.backward 
        self.output.parents = [self.input] # Set the backward function for autograd
        return self.output

    def backward(self, grad):
        """
        Performs the backward pass, computing gradients of the loss with respect to input, weights, and bias.
        
        Args:
            error_wrt_output (np.ndarray): Gradient of loss with respect to layer output.
            **kwargs: Optional L1 and L2 regularization parameters.
        
        Returns:
            np.ndarray: Gradient of loss with respect to layer input.
        """
        grad = self.activation.backward(grad) if self.activation is not None else grad
        
        # Gradient of loss w.r.t. weights

        self.weights.grad = self.input.data.T @ grad

        # Gradient of loss w.r.t. bias
        self.bias.grad = np.sum(grad, axis=0, keepdims=False)
        
        # Gradient of loss w.r.t. input
        self.input.assign_grad(grad @ self.weights.data.T)
        

class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, initialize_type="xavier",activation="none",dropout=None):
        """
        Initializes a 2D convolutional layer.
        
        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of filters (output channels) in the layer.
            kernel_size (int): Size of the square convolutional kernel.
            stride (int, optional): Step size for moving the convolutional kernel. Defaults to 1.
            padding (int, optional): Number of pixels to pad around the input. Defaults to 0.
            initialize_type (str, optional): Method for initializing weights. Defaults to "xavier".
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = get_activation(activation, dropout) if activation != "none" else None

        # Convolution operation utility
        self.convolver = FastConvolver()

        # Shape of the weight tensor (filters)
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size)

        # Initialize weights and biases based on selected method
        self.weights, self.bias = self.initialize_weights(initialize_type)


    def parameters(self):

        return [self.weights,self.bias]
    def initialize_weights(self, initialize_type="xavier"):
        """
        Initializes weights and biases for the Conv2D layer.

        Args:
            initialize_type (str): Type of initialization ('zero', 'random', 'xavier', 'he', 'lecun').

        Returns:
            tuple: Initialized weight and bias tensors.
        """
        if initialize_type == 'zero':
            w = np.zeros(self.kernels_shape)
        elif initialize_type == 'random':
            w = np.random.randn(*self.kernels_shape) * 0.01
        elif initialize_type == 'xavier':
            fan_in = self.input_channels * self.kernel_size * self.kernel_size
            fan_out = self.output_channels * self.kernel_size * self.kernel_size
            bound = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-bound, bound, size=self.kernels_shape)
        elif initialize_type == 'he':
            w = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (self.input_channels * self.kernel_size * self.kernel_size))
        elif initialize_type == 'lecun':
            w = np.random.randn(*self.kernels_shape) * np.sqrt(1 / (self.input_channels * self.kernel_size * self.kernel_size))
        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        # Bias is usually initialized to zero
        b = np.zeros((self.output_channels))



        return Tensor(w, requires_grad=True),Tensor(b, requires_grad=True)  # Return Tensor objects for w, b

    def __call__(self, input,**kwargs):
        """
        Performs the forward pass of the convolution operation.
        
        Args:
            input (np.ndarray): Input tensor of shape (batch_size, input_channels, height, width).
            test (bool, optional): Whether the model is in test mode (disables dropout). Defaults to False.
        
        Returns:
            np.ndarray: Output tensor after convolution.
        """
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got shape {input.data.shape}")
        if input.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {input.data.shape[1]}")

        # print(type(input))
        # print(type(self.weights.data))
        self.input = input
        output, self.col_matrix = self.convolver.convolve(self.input.data, self.weights.data, stride=self.stride, padding=self.padding)
        bias_reshaped = self.bias.data.reshape(1, self.output_channels, 1, 1)
        output = output + bias_reshaped
        # output = output + self.bias.data
        if self.activation is not None:
            output = self.activation(output)
        self.output = Tensor(output,requires_grad=True)
        self.output._grad_fn = self.backward 
        self.output.parents = [self.input]

        return self.output

    def backward(self, grad):
        """
        Computes the backward pass of the convolution operation.
        
        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output tensor.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input tensor.
        """
        B, F, H_out, W_out = self.output.data.shape
        
        # Gradient wrt biases
        # grad = grad.reshape(self.output.shape)
        dbias = np.sum(grad, axis=(0, 2, 3), keepdims=False)
        self.bias.assign_grad(dbias)
        # assert self.bias.grad.shape == self.bias.shape
        
        # Gradient wrt weights
        grad_reshaped = grad.transpose(1, 0, 2, 3).reshape(self.output_channels, -1)  # Shape: (F, B * H_out * W_out)
        grad_kernel_matrix = grad_reshaped @ self.col_matrix  # Use cached `col_matrix`
        dkernel = grad_kernel_matrix.reshape(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        self.weights.assign_grad(dkernel)
        # assert self.grad_w.shape == self.weights.shape
        
        # Gradient wrt input
        kernel_matrix = self.weights.data.reshape(self.output_channels, -1).T  # Shape: (C*k*k, F)
        dout_matrix = grad.transpose(0, 2, 3, 1).reshape(B * H_out * W_out, F)
        dX_col = dout_matrix @ kernel_matrix.T

        # Use col2im_accumulation to fold dX_col back to the padded input shape.
        dInput_padded = self.convolver.col2im_accumulation(
            dX_col=dX_col,
            input_shape=self.input.data.shape, 
            filter_height=self.kernel_size,
            filter_width=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        
        # Remove the padding to recover gradient w.r.t. the original input.
        if self.padding > 0:
            dInput = dInput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dInput = dInput_padded

        self.input.assign_grad(dInput) 
        
        # assert dInput.shape == self.input.shape
        



class MaxPool2d(Module):
    """
    Implements a 2D max pooling layer.

    Args:
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling operation. Defaults to kernel_size.
        padding (int, optional): Amount of zero-padding added to the input. Defaults to 0.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding
        self.cache = {}

    def __call__(self, X, **kwargs):
        """
        Forward pass for max pooling.

        Args:
            X (np.ndarray): Input tensor of shape (batch_size, channels, height, width).
            test (bool, optional): Whether the model is in test mode. Defaults to False.

        Returns:
            np.ndarray: Output tensor after max pooling.
        """
        self.input = X
        B, C, H, W = self.input.data.shape
        H_k, W_k = self.kernel_size
        stride_h, stride_w = self.stride

        padded_X = np.pad(
            self.input.data,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )

        windows = sliding_window_view(padded_X, (H_k, W_k), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]

        max_windows = windows.reshape(B, C, -1, H_k * W_k)
        max_vals = max_windows.max(axis=-1)
        max_indices = max_windows.argmax(axis=-1)

        output = max_vals.reshape(B, C, -1)
        H_out = int(np.sqrt(output.shape[2]))
        output = output.reshape(B, C, H_out, H_out)

        self.cache['out_shape'] = output.shape
        self.cache['input'] = self.input
        self.cache['max_indices'] = (
            max_indices,
            windows.shape,
            (stride_h, stride_w)
        )

        self.output = Tensor(output,requires_grad=True)
        self.output._grad_fn = self.backward
        self.output.parents = [self.input]
        return self.output

    def backward(self, grad_output):
        """
        Backward pass for max pooling.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        # print(grad_output)
        
        X = self.cache['input'].data
        max_indices, window_shape, strides = self.cache['max_indices']
        grad_output = grad_output.reshape(*self.cache['out_shape'])
        B, C, H_out, W_out = grad_output.shape
        stride_h, stride_w = strides
        kernel_H, kernel_W = self.kernel_size
        H_in, W_in = X.shape[2], X.shape[3]
        padding = self.padding

        max_indices_4d = max_indices.reshape(B, C, H_out, W_out)

        # Generate all indices for B, C, H_out, W_out
        b, c, i, j = np.indices((B, C, H_out, W_out))

        # Compute starting positions and offsets
        h_start = i * stride_h
        w_start = j * stride_w
        h_offset = max_indices_4d // kernel_W
        w_offset = max_indices_4d % kernel_W

        # Calculate positions in padded and original input
        h_padded = h_start + h_offset
        w_padded = w_start + w_offset
        h_original = h_padded - padding
        w_original = w_padded - padding

        # Mask for valid positions within original input dimensions
        valid = (h_original >= 0) & (h_original < H_in) & (w_original >= 0) & (w_original < W_in)

        # Extract valid indices and corresponding gradients
        valid_b = b[valid]
        valid_c = c[valid]
        valid_h = h_original[valid]
        valid_w = w_original[valid]
        valid_grad = grad_output[valid]

        # Accumulate gradients using vectorized scatter-add
        # print(type(valid_grad))
        grad_input = np.zeros_like(X)
        np.add.at(grad_input, (valid_b, valid_c, valid_h, valid_w), valid_grad)
        self.input.assign_grad(grad_input)
             
# the input is the output of the conv2d layer which will be [B,C,H,W]
# the output will be [B,C,H,W]
# each filter will have its own mean and variance and beta and gamma
class batchnorm2d(Layer):
    def __init__(self,channels,betanorm=0.9):  
        super().__init__()
        self.eps = 1e-8
        self.mean = None
        self.var = None
        self.running_mean = None
        self.running_variance = None
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        self.input_normalized = None
        self.betanorm = betanorm
        self.initialize_weights(channels)
    def initialize_weights(self, channels):
        running_mean = np.zeros((1, channels, 1, 1))
        running_variance = np.ones((1, channels, 1, 1))
        weights = np.ones((1, channels, 1, 1))  # Gamma initialized to 1
        bias = np.zeros((1, channels, 1, 1))  # Beta initialized to 0
        self.running_mean = Tensor(running_mean, requires_grad=False)
        self.running_variance = Tensor(running_variance, requires_grad=False)
        self.weights = Tensor(weights, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)


    def parameters(self):
        return [self.weights,self.bias]
    
    def __call__(self, input, **kwargs):

        if self.testt==False:
            self.input = input
            self.mean = np.mean(self.input.data, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(self.input.data, axis=(0, 2, 3), keepdims=True)
            self.running_mean.data = (self.betanorm * self.running_mean.data) + ((1 - self.betanorm) * self.mean)
            self.running_variance.data = (self.betanorm * self.running_variance.data) + ((1 - self.betanorm) * self.var)
            self.input_normalized = (self.input.data - self.mean) / np.sqrt(self.var + self.eps)
            output = self.weights.data * self.input_normalized + self.bias.data
            self.output = Tensor(output, requires_grad=True)
            self.output._grad_fn = self.backward
            self.output.parents = [self.input]
            return self.output
        else:
            self.input = input
            self.mean = self.running_mean.data
            self.var = self.running_variance.data
            self.input_normalized = (self.input.data - self.mean) / np.sqrt(self.var + self.eps)
            output = self.weights.data * self.input_normalized + self.bias.data
            self.output = Tensor(output, requires_grad=False)
            return self.output

    
    def backward(self, grad):
        B, C, H, W = self.input.shape

        # Gradients for beta and gamma
        dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True)  # Sum over B, H, W
        dgamma = np.sum(grad * self.input_normalized, axis=(0, 2, 3), keepdims=True)
        self.bias.assign_grad(dbeta)
        self.weights.assign_grad(dgamma)
        # Gradient w.r.t. input (dx)
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        dx_hat = grad * self.weights.data  # Chain rule: dy/dx_hat = gamma * grad
        dvar = np.sum(dx_hat * (self.input.data - self.mean) * -0.5 * std_inv**3, axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(-dx_hat * std_inv, axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2 * (self.input.data - self.mean), axis=(0, 2, 3), keepdims=True)
        dinput = dx_hat * std_inv + dvar * 2 * (self.input.data - self.mean) / (B * H * W) + dmean / (B * H * W)
        self.input.assign_grad(dinput)


class GAP(Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
        self.input_normalized = None
        self.output = None

    def __call__(self, input, **kwargs):
        self.input = input
        #calc mean across H and W for each channel and batch
        output = np.mean(self.input.data, axis=(2, 3),keepdims=False)
        self.output = Tensor(output, requires_grad=True)
        self.output._grad_fn = self.backward
        self.output.parents = [self.input]
        return self.output
    
    def backward(self, grad):
        #each channel gets one value as the gradient 
        B, C, H, W = self.input.shape
        input_grad = np.ones((B, C, H, W)) * grad[:, :, np.newaxis, np.newaxis] / (H * W)
        self.input.assign_grad(input_grad)
