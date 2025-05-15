import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from core.operations import *
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
   
    def parameters(self):
        pass
    
    def trained_parameters(self):
        pass
    
    def __call__(self, input,**kwargs):
        pass

    def backward(self,grad):
        pass

    def initialize_weights(self, shape,initialize_type):
        pass

class Flatten(Module):
    """
    A layer that flattens all dimensions except the batch dimension.
    This is typically used before feeding the output into a fully connected layer.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, input, **kwargs):
        """
        Flattens the input tensor (except for the batch dimension).
        
        Args:
            input (Tensor): Input tensor of shape [B, *], where B is the batch size.
        
        Returns:
            Tensor: Flattened tensor of shape [B, -1].
        """
        self.input = input
        # Flatten all dimensions except the batch dimension
        batch_size = input.shape[0]
        flattened_data = input.data.reshape(batch_size, -1)  # Flatten all dimensions except batch

        self.output = Tensor(flattened_data, requires_grad=True)
        self.output._grad_fn = self.backward
        self.output.parents = [self.input]
        return self.output

    def backward(self, grad):
        """
        Backpropagation for the flatten layer. Since flattening is a simple operation,
        the gradient is passed through unchanged.
        
        Args:
            grad (np.ndarray): Gradient of loss with respect to the output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to the input (same shape as input).
        """
        # The gradient is passed through unchanged since flattening is a simple reshape
        grad_input = grad.reshape(self.input.shape)  # Reshape gradient back to the input shape
        self.input.assign_grad(grad_input)
        return grad_input


class Linear(Layer):
    """
    A fully connected linear layer that performs an affine transformation: 
    output = input * weights + bias. Supports different weight initialization methods
    and optional dropout for regularization.
    """
    
    def __init__(self,input_dim, output_dim, initialize_type="random",activation="none",dropout=None,bias=True):
        """
        Initializes a linear layer with specified weight initialization and optional dropout.
        
        Args:
            shape (tuple): A tuple of (input_dim, output_dim).
            initialize_type (str, optional): Weight initialization method. Defaults to "random".
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__()
        self.activation = get_activation(activation, dropout) if activation != "none" else None
        self.bias_flag = bias
        self.weights, self.bias = self.initialize_weights(input_dim, output_dim, initialize_type)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameters(self):
        if self.bias_flag:
            return [self.weights,self.bias]
        else:
            return [self.weights]
    
    def trained_parameters(self):
        # print("fc params")
        if self.bias_flag:
            return [self.weights,self.bias]
        else:
            return [self.weights]
    def set_parameters(self,weights,bias):
        self.weights = weights
        self.bias = bias
    def initialize_weights(self,input_dim, output_dim, initialize_type="he"):   
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
            w = np.random.randn(input_dim, output_dim) * 0.01

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

        b = np.zeros((output_dim)) if self.bias_flag else None # Bias is always initialized to zeros
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
        original_shape = input.shape

        # Reshape input to 2D for matmul (PyTorch automatically does this)
        reshaped_input = input.data.reshape(-1, self.input_dim)  # Flatten all but the last dimension

        # Linear transformation
        out = reshaped_input @ self.weights.data
        if self.bias_flag:
            out += self.bias.data

        # Restore original shape with output_dim in last place
        new_shape = original_shape[:-1] + (self.output_dim,)  # Keeps batch dimensions intact
        out = out.reshape(new_shape)

        # Apply activation if specified
        if self.activation is not None:
            out = self.activation(out)

        self.output = Tensor(out, requires_grad=True)
        self.output._grad_fn = self.backward
        self.output.parents = [self.input]  # Set backward function for autograd
        return self.output

    def backward(self, grad,**kwargs):
        """
        Performs the backward pass, computing gradients of the loss with respect to input, weights, and bias.
        
        Args:
            grad (np.ndarray): Gradient of loss with respect to layer output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to layer input.
        """
        # Apply activation gradient if activation exists
        grad = self.activation.grad_fn(grad) if self.activation is not None else grad

        # Reshape gradient to match the shape used in forward pass
        reshaped_input = self.input.data.reshape(-1, self.input_dim)  # Flatten input before matrix multiplication
        # Compute gradient of loss w.r.t. weights
        dweights = reshaped_input.T @ grad.reshape(-1, self.output_dim)  # Corrected shape for batch processing
        self.weights.assign_grad(dweights)
        # Compute gradient of loss w.r.t. bias (sum over batch dimensions)
        if self.bias_flag:
            dbias = np.sum(grad, axis=tuple(range(len(grad.shape) - 1)))  # Sum over all non-output dimensions

        self.bias.assign_grad(dbias)
        # Compute gradient of loss w.r.t. input
        grad_input = grad @ self.weights.data.T  # (B, output_dim) x (output_dim, input_dim) -> (B, input_dim)

        # Reshape grad_input to match the original input shape
        grad_input = grad_input.reshape(self.input.shape)

        # Assign gradient to input tensor for autograd
        self.input.assign_grad(grad_input)  
        ret_grad = kwargs['ret_grad'] if 'ret_grad' in kwargs else False
        return grad_input if ret_grad else None
class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, initialize_type="xavier",activation="none",dropout=None,bias=True):
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
        self.bias_flag = bias
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = get_activation(activation, dropout) if activation != "none" else None

        # Convolution operation utility
        if kernel_size == 3 and stride == 1:
            self.convolver = WinogradConv()
            self.engine = "wino"
        elif kernel_size >= 7:
            self.convolver = FFTConvolver()
            self.engine = "fft"
        else:
            self.convolver = FastConvolver()
            self.engine = "im2col"


        # Shape of the weight tensor (filters)
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size)

        # Initialize weights and biases based on selected method
        self.weights, self.bias = self.initialize_weights(initialize_type)


    def parameters(self):
        if self.bias_flag:
            return [self.weights,self.bias]
        else:
            return [self.weights]
        
    def trained_parameters(self):
        # print("conv params")
        if self.bias_flag:
            return [self.weights,self.bias]
        else:
            return [self.weights]
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
        b = np.zeros((self.output_channels)) if self.bias_flag else None



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

        if self.engine == "im2col":
            output, self.col_matrix = self.convolver.convolve(self.input.data, self.weights.data, stride=self.stride, padding=self.padding)
        else:
            output = self.convolver.convolve(self.input.data, self.weights.data, stride=self.stride, padding= self.padding)
            self.col_matrix = None

        if self.bias_flag:
            bias_reshaped = self.bias.data.reshape(1, self.output_channels, 1, 1)
            output = output + bias_reshaped

        # output = output + self.bias.data
        if self.activation is not None:
            output = self.activation(output)
        self.output = Tensor(output,requires_grad=True)
        self.output._grad_fn = self.backward 
        self.output.parents = [self.input]

        return self.output

    def backward(self, grad, **kwargs):
        """
        Computes the backward pass of the convolution operation.

        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output tensor.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input tensor.
        """
        B, F, H_out, W_out = self.output.data.shape
        grad = self.activation.grad_fn(grad) if self.activation is not None else grad

        # ----- bias grad -----
        if self.bias_flag:
            dbias = np.sum(grad, axis=(0, 2, 3))
            self.bias.assign_grad(dbias)

        # ----- im2col engine-----
        if self.engine == "im2col":
            # grad wrt weights via cached col_matrix
            grad_reshaped = grad.transpose(1, 0, 2, 3).reshape(F, -1)
            grad_kernel_matrix = grad_reshaped @ self.col_matrix
            dkernel = grad_kernel_matrix.reshape(
                self.output_channels,
                self.input_channels,
                self.kernel_size,
                self.kernel_size
            )
            self.weights.assign_grad(dkernel)

            # grad wrt input via col2im_accumulation
            kernel_matrix = self.weights.data.reshape(F, -1).T
            dout_matrix = grad.transpose(0, 2, 3, 1).reshape(B * H_out * W_out, F)
            dX_col = dout_matrix @ kernel_matrix.T
            dInput_padded = self.convolver.col2im_accumulation(
                dX_col=dX_col,
                input_shape=self.input.data.shape,
                filter_height=self.kernel_size,
                filter_width=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            if self.padding > 0:
                dInput = dInput_padded[:, :,
                         self.padding:-self.padding,
                         self.padding:-self.padding]
            else:
                dInput = dInput_padded

        # ----- fft or wino engine -----
        else:
            # 1) grad w.r.t. input: cross-corr with 180°-rotated kernel
            W180 = np.flip(self.weights.data, axis=(2, 3))
            pad_full = self.kernel_size - 1
            dX_full = self.convolver.convolve(
                grad,
                W180,
                stride=1,
                padding=pad_full
            )
            # crop back to original input size
            H, W = self.input.data.shape[-2:]
            dInput = dX_full[:, :,
                     pad_full:pad_full + H,
                     pad_full:pad_full + W]

            # 2) grad w.r.t. kernels: correlate input with grad, then rotate back
            dW_corr = self.convolver.convolve(
                self.input.data,
                grad,
                stride=1,
                padding=self.padding
            )
            dkernel = np.flip(dW_corr, axis=(2, 3))
            # ensure shape is exactly (F, C, KH, KW)
            dkernel = dkernel.reshape(
                self.output_channels,
                self.input_channels,
                self.kernel_size,
                self.kernel_size
            )
            self.weights.assign_grad(dkernel)

        # write input grad
        self.input.assign_grad(dInput)

        # return if requested
        return dInput if kwargs.get('ret_grad', False) else None


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
    
    def trained_parameters(self):
        # print("bn params")
        return [self.weights,self.bias,self.running_mean,self.running_variance]
    
    def __call__(self, input, **kwargs):

        if Config.TEST==False:
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


##############################################################################################
#                                                                                            #
#                                                                                            #
#                                                                                            #
#                    Transformers Building Blocks (Attention + ViT Patches)                  #
#                                                                                            #
#                                                                                            #
#                                                                                            #
##############################################################################################


class PatchEmbedding(Layer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = Conv2d(
            input_channels=in_channels,
            output_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def __call__(self, x, **kwargs):
        # x: shape (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)

        # Reshape and transpose
        B = x.shape[0]
        x = x.reshape(B, x.shape[1], -1)  # (B, E, N)
        x = x.transpose(0, 2, 1)  # (B, N, E)
        return x

class PositionalEmbedding(Layer):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = Tensor(
            np.random.randn(1, n_patches + 1, embed_dim) * 0.02, # Multiplied for scale control
            requires_grad=False
        )

    def __call__(self, x, **kwargs):
        return x + self.pos_embed[:, :x.shape[1]]

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = Tensor(eps, requires_grad=True)

    def __call__(self, x, **kwargs):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        output = self.gamma * x_norm + self.beta
        
        return output


class SelfAttention(Layer):
    def __init__(self, dmodel):
        super().__init__()
        self.scale = np.sqrt(dmodel)
        self.softmax = Softmax()
    
    def forward(self, q, k, v, masked=False):
        B, nheads, T, head_dim = q.shape

        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale  # (B, nheads, T, T)

        if masked:
            mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e10
            scores += mask

        weights = self.softmax(scores, axis=-1)  # (B, nheads, T, T)
        out = weights @ v  # (B, nheads, T, head_dim)

        self.cache = (q, k, v, weights)
        return out

    def backward(self, grad_out):
        q, k, v, weights = self.cache
        B, nheads, T, head_dim = q.shape

        grad_weights = grad_out @ v.transpose(0, 1, 3, 2)  # (B, nheads, T, T)
        grad_v = weights.transpose(0, 1, 3, 2) @ grad_out

        grad_scores = weights * (grad_weights - (grad_weights * weights).sum(axis=-1, keepdims=True))
        grad_scores /= self.scale

        grad_q = grad_scores @ k
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ q

        self.q.assign_grad(grad_q)
        self.k.assign_grad(grad_k)
        self.v.assign_grad(grad_v)




class MultiHeadAttention(Layer):
    def __init__(self, dmodel, nheads=1, masked=False, encoder_decoder=False):
        super().__init__()
        assert dmodel % nheads == 0, "dmodel must be divisible by nheads"
        self.dmodel = dmodel
        self.nheads = nheads
        self.head_dim = dmodel // nheads
        self.attn_proj = Linear(dmodel, dmodel * 3, initialize_type='zero')
        self.out_proj = Linear(dmodel, dmodel, initialize_type='zero')
        self.masked = masked
        self.encoder_decoder = encoder_decoder
        self.attn = SelfAttention(self.head_dim)  # Per-head scaled dot-product attention

    def __call__(self, x, **kwargs):
        B, T, _ = x.shape
        self.input = x

        if not self.encoder_decoder:
            qkv = self.attn_proj(x).data  # (B, T, dmodel * 3)
            q, k, v = np.split(qkv, 3, axis=2)
        else:
            q, k, v = kwargs['q'], kwargs['k'], kwargs['v']

        # Reshape for multi-head attention
        def reshape(t): return t.reshape(B, T, self.nheads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = reshape(q), reshape(k), reshape(v)
        self.q, self.k, self.v = q, k, v

        out = self.attn.forward(q, k, v, masked=self.masked)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dmodel)
        out = self.out_proj(Tensor(out))

        out.parents = [x]
        out._grad_fn = self.backward
        return out

    def backward(self, grad_out, **kwargs):
        B, T, _ = grad_out.shape
        grad_attn_out = self.out_proj.backward(grad_out, ret_grad=True)
        grad_attn_out = grad_attn_out.reshape(B, T, self.nheads, self.head_dim).transpose(0, 2, 1, 3)

        grad_q, grad_k, grad_v = self.attn.backward(grad_attn_out)

        # Reconstruct gradients to full shape
        def restore(t): return t.transpose(0, 2, 1, 3).reshape(B, T, self.dmodel)
        grad_q, grad_k, grad_v = restore(grad_q), restore(grad_k), restore(grad_v)

        if not self.encoder_decoder:
            grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=2)
            grad_x = self.attn_proj.backward(grad_qkv, ret_grad=True)
            self.input.assign_grad(grad_x)
            return grad_x if kwargs.get('ret_grad', False) else None
        else:
            self.q.assign_grad(grad_q)
            self.k.assign_grad(grad_k)
            self.v.assign_grad(grad_v)
            return None

