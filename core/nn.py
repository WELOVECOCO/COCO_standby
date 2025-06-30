
from core.new_tensor import Tensor
import numpy as np
# from core.config import Config

TEST = False

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
                                                #
    def backward(self,grad):
        pass

    def initialize_weights(self, shape,initialize_type):
        pass



##############################################################################################
#                                                                                            #
#                                                                                            #
#                                                                                            #
#                   BASIC LAYERS                                                             #
#                                                                                            #
#                                                                                            #
#                                                                                            #
##############################################################################################


class Flatten(Module):
    """
    A layer that flattens all dimensions except the batch dimension.
    This is typically used before feeding the output into a fully connected layer.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, input, **kwargs):
        batch_size = input.shape[0]
        flattened_data = input.reshape(batch_size, -1)  # Flatten all dimensions except batch

        return flattened_data

class Linear(Layer):
    """
    A fully connected linear layer that performs an affine transformation: 
    output = input * weights + bias. Supports different weight initialization methods
    and optional dropout for regularization.
    """
    
    def __init__(self,input_dim, output_dim, initialize_type="random",bias=True):
        """
        Initializes a linear layer with specified weight initialization and optional dropout.
        
        Args:
            shape (tuple): A tuple of (input_dim, output_dim).
            initialize_type (str, optional): Weight initialization method. Defaults to "random".
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__()
       
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
        out = input @ self.weights
        if self.bias_flag:
            out = out + self.bias.reshape(1, -1)

        return out
    

class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, initialize_type="xavier",bias=True):

        super().__init__()
        self.bias_flag = bias
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size)
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

    def __call__(self, input, **kwargs):
       
        b, c, _, _ = input.shape
        F, _, hk, wk = self.weights.data.shape

        
        if self.padding > 0:
            input = input.pad(((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        h_p, w_p = input.shape[2], input.shape[3]

        # Compute output dimensions
        hout = (h_p - hk) // self.stride + 1
        wout = (w_p - wk) // self.stride + 1

        # Strides of input data
        batch_stride, channel_stride, height_stride, width_stride = input.data.strides

        # Shape and strides for as_strided to extract sliding windows
        shape = (b, c, hout, wout, hk, wk)
        strides = (
            batch_stride,
            channel_stride,
            height_stride * self.stride,
            width_stride * self.stride,
            height_stride,
            width_stride,
        )

        
        strided_input = input.as_strided(shape, strides)

        # Rearrange into 2D matrix: (B * hout * wout, C * hk * wk)
        col_matrix = strided_input.transpose(0, 2, 3, 1, 4, 5).reshape(b * hout * wout, c * hk * wk)

        # Flatten the kernels: (F, C * hk * wk) → then transpose → (C * hk * wk, F)
        reshaped_kernel = self.weights.reshape(F, c * hk * wk).T

        # Matrix multiplication → (B * hout * wout, F)
        out = col_matrix @ reshaped_kernel

        # Reshape back to (B, F, hout, wout)
        out_reshaped = out.reshape(b, hout, wout, F).transpose(0, 3, 1, 2)

        # Add bias if enabled
        if self.bias_flag:
            bias_reshaped = self.bias.reshape(1, self.output_channels, 1, 1)
            out_reshaped = out_reshaped + bias_reshaped

        return out_reshaped




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

    def __call__(self, x, **kwargs):
        B, C, H, W = x.shape
        H_k, W_k = self.kernel_size
        stride_h, stride_w = self.stride

        # Apply padding manually
        if self.padding > 0:
            x = x.pad(
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )

        H_padded, W_padded = x.shape[2], x.shape[3]
        H_out = (H_padded - H_k) // stride_h + 1
        W_out = (W_padded - W_k) // stride_w + 1

        # Strides of the input array
        s0, s1, s2, s3 = x.data.strides

        # Compute shape and strides for as_strided
        shape = (B, C, H_out, W_out, H_k, W_k)
        strides = (s0, s1, s2 * stride_h, s3 * stride_w, s2, s3)

        # Get windowed view
        windows = x.as_strided(shape=shape, strides=strides)

        # Perform max pooling
        max_vals = windows.reshape(B, C, H_out, W_out, -1).max(axis=-1)

        return max_vals


class GAP(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, **kwargs):
        return input.mean(axis=(2, 3),keepdims=True)
    


##############################################################################################
#                                                                                            #
#                                                                                            #
#                                                                                            #
#                   ALOT OF NORMALIZATIONS                                                   #
#                                                                                            #
#                                                                                            #
#                                                                                            #
##############################################################################################




class GeneralNorm(Layer):
    def __init__(self, axes, param_shape):
        super().__init__()
        self.axes = axes
        self.gamma = Tensor(np.ones(param_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(param_shape), requires_grad=True)

    def __call__(self, input, **kwargs):
        mean = input.mean(axis=self.axes, keepdims=True)
        var = input.var(axis=self.axes, keepdims=True)
        x_norm = (input - mean) / (var + 1e-5).sqrt()
        return self.gamma * x_norm + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class ConvBatchNorm2D(GeneralNorm):
    def __init__(self, channels):
        super().__init__(axes=(0, 2, 3), param_shape=(1, channels, 1, 1))
        self.eps = Tensor(1e-5, requires_grad=False,name="eps")  # Small constant for numerical stability
        self.running_mean = Tensor(np.zeros((1, channels, 1, 1)), requires_grad=False,name="running_mean")
        self.running_variance = Tensor(np.ones((1, channels, 1, 1)), requires_grad=False,name="running_variance")
        self.betanorm = Tensor(0.9, requires_grad=False,name="betanorm")  # Beta normalization factor
        self.gamma = Tensor(np.ones((1, channels, 1, 1)), requires_grad=True,name="gamma")
        self.beta = Tensor(np.zeros((1, channels, 1, 1)), requires_grad=True,name="beta")

    def __call__(self, input, **kwargs):
        mean = input.mean(axis=(0, 2, 3), keepdims=True)
        var = input.var(axis=(0, 2, 3), keepdims=True)
        # print(f"mean shape: {mean.shape}, var shape: {var.shape}")
        self.running_mean = (self.running_mean * self.betanorm) + (mean * (1 - self.betanorm))
        self.running_variance = (self.running_variance * self.betanorm) + (var * (1 - self.betanorm))
        if TEST==False:
            
            x_norm = (input - mean) / (var + 1e-5).sqrt()
        
        else:
            x_centered = input - self.running_mean
            scale = (self.running_variance + 1e-5).sqrt()
            x_norm = x_centered / scale

        return self.gamma * x_norm + self.beta
    

    def parameters(self):
        return [self.gamma, self.beta]

    def trained_parameters(self):
        return [self.gamma, self.beta, self.running_mean, self.running_variance]


class InstanceNorm2D(GeneralNorm):
    def __init__(self, channels):
        super().__init__(axes=(2, 3), param_shape=(1, channels, 1, 1))

class LayerNorm2D(GeneralNorm):
    def __init__(self, channels):
        super().__init__(axes=(1, 2, 3), param_shape=(1, channels, 1, 1))

class LayerNorm1D(GeneralNorm):
    def __init__(self, dim):
        super().__init__(axes=-1, param_shape=(1, dim))

class LayerNorm3D(GeneralNorm):
    def __init__(self, dim):
        super().__init__(axes=-1, param_shape=(1, 1, dim))


class GroupNorm(Layer):
    def __init__(self, num_channels, num_groups):
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_size = num_channels // num_groups
        self.gamma = Tensor(np.ones((1, num_channels, 1, 1)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_channels, 1, 1)), requires_grad=True)

    def __call__(self, x, **kwargs):
        B, C, H, W = x.shape
        G = self.num_groups
        assert C == self.num_channels

        # Reshape to (B, G, C//G, H, W)
        x_reshaped = x.reshape(B, G, self.group_size, H, W)

        # Compute mean and var across (2, 3, 4)
        mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)

        # Normalize
        x_norm = (x_reshaped - mean) / (var + 1e-5).sqrt()

        # Reshape back to (B, C, H, W)
        x_norm = x_norm.reshape(B, C, H, W)

        
        return self.gamma * x_norm + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]

class rmsnorm(Layer):
    def __init__(self, dim, eps=1e-6):
        #shape is (b,t,dim)
        super().__init__()
        self.dim = dim
        self.eps = eps
        shape = (1,1, dim)
        self.gamma = Tensor(np.ones(shape), requires_grad=True)

    def __call__(self, x, **kwargs):
        mean_square = (x ** 2).mean(axis=-1, keepdims=True)
        x_norm = x / np.sqrt(mean_square + self.eps)
        return self.gamma * x_norm
    
    def parameters(self):
        return [self.gamma]



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

class SelfAttention(Layer):
    def __init__(self, dmodel,d_head=None,masked=False,out_proj=False):
        super().__init__()
        self.dhead = d_head if d_head is not None else dmodel
        self.scale = Tensor(self.dhead, requires_grad=False).sqrt()

        self.q_proj = Linear(dmodel, self.dhead , initialize_type='random')
        self.k_proj = Linear(dmodel, self.dhead , initialize_type='random')
        self.v_proj = Linear(dmodel, self.dhead , initialize_type='random')
        self.out_proj = Linear(dmodel, self.dhead , initialize_type='random') if out_proj else None
        
        self.masked = masked

    
    def __call__(self, input,q=None, k=None, v=None):
        
        if q is None and k is None and v is None:
            q = self.q_proj(input)
            k = self.k_proj(input)
            v = self.v_proj(input)

        qk = q @ k.transpose(0, 2, 1) / self.scale  # Scaled dot-product attention scores (B, T, T)

        if self.masked:
            B, T, _ = qk.shape
            mask = Tensor(np.triu(np.ones((T, T)), k=1)* -1e10, requires_grad=False).reshape(1, T, T) 
            qk = qk + mask  

        attn_weights = qk.softmax(axis=-1)  
        attn_output = attn_weights @ v  # (B, T, dhead)

        if self.out_proj:
            attn_output = self.out_proj(attn_output)
        
        
        return attn_output
        

    




class MultiHeadAttention(Layer):
    def __init__(self, dmodel, n_heads, masked=False):
        super().__init__()
        assert dmodel % n_heads == 0, "dmodel must be divisible by n_heads"
        
        self.dmodel = dmodel
        self.n_heads = n_heads
        self.d_head = dmodel // n_heads
        self.masked = masked
        self.scale = Tensor(self.d_head).sqrt()  # Scale factor for attention scores

        # Single linear layer for q, k, v: (B, T, D) -> (B, T, 3D)
        self.qkv_proj = Linear(dmodel, 3 * dmodel, initialize_type='random')
        
        # Output projection: (B, T, D)
        self.out_proj = Linear(dmodel, dmodel, initialize_type='random')

    def __call__(self, x,q=None, k=None, v=None):
        b, t, _ = x.shape
        
        if q is None and k is None and v is None:
            qkv = self.qkv_proj(x)  # (B, T, 3D)
            q, k, v = qkv.split(indices_or_sections=3, axis=2)  # Each is (B, T, D)

        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        def reshape_heads(tensor):  # (B, T, D) -> (B, n_heads, T, d_head)
            return tensor.reshape(b, t, self.n_heads, self.d_head).transpose(0 ,2, 1, 3)  # (B, n_heads, T, d_head)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

      
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale  # (B, n_heads, T, T)

        if self.masked:
            _, _, T, _ = scores.shape
            mask_data = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e10
            mask = Tensor(mask_data, requires_grad=False, device=x.device).reshape(1, 1, T, T)
            scores = scores + mask

        weights = scores.softmax(axis=-1)  # (B, n_heads, T, T)
        attn = weights @ v  # (B, n_heads, T, d_head)

       
        attn = attn.transpose(0,2,1,3).reshape(b, t, self.dmodel)

      
        return self.out_proj(attn)

        

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x, **kwargs):
        if TEST:
            return x  # No dropout during evaluation
        mask = Tensor(np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p))
        return x * mask




class TransformerEncoderLayer(Layer):
    def __init__(self, dmodel, n_heads, dim_feedforward=2048, dropout=0.1,masked=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(dmodel, n_heads, masked=masked)
        self.linear1 = Linear(dmodel, dim_feedforward, initialize_type='random')
        self.linear2 = Linear(dim_feedforward, dmodel, initialize_type='random')
        self.norm1 = LayerNorm3D(dmodel)
        self.norm2 = LayerNorm3D(dmodel)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        

    def __call__(self, x): #x is of shape (B, T, D)
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)

        return x
    


