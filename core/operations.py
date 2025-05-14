import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from numpy.lib.stride_tricks import as_strided

class FastConvolver:
    """
    Implements efficient 2D convolution using the im2col approach.

    This class transforms input feature maps into column format to enable
    matrix multiplication-based convolution, significantly improving performance
    compared to naive implementations.

    Features:
    - Supports multi-channel inputs and multiple filters.
    - Allows custom stride and padding configurations.
    - Provides utility functions for im2col and col2im transformations.

    Example usage:
        convolver = FastConvolver()
        output, col_matrix = convolver.convolve(input_data, kernels, stride=1, padding=1)

    """
    def __init__(self):
        pass

    def _im2col(self, input_data, kernel_shape, stride=1):
        """
                Converts an input batch into a columnized matrix for efficient convolution.

                This function extracts patches from the input tensor and flattens them into
                rows, enabling efficient matrix multiplication.

                Parameters:
                    input_data (numpy.ndarray): Padded input data of shape (B, C, H, W).
                    kernel_shape (tuple): Tuple (C, H_k, W_k) describing the kernel dimensions.
                    stride (int): Stride of the convolution (applied to height & width).

                Returns:
                    col_matrix (numpy.ndarray): 2D array of shape (B * H_out * W_out, C * H_k * W_k)
                        where each row corresponds to a flattened receptive field.

                Notes:
                    - Assumes input_data is already padded.
                    - Uses `sliding_window_view` for efficient patch extraction.
                """
        B, C, H, W = input_data.shape
        # kernel_shape is expected to be (C, H_k, W_k)
        _, H_k, W_k = kernel_shape

        # Extract sliding windows. This returns an array of shape:
        # (B, C, H - H_k + 1, W - W_k + 1, H_k, W_k)
        windows = sliding_window_view(input_data, (H_k, W_k), axis=(2, 3))

        # Apply the stride by slicing the spatial dimensions:
        windows = windows[:, :, ::stride, ::stride, :, :]
        # Now, windows.shape is (B, C, H_out, W_out, H_k, W_k) where:
        # H_out = (H - H_k + 1) // stride  and  W_out = (W - W_k + 1) // stride

        B, C, H_out, W_out, H_k, W_k = windows.shape

        # Rearrange axes so that each patch becomes a row:
        # from (B, C, H_out, W_out, H_k, W_k) to (B, H_out, W_out, C, H_k, W_k)
        # and then reshape to (B * H_out * W_out, C * H_k * W_k)
        col_matrix = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * H_out * W_out, C * H_k * W_k)
        return col_matrix

    def _transform_kernels(self, kernels):
        """
        Converts convolution filters into a matrix form for multiplication.

        Parameters:
            kernels (numpy.ndarray): Convolution filters of shape (F, C, H_k, W_k),
                                     where F is the number of filters.

        Returns:
            numpy.ndarray: Reshaped kernel matrix of shape (C * H_k * W_k, F),
                           where each column represents a flattened kernel.

        Notes:
            - This transformation allows convolution to be performed using
              standard matrix multiplication.
        """
        F, C, H_k, W_k = kernels.shape
        # Reshape each kernel to a vector and then transpose so that
        # the kernel matrix is of shape (C*H_k*W_k, F)
        return kernels.reshape(F, C * H_k * W_k).T

    def _col2im(self, result_matrix, B, H_out, W_out, num_filters):
        """
        Reshape the result matrix back to the convolution output dimensions.
        (Note: This version is for the forward pass and only performs reshaping,
         not the full accumulation needed for a backward col2im with overlaps.)

        Parameters:
            result_matrix (numpy.ndarray): Matrix of shape (B * H_out * W_out, num_filters).
            B (int): Batch size.
            H_out (int): Output height.
            W_out (int): Output width.
            num_filters (int): Number of filters (output channels).

        Returns:
            numpy.ndarray: Convolution output of shape (B, num_filters, H_out, W_out).
        """
        return result_matrix.reshape(B, H_out, W_out, num_filters).transpose(0, 3, 1, 2)
    
    def col2im_accumulation(self,dX_col, input_shape, filter_height, filter_width, stride, padding):
        """
        Reconstructs the gradient tensor by summing overlapping regions.

        Used in backpropagation to redistribute gradients from the im2col representation
        back to the original input shape.

        Parameters:
            dX_col (numpy.ndarray): 2D array of shape (B * H_out * W_out, C * filter_height * filter_width),
                                    representing gradient patches.
            input_shape (tuple): Original input shape (B, C, H, W).
            filter_height (int): Height of the convolution filter.
            filter_width (int): Width of the convolution filter.
            stride (int): Stride used in the forward pass.
            padding (int): Amount of zero-padding applied to the input.

        Returns:
            numpy.ndarray: Reconstructed gradient of shape (B, C, H + 2*padding, W + 2*padding).

        Notes:
            - Accumulates overlapping regions correctly.
            - This version assumes simple summation without additional normalization.
        """
        B, C, H, W = input_shape
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding

        # Calculate output spatial dims for padded input:
        H_out = (H_padded - filter_height) // stride + 1
        W_out = (W_padded - filter_width) // stride + 1

        # Initialize the gradient array for padded input.
        dInput_padded = np.zeros((B, C, H_padded, W_padded))

        # Reshape dX_col into (B, H_out, W_out, C, filter_height, filter_width)
        dX_col_reshaped = dX_col.reshape(B, H_out, W_out, C, filter_height, filter_width)
        # Permute to (B, C, H_out, W_out, filter_height, filter_width)
        dX_col_reshaped = dX_col_reshaped.transpose(0, 3, 1, 2, 4, 5)

        # Accumulate gradients into the padded input.
        for i in range(filter_height):
            for j in range(filter_width):
                dInput_padded[:, :, i: i + stride * H_out: stride, j: j + stride * W_out: stride] += dX_col_reshaped[:, :, :, :, i, j]

        return dInput_padded

    def convolve(self, input_data, kernels, stride=1, padding=0):
        """
        Performs 2D convolution using the im2col approach.

        This method extracts patches from the input feature map,
        converts them into a column matrix, and performs convolution using
        matrix multiplication.

        Parameters:
            input_data (numpy.ndarray): Input tensor of shape (B, C, H, W).
            kernels (numpy.ndarray): Filters of shape (F, C, H_k, W_k),
                                     where F is the number of filters.
            stride (int): Stride value for convolution.
            padding (int): Amount of zero-padding applied before convolution.

        Returns:
            conv_output (numpy.ndarray): Output tensor of shape (B, F, H_out, W_out).
            col_matrix (numpy.ndarray): Columnized patches from the padded input.

        Notes:
            - Uses `im2col` for efficient patch extraction.
            - Output dimensions are computed as:
                H_out = (H + 2 * padding - H_k) // stride + 1
                W_out = (W + 2 * padding - W_k) // stride + 1
        """
        B, C, H, W = input_data.shape
        F, C, H_k, W_k = kernels.shape

        # Compute output dimensions based on padded input size.
        H_out = (H + 2 * padding - H_k) // stride + 1
        W_out = (W + 2 * padding - W_k) // stride + 1

        # Pad the input (only on spatial dimensions)
        padded_input = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

        # Convert the padded input into columns
        col_matrix = self._im2col(padded_input, kernel_shape=(C, H_k, W_k), stride=stride)

        # Transform kernels into a matrix for multiplication
        kernel_matrix = self._transform_kernels(kernels)

        # Matrix multiplication:
        #   (B*H_out*W_out, C*H_k*W_k) @ (C*H_k*W_k, F) -> (B*H_out*W_out, F)
        result_matrix = col_matrix @ kernel_matrix

        # Reshape the result into (B, F, H_out, W_out)
        conv_output = self._col2im(result_matrix, B, H_out, W_out, F)
        return conv_output, col_matrix


###########################################################################################
##
##                     WINOGRAD CONVOLUTION
##
############################################################################################

'''
#STEPS :
to calc F(rxr,mxm) where r is output size and m is filter size

get the input shape (B, C, H, W)
get the kernel shape (F, C, H_k, W_k)
extract the input tiles 4x4 (m-r+1) from the input tensor

for each filter:
    calc the transformed kernel tile (G * kernel_tile * GT)
for each image in batch:
 for each filter:
    for each tile
        - apply the B matrix to the input tile (B * input_tile * BT) [pre defined additions]
        - elementwise multiply the transformed input tile with the transformed kernel tile
        - apply the A matrix to the result (A * result * AT) [pre defined additions]

'''

# def winograd_kernel_transform_manual(k):
#     #k is 3x3
#     output = np.zeros((4, 4), dtype=k.dtype)
#     temp = np.zeros((4, 3), dtype=k.dtype)
#     temp[0,0] = k[0,0]
#     temp[0,1] = k[0,1]
#     temp[0,2] = k[0,2]
#     temp[1,0] = 0.5*(k[0,0] + k[1,0] + k[2,0])
#     temp[1,1] = 0.5*(k[0,1] + k[1,1] + k[2,1])
#     temp[1,2] = 0.5*(k[0,2] + k[1,2] + k[2,2])
#     temp[2,0] = 0.5*(k[0,0] - k[1,0] + k[2,0])
#     temp[2,1] = 0.5*(k[0,1] - k[1,1] + k[2,1])
#     temp[2,2] = 0.5*(k[0,2] - k[1,2] + k[2,2])
#     temp[3,0] = k[2,0]
#     temp[3,1] = k[2,1]
#     temp[3,2] = k[2,2]

#     output[0,0] = temp[0,0]
#     output[0,1] = 0.5*(temp[0,0]+temp[0,1]+temp[0,2])
#     output[0,2] = 0.5*(temp[0,0]-temp[0,1]+temp[0,2])
#     output[0,3] = temp[0,2]
#     output[1,0] = temp[1,0]
#     output[1,1] = 0.5*(temp[1,0]+temp[1,1]+temp[1,2])
#     output[1,2] = 0.5*(temp[1,0]-temp[1,1]+temp[1,2])
#     output[1,3] = temp[1,2]
#     output[2,0] = temp[2,0]
#     output[2,1] = 0.5*(temp[2,0]+temp[2,1]+temp[2,2])
#     output[2,2] = 0.5*(temp[2,0]-temp[2,1]+temp[2,2])
#     output[2,3] = temp[2,2]
#     output[3,0] = temp[3,0]
#     output[3,1] = 0.5*(temp[3,0]+temp[3,1]+temp[3,2])
#     output[3,2] = 0.5*(temp[3,0]-temp[3,1]+temp[3,2])
#     output[3,3] = temp[3,2]

#     return output

    
# def winograd_output_transform_manual(v):
#     """
#     Manual output transform using Winograd F(2x2, 3x3) A^T matrix.
#     """
#     m00 = v[0,0] + v[0,1] + v[0,2]
#     m01 = v[0,1] - v[0,2] - v[0,3]
#     m10 = v[1,0] + v[1,1] + v[1,2]
#     m11 = v[1,1] - v[1,2] - v[1,3]
#     m20 = v[2,0] + v[2,1] + v[2,2]
#     m21 = v[2,1] - v[2,2] - v[2,3]
#     m30 = v[3,0] + v[3,1] + v[3,2]
#     m31 = v[3,1] - v[3,2] - v[3,3]

#     o00 = m00 + m10 + m20
#     o01 = m01 + m11 + m21
#     o10 = m10 - m20 - m30
#     o11 = m11 - m21 - m31

#     return np.array([[o00, o01],
#                      [o10, o11]], dtype=v.dtype)
'''
B I B.T 
G k G.T
z = t^ * k^
A z A.T




'''

class WinogradConv:
    """
    Implements the Winograd convolution algorithm for efficient 2D convolution.

    This class provides methods to perform Winograd convolution using the F(2x2, 3x3) algorithm,
    which reduces the number of multiplications required for convolution operations.

    Example usage:
        conv = WinogradConv()
        output = conv.forward(input_data, kernels)

    """

    def __init__(self):
        pass

# --- Optimized Winograd Transformations ---
    def winograd_input_transform_manual(self, d):
        assert d.shape == (4, 4), "Input must be 4x4"
        temp = np.zeros_like(d)
        temp[:, 0] = d[:, 0] - d[:, 2]
        temp[:, 1] = d[:, 1] + d[:, 2]
        temp[:, 2] = d[:, 2] - d[:, 1]
        temp[:, 3] = d[:, 1] - d[:, 3]

        V = np.zeros_like(d)
        V[0, :] = temp[0, :] - temp[2, :]
        V[1, :] = temp[1, :] + temp[2, :]
        V[2, :] = temp[2, :] - temp[1, :]
        V[3, :] = temp[1, :] - temp[3, :]
        return V

    def winograd_kernel_transform_manual(self,k):
        assert k.shape == (3, 3), "Kernel must be 3x3"
        temp = np.zeros((4, 3), dtype=k.dtype)
        temp[0, :] = k[0, :]
        temp[3, :] = k[2, :]
        col_sums = 0.5 * (k[0, :] + k[1, :] + k[2, :])
        col_diffs = 0.5 * (k[0, :] - k[1, :] + k[2, :])
        temp[1, :] = col_sums
        temp[2, :] = col_diffs

        output = np.zeros((4, 4), dtype=k.dtype)
        output[:, 0] = temp[:, 0]
        output[:, 3] = temp[:, 2]
        output[:, 1] = 0.5 * (temp[:, 0] + temp[:, 1] + temp[:, 2])
        output[:, 2] = 0.5 * (temp[:, 0] - temp[:, 1] + temp[:, 2])
        return output

    def winograd_output_transform_manual(self,v):
        m = np.zeros((4, 2), dtype=v.dtype)
        m[:, 0] = v[:, 0] + v[:, 1] + v[:, 2]
        m[:, 1] = v[:, 1] - v[:, 2] - v[:, 3]

        o00 = m[0, 0] + m[1, 0] + m[2, 0]
        o01 = m[0, 1] + m[1, 1] + m[2, 1]
        o10 = m[1, 0] - m[2, 0] - m[3, 0]
        o11 = m[1, 1] - m[2, 1] - m[3, 1]
        return np.array([[o00, o01], [o10, o11]], dtype=v.dtype)

    # --- Helper: Extract 4x4 Tiles with Stride 2 ---
    def extract_tiles(self,image, tile_size=4, stride=2):
        #stride = 2 is used for f(2x2, 3x3) winograd conv to have a r-1 overlap
        # image.shape = (C, H, W)
        C, H, W = image.shape
        num_h = (H - tile_size) // stride + 1
        num_w = (W - tile_size) // stride + 1

        shape = (C, num_h, num_w, tile_size, tile_size)
        strides = image.strides[:-2] + (stride * image.strides[-2], stride * image.strides[-1]) + image.strides[-2:]
        tiles = as_strided(image, shape=shape, strides=strides)
        tiles = tiles.reshape(C, -1, tile_size, tile_size)
        # tiles.shape = (C, num_tiles_h * num_tiles_w, 4, 4)
        return tiles

    # --- Main Winograd Convolution Function ---
    def convolve(self,X, W,stride=1, padding=0):
        """
        Winograd convolution (F(2×2, 3×3)).

        X: input tensor, shape (N, C, H, W)
        W: filter tensor, shape (K, C, 3, 3)
        returns Y: output tensor, shape (N, K, H_out, W_out)
        """
        

        # --- unpack shapes ---
        N, C, H, W_x = X.shape       # N=batch size, C=channels, H×W_x=spatial dims
        K, C_w, R, S = W.shape       # K=number of filters, C_w must match C, R=S=3

        # sanity checks
        assert (R, S) == (3, 3), "Only 3×3 kernels supported"
        assert C == C_w,           "Mismatch between input and filter channels"
        assert H >= 4 and W_x >= 4, "Input must be at least 4x4"
        assert H % 2 == 0 and W_x % 2 == 0, "Input height and width must be even"
        assert stride == 1, "Stride must be 1 for Winograd convolution"

        if padding > 0:
            
            X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        # Winograd tile settings:

        m = 2                       # we want 2×2 output tiles
        alpha = m + R - 1           # transform tile is 4×4 (since 2 + 3 - 1 = 4)
        stride = 2                  # move tiles by 2 pixels each time

        # --- Step 1: transform all filters into Winograd domain ---
        # U[k, c, i, j] will hold each filter’s channel-transformed 4×4 tile
        U = np.empty((K, C, alpha, alpha), dtype=W.dtype)
        for k in range(K):
            for c in range(C):
                # take the 3×3 weights for filter k, channel c
                # apply G · W · Gᵀ to get a 4×4 Winograd-domain patch
                U[k, c] = self.winograd_kernel_transform_manual(W[k, c])

        # figure out how many 4×4 tiles fit across height/width
        n_tiles_h = (H - alpha) // stride + 1
        n_tiles_w = (W_x - alpha) // stride + 1
        P = n_tiles_h * n_tiles_w    # total number of tiles per image

        # prepare the output array: each 2×2 tile will be placed back into spatial grid
        Y = np.zeros((N, K, n_tiles_h * m, n_tiles_w * m), dtype=X.dtype)

        # --- Step 2: loop over each image in the batch ---
        for n in range(N):
            # slice out all overlapping 4×4 patches from each channel of image n
            # result shape: (C, P, 4, 4)
            tiles = self.extract_tiles(X[n], tile_size=alpha, stride=stride)

            # transform each 4×4 input patch into Winograd domain: Bᵀ · d · B
            V = np.empty((C, P, alpha, alpha), dtype=X.dtype)
            for c in range(C):
                for p in range(P):
                    V[c, p] = self.winograd_input_transform_manual(tiles[c, p])

            # --- Step 3: perform the “convolution” as channel-wise GEMMs ---
            # M[i, j, k, p] will hold the sum over channels for position (i,j)
            M = np.empty((alpha, alpha, K, P), dtype=X.dtype)
            for i in range(alpha):
                for j in range(alpha):
                    # build a (K × C) matrix: filters × channels at coord (i,j)
                    U_slice = U[:, :, i, j]
                    # build a (C × P) matrix: channels × tiles at coord (i,j)
                    V_slice = V[:, :, i, j]
                    # multiply to sum over C in one go → get (K × P) results
                    M[i, j] = U_slice @ V_slice

            # --- Step 4: invert the Winograd transform for each tile & filter ---
            for p in range(P):
                # compute the top-left corner in output feature map
                row = (p // n_tiles_w) * m
                col = (p %  n_tiles_w) * m
                for k in range(K):
                    # M[:, :, k, p] is the 4×4 Winograd result for filter k, tile p
                    # apply Aᵀ · M · A to get the final 2×2 output patch
                    y_patch = self.winograd_output_transform_manual(M[:, :, k, p])
                    # write that 2×2 patch back into Y
                    Y[n, k, row:row + m, col:col + m] = y_patch

        return Y

