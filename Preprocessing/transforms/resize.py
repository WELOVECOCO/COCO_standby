
import numpy as np
from core.tensor import Tensor

class Resize:
    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size as (new_height, new_width).
        """
        self.size = size

    def __call__(self, image):
        # Determine if the image is a Tensor or numpy array.
        if isinstance(image, Tensor):
            arr = image.data
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image
            is_tensor = False

        # Handle both 3D (color) and 2D (grayscale) images.
        if arr.ndim == 3:
            # Assume channel-first if first dimension is 1,3,4; otherwise, channel-last.
            if arr.shape[0] in (1, 3, 4):
                new_arr = self._resize_channel_first(arr)
            else:
                new_arr = self._resize_channel_last(arr)
        elif arr.ndim == 2:
            new_arr = self._resize_channel(arr, *self.size)
        else:
            raise ValueError("Unsupported image shape for Resize transform")

        return Tensor(new_arr, requires_grad=requires_grad) if is_tensor else new_arr

    def _resize_channel(self, channel, new_h, new_w):
        H, W = channel.shape
        row_idx = (np.linspace(0, H, new_h, endpoint=False)).astype(np.int32)
        col_idx = (np.linspace(0, W, new_w, endpoint=False)).astype(np.int32)
        return channel[row_idx[:, None], col_idx]

    def _resize_channel_first(self, arr):
        channels, H, W = arr.shape
        new_h, new_w = self.size
        new_arr = np.empty((channels, new_h, new_w), dtype=arr.dtype)
        for c in range(channels):
            new_arr[c] = self._resize_channel(arr[c], new_h, new_w)
        return new_arr

    def _resize_channel_last(self, arr):
        H, W, C = arr.shape
        new_h, new_w = self.size
        new_arr = np.empty((new_h, new_w, C), dtype=arr.dtype)
        for c in range(C):
            new_arr[..., c] = self._resize_channel(arr[..., c], new_h, new_w)
        return new_arr
