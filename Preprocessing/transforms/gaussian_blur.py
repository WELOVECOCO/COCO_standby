
import numpy as np
from core.tensor import Tensor
from scipy.ndimage import gaussian_filter

class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        """
        Args:
            sigma (tuple): Range for sigma.
        """
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(image, Tensor):
            arr = image.data
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image
            is_tensor = False

        sigma_val = np.random.uniform(*self.sigma)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                blurred = np.empty_like(arr)
                for c in range(arr.shape[0]):
                    blurred[c] = gaussian_filter(arr[c], sigma=sigma_val)
            else:
                blurred = np.empty_like(arr)
                for c in range(arr.shape[2]):
                    blurred[..., c] = gaussian_filter(arr[..., c], sigma=sigma_val)
        elif arr.ndim == 2:
            blurred = gaussian_filter(arr, sigma=sigma_val)
        else:
            raise ValueError("Unsupported image shape for GaussianBlur")
        return Tensor(blurred, requires_grad=requires_grad) if is_tensor else blurred
