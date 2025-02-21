
import numpy as np
from core.tensor import Tensor


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of flipping the image horizontally.
        """
        self.p = p

    def __call__(self, image):
        if np.random.rand() < self.p:
            if isinstance(image, Tensor):
                arr = image.data
                requires_grad = image.requires_grad
            else:
                arr = image

            if arr.ndim == 3:
                # For channel-first images (C, H, W), flip the width axis (axis=2).
                # For channel-last images (H, W, C), flip the width axis (axis=1).
                if arr.shape[0] in (1, 3, 4):
                    flipped = np.flip(arr, axis=2)
                else:
                    flipped = np.flip(arr, axis=1)
            elif arr.ndim == 2:
                flipped = np.flip(arr, axis=1)
            else:
                raise ValueError("Unsupported image shape for RandomHorizontalFlip")

            return Tensor(flipped, requires_grad=requires_grad) if isinstance(image, Tensor) else flipped
        else:
            return image
