
import numpy as np
from core.tensor import Tensor


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        """
        Args:
            p (float): Probability to apply.
            scale (tuple): Proportion range of the erased area.
            ratio (tuple): Aspect ratio range of the erased area.
            value: Fill value.
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image

        if isinstance(image, Tensor):
            arr = image.data
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image
            is_tensor = False

        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                h, w = arr.shape[1:3]
            else:
                h, w = arr.shape[0:2]
        elif arr.ndim == 2:
            h, w = arr.shape
        else:
            raise ValueError("Unsupported image shape for RandomErasing")

        area = h * w
        target_area = np.random.uniform(*self.scale) * area
        aspect_ratio = np.random.uniform(*self.ratio)
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        if h_erase < h and w_erase < w:
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3, 4):
                    i = np.random.randint(0, h - h_erase)
                    j = np.random.randint(0, w - w_erase)
                    arr[:, i:i + h_erase, j:j + w_erase] = self.value
                else:
                    i = np.random.randint(0, h - h_erase)
                    j = np.random.randint(0, w - w_erase)
                    arr[i:i + h_erase, j:j + w_erase, :] = self.value
            elif arr.ndim == 2:
                i = np.random.randint(0, h - h_erase)
                j = np.random.randint(0, w - w_erase)
                arr[i:i + h_erase, j:j + w_erase] = self.value

        return Tensor(arr, requires_grad=requires_grad) if is_tensor else arr
