
import numpy as np
from core.tensor import Tensor

class RandomCrop:
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple): Desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, image):
        if isinstance(image, Tensor):
            arr = image.data
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image
            is_tensor = False

        h, w = arr.shape[:2]
        new_h, new_w = self.output_size
        if new_h > h or new_w > w:
            raise ValueError("Output size must be smaller than input size")
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        cropped = arr[top:top+new_h, left:left+new_w]
        return Tensor(cropped, requires_grad=requires_grad) if is_tensor else cropped
