
import numpy as np
from core.tensor import Tensor

class ToTensor:
    def __init__(self, dtype=np.float32, requires_grad=True):
        self.dtype = dtype
        self.requires_grad = requires_grad

    def __call__(self, image):
        # If the image is already a Tensor, return it.
        if isinstance(image, Tensor):
            return image

        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=self.dtype)
        else:
            image = image.astype(self.dtype)
        return Tensor(image, requires_grad=self.requires_grad)
