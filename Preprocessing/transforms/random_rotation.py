
import numpy as np
from core.tensor import Tensor
from scipy.ndimage import rotate

class RandomRotation:
    def __init__(self, degrees, reshape=False, mode='constant', cval=0):
        """
        Args:
            degrees (float or tuple): Range of degrees for rotation.
            reshape (bool): Whether to reshape the output.
            mode (str): How to fill points outside boundaries.
            cval (int/float): Fill value.
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.reshape = reshape
        self.mode = mode
        self.cval = cval

    def __call__(self, image):
        if isinstance(image, Tensor):
            arr = image.data
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image
            is_tensor = False

        angle = np.random.uniform(*self.degrees)
        rotated = rotate(arr, angle, reshape=self.reshape, mode=self.mode, cval=self.cval)
        return Tensor(rotated, requires_grad=requires_grad) if is_tensor else rotated
