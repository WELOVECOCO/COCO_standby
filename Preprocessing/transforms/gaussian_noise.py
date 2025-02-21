
import numpy as np
from core.tensor import Tensor

class GaussianNoise:
    def __init__(self, mean=0.0, std=(0.0, 0.1)):
        """
        Args:
            mean (float): Mean of the noise.
            std (tuple): Range for the standard deviation.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, Tensor):
            arr = image.data.astype(np.float32)
            is_tensor = True
            requires_grad = image.requires_grad
        else:
            arr = image.astype(np.float32)
            is_tensor = False

        noise_std = np.random.uniform(*self.std)
        noise = np.random.normal(self.mean, noise_std, arr.shape)
        noisy = arr + noise
        return Tensor(noisy, requires_grad=requires_grad) if is_tensor else noisy
