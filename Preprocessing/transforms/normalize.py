
import numpy as np
from core.tensor import Tensor


class Normalize:
    def __init__(self, mean, std):
        """
        Args:
            mean (sequence): Means for each channel.
            std (sequence): Standard deviations for each channel.
        """
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("Normalize transform expects a Tensor as input.")

        img = tensor.data

        if img.ndim == 3:
            # If image is channel-first: (C, H, W)
            if img.shape[0] == len(self.mean):
                norm_data = (img - self.mean[:, None, None]) / self.std[:, None, None]
            # If image is channel-last: (H, W, C)
            elif img.shape[-1] == len(self.mean):
                norm_data = (img - self.mean[None, None, :]) / self.std[None, None, :]
            else:
                raise ValueError("Unexpected number of channels in image")
        else:
            raise ValueError("Normalize transform expects a 3D image")
        return Tensor(norm_data, requires_grad=tensor.requires_grad)
