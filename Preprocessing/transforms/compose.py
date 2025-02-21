
import numpy as np

class Compose:
    def __init__(self, transforms):
        """
        Args:
            transforms (list): List of transform objects to be applied sequentially.
        """
        self.transforms = transforms

    def __call__(self, images):
        """
        Applies the transformation pipeline. If the input is a batch (numpy array with 4 dimensions
        or a list of images), each sample is processed individually.

        Args:
            images: A single image or a batch of images.
        Returns:
            Transformed image or list of transformed images.
        """
        # Check for a batch represented as a numpy array with shape (N, H, W, C) or (N, C, H, W)
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return [self._apply_transforms(img) for img in images]
        # If the input is a list, assume a list of images.
        elif isinstance(images, list):
            return [self._apply_transforms(img) for img in images]
        else:
            # Single image case.
            return self._apply_transforms(images)

    def _apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
