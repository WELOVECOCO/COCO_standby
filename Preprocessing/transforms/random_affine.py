
import numpy as np
from core.tensor import Tensor
from scipy.ndimage import affine_transform


class RandomAffine:
    def __init__(self, degrees=0, translate=None, scale=None, shear=0, mode='constant', cval=0):
        """
        Args:
            degrees (float or tuple): Rotation range.
            translate (tuple): Maximum horizontal and vertical translations (as fractions).
            scale (tuple): Scaling factor range.
            shear (float or tuple): Shear angle range in degrees.
            mode (str): How to fill points outside boundaries.
            cval (int/float): Fill value.
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

        self.translate = translate
        self.scale = scale if scale is not None else (1.0, 1.0)
        if isinstance(shear, (int, float)):
            self.shear = (-shear, shear)
        else:
            self.shear = shear
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

        if arr.ndim == 3:
            H, W, _ = arr.shape
            center = np.array([H / 2, W / 2])
        elif arr.ndim == 2:
            H, W = arr.shape
            center = np.array([H / 2, W / 2])
        else:
            raise ValueError("Unsupported image shape for RandomAffine")

        angle = np.random.uniform(*self.degrees)
        if self.translate:
            max_dx = self.translate[0] * W
            max_dy = self.translate[1] * H
            translations = (np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)
        scale_factor = np.random.uniform(*self.scale) if self.scale != (1.0, 1.0) else 1.0
        shear_angle = np.random.uniform(*self.shear)

        angle_rad = np.deg2rad(angle)
        shear_rad = np.deg2rad(shear_angle)
        rotation = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                             [np.sin(angle_rad), np.cos(angle_rad)]])
        shear_mat = np.array([[1, np.tan(shear_rad)],
                              [0, 1]])
        scale_mat = np.array([[scale_factor, 0],
                              [0, scale_factor]])
        M = rotation @ shear_mat @ scale_mat
        offset = center - M @ center + np.array(translations)

        if arr.ndim == 3:
            H, W, C = arr.shape
            transformed = np.empty_like(arr)
            for c in range(C):
                transformed[..., c] = affine_transform(arr[..., c], M, offset=offset, mode=self.mode, cval=self.cval)
        else:
            transformed = affine_transform(arr, M, offset=offset, mode=self.mode, cval=self.cval)

        return Tensor(transformed, requires_grad=requires_grad) if is_tensor else transformed
