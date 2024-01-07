import random

import numpy as np


class BaseTransform:
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, force_apply: bool = False, **data):
        if (random.random() < self.p) or self.always_apply or force_apply:
            return self.apply(**data)
        else:
            return data

    def apply(self, **data):
        raise NotImplementedError


class HorizontalFlip(BaseTransform):
    """Flip the skeletons around the y-axis

    Args:
        flip_axis (float): The x position of the y-axis to flip around.
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, flip_axis: float, always_apply: bool = False, p: float = 1):
        self.flip_axis = flip_axis
        super().__init__(always_apply, p)

    def apply(self, **data):
        data["keypoints"][:, :, 0] = 2 * self.flip_axis - data["keypoints"][:, :, 0]
        return data
