import random
from typing import Union

import numpy as np
from scipy import interpolate

from . import random_utils
from .transform import BaseTransform


class RandomOcclusion(BaseTransform):
    """Set random keypoints througout the sequence invalid.

    Args:
        chance (float): the probability of setting a keypoint to invalid or a list of probabilities for each keypoint.
        p (float): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, chance: Union[float, list[float]], always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.chance = chance

    def apply(self, **data):
        num_joints = data["invalid"].shape[1]
        occluded_joints = []
        for i in range(num_joints):
            chance = self.chance if type(self.chance) == float else self.chance[i]
            if random_utils.uniform() < chance:
                occluded_joints.append(i)
        data["invalid"][:, occluded_joints] = True
        return data


class WholeOcclusion(BaseTransform):
    """Set all keypoints to invalid.

    Args:
        p (float): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, **data):
        data["invalid"][:] = True
        return data


class SpecificOcclusion(BaseTransform):
    """Set the specified keypoints to invalid.

    Args:
        joints (list[int]): The indeces of joints that should be set to invalid.
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, joints: list[int], always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.joints = joints

    def apply(self, **data):
        data["invalid"][:, self.joints] = True
        return data


class InterpolateOcclusions(BaseTransform):
    """Interpolates occluded keypoints with linear interpolation.
    Only interpolates between two valid keypoints (no extrapolating).

    Args:
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_frames = 25

    def apply(self, **data):
        # T V C
        for k in range(data["keypoints"].shape[1]):
            kp = data["keypoints"][:, k]
            length = data["keypoints"].shape[0]
            invalid = data["invalid"][:, k]
            valid = np.logical_not(invalid)
            valid_ids = np.arange(length)[valid]
            # At least two valid points needed to interpolate in between
            if valid_ids.size < 2:
                continue
            valid_kps = kp[valid]
            f = interpolate.interp1d(valid_ids, valid_kps, axis=0)
            min_valid_id = np.min(valid_ids)
            max_valid_id = np.max(valid_ids)

            interpolate_map = invalid.copy()

            # Prevent extrapolating
            interpolate_map[:min_valid_id] = False
            interpolate_map[max_valid_id:] = False
            interpolate_ids = np.arange(length)[interpolate_map]

            data["keypoints"][interpolate_map, k] = f(interpolate_ids)
            data["invalid"][interpolate_map, k] = False

        return data
