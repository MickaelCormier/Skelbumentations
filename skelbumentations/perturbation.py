from typing import List, Tuple, Union

import numpy as np

from . import random_utils
from .transform import BaseTransform


class Jittering(BaseTransform):
    """Apply Jittering to one random keypoint at each frame.
    The keypoint is moved in a random direction with a random distance.
    The displacement is restricted to the minimal bounding-box of the skeleton.
    Only works with 2D coordinates.

    Args:
        angles (Tuple[int, int], optional): The range of angles to choose from randomly. Defaults to (0,360).
        distance (Tuple[float, float], optional): The range of distances to choose from randomly. Defaults to (0,1).
        keypoint: (Union[None, int], optional): The index of the keypoint to jitter.
        None means a random keypoint is choosen at every frame. Defaults to None.
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(
        self,
        angles: Tuple[int, int] = (0, 360),
        distance: Tuple[float, float] = (0, 1),
        keypoint: Union[None, int] = None,
        always_apply: bool = False,
        p: float = 1,
    ):
        if not angles[0] <= angles[1]:
            raise ValueError("Invalid angle range. First angle has to be smaller or equal than the second angle")
        if not distance[0] <= distance[1]:
            raise ValueError(
                "Invalid distance range. First distance has to be smaller or equal than the second distance"
            )
        self.radians = np.deg2rad(angles)
        self.distance = distance
        self.keypoint = keypoint
        super().__init__(always_apply, p)

    def apply(self, **data):
        assert data["keypoints"].shape[2] == 2, "Jittering only works with 2D coordinates"
        keypoints = np.transpose(data["keypoints"], (2, 0, 1))  # C T V
        valid = np.invert(data["invalid"])

        # Calculate bounding boxes
        bottom_left_bb = np.min(keypoints, axis=2, where=valid, initial=np.inf)
        bottom_left_bb[bottom_left_bb == np.inf] = 0
        top_right_bb = np.max(keypoints, axis=2, where=valid, initial=np.NINF)
        top_right_bb[top_right_bb == np.NINF] = 0

        for i in range(data["keypoints"].shape[0]):
            if self.keypoint:
                keypoint_id = self.keypoint
            else:
                keypoint_id = random_utils.randint(0, data["keypoints"].shape[1])
            if data["invalid"][i, keypoint_id]:  # Skip invalid keypoints
                continue

            keypoint = data["keypoints"][i, keypoint_id]
            direction = self._get_random_direction()
            distance = self._get_random_distance()
            new_kp = keypoint + distance * direction
            new_kp = self._clip_bb(new_kp, bottom_left_bb[:, i], top_right_bb[:, i], direction)
            data["keypoints"][i, keypoint_id] = new_kp
        return data

    def _get_random_direction(self):
        radian = random_utils.uniform(self.radians[0], self.radians[1])
        return np.array([np.cos(radian), np.sin(radian)])

    def _get_random_distance(self) -> float:
        return random_utils.uniform(self.distance[0], self.distance[1])

    def _clip_bb(self, keypoint, bottom_left, top_right, direction):
        # Check if keypoint is outside of bb
        if (keypoint < bottom_left).any() or (keypoint > top_right).any():
            db = keypoint - bottom_left
            dt = keypoint - top_right
            db[db > 0] = 0
            dt[dt < 0] = 0
            temp_direction = direction.copy()
            temp_direction[temp_direction == 0.0] = np.inf
            db /= temp_direction
            dt /= temp_direction
            d = np.concatenate((db, dt))
            dm = np.max(d)  # Distance in direction keypoint is outside of bb
            keypoint = keypoint - dm * direction
        return keypoint


class MovePerturbation(BaseTransform):
    """Apply move pertubations to keypoints. The distance in each dimension is determined
    by the normal distribution.

    Args:
        variance (float): the variance of the distance.
        joints (list(int)): the keypoints the transform should be applied on. None if all. Defaults to None.
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(
        self, variance: float, joints: Union[list[int], None] = None, always_apply: bool = False, p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.variance = variance
        self.joints = joints

    def apply(self, **data):
        shape = data["keypoints"].shape
        if self.joints:
            shape = list(shape)
            shape[1] = len(self.joints)
        d = random_utils.normal(0, self.variance, shape)
        if self.joints:
            d[data["invalid"][:, self.joints]] = 0.0
            data["keypoints"][:, self.joints] += d
        else:
            d[data["invalid"]] = 0.0
            data["keypoints"] += d
        return data


class SwapPerturbation(BaseTransform):
    """Apply swap pertubations to keypoints. Two randomly selected keypoints are swapped.

    Args:
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, **data):
        length = data["keypoints"].shape[1]
        i1 = i2 = 0
        while i1 == i2:
            [i1, i2] = random_utils.randint(0, length, 2)
        temp = data["keypoints"][:, i1].copy()
        data["keypoints"][:, i1] = data["keypoints"][:, i2]
        data["keypoints"][:, i2] = temp

        return data


class MirrorPerturbation(BaseTransform):
    """Apply mirror pertubation.

    Args:
        opposite_points (list[list[int]]): Pairs of opposite keypoint indices that should be swapped.
        p (float, optional): probability of applying the transform. Defaults to 1.0.
    """

    def __init__(
        self,
        opposite_points: list[list[int]],
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.opposite_points = opposite_points

    def apply(self, **data):
        for [p, q] in self.opposite_points:
            temp = data["keypoints"][:, p].copy()
            data["keypoints"][:, p] = data["keypoints"][:, q]
            data["keypoints"][:, q] = temp

        return data
