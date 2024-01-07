import random
from typing import List, Union

import numpy as np

from . import random_utils


class BaseSelect:
    def __init__(self, transforms, p: float):
        self.transforms = transforms
        self.p = p

    def __call__(self, force_apply: bool = False, **data):
        if not (force_apply or random.random() < self.p):
            return data

        select_ids = self.get_selection(**data)
        select_data = {}
        for key, d in data.items():
            select_data[key] = d[select_ids]

        for t in self.transforms:
            select_data = t(**select_data)

        for key, d in data.items():
            d[select_ids] = select_data[key]

        return data

    def get_selection(self, **data):
        raise NotImplementedError


class SelectRandomFrames(BaseSelect):
    """Randomly select frames. The transforms inside this selection are only applied to this selection.

    Args:
        transforms: list of transformations to apply on selection.
        p (float, optional): probability of applying the transforms. Defaults to 1.0.
        max_num (int, optional): maximum number of frames to select. Defaults to 1.
        min_num (int, optional): minimum number of frames to select. Defaults to 1.
        contiguous (bool, optional): Wether the frames should be selected contiguously. Defaults to True.
    """

    def __init__(
        self,
        transforms,
        p: float = 1.0,
        max_num: int = 1,
        min_num: int = 1,
        contiguous: bool = True,
    ):
        super().__init__(transforms, p)
        assert max_num >= min_num, "max_num cannot be smaller than min_num"
        self.min_num = min_num
        self.max_num = max_num
        self.contiguous = contiguous

    def get_selection(self, **data):
        length = len(data["keypoints"])
        size = random_utils.randint(self.min_num, self.max_num + 1)
        if self.contiguous:
            start = random_utils.randint(0, length - size + 1)
            end = start + size
            select_ids = np.arange(start, end)
        else:
            select_ids = random_utils.choice(length, size=size, replace=False)

        return select_ids


class SelectFrames(BaseSelect):
    """Select specific frames. The transforms inside this selection are only applied to this selection.

    Args:
        transforms: list of transformations to apply on selection.
        frames (list[int]): the indices of the frames to be selected.
        p (float, optional): probability of applying the transforms. Defaults to 1.0.
    """

    def __init__(self, transforms, frames: List[int], p: float = 1.0):
        super().__init__(transforms, p)
        self.frames = frames

    def get_selection(self, **data):
        length = len(data["keypoints"])
        assert all(f in range(length) for f in self.frames), "A frame number is out of range"
        return self.frames


class SelectRandomWithBorder(BaseSelect):
    """Randomly select frames with addtional border frames. The transforms inside this selection are only applied to this selection.

    Args:
        inner_transforms: list of transformations to apply on inner selection.
        transforms: list of transformations to apply on inner selection and border after the inner transforms.
        p (float, optional): probability of applying the transforms. Defaults to 1.0.
        max_num (int, optional): maximum number of frames to select. Defaults to 1.
        min_num (int, optional): minimum number of frames to select. Defaults to 1.
        border_num (int, optional): number of border frames on each side. Defaults to 1.
    """

    def __init__(
        self,
        inner_transforms,
        transforms,
        p: float = 1.0,
        min_num: int = 1,
        max_num: int = 25,
        border_num=1,
    ):
        inner_size = random_utils.randint(min_num, max_num + 1)
        self.size = inner_size + 2 * border_num
        self.border_num = border_num
        inner_frames = np.arange(border_num, border_num + inner_size)
        combined_transforms = [SelectFrames(inner_transforms, inner_frames, p=1.0)] + transforms
        super().__init__(combined_transforms, p)

    def get_selection(self, **data):
        length = len(data["keypoints"])
        start = random_utils.randint(0, length - self.size + 1)
        end = start + self.size
        select_ids = np.arange(start, end)

        return select_ids


class SelectHighMovement(BaseSelect):
    """Select frames with the highest mean keypoint movement. Keypoint movement is the distance of a keypoint between two frames.

    Args:
        transforms: list of transformations to apply on selection.
        p (float, optional): probability of applying the transforms. Defaults to 1.0.
        max_num (int, optional): maximum number of frames to select. Defaults to 20.
        min_num (int, optional): minimum number of frames to select. Defaults to 1.
        part (list[int] | None, optional): keypoints indices of keypoints to include in the movement calculation.
            None means all the keypoints are used. Defaults to None.
        contiguous (bool, optional): Wether the frames should be selected contiguously.
            If so, the contiguous frames with the highest keypoint movement are selected. Defaults to True.
    """

    def __init__(
        self,
        transforms,
        p: float = 1.0,
        min_num: int = 1,
        max_num: int = 20,
        part: Union[List[int], None] = None,
        contiguous: bool = True,
    ):
        super().__init__(transforms, p)
        self.size = random_utils.randint(min_num, max_num + 1)
        self.part = part
        self.contiguous = contiguous

    def get_selection(self, **data):
        keypoints = data["keypoints"]
        invalid = data["invalid"]
        length = len(keypoints)
        if self.part:
            keypoints = keypoints[:, self.part]
            invalid = invalid[:, self.part]
        movement = keypoints[1:] - keypoints[:-1]
        movement = np.linalg.norm(movement, axis=2)
        for f, frame in enumerate(movement):
            for k, kp in enumerate(frame):
                if invalid[f, k] or invalid[f + 1, k]:
                    movement[f, k] = 0.0
        movement = np.mean(movement, axis=1)
        if self.contiguous:
            max_sum = 0
            max_index = 0
            for i in range(length - self.size):
                sum = np.sum(movement[i : i + self.size])
                if sum > max_sum:
                    max_sum = sum
                    max_index = i
            ids = np.arange(start=max_index, stop=max_index + self.size)
        else:
            ids = np.argsort(movement)[-self.size :]

        return ids
