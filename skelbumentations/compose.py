import random
from typing import Any

import numpy as np

from . import random_utils
from .select import BaseSelect
from .transform import BaseTransform


class BaseCompose:
    def __init__(self, transforms, p: float):
        if isinstance(transforms, (BaseCompose, BaseSelect, BaseTransform)):
            print("transforms is single transform, but a sequence is expected! Transform will be wrapped into list.")
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

    def __len__(self) -> int:
        return len(self.transforms)


class Compose(BaseCompose):
    """Compose transforms

    Args:
        transforms: list of transformations to compose.
        p (float, optional):  probability of applying all list of transforms. Defaults to 1.0.
        set_invalid_to_zero (bool, optional): if invalid keypoints should be set to zero. Defaults to True.
    """

    def __init__(self, transforms, p: float = 1.0, set_invalid_to_zero: bool = True):
        super().__init__(transforms, p)
        self.set_invalid_to_zero = set_invalid_to_zero

    def __call__(self, *args, force_apply: bool = False, **data):
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(keypoints=keypoints)"
            )

        data = self.check_args(**data)

        transforms = self.transforms

        for t in transforms:
            data = t(**data)

        if self.set_invalid_to_zero:
            data["keypoints"][data["invalid"]] = 0.0

        return data

    def check_args(self, **data):
        if "keypoints" not in data:
            raise KeyError("You have to pass keypoints to the augmentations")

        if "invalid" not in data:
            data["invalid"] = np.full(data["keypoints"].shape[:-1], False)
        else:
            assert data["invalid"].shape == data["keypoints"].shape[:-1]

        if "peturbation" not in data:
            data["peturbation"] = np.full(data["keypoints"].shape[:-1], False)
        else:
            assert data["peturbation"].shape == data["keypoints"].shape[:-1]
        return data


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms: list of transformations to compose.
        p (float): probability of applying selected transform. Default: 1.0.
    """

    def __init__(self, transforms, p: float = 1.0):
        super().__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, force_apply: bool = False, **data):
        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx: int = random_utils.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data


class SomeOf(BaseCompose):
    """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms: list of transformations to compose.
        n (int): number of transforms to apply.
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying selected transform. Default: 1.0.
    """

    def __init__(self, transforms, n: int, replace: bool = True, p: float = 1.0):
        super().__init__(transforms, p)
        self.n = n
        self.replace = replace
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, force_apply: bool = False, **data):
        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx = random_utils.choice(len(self.transforms), size=self.n, replace=self.replace, p=self.transforms_ps)
            for i in idx:
                t = self.transforms[i]
                data = t(force_apply=True, **data)
        return data


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`.
        Either set first and second arguments, or the transforms argument.

    Args:
        first: first one of the two transforms.
        second: second one of the two transforms.
        transforms: The two transforms.
        p (float, optional): probability of applying first transform. Defaults to 1.0.
    """

    def __init__(self, first=None, second=None, transforms=None, p: float = 0.5):
        if transforms is None:
            if first is None or second is None:
                raise ValueError("You must set both first and second or set transforms argument.")
            transforms = [first, second]
        super().__init__(transforms, p)
        if len(self.transforms) != 2:
            print("Length of transforms is not equal to 2.")

    def __call__(self, force_apply: bool = False, **data):
        if random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class Sequential(BaseCompose):
    """Sequentially applies all transforms.

    Note:
        This transform is not intended to be a replacement for `Compose`. Instead, it should be used inside `Compose`
        the same way `OneOf` or `OneOrOther` are used.

    Args:
        p (float, optional): probability of applying selected transform. Defaults to 1.0.
    """

    def __init__(self, transforms, p: float = 1.0):
        super().__init__(transforms, p)

    def __call__(self, **data):
        if random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class NoOp(BaseCompose):
    """No Operation. This does not change the data in any way.

    Args:
        p (float, optional): probability of applying selected transform. Defaults to 1.0.
    """

    def __init__(self, p: float = 1.0):
        super().__init__([], p)

    def __call__(self, **data):
        return data
