from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from skelbumentations.compose import (
    Compose,
    NoOp,
    OneOf,
    OneOrOther,
    Sequential,
    SomeOf,
)


@pytest.fixture
def data():
    return {"keypoints": np.ones((100, 17, 3)), "invalid": np.full((100, 17), False)}


def test_one_or_other(data):
    first = MagicMock()
    second = MagicMock()
    augmentation = OneOrOther(first, second, p=1)
    augmentation(**data)
    assert first.called != second.called


def test_compose(data):
    first = MagicMock()
    second = MagicMock()
    augmentation = Compose([first, second], p=1)
    augmentation(**data)
    assert first.called
    assert second.called


@pytest.mark.parametrize("invalid_to_zero", [True, False])
def test_compose_invalid_to_zero(data, invalid_to_zero):
    augmentation = Compose([], p=1, set_invalid_to_zero=invalid_to_zero)
    data["invalid"][:] = True
    data = augmentation(**data)
    if invalid_to_zero:
        np.testing.assert_equal(data["keypoints"], np.zeros((100, 17, 3)))
    else:
        np.testing.assert_equal(data["keypoints"], np.ones((100, 17, 3)))


def test_compose_missing_invalid(data):
    augmentation = Compose([], p=1)
    result = augmentation(keypoints=data["keypoints"])
    assert "invalid" in result
    assert "keypoints" in result
    np.testing.assert_equal(data["invalid"], result["invalid"])


def test_one_of(data):
    transforms = [Mock(p=1) for _ in range(10)]
    augmentation = OneOf(transforms, p=1)
    augmentation(**data)
    assert len([transform for transform in transforms if transform.called]) == 1


@pytest.mark.parametrize("N", [1, 2, 5, 10])
@pytest.mark.parametrize("replace", [True, False])
def test_n_of(data, N, replace):
    transforms = [Mock(p=1, side_effect=lambda **kw: {"keypoints": kw["keypoints"]}) for _ in range(10)]
    augmentation = SomeOf(transforms, N, p=1, replace=replace)
    augmentation(**data)
    if not replace:
        assert len([transform for transform in transforms if transform.called]) == N
    assert sum([transform.call_count for transform in transforms]) == N


def test_sequential(data):
    transforms = [Mock(side_effect=lambda **kw: kw) for _ in range(10)]
    augmentation = Sequential(transforms, p=1)
    augmentation(**data)
    assert len([transform for transform in transforms if transform.called]) == len(transforms)


def test_noop(data):
    augmentation = NoOp()
    result = augmentation(**data)
    for key in data:
        np.testing.assert_equal(data[key], result[key])
