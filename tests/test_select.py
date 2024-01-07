import random
from typing import Any

import numpy as np
import pytest

from skelbumentations.select import (
    BaseSelect,
    SelectFrames,
    SelectHighMovement,
    SelectRandomFrames,
    SelectRandomWithBorder,
)


@pytest.fixture
def data():
    invalid = np.full((100, 17), False)
    invalid[20] = True
    keypoints = np.ones((100, 17, 3), dtype=float)
    keypoints[20] = 0
    keypoints[5, 10] = 5
    keypoints[70, 2] = 3
    return {"keypoints": keypoints, "invalid": invalid}


def test_base_select(data):
    select = SelectFrames([MockTransform()], [50, 55, 60])
    expected_data = {"keypoints": data["keypoints"].copy(), "invalid": data["invalid"].copy()}
    expected_data["invalid"][55] = True
    expected_data["keypoints"][55] = 11
    data = select(**data)
    np.testing.assert_equal(data["invalid"], expected_data["invalid"])
    np.testing.assert_equal(data["keypoints"], expected_data["keypoints"])


def test_select_random_frames_full_size(data):
    select = SelectRandomFrames([], max_num=100, min_num=100, contiguous=True)
    selection = select.get_selection(**data)
    expected_selection = np.arange(100)
    np.testing.assert_equal(selection, expected_selection)


def test_select_random_frames(data):
    select = SelectRandomFrames([], max_num=10, min_num=5, contiguous=True)
    random.seed(1998)
    selection = select.get_selection(**data)
    expected_selection = np.arange(85, 94)
    np.testing.assert_equal(selection, expected_selection)


def test_select_frames(data):
    expected_selection = [4, 8, 44]
    select = SelectFrames([], expected_selection)
    selection = select.get_selection(**data)
    np.testing.assert_equal(selection, expected_selection)


def test_select_high_movement(data):
    select = SelectHighMovement([], max_num=5, min_num=3, contiguous=True)
    random.seed(1998)
    selection = select.get_selection(**data)
    expected_selection = [3, 4, 5]
    np.testing.assert_equal(selection, expected_selection)


def test_select_high_part_movement(data):
    select = SelectHighMovement([], max_num=1, min_num=1, contiguous=True, part=[2])
    selection = select.get_selection(**data)
    expected_selection = [69]
    np.testing.assert_equal(selection, expected_selection)


def test_random_frames_with_border_select():
    invalid = np.full((100, 17), False)
    keypoints = np.ones((100, 17, 3), dtype=float)
    augment = SelectRandomWithBorder(
        inner_transforms=[MockFill(2, 2)], transforms=[MockFill(3, 3)], min_num=50, max_num=50, border_num=25
    )
    result = augment(keypoints=keypoints, invalid=invalid)
    expected_keypoints = keypoints.copy()
    expected_keypoints[:, 3] = 3.0
    expected_keypoints[25:75, 2] = 2.0
    np.testing.assert_equal(result["invalid"], invalid)
    np.testing.assert_equal(result["keypoints"], expected_keypoints)


class MockTransform:
    def __call__(self, **data):
        data["invalid"][1] = True
        data["keypoints"][1] = 11
        return data


class MockFill:
    def __init__(self, number, joint):
        self.number = number
        self.joint = joint

    def __call__(self, **data):
        data["keypoints"][:, self.joint] = self.number
        return data
