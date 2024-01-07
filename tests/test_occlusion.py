import random

import numpy as np

from skelbumentations import RandomOcclusion, SpecificOcclusion, WholeOcclusion


def test_random_occlusion():
    augment = RandomOcclusion(0.2, p=1)
    random.seed(1998)
    result = augment(keypoints=np.ones((100, 17, 3)), invalid=np.full((100, 17), False))["invalid"]
    expected_result = np.full((100, 17), False)
    expected_result[:, [6, 9]] = True
    np.testing.assert_equal(result, expected_result)


def test_whole_occlusion():
    augment = WholeOcclusion()
    result = augment(keypoints=np.ones((100, 17, 3), dtype=float), invalid=np.full((100, 17), False))["invalid"]
    expected_result = np.full((100, 17), True)
    np.testing.assert_equal(result, expected_result)


def test_specific_occlusion():
    augment = SpecificOcclusion(joints=[1, 3])
    result = augment(keypoints=np.ones((100, 17, 3), dtype=float), invalid=np.full((100, 17), False))["invalid"]
    expected_result = np.full((100, 17), False)
    expected_result[:, [1, 3]] = True
    np.testing.assert_equal(result, expected_result)
