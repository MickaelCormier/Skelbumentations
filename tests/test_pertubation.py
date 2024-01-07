import random

import numpy as np

from skelbumentations import MovePerturbation, random_utils


def test_move_pertubation():
    augment = MovePerturbation(variance=0.5)
    data = np.ones((100, 17, 3), dtype=float)
    data[5] = 0.0
    invalid = np.full((100, 17, 3), False)
    invalid[5] = True
    data = augment(keypoints=data, invalid=invalid)
    assert (data["keypoints"][5] == 0.0).all()
