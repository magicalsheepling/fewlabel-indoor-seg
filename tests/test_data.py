import numpy as np
import pytest

from fewlabel.data import image_to_hwc, label_budget_indices, label_to_hw

W, H = 640, 480  # NYUv2 frame size


def test_image_axes():
    frame = np.zeros((3, W, H), dtype=np.uint8)
    frame[1, 10, 20] = 255  # channel 1, x=10, y=20
    out = image_to_hwc(frame)
    assert out.shape == (H, W, 3)
    assert out[20, 10, 1] == 255


def test_label_axes():
    label = np.zeros((W, H), dtype=np.uint16)
    label[10, 20] = 7
    out = label_to_hw(label)
    assert out.shape == (H, W)
    assert out[20, 10] == 7


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        image_to_hwc(np.zeros((H, W, 3)))


def test_budget_is_reproducible_and_sized():
    a = label_budget_indices(795, 0.1, seed=0)
    b = label_budget_indices(795, 0.1, seed=0)
    assert np.array_equal(a, b)
    assert len(a) == 80 and len(set(a.tolist())) == 80
    assert len(label_budget_indices(795, 1.0, seed=0)) == 795
