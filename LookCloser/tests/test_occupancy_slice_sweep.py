import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from export_occupancy_slice_sweep import slice_rgb


def test_xy_orientation_and_support_colors():
    active = np.zeros((3, 4, 2), bool)
    support = active.copy()
    active[2, 3, 1] = True
    support[0, 0, 1] = True
    active[1, 1, 1] = support[1, 1, 1] = True
    rgb = slice_rgb(active, support, 2, 1)
    assert rgb.shape == (4, 3, 3)
    np.testing.assert_array_equal(rgb[0, 2], [240, 45, 45])
    np.testing.assert_array_equal(rgb[3, 0], [30, 65, 100])
    np.testing.assert_array_equal(rgb[2, 1], [232, 232, 220])


def test_yz_orientation_keeps_xyz_storage_order():
    active = np.zeros((3, 4, 2), bool)
    active[1, 3, 0] = True
    rgb = slice_rgb(active, active.copy(), 0, 1)
    assert rgb.shape == (2, 4, 3)
    np.testing.assert_array_equal(rgb[1, 3], [232, 232, 220])
    assert np.count_nonzero((rgb == [232, 232, 220]).all(-1)) == 1
