import unittest
import numpy as np
from restore_union_interior import restore

class RestoreTests(unittest.TestCase):
    def test_depth_gate_hole_is_restored_from_raw(self):
        raw = np.full((50,50,3), 140, np.uint8)
        old = raw.copy(); old[20:30,20:30] = 0
        alpha = np.zeros((50,50)); alpha[5:45,5:45] = 1; alpha[20:30,20:30] = 0
        matte = np.zeros_like(alpha); matte[5:45,5:45] = 1
        result, _ = restore(raw, old, alpha, matte)
        np.testing.assert_array_equal(result[20:30,20:30], raw[20:30,20:30])

    def test_matte_cannot_remove_foreground_and_real_gap_stays(self):
        raw = np.full((50,50,3), 140, np.uint8)
        old = np.zeros_like(raw)
        alpha = np.zeros((50,50)); alpha[5:45,5:45] = 1; alpha[18:32,18:32] = 0
        result, weight = restore(raw, old, alpha, np.zeros_like(alpha))
        self.assertTrue(np.all(result[10:13,10:13] == 140))
        self.assertTrue(np.all(weight[18:32,18:32] == 0))
        self.assertTrue(np.all(result[18:32,18:32] == 0))

if __name__ == '__main__': unittest.main()
