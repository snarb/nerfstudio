"""Numerical conservation tests for read-only signed-height contributions."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from marshawn_signed_height_bins import partition_weights, accumulate_parts


class HeightPartitionTests(unittest.TestCase):
    def test_known_exponential_and_original_approximation(self):
        parts, record = partition_weights([0], [2], [1], [.7], [-1], [1], [0], transmittance=[1])
        expected = .7 * np.array([1 - np.exp(-1), np.exp(-1) - np.exp(-2)]) / (1 - np.exp(-2))
        np.testing.assert_allclose(parts[0], expected, rtol=1e-15)
        self.assertAlmostEqual(parts.sum(), .7, places=15)
        self.assertAlmostEqual(record['analytic_minus_original_weight']['sum'], 1 - np.exp(-2) - .7)

    def test_downward_direction_reverses_bins(self):
        up, _ = partition_weights([0], [2], [1], [.8], [-1], [1], [0])
        down, _ = partition_weights([0], [2], [1], [.8], [1], [-1], [0])
        np.testing.assert_array_equal(up[:, ::-1], down)

    def test_parallel_endpoints_zero_length(self):
        parts, record = partition_weights([0, 0, 0, 1], [1, 1, 1, 1], [1] * 4, [.2] * 4,
                                          [-1, 0, 1, -1], [0, 0, 0, 1], [-1, 0, 1])
        np.testing.assert_array_equal(parts, np.array([[0,.2,0,0],[0,0,.2,0],[0,0,0,.2],[0,0,.2,0]]))
        self.assertEqual(record['zero_length_nonzero_weight_samples'], 1)

    def test_zero_tiny_and_saturated_density(self):
        parts, record = partition_weights([0]*4, [2]*4, [0, 1e-300, 1e-12, 1e4], [.5]*4,
                                          [-1]*4, [1]*4, [0])
        np.testing.assert_allclose(parts[:3], .25, atol=3e-13, rtol=0)
        np.testing.assert_allclose(parts[3], [.5, 0], atol=0, rtol=0)
        self.assertEqual(record['zero_density_nonzero_weight_samples'], 1)

    def test_cut_at_sample_endpoints_and_no_cuts(self):
        parts, _ = partition_weights([0], [1], [2], [.4], [0], [1], [0,1])
        np.testing.assert_array_equal(parts, [[0,.4,0]])
        parts, _ = partition_weights([0], [1], [2], [.4], [0], [1], [])
        np.testing.assert_array_equal(parts, [[.4]])

    def test_random_conservation_streamed_rgb(self):
        rng = np.random.default_rng(24)
        count = 1000
        starts = rng.uniform(0, 2, count)
        ends = starts + rng.uniform(.00001, .2, count)
        density = np.exp(rng.uniform(-20, 8, count))
        weights = rng.uniform(0, .03, count)
        rgb = rng.uniform(0, 1, (count,3))
        origins = rng.uniform(-.1, .1, count)
        directions = rng.uniform(-1, 1, count)
        ray_ids = rng.integers(0, 32, count)
        parts, _ = partition_weights(starts, ends, density, weights, origins, directions, [-.02,-.015,-.01,0,.01,.015,.02])
        np.testing.assert_allclose(parts.sum(1), weights, rtol=2e-15, atol=0)
        mass, color = accumulate_parts(parts, rgb, ray_ids, 32)
        np.testing.assert_allclose(mass.sum(1), np.bincount(ray_ids, weights=weights, minlength=32), rtol=2e-15)
        for channel in range(3):
            np.testing.assert_allclose(color[:,:,channel].sum(1), np.bincount(ray_ids, weights=weights*rgb[:,channel], minlength=32), rtol=2e-15)
        first = accumulate_parts(parts[:407],rgb[:407],ray_ids[:407],32)
        second = accumulate_parts(parts[407:],rgb[407:],ray_ids[407:],32)
        np.testing.assert_allclose(first[1]+second[1],color,rtol=2e-15)

    def test_empty(self):
        parts, record = partition_weights([],[],[],[],[],[],[0])
        self.assertEqual(parts.shape,(0,2))
        self.assertEqual(record['max_sample_weight_reconstruction_error'],0)

    def test_invalid_inputs(self):
        for sigma, cuts, end in [(-1,[0],1),(1,[0,0],1),(1,[np.nan],1),(1,[0],-1)]:
            with self.assertRaises(ValueError):
                partition_weights([0],[end],[sigma],[.2],[0],[1],cuts)


if __name__ == '__main__':
    unittest.main()
