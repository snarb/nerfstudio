"""Reject unfinished, regressed or invalid results at the autonomous launch gate."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/watch_marshawn_exp_followup_release.py'
spec = importlib.util.spec_from_file_location('conditional_release', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConditionalReleaseTest(unittest.TestCase):
    def setUp(self):
        self.policy = dict(maximum_actor_lpips=.06, maximum_actor_ratio=1.03,
                           reference_actor_lpips=dict(black=.0656, grey=.0737))
        self.result = dict(technical_smoke=False, final_step=126316, plateau=dict(stopped=True),
                           selection=dict(metrics=dict(psnr=27., ssim=.8, lpips=.054),
                                          by_actor=dict(black=dict(lpips=.06), grey=dict(lpips=.048))))

    def test_completed_passing_parent(self):
        self.assertTrue(module.eligible(self.result, self.policy))

    def test_plateau_and_ancestry_required(self):
        for changes in [dict(final_step=15188), dict(plateau=dict(stopped=False)), dict(technical_smoke=True)]:
            with self.subTest(changes=changes):
                self.assertFalse(module.eligible(dict(self.result, **changes), self.policy))

    def test_one_actor_regression_blocks_good_aggregate(self):
        self.result['selection']['by_actor']['black']['lpips'] = .08
        self.assertFalse(module.eligible(self.result, self.policy))

    def test_absolute_quality_guard(self):
        self.result['selection']['metrics']['lpips'] = .061
        self.assertFalse(module.eligible(self.result, self.policy))

    def test_nonfinite_or_boolean_metrics_rejected(self):
        for value in [float('nan'), float('inf'), True]:
            with self.subTest(value=value):
                result = copy.deepcopy(self.result)
                result['selection']['metrics']['psnr'] = value
                with self.assertRaises(ValueError):
                    module.eligible(result, self.policy)


if __name__ == '__main__':
    unittest.main()
