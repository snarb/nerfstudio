"""CPU physical-selector contracts, with no CUDA or optimizer creation."""
from pathlib import Path
import sys
import unittest
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here if (here/'marshawn_frozen_semantic_selectors.py').exists() else here.parent/'scripts'))
import torch
from marshawn_frozen_semantic_selectors import selector_memberships


class PhysicalSelectors(unittest.TestCase):
    def test_exact_zero_and_predeclared_thresholds(self):
        p = torch.tensor([0., .001, .0999, .1, .3, .5, .99])
        before = p.clone(); result = selector_memberships(p)
        torch.testing.assert_close(result['hard_theta01'], torch.tensor([0., 0., 0., 1., 1., 1., 1.]))
        torch.testing.assert_close(result['hard_theta05'], torch.tensor([0., 0., 0., 0., 0., 1., 1.]))
        self.assertTrue(torch.equal(result['exactzero_smoothstep01_05'][:4], torch.zeros(4)))
        self.assertTrue(torch.equal(result['exactzero_smoothstep01_05'][5:], torch.ones(2)))
        self.assertTrue(torch.equal(p, before))

    def test_dense_background_soft_leak_and_exact_zero(self):
        sigma = torch.tensor([1e9]); variants = selector_memberships(torch.tensor([.001]))
        self.assertEqual((-torch.expm1(-sigma*variants['soft_probability']*.003)).item(), 1.)
        for name in ['hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']:
            self.assertEqual((-torch.expm1(-sigma*variants[name]*.003)).item(), 0.)

    def test_recomputed_visibility_reveals_originally_hidden_interval(self):
        sigma = torch.tensor([1e9, 1.]); probability = torch.tensor([.001, .99])
        def second_weight(selected):
            tau = sigma*selected
            return torch.exp(-tau[0])*(-torch.expm1(-tau[1]))
        self.assertEqual(second_weight(torch.ones(2)).item(), 0.)
        self.assertEqual(second_weight(selector_memberships(probability)['soft_probability']).item(), 0.)
        self.assertGreater(second_weight(selector_memberships(probability)['hard_theta05']).item(), .6)

    def test_unobserved_prior_is_retained_not_certified(self):
        variants = selector_memberships(torch.tensor([.99]))
        for name in ['hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']:
            self.assertEqual(variants[name].item(), 1.)

    def test_explicit_removal_prior_is_zeroed_without_emptiness_certificate(self):
        variants = selector_memberships(torch.tensor([.01]))
        for name in ['hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']:
            self.assertEqual(variants[name].item(), 0.)
        self.assertGreater(variants['soft_probability'].item(), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()
