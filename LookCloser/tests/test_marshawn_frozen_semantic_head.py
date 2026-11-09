"""CPU mathematical contracts; no optimizer, CUDA, field import, or fitting."""
import sys
import tempfile
import unittest
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here if (here/'marshawn_frozen_semantic_head.py').is_file() else here.parent/'scripts'))
import numpy as np
import torch
from marshawn_frozen_semantic_head import (FrozenActorGrid, rendered_classes,
    conditional_class_objective, ray_capacity, assert_exact_restoration,
    tensor_digest, export_density, validate_cache)


class FrozenSemanticTests(unittest.TestCase):
    def test_explicit_initial_prior_does_not_consume_rng(self):
        before = torch.get_rng_state().clone()
        h99 = FrozenActorGrid([[0, 0, 0], [1, 1, 1]], np.eye(4), resolution=2, initial_probability=.99)
        h01 = FrozenActorGrid([[0, 0, 0], [1, 1, 1]], np.eye(4), resolution=2, initial_probability=.01)
        point = torch.tensor([[.5, .5, .5]])
        self.assertAlmostEqual(float(h99(point)), .99, places=6)
        self.assertAlmostEqual(float(h01(point)), .01, places=6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_partial_opacity_and_frozen_weight_gradient(self):
        w = torch.tensor([.3, .2, .05], requires_grad=True)
        p = torch.tensor([.8, .4, .2], requires_grad=True)
        q = rendered_classes(p, w, torch.tensor([0, 0, 1]), 3)
        torch.testing.assert_close(q.sum(-1), torch.ones(3))
        torch.testing.assert_close(q[:, 0], torch.tensor([.32, .01, 0.]))
        self.assertTrue(torch.all(q[:, 0] <= q[:, :2].sum(-1)))
        conditional_class_objective(q, torch.tensor([1., 0., 1.]), torch.ones(3)).backward()
        self.assertIsNone(w.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertLess(p.grad[0].item(), 0)
        self.assertGreater(p.grad[2].item(), 0)

    def test_unknown_contributes_no_class_gradient(self):
        p = torch.tensor([.4, .7], requires_grad=True)
        q = rendered_classes(p, torch.tensor([.8, .9]), torch.tensor([0, 1]), 2)
        conditional_class_objective(q, torch.tensor([0., 1.]), torch.tensor([1., 0.])).backward()
        self.assertEqual(p.grad[1].item(), 0)

    def test_transform_once_axes_and_box_cap(self):
        T = np.eye(4); T[0, 3] = 2
        head = FrozenActorGrid([[0, 0, 0], [1, 1, 1]], T, resolution=2)
        with torch.no_grad():
            head.logits[:, :, 0, :, :] = -5
            head.logits[:, :, 1, :, :] = 5
        points = torch.tensor([[-2., 0., 0.], [-1., 0., 0.], [-.5, 0., 0.]])
        pred = head(points)
        self.assertLess(pred[0], .01); self.assertGreater(pred[1], .99)
        self.assertEqual(pred[2].item(), 0)
        total, cap = ray_capacity(torch.tensor([.4, .5]), torch.tensor([True, False]), torch.tensor([0, 0]), 1)
        torch.testing.assert_close(total, torch.tensor([.9])); torch.testing.assert_close(cap, torch.tensor([.4]))

    def test_conditional_labels_do_not_solve_far_background(self):
        # An opaque-ish background behind partial foreground receives positive
        # class gradients on a positive ray. The proposed head must admit this
        # ambiguity rather than claiming conditional CE automatically fixes it.
        p = torch.tensor([.5, .5], requires_grad=True)
        q = rendered_classes(p, torch.tensor([.6, .39]), torch.tensor([0, 0]), 1)
        conditional_class_objective(q, torch.ones(1), torch.ones(1)).backward()
        self.assertTrue((p.grad < 0).all())
        self.assertAlmostEqual(p.grad[1].item()/p.grad[0].item(), .39/.6, places=5)

    def test_export_recomposites_instead_of_partitioning(self):
        sigma = torch.tensor([1., 1.], requires_grad=True)
        actor = export_density(sigma, torch.tensor([0., 1.]))
        alpha = 1-torch.exp(-actor)
        exported_actor_mass = alpha[1]*(1-alpha[0])
        original_second_mass = (1-torch.exp(torch.tensor(-1.)))*torch.exp(torch.tensor(-1.))
        self.assertGreater(exported_actor_mass, original_second_mass)
        self.assertFalse(actor.requires_grad)

    def test_tiny_nonactor_probability_can_leave_dense_background_opaque(self):
        # Excellent nonactor classification can coexist with fully opaque soft
        # physical export. This is deliberately NOT a clean-asset assertion.
        sigma = torch.tensor([1e9])
        probability = torch.tensor([.001])
        exported_sigma = export_density(sigma, probability)
        physical_alpha = -torch.expm1(-exported_sigma*.003)
        self.assertLess(probability.item(), .01)
        self.assertEqual(physical_alpha.item(), 1.)
        hard_sigma = sigma*(probability >= .1)
        self.assertEqual((-torch.expm1(-hard_sigma*.003)).item(), 0.)

    def test_strict_restore_and_field_digest(self):
        m = torch.nn.Linear(2, 1)
        frozen = {k: v.detach().clone() for k, v in m.state_dict().items()}
        before = tensor_digest(m); assert_exact_restoration(m, frozen)
        with torch.no_grad(): m.weight.add_(.01)
        self.assertNotEqual(before, tensor_digest(m))
        with self.assertRaises(ValueError): assert_exact_restoration(m, frozen)
        with self.assertRaises(ValueError): assert_exact_restoration(m, {})

    def test_cache_rejects_eval_and_wrong_model(self):
        train = [f'train{i}' for i in range(63)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.npz'
            base = dict(camera=np.array(['DC']), model_digest=np.array('frozen'), pixel_xy=np.zeros((1, 2), int),
                semantic_target=np.ones(1), label_weight=np.ones(1), xyz=np.zeros((1, 3)), weight=np.array([.9]), ray_indices=np.zeros(1, int))
            np.savez(path, **base)
            with self.assertRaises(ValueError): validate_cache(path, train, 'frozen')
            base['camera'] = np.array([train[0]]); np.savez(path, **base)
            with self.assertRaises(ValueError): validate_cache(path, train, 'different')
            self.assertEqual(len(validate_cache(path, train, 'frozen')['camera']), 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
