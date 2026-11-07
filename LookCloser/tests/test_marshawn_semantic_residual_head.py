"""Small CPU grids only: optional residual semantics, no field or GPU launch."""
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_frozen_semantic_head import (
    FrozenActorGrid, conditional_class_objective, rendered_classes, tensor_digest,
)
from marshawn_semantic_residual_head import FrozenResidualActorGrid, dense_fp32_head_memory


def coarse_grid(resolution=3, bounds=((0, 0, 0), (1, 1, 1))):
    coarse = FrozenActorGrid(bounds, torch.eye(4), resolution=resolution)
    with torch.no_grad():
        coarse.logits.copy_(torch.linspace(-6, 6, resolution**3).reshape_as(coarse.logits))
    return coarse


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_zero_residual_bitwise_identity_probabilities_classes_and_objective():
    coarse = coarse_grid()
    before_rng = torch.get_rng_state().clone()
    head = FrozenResidualActorGrid(coarse, resolution=7)
    assert torch.equal(before_rng, torch.get_rng_state())
    xyz = torch.tensor([[0., 0., 0.], [1., 1., 1.], [.2, .4, .7],
                        [.5, .5, .5], [-.01, .5, .5], [.4, .8, .2]])
    assert torch.count_nonzero(head.residual_logits) == 0
    assert torch.equal(head(xyz), coarse(xyz))
    weights = torch.tensor([.2, .3, .15, .01, .1, .5])
    ids = torch.tensor([0, 0, 1, 1, 2, 3])
    original = rendered_classes(coarse(xyz), weights, ids, 5)
    refined = rendered_classes(head(xyz), weights, ids, 5)
    assert torch.equal(original, refined)
    target = torch.tensor([1., 0., 1., .3, 1.])
    confidence = torch.tensor([1., 1., 0., .2, 1.])
    assert torch.equal(conditional_class_objective(original, target, confidence),
                       conditional_class_objective(refined, target, confidence))
    assert not torch.cuda.is_initialized()


def test_anisotropic_xyz_affine_logit_sum_matches_analytic_solution():
    bounds = torch.tensor([[-.84, -.21, -.07], [.15, .42, .63]])
    coarse = coarse_grid(resolution=3, bounds=bounds)
    head = FrozenResidualActorGrid(coarse, resolution=6)
    with torch.no_grad():
        x, y, z = torch.meshgrid(*(torch.linspace(0, 1, 3),)*3, indexing='ij')
        head.coarse_logits.copy_((1+2*x+3*y+4*z)[None, None])
        x, y, z = torch.meshgrid(*(torch.linspace(0, 1, 6),)*3, indexing='ij')
        head.residual_logits.copy_((-.5+5*x-6*y+7*z)[None, None])
    unit = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.], [.13, .59, .87], [.83, .27, .16]])
    expected = (.5+7*unit[:, 0]-3*unit[:, 1]+11*unit[:, 2]).sigmoid()
    torch.testing.assert_close(head(bounds[0]+unit*(bounds[1]-bounds[0])), expected, rtol=2e-6, atol=1e-7)


def test_nonnested_fine_knots_preserve_sharp_coarse_knot_without_resampling():
    coarse = coarse_grid(resolution=4)
    with torch.no_grad():
        coarse.logits.fill_(-8)
        coarse.logits[0, 0, 1, 1, 1] = 8
    # Coarse denominator3 and fine denominator5 do not share this knot.
    head = FrozenResidualActorGrid(coarse, resolution=6)
    xyz = torch.tensor([[1/3]*3, [.31, .34, .36], [.2, .2, .2], [.4, .4, .4]])
    assert torch.equal(head(xyz), coarse(xyz))
    assert head(xyz)[0] > .999


def test_closed_box_faces_nextafter_outside_and_zero_outside_gradient():
    head = FrozenResidualActorGrid(coarse_grid(), resolution=5)
    points = []
    for axis in range(3):
        for boundary, direction in [(0., -torch.inf), (1., torch.inf)]:
            p = torch.full((3,), .5); p[axis] = boundary
            points.append(p)
            out = p.clone(); out[axis] = torch.nextafter(torch.tensor(boundary), torch.tensor(direction))
            points.append(out)
    xyz = torch.stack(points)
    pred = head(xyz)
    assert (pred[::2] > 0).all()
    assert torch.equal(pred[1::2], torch.zeros(6))
    pred[1::2].sum().backward()
    assert torch.count_nonzero(head.residual_logits.grad) == 0


def test_partial_coverage_only_changes_visited_basis_support():
    coarse = coarse_grid()
    coarse_before = tensor_digest(coarse)
    head = FrozenResidualActorGrid(coarse, resolution=9)
    frozen_before = head.coarse_logits.clone()
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    observed = torch.tensor([[.04, .05, .06]])
    unobserved = torch.tensor([[.5, .5, .5], [.9, .9, .9], [1., 1., 1.]])
    known_before = head(observed).detach().clone()
    far_before = head(unobserved).detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (-head(observed).log().mean()).backward(); optimizer.step()
    assert head(observed).item() > known_before.item()
    assert torch.equal(head(unobserved), far_before)
    nonzero = (head.residual_logits != 0).nonzero()
    assert 0 < len(nonzero) <= 8
    assert (nonzero[:, -3:] <= 1).all()
    assert torch.equal(head.coarse_logits, frozen_before)
    assert tensor_digest(coarse) == coarse_before
    assert list(dict(head.named_parameters())) == ['residual_logits']
    assert head.coarse_logits.grad is None


def test_copied_base_has_no_alias_and_does_not_freeze_source_parameter():
    coarse = coarse_grid(); before = tensor_digest(coarse)
    head = FrozenResidualActorGrid(coarse, resolution=5)
    assert coarse.logits.requires_grad
    assert tensor_digest(coarse) == before
    frozen = head.coarse_logits.clone()
    with torch.no_grad():
        coarse.logits.add_(4)
    assert torch.equal(head.coarse_logits, frozen)
    assert head.bounds.data_ptr() != coarse.bounds.data_ptr()
    assert head.world_to_box.data_ptr() != coarse.world_to_box.data_ptr()


def test_positive_and_negative_observations_refine_separately_in_one_coarse_cell():
    coarse = coarse_grid(resolution=2)
    with torch.no_grad(): coarse.logits.zero_()
    head = FrozenResidualActorGrid(coarse, resolution=9)
    xyz = torch.tensor([[.04, .05, .06], [.94, .95, .96]])
    classes = rendered_classes(head(xyz), torch.tensor([.8, .8]), torch.arange(2), 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    conditional_class_objective(classes, torch.tensor([1., 0.]), torch.ones(2)).backward()
    optimizer.step()
    pred = head(xyz)
    assert pred[0] > .5 and pred[1] < .5
    # Fine supports are disjoint even though the coarse cell is shared.
    assert torch.equal(head(torch.tensor([[.5]*3])), torch.tensor([.5]))
    assert torch.equal(coarse(xyz), torch.tensor([.5, .5]))


def test_native_arrays_and_positions_are_immutable_and_receive_no_gradients():
    head = FrozenResidualActorGrid(coarse_grid(), resolution=5)
    xyz = torch.tensor([[.2, .2, .2], [.8, .8, .8], [.3, .4, .6]], requires_grad=True)
    sigma = torch.tensor([2., 3., 4.], requires_grad=True)
    rgb = torch.tensor([[.1, .2, .3]]*3, requires_grad=True)
    weights = torch.tensor([.2, .4, .3], requires_grad=True)
    originals = [x.detach().clone() for x in (xyz, sigma, rgb, weights)]
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    classes = rendered_classes(head(xyz), weights, torch.tensor([0, 0, 1]), 3)
    conditional_class_objective(classes, torch.tensor([1., 0., 1.]), torch.ones(3)).backward()
    optimizer.step()
    for value, original in zip((xyz, sigma, rgb, weights), originals):
        assert value.grad is None and torch.equal(value, original)
    assert head.residual_logits.grad is not None


def test_unknown_zero_alpha_and_outside_queries_do_not_train_residual():
    head = FrozenResidualActorGrid(coarse_grid(), resolution=5)
    xyz = torch.tensor([[.1, .1, .1], [.9, .9, .9], [2., 2., 2.]])
    classes = rendered_classes(head(xyz), torch.tensor([.9, 0., .7]), torch.arange(3), 3)
    conditional_class_objective(classes, torch.tensor([1., 1., 0.]), torch.tensor([0., 1., 1.])).backward()
    assert torch.count_nonzero(head.residual_logits.grad) == 0


def test_serialized_state_and_adam_resume_are_bitwise_exact():
    base = coarse_grid()
    full = FrozenResidualActorGrid(base, resolution=7)
    optimizer = torch.optim.Adam(full.parameters(), lr=.03, foreach=False)
    generator = torch.Generator().manual_seed(2718)

    def step(head, opt, rng):
        xyz = torch.rand((12, 3), generator=rng)
        weights = torch.linspace(.05, .15, 12)
        ids = torch.arange(4).repeat_interleave(3)
        classes = rendered_classes(head(xyz), weights, ids, 4)
        target = torch.tensor([1., 0., .7, 1.])
        confidence = torch.tensor([1., 1., .4, 0.])
        opt.zero_grad(set_to_none=True)
        conditional_class_objective(classes, target, confidence).backward(); opt.step()

    for _ in range(3): step(full, optimizer, generator)
    stream = io.BytesIO()
    torch.save(dict(head=full.state_dict(), optimizer=optimizer.state_dict(), rng=generator.get_state()), stream)
    stream.seek(0); saved = torch.load(stream, map_location='cpu', weights_only=True)
    resumed = FrozenResidualActorGrid(base, resolution=7)
    resumed.load_state_dict(saved['head'], strict=True)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.03, foreach=False)
    resumed_optimizer.load_state_dict(saved['optimizer'])
    resumed_rng = torch.Generator(); resumed_rng.set_state(saved['rng'])
    assert_nested_equal(full.state_dict(), resumed.state_dict())
    assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    for _ in range(4):
        step(full, optimizer, generator); step(resumed, resumed_optimizer, resumed_rng)
        assert_nested_equal(full.state_dict(), resumed.state_dict())
        assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
        assert torch.equal(generator.get_state(), resumed_rng.get_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('resolution', [1, 3, 3.5, True])
def test_invalid_or_nonfiner_resolutions_rejected(resolution):
    with pytest.raises(ValueError): FrozenResidualActorGrid(coarse_grid(), resolution=resolution)


def test_nonidentity_frame_and_nonfinite_coarse_rejected():
    coarse = coarse_grid()
    with torch.no_grad(): coarse.world_to_box[0, 3] = .1
    with pytest.raises(ValueError): FrozenResidualActorGrid(coarse, resolution=5)
    with torch.no_grad():
        coarse.world_to_box.copy_(torch.eye(4)); coarse.logits[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError): FrozenResidualActorGrid(coarse, resolution=5)


def test_128_256_512_memory_counts_without_allocating_grids():
    expected = [(128, 2_097_152, 8), (256, 16_777_216, 64), (512, 134_217_728, 512)]
    for resolution, nodes, mib in expected:
        row = dense_fp32_head_memory(resolution)
        assert row['nodes'] == nodes and row['grid_bytes'] == mib*2**20
        assert row['hierarchical_inference_bytes'] == (8+mib)*2**20
        assert row['hierarchical_Adam_minimum_bytes'] == (8+4*mib)*2**20
    assert not torch.cuda.is_initialized()
