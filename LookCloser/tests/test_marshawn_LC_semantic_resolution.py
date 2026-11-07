"""CPU contracts for independent128/256 heads; no model or GPU execution."""
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_frozen_semantic_head import FrozenActorGrid, rendered_classes, conditional_class_objective
from marshawn_LC_semantic_resolution_contract import (
    FAMILY, IDENTITY, UNKNOWN, node_resolution, validate_spec, validate_qualification,
    checkpoint_metadata, validate_saved_state, validate_resume, validate_matched_baseline,
    physical_node_resolution,
)
from probe_marshawn_LC_semantic_resolution_head import run


BOUNDS = [[-.84, -.21, -.07], [.15, .42, .63]]


@pytest.fixture
def spec(tmp_path):
    files = {}
    for name in ['manifest', 'capture_receipt', 'packet_history_reference', 'label_qualification']:
        p = tmp_path/(name+'.json'); p.write_text(json.dumps({'bound': name})); files[name] = str(p)
    return dict(files, phase='CPU_synthetic', resolution_experiment_family=FAMILY,
                seed=42, learning_rate=.03, initial_probability=.01, world_to_box=IDENTITY,
                qualified_UNKNOWN_count=4, native_model_updates=0, no_NGP_head_transfer=True,
                grid_resolution=128, updates=2064, capacity_policy='KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND',
                torch_version='2.7.1+cu128', cuda_version='12.8',
                camera_history_reference=files['packet_history_reference'])


@pytest.mark.parametrize('resolution', [128, 256])
def test_actual_dynamic_shape_constant_prior_floor_and_serialization(resolution):
    before = torch.get_rng_state().clone()
    head = FrozenActorGrid(BOUNDS, IDENTITY, node_resolution(resolution), initial_probability=.01)
    assert torch.equal(before, torch.get_rng_state())
    assert head.logits.shape == (1, 1, resolution, resolution, resolution)
    xyz = torch.tensor([BOUNDS[0], BOUNDS[1], [-.3, .1, .3], [-1., 0., 0.]])
    p = head(xyz)
    torch.testing.assert_close(p[:3], torch.full((3,), .01), rtol=1e-6, atol=1e-8)
    assert p[-1] == 0
    # This exercises actual128/256 storage. Optimizer resume is tested below too.
    stream = io.BytesIO(); torch.save(head.state_dict(), stream); stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=True)
    validate_saved_state(saved, resolution, BOUNDS)
    restored = FrozenActorGrid(BOUNDS, IDENTITY, resolution, initial_probability=.99)
    restored.load_state_dict(saved, strict=True)
    assert torch.equal(restored.logits, head.logits) and torch.equal(restored(xyz), p)
    assert not torch.cuda.is_initialized()


def test_empty_packed_rays_keep_escape_one_and_zero_objective():
    p = torch.empty(0, requires_grad=True)
    classes = rendered_classes(p, torch.empty(0), torch.empty(0, dtype=torch.long), 4)
    assert torch.equal(classes, torch.tensor([[0., 0., 1.]]*4))
    objective = conditional_class_objective(classes, torch.tensor([0., 1., 1., 0.]), torch.ones(4))
    assert objective == 0
    objective.backward()
    assert p.grad.numel() == 0


@pytest.mark.parametrize('resolution', [128, 256])
def test_actual_resolution_CPU_32_plus32_vs64_exact_resume(spec, resolution):
    torch.set_num_threads(2)
    spec['grid_resolution'] = resolution
    xyz = torch.tensor([[-.3, .1, .3], [-.1, .2, .4], [-1., 0., 0.]])
    weights = torch.tensor([.4, .3, .1]); ids = torch.tensor([0, 1, 1])
    targets = torch.tensor([1., 0., 1.]); confidence = torch.tensor([1., 1., 0.])

    def setup():
        h = FrozenActorGrid(BOUNDS, IDENTITY, resolution, initial_probability=.01)
        return h, torch.optim.Adam([h.logits], lr=.03, foreach=False), np.random.default_rng(42)

    def advance(h, opt, rng, count):
        history = []
        for _ in range(count):
            packet = int(rng.integers(60)); history.append(packet)
            q = rendered_classes(h(xyz), weights, ids, 3)
            opt.zero_grad(set_to_none=True)
            conditional_class_objective(q, targets, confidence).backward(); opt.step()
        return history

    h, opt, rng = setup(); history = advance(h, opt, rng, 32)
    stream = io.BytesIO()
    torch.save(dict(checkpoint_metadata(spec), head=h.state_dict(), optimizer=opt.state_dict(),
                    sampler=rng.bit_generator.state, RNG=torch.get_rng_state()), stream)
    stream.seek(0); saved = torch.load(stream, weights_only=False)
    validate_resume(saved, spec, BOUNDS)
    continued, Adam, sampler = setup()
    continued.load_state_dict(saved['head'], strict=True); Adam.load_state_dict(saved['optimizer'])
    sampler.bit_generator.state = saved['sampler']; torch.set_rng_state(saved['RNG'])
    assert torch.equal(continued.logits, h.logits)
    for key, value in Adam.state_dict()['state'][0].items():
        assert torch.equal(value, opt.state_dict()['state'][0][key])
    resumed_history = history+advance(continued, Adam, sampler, 32)
    reference, reference_Adam, reference_sampler = setup()
    full_history = advance(reference, reference_Adam, reference_sampler, 64)
    assert resumed_history == full_history and torch.equal(continued.logits, reference.logits)
    for key, value in Adam.state_dict()['state'][0].items():
        assert torch.equal(value, reference_Adam.state_dict()['state'][0][key])
    assert sampler.bit_generator.state == reference_sampler.bit_generator.state
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field,value', [('grid_resolution', 512), ('grid_resolution', 128.),
    ('grid_resolution', True), ('initial_probability', .99), ('seed', 43),
    ('learning_rate', .01), ('qualified_UNKNOWN_count', 2), ('world_to_box', None)])
def test_unreviewed_scientific_dimensions_rejected(spec, field, value):
    spec[field] = value
    with pytest.raises(ValueError): validate_spec(spec)


def test_exact_four_unknown_overlay_and_targets_are_required():
    labels = dict(unknown_queries=[dict(camera=c, global_query_index=i, pixel_xy=list(p),
        label_weight=0, semantic_target=1) for c, i, p in sorted(UNKNOWN)])
    validate_qualification(labels)
    bad = copy.deepcopy(labels); bad['unknown_queries'][-1]['semantic_target'] = 0
    with pytest.raises(ValueError): validate_qualification(bad)
    bad = copy.deepcopy(labels); bad['unknown_queries'].pop()
    with pytest.raises(ValueError): validate_qualification(bad)
    bad = copy.deepcopy(labels); bad['unknown_queries'][0]['pixel_xy'][0] += 1
    with pytest.raises(ValueError): validate_qualification(bad)


def test_resume_rejects_resolution_label_data_packet_and_actual_optimizer_drift(spec):
    head = FrozenActorGrid(BOUNDS, IDENTITY, 128, initial_probability=.01)
    parent = dict(checkpoint_metadata(spec), head=head.state_dict(), optimizer={'param_groups': [{'lr': .03}]})
    validate_resume(parent, spec, BOUNDS)
    for key, value in [('grid_resolution', 256), ('qualified_UNKNOWN_count', 2),
                       ('source_manifest_sha256', 'wrong'), ('capture_adapter_sha256', 'wrong'),
                       ('packet_history_reference_sha256', 'wrong'), ('initial_seed', 43)]:
        bad = dict(parent); bad[key] = value
        with pytest.raises(ValueError): validate_resume(bad, spec, BOUNDS)
    bad = dict(parent, optimizer={'param_groups': [{'lr': .02}]})
    with pytest.raises(ValueError): validate_resume(bad, spec, BOUNDS)
    Path(spec['packet_history_reference']).write_text('changed packet reference')
    with pytest.raises(ValueError): validate_resume(parent, spec, BOUNDS)


def test_shape_floor_bounds_and_dtype_cannot_be_relabelled_as_another_head():
    head = FrozenActorGrid(BOUNDS, IDENTITY, 128, initial_probability=.01)
    state = head.state_dict()
    with pytest.raises(ValueError): validate_saved_state(state, 256, BOUNDS)
    bad = dict(state, world_to_box=torch.eye(4)); bad['world_to_box'][0, 3] = .1
    with pytest.raises(ValueError): validate_saved_state(bad, 128, BOUNDS)
    bad = dict(state, bounds=state['bounds']+.01)
    with pytest.raises(ValueError): validate_saved_state(bad, 128, BOUNDS)
    bad = dict(state, logits=state['logits'].half())
    with pytest.raises(ValueError): validate_saved_state(bad, 128, BOUNDS)


def test_matched_comparison_rejects_two_unknown_or_changed_packets(spec):
    baseline = dict(spec); spec['grid_resolution'] = 256
    validate_matched_baseline(spec, baseline)
    bad = dict(baseline, qualified_UNKNOWN_count=2)
    with pytest.raises(ValueError): validate_matched_baseline(spec, bad)
    bad = dict(baseline, camera_history_reference='another stream')
    with pytest.raises(ValueError): validate_matched_baseline(spec, bad)
    history128 = np.random.default_rng(42).integers(60, size=2064)
    history256 = np.random.default_rng(42).integers(60, size=2064)
    assert np.array_equal(history128, history256)


def test_proposed_physical_constructor_requires_matching_explicit256_metadata(spec):
    head = FrozenActorGrid(BOUNDS, IDENTITY, 128, initial_probability=.01)
    render_spec = dict(expected_grid_resolution=128, expected_head_bounds=BOUNDS)
    assert physical_node_resolution({'head': head.state_dict()}, {'grid_resolution': 128}, render_spec) == 128
    render_spec['expected_grid_resolution'] = 256
    with pytest.raises(ValueError): physical_node_resolution({'head': head.state_dict()}, {'grid_resolution': 128}, render_spec)


def test_dry_run_and_false_release_stop_before_GPU(spec, tmp_path, capsys):
    baseline = dict(spec)
    baseline_path = tmp_path/'baseline.json'; baseline_path.write_text(json.dumps(baseline))
    release = tmp_path/'root_release.json'; release.write_text(json.dumps({'reviewer': 'root', 'LC_resolution_technical_head_authorized': False}))
    spec.update(grid_resolution=256, matched_four_UNKNOWN128_spec=str(baseline_path),
                root_release=str(release), authorization_key='LC_resolution_technical_head_authorized')
    path = tmp_path/'spec.json'; path.write_text(json.dumps(spec))
    run(path, False)
    dry = json.loads(capsys.readouterr().out)
    assert dry['grid_resolution'] == 256 and not dry['CUDA_started'] and not dry['optimizer_created']
    with pytest.raises(ValueError, match='root head-technical release'): run(path, True)
    assert not torch.cuda.is_initialized()
