import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import export_marshawn_semantic_cell_candidates_cpu as m
from render_marshawn_semantic_cell_slices_cpu import render


@pytest.fixture
def head_case():
    state = dict(logits=torch.zeros((1, 1, 128, 128, 128)), bounds=torch.from_numpy(m.HEAD_BOUNDS.copy()), world_to_box=torch.eye(4))
    cameras = [f'T{i:02}' for i in range(63)]
    hashes = dict(head='h'*64, spec='s'*64, labels='l'*64)
    payload = dict(head=state, head_digest=m.tensor_head_digest(state), model_digest=m.SOURCE_PIPELINE,
        source_kind=m.SOURCE_KIND, LC_capture_receipt_sha256=m.CAPTURE_SHA, world_to_box_is_identity=True,
        iteration=2064, initial_probability=.99, spec_sha256=hashes['spec'], label_qualification_sha256=hashes['labels'],
        camera_history=[cameras[0]]*2064, optimizer={'state': {0: {'step': torch.tensor(2064.)}}})
    complete = dict(source_kind=m.SOURCE_KIND, LC_capture_receipt_sha256=m.CAPTURE_SHA,
        world_to_box_is_identity=True, iteration=2064, actual_head_updates=2064, initial_probability=.99,
        checkpoint_sha256=hashes['head'], native_checkpoint_and_caches_unchanged=True,
        native_model_not_loaded_or_optimized=True, source_RGB_targets_loaded=False,
        camera_history=payload['camera_history'].copy(), phase='direct_LC_prior099_fresh2064')
    spec = dict(world_to_box=np.eye(4).tolist(), updates=2064, grid_resolution=128, seed=42,
        learning_rate=.03, native_model_updates=0, initial_probability=.99,
        phase=complete['phase'], authorization_key='LC_first2064_head_authorized')
    contract = dict(source_kind=m.SOURCE_KIND, LC_capture_receipt_sha256=m.CAPTURE_SHA,
        source_pipeline_digest=m.SOURCE_PIPELINE, native_source={'checkpoint_sha256': m.SOURCE_CHECKPOINT},
        native_world_to_box=np.eye(4).tolist(), bounds=m.HEAD_BOUNDS.tolist(), train_cameras=cameras)
    return payload, complete, spec, contract, hashes


def test_completed_head_validates_without_cuda_or_rng_change(head_case):
    before = torch.get_rng_state().clone()
    out = m.validate_head(*head_case)
    assert out.shape == (128, 128, 128) and np.isfinite(out).all()
    assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['capture', 'pipeline', 'source_checkpoint', 'technical', 'prior', 'frame', 'digest', 'RGB', 'eval_camera', 'spec', 'labels'])
def test_wrong_head_provenance_rejected(head_case, case):
    p, c, s, contract, hashes = head_case
    if case == 'capture': p['LC_capture_receipt_sha256'] = 'wrong'
    elif case == 'pipeline': p['model_digest'] = 'wrong'
    elif case == 'source_checkpoint': contract['native_source']['checkpoint_sha256'] = 'wrong'
    elif case == 'technical': p['iteration'] = 64
    elif case == 'prior': p['initial_probability'] = .01
    elif case == 'frame': p['head']['world_to_box'][0, 3] = .01
    elif case == 'digest': p['head']['logits'][0, 0, 1, 2, 3] = 1
    elif case == 'RGB': c['source_RGB_targets_loaded'] = True
    elif case == 'eval_camera': p['camera_history'][0] = 'DC'; c['camera_history'][0] = 'DC'
    elif case == 'spec': p['spec_sha256'] = 'wrong'
    elif case == 'labels': p['label_qualification_sha256'] = 'wrong'
    with pytest.raises(ValueError): m.validate_head(p, c, s, contract, hashes)


def test_fresh_core_halo_then_support_differs_from_filter_first_without_source_mutation():
    core = np.zeros((1, 5, 5, 5), bool); core[0, 2, 2, 2] = True
    admission = np.ones_like(core); envelope = np.zeros_like(core); envelope[0, 3, 2, 2] = True
    before = core.copy()
    out = m.compose_candidates(core, core.copy(), admission, envelope)
    assert out['active_r0'].sum() == 0 and out['active_r1'].sum() == 1
    assert out['alternative_filter_then_dilate_r1'].sum() == 0
    assert np.array_equal(core, before)
    assert not out['active_r1'][0, 2, 2, 2]  # +1 cannot override rejected semantic support.
    with pytest.raises(ValueError): m.compose_candidates(core, ~core, admission, envelope)


def test_cpu_probability_interpolates_logits_not_probabilities():
    nodes = np.zeros((2, 2, 2)); nodes[1] = 6
    p, logit, inside = m.reference_probabilities(nodes, [[0]*3, [1]*3], [[.25, .5, .5], [-1, .5, .5]])
    assert logit[0] == 1.5 and p[0] == pytest.approx(1/(1+np.exp(-1.5)))
    assert abs(p[0]-(.75*.5+.25/(1+np.exp(-6)))) > .1
    assert p[1] == 0 and not inside[1]


def make_candidate(tmp_path, size=4):
    root = tmp_path/'candidate'; root.mkdir()
    core = np.zeros((1, size, size, size), bool); core[0, 1, 1, 1] = True
    arrays = m.compose_candidates(core, core.copy(), np.ones_like(core), np.ones_like(core))
    np.savez_compressed(root/'candidate.npz', **arrays, cascade_aabbs=np.array([[0, 0, 0, 1, 1, 1]]), resolution=np.array([size]*3))
    metadata = dict(schema='LC_SEMANTIC_CELL_CANDIDATE_V1', source_checkpoint_sha256=m.SOURCE_CHECKPOINT,
        head_sha256='head', source_grid_sha256='grid', selector='hard_theta05', initial_probability=.99,
        phi_installed_or_changed=False, candidate_sha256=m.sha(root/'candidate.npz'))
    m.write(root/'metadata.json', metadata)
    return root


def test_candidate_api_requires_exact_phi_metadata_and_never_installs_it(tmp_path):
    root = make_candidate(tmp_path)
    kwargs = dict(expected_metadata_sha256=m.sha(root/'metadata.json'), expected_head_sha256='head',
        expected_source_grid_sha256='grid', expected_selector='hard_theta05', radius=1)
    arrays, meta = m.load_candidate(root, **kwargs)
    assert not arrays['active_r1'].flags.writeable and meta['phi_installed_or_changed'] is False
    for key, wrong in [('expected_head_sha256', 'other'), ('expected_selector', 'hard_theta03'), ('expected_metadata_sha256', 'changed'), ('radius', 2)]:
        with pytest.raises(ValueError): m.load_candidate(root, **dict(kwargs, **{key: wrong}))


def test_slice_atlas_includes_every_axis_slice_once(tmp_path):
    root = make_candidate(tmp_path, 128)
    receipt = render(root, tmp_path/'pages')
    assert receipt['total_slices'] == 384 and len(receipt['pages']) == 6
    assert len({(x['cascade'], x['axis'], x['index']) for x in receipt['slices']}) == 384
    assert not receipt['physical_free_space_claim'] and not receipt['native_model_loaded']


def test_unreviewed_selector_rejected_before_any_file_load(tmp_path):
    with pytest.raises(ValueError): m.export(tmp_path/'missing_binding.json', tmp_path/'out', ['hard_theta03'])
    with pytest.raises(ValueError): m.export(tmp_path/'missing_binding.json', tmp_path/'out', ['hard_theta01']*2)


def test_missing_head_and_unsealed_template_do_not_create_binding_or_candidates(tmp_path):
    output = tmp_path/'binding.json'
    with pytest.raises(FileNotFoundError): m.seal_binding({'head': tmp_path/'not_completed.pt'}, output)
    assert not output.exists()
    m.write(output, dict(schema='LC_SEMANTIC_CELL_CPU_V1', ready=False, files={}))
    with pytest.raises(ValueError): m.validate_binding(output)
    with pytest.raises(ValueError): m.seal_binding({}, output)
