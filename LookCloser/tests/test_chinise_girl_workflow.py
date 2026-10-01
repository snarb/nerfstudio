"""Fresh-machine portability and explicit checkpoint-bound review gates."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_luster_video as preparation
import launch_luster_video_stage as launcher
import run_chinise_girl_video as workflow


def write(root, name, value):
    workflow.write(root / name, value)


def test_environment_respects_new_machine_cuda_and_cache(monkeypatch):
    monkeypatch.setenv('CUDA_HOME', '/opt/new-cuda')
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', '/scratch/extensions')
    env = preparation.environment()
    assert env['CUDA_HOME'] == '/opt/new-cuda'
    assert env['TORCH_EXTENSIONS_DIR'] == '/scratch/extensions'
    assert '/opt/new-cuda/bin' in env['PATH']
    assert str(preparation.REPO) in env['PYTHONPATH']
    monkeypatch.delenv('CUDA_HOME')
    assert 'CUDA_HOME' not in preparation.environment()


def test_cold_request_needs_no_historical_local_experiment(tmp_path, monkeypatch, capsys):
    write(tmp_path, 'preparation_complete.json', {})
    write(tmp_path, 'frames/000470/data/video_rois.json', {})
    monkeypatch.setattr(sys, 'argv', ['launch', str(tmp_path), '000470', '8000',
                                    '--fr', '1', '--reason', 'fresh seed', '--dry-run'])
    launcher.main()
    request = json.loads(capsys.readouterr().out)
    assert request['data'] == str(tmp_path / 'frames/000470/data')
    assert request['model']['correct_sh_directions']
    assert request['model']['density_activation'] == 'trunc_exp'
    assert request['model']['feature_reweighting_strength'] == 1
    assert request['model']['log2_hashmap_size'] == 23
    assert request['lr'] == .01 and request['rays'] == 4096
    assert 'resume' not in request and 'warm_start' not in request
    assert '/home/brans' not in json.dumps(request)


def test_first_frame_470_is_ingested_on_fresh_machine(tmp_path, monkeypatch):
    import numpy as np
    calls = []
    def ingest(base, frame, host):
        calls.append((frame, host))
    def execute(argv, **kwargs):
        script = Path(argv[1]).name; base = Path(argv[2])
        if script == 'prepare_luster_frame.py':
            write(base, 'data/complete.json', {'transforms_sha256': 'old'})
            write(base, 'data/transforms.json', {'frames': []})
            write(base, 'data/bounds_audit.json', {'normalization': np.eye(4).tolist(), 'scale': 1.,
                                                 'bounds': [[-1,-1,-1], [1,1,1]]})
        elif script == 'audit_luster_data.py':
            write(base, 'data/audit_preprocessing.json', {})
    monkeypatch.setattr(preparation, 'ingest', ingest)
    monkeypatch.setattr(preparation.subprocess, 'run', execute)
    monkeypatch.setattr(sys, 'argv', ['prepare', str(tmp_path), '--start', '470', '--end', '470', '--host', 'new-dev3'])
    preparation.main()
    assert calls == [('000470', 'new-dev3')]
    assert workflow.read(tmp_path / 'preparation_complete.json')['frames'] == ['000470']


def test_campaign_cannot_silently_change_range_or_recipe(tmp_path):
    args = SimpleNamespace(start=470, end=471, host='dev3', archive_root='/archive/new')
    config = workflow.initialize(tmp_path, args)
    assert workflow.initialize(tmp_path, args) == config
    args.end = 472
    with pytest.raises(ValueError, match='changed'):
        workflow.initialize(tmp_path, args)


def candidate(root, final=False):
    run = root / 'frames/000470/runs/s008000'
    checkpoint = run / 'model.ckpt'; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'weights')
    selected = dict(step=8000, checkpoint=str(checkpoint), eval_all_psnr=29., eval_all_ssim=.94,
                    eval_all_lpips=.07, per_view=[dict(split='eval', foreground_psnr=25.)])
    write(root, 'frames/000470/runs/s008000/selection.json', selected)
    if final:
        write(root, 'snapshots/000470.json', dict(run=str(run), selection=str(run/'selection.json'),
              render_receipt=str(root/'render.json'), archived_checkpoint={'sha256': workflow.sha(checkpoint)}))
    gate = workflow.gate_identity(root, '000470', run, 'final' if final else 'seed_stage')
    write(root, 'fresh_status.json', dict(phase='visual_review_required', **gate))
    evidence = root / 'reviewed_crop.png'; evidence.write_bytes(b'review evidence')
    return gate, SimpleNamespace(decision='accept' if final else 'continue', reason='Inspected train/eval detail and silhouettes',
                                evidence=[str(evidence)], allow_numeric_exception=False)


def test_seed_gate_stops_then_resumes_only_after_matching_review(tmp_path):
    gate, args = candidate(tmp_path)
    with pytest.raises(SystemExit) as error:
        workflow.require_review(tmp_path, gate)
    assert error.value.code == 2
    workflow.record_review(tmp_path, args)
    assert workflow.require_review(tmp_path, gate) == 'continue'
    with pytest.raises(ValueError, match='Stale'):
        workflow.require_review(tmp_path, dict(gate, checkpoint_sha256='other'))


def test_failed_numeric_gate_requires_explicit_exception(tmp_path):
    gate, args = candidate(tmp_path, final=True)
    with pytest.raises(ValueError, match='Numeric screen failed'):
        workflow.record_review(tmp_path, args)
    assert not workflow.review_path(tmp_path, gate).exists()
    args.allow_numeric_exception = True
    workflow.record_review(tmp_path, args)
    review = workflow.read(workflow.review_path(tmp_path, gate))
    assert review['accepted']
    assert review['numeric_gate_override']['checkpoint_sha256'] == gate['checkpoint_sha256']


def test_changed_checkpoint_cannot_inherit_pending_seed_review(tmp_path):
    gate, args = candidate(tmp_path)
    (Path(gate['run']) / 'model.ckpt').write_bytes(b'different')
    with pytest.raises(ValueError, match='changed'):
        workflow.record_review(tmp_path, args)


def test_lock_prevents_competing_workflow(tmp_path):
    with workflow.exclusive(tmp_path):
        with pytest.raises(RuntimeError, match='Another workflow'):
            with workflow.exclusive(tmp_path):
                pytest.fail('Lock acquired twice')


def test_range_is_real_consecutive_frames():
    assert workflow.frame_range(470,529) == [f'{n:06d}' for n in range(470,530)]
    for start,end in [(529,470),(-1,3),(999999,1000000)]:
        with pytest.raises(ValueError):
            workflow.frame_range(start,end)


def test_frozen_recipe_builds_the_actual_trainer_config(tmp_path):
    from blur_runtime import configuration
    recipe = workflow.read(preparation.CONFIG / 'recipe.json')
    write(tmp_path, 'data/transforms.json', {'blur_aabb': [[-1,-1,-1],[1,1,1]]})
    request = dict(recipe, data=str(tmp_path/'data'), output=str(tmp_path/'run'))
    cfg = configuration(request)
    model = cfg.pipeline.model
    assert model.density_activation == 'trunc_exp' and model.correct_sh_directions
    assert model.density_normalization == 'none' and model.depth_loss_mult == 0
    assert model.adaptive_coarse_step_size == .001 and model.background_opacity_loss_mult == .02
    assert model.adaptive_warmup_steps == 4096 and model.log2_hashmap_size == 23
    assert cfg.pipeline.datamanager.train_num_rays_per_batch == 4096
    assert not cfg.pipeline.datamanager.dataparser.auto_scale_poses


def test_approved_early_stage_survives_recorded_checkpoint_pruning(tmp_path):
    gate,args=candidate(tmp_path)
    workflow.record_review(tmp_path,args)
    checkpoint=Path(gate['run'])/'model.ckpt'
    (tmp_path/'checkpoint_pruning.jsonl').write_text(json.dumps(dict(path=str(checkpoint),sha256=workflow.sha(checkpoint)))+'\n')
    checkpoint.unlink()
    current=workflow.gate_identity(tmp_path,'000470',Path(gate['run']),'seed_stage')
    assert current==gate
    assert workflow.require_review(tmp_path,current)=='continue'
