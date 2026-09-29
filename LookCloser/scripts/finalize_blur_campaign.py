"""Audit completed blur evidence, recipes, selections and encoded review paths."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg

from blur_runtime import configuration, sha, write
from nerfstudio.configs.method_configs import method_configs
from supervise_blur_campaign import aggregate_job_seconds, consumed_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    evidence = json.loads((args.output.parent / 'campaign_evidence.json').read_text())
    runs = evidence['runs']
    assert runs and all(run['complete'] for run in runs.values())
    pairs = [p for p in evidence['pairs'] if 'same_initial_parameters' in p]
    assert all(p['same_initial_parameters'] and p['same_first_32_batches'] and p['both_complete'] for p in pairs)
    for name, run in runs.items():
        assert run['identity']['seed'] == 42
        top = max(h['full']['psnr'] for h in run['history'])
        candidates = [h for h in run['history'] if h['full']['psnr'] >= top - .07]
        expected = min(candidates, key=lambda h: (h['full']['lpips'], -h['full']['psnr']))
        assert run['selected']['step'] == expected['step'], name
        assert (args.root / name / 'best.pt').exists()
        assert (args.root / name / f"eval_{expected['step']:06d}" / 'metrics.json').exists()
    recipes = []
    for path in sorted((repo / 'recipes/blur_fixes').glob('*.json')):
        request = json.loads(path.read_text())
        cfg = configuration(request)
        model = cfg.pipeline.model
        assert request['seed'] == 42 and cfg.load_dir is None and cfg.load_checkpoint is None
        assert model.density_normalization == request['model']['density_normalization']
        recipes.append(dict(recipe=str(path.relative_to(repo)), sha256=sha(path),
                            activation=model.density_activation, normalization=model.density_normalization,
                            correct_sh=model.correct_sh_directions))
    model = method_configs['lookcloser'].pipeline.model
    assert (model.density_activation, model.density_normalization, model.correct_sh_directions) == ('softplus', 'none', False)
    paths = []
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    for path in sorted((args.root / 'paths').glob('*/complete.json')):
        receipt = json.loads(path.read_text())
        assert receipt['frames'] == 24 and receipt['finite_rgb_checked']
        if 'sources' in receipt:
            assert receipt['camera_paths_equal']
        video = path.parent / 'learned_rgb.mp4'
        assert sha(video) == receipt['video_sha256']
        # Recheck even older receipts that predate the explicit decode flag.
        decoded = subprocess.run([encoder, '-v', 'error', '-threads', '2', '-i', str(video),
                                  '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'],
                                 capture_output=True, text=True, check=True, timeout=60)
        frames = [int(line.split('=')[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
        assert frames and frames[-1] == 24, (path, decoded.stdout)
        paths.append(dict(path=str(path.parent), frames=24, full_video_decode_checked=True,
                          receipt_sha256=sha(path), video_sha256=receipt['video_sha256']))
    manual = json.loads((args.root / 'manual_checks.jsonl').read_text().splitlines()[-1])
    assert not manual['runs'] and not manual['processes'] and not manual['gpu']
    tests = (args.root / 'retained_formula_tests.log').read_text()
    assert '13 passed' in tests and 'FAILED' not in tests
    source_files = ['../nerfstudio/fields/lookcloser_field.py', '../nerfstudio/models/lookcloser.py',
                    '../nerfstudio/pipelines/lookcloser_pipeline.py', 'scripts/blur_runtime.py',
                    'scripts/finalize_blur_campaign.py']
    reference = runs['f0_original']['selected']['full']
    gates = {}
    for name in ['f6_safe_exp_sh_transfer', 'f7_canonical_identity']:
        values = runs[name]['selected']['full']
        delta = {key: values[key] - reference[key] for key in reference}
        passed = delta['psnr'] >= -.1 and delta['ssim'] >= -.005 and delta['lpips'] <= .01
        gates[name] = dict(delta=delta, passed=passed)
    assert gates['f6_safe_exp_sh_transfer']['passed']
    assert not gates['f7_canonical_identity']['passed']
    audit = dict(completed_runs=len(runs), seed=42, executed_one_condition_pairs=len(pairs),
                 all_paired_initial_weights_and_first_32_batches_match=True,
                 all_checkpoint_selections_verified=True,
                 prepared_unrun_pairs=[dict(parent=p['parent'], child=p['child']) for p in evidence['pairs']
                                       if 'same_initial_parameters' not in p],
                 tests='13 passed; 2 existing AMP deprecation warnings', standard_defaults_unchanged=True,
                 fight_gates=gates, recipes=recipes, paths=paths,
                 source_sha256={f:sha(repo/f) for f in source_files},
                 recorded_active_gpu_hours=consumed_seconds(args.root)/3600,
                 aggregate_concurrent_job_hours=aggregate_job_seconds(args.root)/3600,
                 active_training_workers=0, active_controllers=0, last_supervision_check=manual['time'])
    write(args.output, audit)
    for path in [repo/'experiments/blur_ablation_fresh.md', repo/'recipes/blur_fixes/README.md']:
        links = re.findall(r'\]\(([^)]+)\)', path.read_text())
        assert all((path.parent/link).exists() for link in links), path
    print(json.dumps({k:audit[k] for k in ['completed_runs', 'executed_one_condition_pairs',
                                         'recorded_active_gpu_hours', 'active_training_workers']}))


if __name__ == '__main__':
    main()
