"""Rank archived fixed-domain actor measurements without changing any runner.

Raw LPIPS ranking follows the current user objective. Historical checkpoint
selectors and training plateau trackers remain recorded by their own runners.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = {
    'R0': 'semantic_actor_progress', 'R1': 'semantic_actor_r1_progress',
    'W32': 'semantic_actor_w32_progress', 'bag-corrected': 'semantic_actor_bagloss_progress',
    'source-point': 'semantic_actor_sourcepoint_progress', 'distortion0.1': 'semantic_cold_lambda01_progress',
    'half-step': 'semantic_cold_halfstep_progress', 'quarter-step': 'semantic_cold_quarterstep_progress',
    'focused-frequency': 'semantic_frequency_focus_progress',
    'focused-frequency+distortion0.1': 'semantic_frequency_focus_distortion_progress',
    'quarter-step+distortion0.1': 'semantic_quarter_distortion_progress',
    'TRAIN4K': 'semantic_4k_original_r0',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def domain(record):
    return record['view'], record['actor'], tuple(record['crop_xyxy']), record['actor_pixels'], record['domain']


def main(a):
    if a.output.exists():
        raise ValueError('Preserve prior review generation')
    root = a.campaign/'independent_occupancy'
    reference = root/'semantic_actor_progress/step-000111316/renders/metrics.json'
    expected = sorted(domain(r) for r in json.loads(reference.read_text())['records'])
    rows, missing, inputs = [], [], {str(reference):sha(reference)}
    for label, folder in CANDIDATES.items():
        files = sorted((root/folder).glob('step-*/renders/metrics.json'))
        if not files:
            missing.append(label)
            continue
        values = []
        for file in files:
            d = json.loads(file.read_text())
            if len(d['records']) != 6 or sorted(domain(r) for r in d['records']) != expected:
                raise ValueError('Different fixed actor evaluation domain: '+str(file))
            metrics = {k:float(d['metrics'][k]) for k in ('psnr','ssim','lpips')}
            for key, value in metrics.items():
                if not math.isfinite(value) or abs(value-sum(r[key] for r in d['records'])/6)>1e-6:
                    raise ValueError('Invalid equal actor/view aggregation')
            inputs[str(file)] = sha(file)
            values.append(dict(label=label, step=d['step'], **metrics,
                by_view={v:{k:r[k] for k in metrics} for v,r in d['by_view'].items()},
                by_actor={v:{k:r[k] for k in metrics} for v,r in d['by_actor'].items()},
                source_metrics=str(file), source_metrics_sha256=sha(file)))
        best = min(values, key=lambda r:(r['lpips'],-r['psnr'],r['step']))
        best.update(observed_through_step=max(r['step'] for r in values), archived_evaluations=len(values))
        rows.append(best)
    rows.sort(key=lambda r:(r['lpips'],-r['psnr'],r['step']))
    report = dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        kind='PROVISIONAL_ARCHIVED_V1_ACTOR_METRIC_REVIEW', ranking='Raw minimum LPIPS; PSNR only on exact ties',
        rows=rows, candidates_without_archived_evaluation=missing, inputs_sha256=inputs,
        helper_sha256=sha(Path(__file__)), historical_selectors_or_training_changed=False,
        source_models_or_renders_changed=False, geometry_or_global_artifact_clearance_certified=False,
        limitation='Snapshots may belong to running models and have different ages. This is an image-metric shortlist, not final model promotion; independent geometry/contour/exterior review is required.')
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Provisional Marshawn actor metric review', '', report['limitation'], '',
           '| Candidate | Raw LPIPS best step | PSNR | SSIM | LPIPS | Observed through |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|{label}|{step}|{psnr:.5f}|{ssim:.6f}|{lpips:.8f}|{observed_through_step}|'.format(**r))
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(candidates=len(rows),first3=[{k:r[k] for k in ('label','step','psnr','ssim','lpips')} for r in rows[:3]])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
