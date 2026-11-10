"""Read-only export audit; pending frames never count as verified completion.

Run on each artifact host. For relocated exports pass --frames-dir pointing at
the archive and --queue pointing at the original queue copied to that host.
This checks persisted evidence, not visual quality or independent tensor parity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root, queue=None, frames_dir=None):
    root = Path(root)
    frames_dir = Path(frames_dir) if frames_dir else root / 'frames'
    recipe = read(root / 'selected_recipe.json')
    recipe_sha = sha(root / 'selected_recipe.json')
    box_sha = sha(root / 'box.json')
    frames = []
    for job in read(queue or root / 'queue.json')['jobs']:
        match = re.fullmatch(r'frame(\d{6})', job['id'])
        if match:
            frames.append(int(match.group(1)))
    if not frames or len(frames) != len(set(frames)):
        raise ValueError('Queue must contain unique production frame IDs')
    results = []
    for frame in frames:
        folder = frames_dir / f'{frame:06d}'
        receipt = folder / 'frame_complete.json'
        if not receipt.exists():
            results.append(dict(frame=frame, status='pending'))
            continue
        try:
            def require(condition, message):
                if not condition:
                    raise ValueError(message)

            completion = read(receipt)
            manifest = read(folder / 'model/manifest.json')
            spec_path = folder / 'train/spec.json'
            spec = read(spec_path)
            cold_path = folder / 'train/cold_provenance.json'
            cold = read(folder / 'train/cold_setup_audit.json')
            provenance = read(cold_path)
            budget = read(folder / 'train/attempt_budget.json')
            native = read(folder / 'native/native_ready.json')
            dataset = folder / Path(spec['data']).name
            ready = read(dataset / 'ready.json')
            transforms = read(dataset / 'transforms.json')
            stage = read(folder / 'train/stage_complete.json')
            evaluations = read(folder / 'train/metrics.json')
            require(completion['frame'] == frame, 'Frame identity differs')
            require(native['frame'] == ready['frame'] == frame and
                    all(row['frame'] == frame for row in native['records']), 'Input data belongs to another frame')
            require(completion['recipe_sha256'] == recipe_sha, 'Recipe differs')
            require(completion['aabb_sha256'] == box_sha, 'AABB differs')
            require(completion['train_cameras'] == ready['train'] == 72, 'Not all72TRAIN')
            require(len(native['records']) == 72 and native['native_no_resize'], 'Native72-camera preparation absent')
            filenames = [row['file_path'] for row in transforms['frames']]
            require(len(filenames) == len(set(filenames)) == 72 and
                    len(transforms['train_filenames']) == 72 and
                    set(transforms['train_filenames']) == set(filenames), 'Actual train split is not all72 images')
            require(sha(dataset / 'transforms.json') == ready['transforms_sha256'], 'Transforms checksum differs')
            source_rows = {row['camera']: row for row in native['records']}
            require(len(source_rows) == 72, 'Native camera IDs repeat')
            require({row['physical_camera'] for row in transforms['frames']} == set(source_rows), 'Physical camera coverage differs')
            for row in transforms['frames']:
                x0, y0, x1, y1 = source_rows[row['physical_camera']]['native_crop_xyxy']
                require(row['w'] == x1 - x0 and row['h'] == y1 - y0, 'Training crop dimensions changed')
            require(ready.get('in_sample') and completion['in_sample_metrics'], 'Final metric domain differs')
            require(completion['completed_iterations'] == recipe['steps'] <= 150000, 'Iteration target differs')
            require(stage['completed_iterations'] == completion['completed_iterations'] and not stage['technical'], 'Stage completion differs')
            require(all(spec.get(key) == value for key, value in recipe['parameters'].items()), 'Training parameters differ')
            require(completion['completed_iterations'] <= budget['reserved_attempts'] <= 150000, 'Attempt ceiling differs')
            require(spec['hard_max_iterations'] <= 150000, 'Spec exceeds ceiling')
            require(cold['cold'] and all(v == 0 for v in cold['optimizer_states'].values()), 'Cold optimizer evidence absent')
            require(all(cold[k] == 0 for k in ('occupied', 'nonzero_ema', 'nonzero_frequency')), 'Cold grid evidence absent')
            require(sha(cold_path) == completion['cold_provenance_sha256'], 'Cold provenance changed')
            require(provenance['spec_sha256'] == sha(spec_path) and not provenance['technical'], 'Spec provenance differs')
            require(manifest['all_pipeline_tensors_roundtrip_equal'] and manifest['optimizer_removed'], 'Export verification receipt absent')
            require(completion['inference_only_export'], 'Export is not inference-only')
            require(manifest['step'] == completion['selected']['step'], 'Selected checkpoint differs')
            require(0 <= manifest['step'] < completion['completed_iterations'], 'Selected step outside training')
            require(manifest['inference_cap'] == recipe['selected_inference_cap'], 'Inference cap differs')
            scored = [row for row in evaluations if row['cap'] == recipe['selected_inference_cap']]
            best = min(scored, key=lambda row: (row['metrics']['lpips'], -row['metrics']['psnr']))
            require(best == completion['selected'], 'Selected model is not the best committed actor LPIPS')
            metrics = {k: completion['selected']['metrics'][k] for k in ('psnr', 'ssim', 'lpips')}
            require(metrics == manifest['metrics'] and all(math.isfinite(v) for v in metrics.values()), 'Metrics differ or are nonfinite')
            digest = sha(folder / 'model' / manifest['checkpoint'])
            require(digest == manifest['sha256'] == completion['checkpoint_sha256'], 'Model bytes differ')
            results.append(dict(frame=frame, status='verified', model_sha256=digest,
                                selected_iterations=manifest['step'] + 1, metrics=metrics))
        except (OSError, ValueError, KeyError, TypeError) as error:
            results.append(dict(frame=frame, status='failed', error=str(error)))
    return summarize(results)


def summarize(results):
    verified = [x for x in results if x['status'] == 'verified']
    digests = [x['model_sha256'] for x in verified]
    unique_models = len(digests) == len(set(digests))
    return dict(time=time.time(), expected=len(results), verified=len(verified),
                pending=sum(x['status'] == 'pending' for x in results),
                failed=sum(x['status'] == 'failed' for x in results),
                unique_model_bytes=unique_models,
                complete=bool(results) and len(verified) == len(results) and unique_models,
                scope='Persisted export evidence and streamed SHA; visual review remains separate',
                results=results)


def combine(paths):
    """Require the entire accepted361-frame sequence, exactly once."""
    rows = {}
    expected = set(range(908, 1269))
    for path in paths:
        report = read(path)
        if report['expected'] != len(report['results']):
            raise ValueError('Report count differs: ' + str(path))
        for row in report['results']:
            frame = row['frame']
            if frame not in expected or frame in rows:
                raise ValueError('Duplicate or unexpected frame: ' + str(frame))
            if row['status'] not in ('verified', 'failed', 'pending'):
                raise ValueError('Unknown audit status')
            rows[frame] = row
    missing = sorted(expected - rows.keys())
    result = summarize([rows.get(frame, dict(frame=frame, status='pending')) for frame in sorted(expected)])
    result['missing_from_input_reports'] = missing
    result['scope'] += '; exact sequence908–1268, no duplicate frame IDs'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--combine', nargs='+', metavar='REPORT')
    parser.add_argument('--queue')
    parser.add_argument('--frames-dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.combine:
        if args.root or args.queue or args.frames_dir:
            parser.error('--combine cannot be mixed with host audit arguments')
        report = combine(args.combine)
    else:
        if not args.root:
            parser.error('--root is required for a host audit')
        report = audit(args.root, args.queue, args.frames_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)
    summary = {k: v for k, v in report.items() if k not in ('results', 'missing_from_input_reports')}
    if 'missing_from_input_reports' in report:
        summary['missing_from_input_reports_count'] = len(report['missing_from_input_reports'])
    print(json.dumps(summary))
    raise SystemExit(1 if report['failed'] or not report['unique_model_bytes'] or args.require_complete and not report['complete'] else 0)
