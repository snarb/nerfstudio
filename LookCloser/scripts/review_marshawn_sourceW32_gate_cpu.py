"""Saved-render and actual-grid diagnostics; no model, target or grid mutation."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image
import torch
from marshawn_source_point_admission import SourcePointAdmission
from marshawn_source_margin_admission import GeometricSourcePointAdmission
from marshawn_semantic_cell_admission import WholeCellAdmission


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(row):
    return {key: row['metrics'][key] for key in ('psnr', 'ssim', 'lpips')}


def review(root, queries, step):
    torch.set_num_threads(4)
    xyz, labels = [], []
    hashes = {}
    for name in ('clothing103.json', 'shoe6.json'):
        path = queries/name
        hashes[str(path)] = sha(path)
        for point in read(path)['pixels']:
            xyz.append(point['witness_xyz'])
            labels.append({k: point.get(k) for k in ('camera_name', 'witness_id', 'actor', 'region')})
    xyz = np.array(xyz)
    assert xyz.shape == (109, 3)
    source = root/'margin/source_grid.npz'
    manifest = read(root/'margin/manifest.json')
    assert sha(source) == manifest['source_grid_sha256']
    for helper in (SourcePointAdmission, GeometricSourcePointAdmission):
        path = Path(inspect.getfile(helper))
        assert sha(path) == manifest['helper_sha256'][path.name]
    with np.load(source, allow_pickle=False) as data:
        lookup = GeometricSourcePointAdmission(torch.from_numpy(data['semantic_gate_pad1']),
            data['aabb'], margin=.002, epsilon=2e-7, base_class=SourcePointAdmission)
        retained = lookup(torch.from_numpy(xyz)).numpy()
        cell_lookup = WholeCellAdmission(data['semantic_gate_pad1'], data['aabb'], epsilon=2e-7)
    cell_helper = Path(inspect.getfile(WholeCellAdmission))
    assert sha(cell_helper) == sha(root/'code/LookCloser/scripts'/cell_helper.name)
    assert retained.all()
    result = dict(step=step, CPU_only=True, no_model_loaded=True, inputs=hashes,
        source_grid_sha256=sha(source), fixed_W32_point_admission_retained=int(retained.sum()),
        sparse_unique_points=len(np.unique(xyz, axis=0)), arms={}, contours={})
    archives = {}
    for arm in ('original_mapped', 'source_margin002'):
        folder = root/'review_archives'/arm/f'step-{step:09d}'
        assert read(folder/'fetch_verified.json')['passed']
        archive = read(folder/'receipt.json')
        for name, digest in archive['files_sha256'].items():
            assert sha(folder/name) == digest, name
        assert sha(folder/'margin_manifest.json') == sha(root/'margin/manifest.json')
        assert sha(folder/'spec.json') == sha(root/'specs'/('production_'+arm)/'spec.json')
        grid = read(folder/'grid.json')
        assert grid['schema'] == 2 and grid['radius'] == 0 and not grid['cpu_reconstruction_used']
        assert sha(folder/'grid.npz') == grid['grid_archive_sha256']
        with np.load(folder/'grid.npz', allow_pickle=False) as data:
            core, active, admission, ema = [data[k][0] for k in ('core','active','admission','ema')]
            assert np.array_equal(active, core & admission)
            assert np.isfinite(ema).all()
            bounds = data['cascade_aabbs'][0].reshape(2,3)
            indices = np.floor((xyz-bounds[0])/(bounds[1]-bounds[0])*128).astype(int)
            assert ((indices>=0)&(indices<128)).all()
            occupied = active[tuple(indices.T)]
            native = dict(core_cells=int(core.sum()), active_cells=int(active.sum()),
                admission_cells=int(admission.sum()), outside_admission=int((active & ~admission).sum()),
                EMA_finite=True, EMA_max=float(ema.max()), actual_capture_step=grid['core_update_step'],
                witness_cell_active_count=int(occupied.sum()),
                witness_cell_inactive=[labels[i] for i in np.flatnonzero(~occupied)],
                authority=grid['authority'], checkpoint_sha256=grid['checkpoint_sha256'])
            if arm == 'source_margin002':
                active_indices = np.argwhere(active)
                pitch = (bounds[1]-bounds[0])/128
                lower = bounds[0]+active_indices*pitch
                # Expanded-cell/source intersection is conservative. Its
                # outside-domain rule admits extra boundary cells, never
                # labels UNKNOWN as definitely excluded.
                intersects = cell_lookup.boxes_admitted(lower-.002, lower+pitch+.002)
                native['whole_cells_disjoint_from_bound_point_gate'] = int((~intersects).sum())
                native['wholecell_helper_sha256'] = sha(cell_helper)
                native['wholecell_diagnostic_scope'] = 'No possible admitted point in excluded whole cells; not physical empty-space ground truth.'
        current = read(folder/'renders/metrics.json')
        zero = read(root/'runs'/('zero_'+arm)/'branch_zero/renders/step-000126316/metrics.json')
        result['arms'][arm] = dict(metrics=score(current), own_zero_metrics=score(zero),
            own_zero_delta={k:score(current)[k]-score(zero)[k] for k in score(current)},
            by_actor={name:{k:row[k] for k in ('psnr','ssim','lpips')} for name,row in current['by_actor'].items()},
            by_view={name:{k:row[k] for k in ('psnr','ssim','lpips')} for name,row in current['by_view'].items()}, native_grid=native,
            own_runtime_selector_required=True, region_diagnostics=read(folder/'region_diagnostics.json'))
        archives[arm] = folder/'renders'
    for cam, x, y in [('KA',1028,331), ('HC',1052,576), ('DC',850,85)]:
        values = {}
        dirs = {**archives, 'sourceA':root/'runs/zero_original_mapped/branch_zero/renders/step-000126316',
                'sourceW32zero':root/'runs/zero_source_margin002/branch_zero/renders/step-000126316'}
        for name, folder in dirs.items():
            alpha = np.array(Image.open(folder/(cam+'_alpha.png'))).astype(float)/65535
            rgb = np.array(Image.open(folder/(cam+'.png'))).astype(float)/255
            values[name] = {'alpha':alpha[y,x-1:x+2].tolist(), 'RGB':rgb[y,x-1:x+2].tolist()}
        result['contours'][cam] = dict(center=[x,y], neighbors='x-1,x,x+1', values=values)
    assert not torch.cuda.is_initialized()
    result['limitations'] = ['109 observations are 41 unique points, not whole-hair/feet coverage.',
        'Inactive native witness cells are diagnostic; recovered feature uncertainty and native ray traversal must be examined before calling geometry missing.',
        'Neither high EMA nor admitted density certifies a physical surface; skipped space is UNKNOWN.',
        'Alpha is predicted opacity, not compared to alpha-one ground truth. V1 unchanged.']
    out = root/'reviews'/f'step-{step:09d}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'CPU_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(step=step, metrics={k:v['metrics'] for k,v in result['arms'].items()},
        active_witnesses={k:v['native_grid']['witness_cell_active_count'] for k,v in result['arms'].items()})))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    args=p.parse_args()
    review(args.root, args.queries, args.step)
