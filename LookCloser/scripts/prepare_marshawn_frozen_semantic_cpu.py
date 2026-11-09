"""Seal TRAIN-only semantic queries for a frozen native NGP; never launch CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(c, output):
    output.mkdir(parents=True, exist_ok=False)
    data = c/'data'
    native_data = c/'native_semantic_grid_20000/native_4k_continuation_plan/adapter/data_HD_identity'
    ready = json.loads((data/'ready.json').read_text())
    current = json.loads((data/'transforms.json').read_text())
    native = json.loads((native_data/'transforms.json').read_text())
    T = np.array(ready['world_to_box'])
    trains = [f for f in current['frames'] if f['file_path'] in current['train_filenames']]
    native_frames = {f['physical_camera']: f for f in native['frames']}
    names = [f['physical_camera'] for f in trains]
    assert len(names) == len(set(names)) == 63 and not set(names) & {'DC', 'HC', 'KA'}
    assert set(names) == {f['physical_camera'] for f in native['frames'] if f['file_path'] in native['train_filenames']}
    support_root = c/'native_semantic_grid_20000/occlusion_v2/W64'
    support_manifest = support_root/'support_manifest.json'
    support_rows = {r['camera']: r for r in json.loads(support_manifest.read_text())['records']}
    bag_proof = c/'native_semantic_grid_20000/bag_qualification/train_bag_qualification.json'
    bags = {r['camera']: r for r in json.loads(bag_proof.read_text())['records']}
    bound = {str(p): sha(p) for p in [data/'ready.json', data/'transforms.json', native_data/'transforms.json', support_manifest, bag_proof]}
    rng = np.random.default_rng(42)
    rows = []
    (output/'queries').mkdir()
    for frame in trains:
        name = frame['physical_camera']
        original = native_frames[name]
        assert np.array_equal(np.array(frame['transform_matrix']), T @ np.array(original['transform_matrix'])), name
        assert all(frame[k] == original[k] for k in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'])
        assert all(frame.get(k, 0) == original.get(k, 0) == 0 for k in ['k1', 'k2', 'k3', 'k4', 'p1', 'p2'])
        sr = support_rows[name]
        raw_path, support_path = Path(sr['source_archive']), Path(sr['archive'])
        assert sha(raw_path) == sr['source_archive_sha256'] and sha(support_path) == sr['archive_sha256']
        bound[str(raw_path)] = sr['source_archive_sha256']
        bound[str(support_path)] = sr['archive_sha256']
        with np.load(raw_path) as z:
            coverage = z['nominal_coverage'].copy()
            ineligible = z['source_ineligible'].copy()
        with np.load(support_path) as z:
            support, unknown, valid = [z[k].copy() for k in ['support', 'unknown', 'valid']]
        assert coverage.shape == valid.shape == (frame['h'], frame['w'])
        assert valid.dtype == support.dtype == unknown.dtype == np.bool_
        eligible = valid & ~ineligible
        # Conservative categorical positives only; coverage is never alpha GT.
        interior = coverage >= .999
        positive = interior & (distance_transform_edt(np.pad(interior, 1))[1:-1, 1:-1] > 2) & eligible & ~unknown
        clear = eligible & ~support & ~unknown
        bag = np.zeros_like(valid)
        shell = np.zeros_like(valid)
        if name in bags:
            b = bags[name]
            for path_key, hash_key in [('exclusion_HD', 'exclusion_sha256'), ('unknown_HD', 'unknown_sha256')]:
                p = Path(b[path_key]); assert sha(p) == b[hash_key]; bound[str(p)] = b[hash_key]
            bag, shell = np.load(b['exclusion_HD']), np.load(b['unknown_HD'])
            assert not (bag & shell).any()
            bag = bag & eligible
        positive &= ~bag & ~shell
        clear &= ~shell
        near = clear & (distance_transform_edt(~support) <= 24)
        far = clear & ~near
        categories = [('actor_interior', positive, 512, 1), ('near_qualified_clear', near, 512, 0),
                      ('far_qualified_clear', far, 128, 0), ('qualified_bag_interior', bag, 128, 0)]
        pixels, targets, kinds = [], [], []
        counts = {}
        for kind, mask, count, target in categories:
            indices = np.flatnonzero(mask)
            counts[kind] = dict(available=int(len(indices)), sampled=min(count, len(indices)))
            chosen = rng.choice(indices, min(count, len(indices)), replace=False)
            y, x = np.unravel_index(chosen, mask.shape)
            pixels.extend(np.stack([x, y], -1).tolist())
            targets.extend([target]*len(chosen)); kinds.extend([kind]*len(chosen))
        path = output/'queries'/f'{name}.npz'
        pixel_array = np.array(pixels, np.int32).reshape(-1, 2)
        np.savez_compressed(path, pixel_xy=pixel_array, semantic_target=np.array(targets, np.float32),
            label_weight=np.ones(len(targets), np.float32), category=np.array(kinds), camera=np.array(name),
            nominal_coverage_at_query=coverage[pixel_array[:, 1], pixel_array[:, 0]])
        rows.append(dict(camera=name, archive=str(path), archive_sha256=sha(path), counts=counts,
            unknown_excluded_pixels=int((unknown | shell | ineligible | ~valid).sum()),
        note='Qualified bag interior may override broad mat UNKNOWN; its separately reviewed shell remains UNKNOWN. Empty-camera records remain explicit; do not invent labels where full-raster visibility is UNKNOWN.'))
    state_path = c/'native_resolution_pair_100000/state.json'
    selection = json.loads(state_path.read_text())['arms']['HD']['selection']
    assert selection['step'] == 190000
    config = next((c/'native_resolution_pair_100000/configs').glob('*/HD.yml'))
    # Hash the actual selected source; do not deserialize field tensors on CPU.
    assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
    bound[str(state_path)] = sha(state_path); bound[str(config)] = sha(config)
    bound[selection['checkpoint']] = selection['checkpoint_sha256']
    source_release = c/'native_semantic_grid_20000/native_continuation_plan/root_release.json'
    root_release = json.loads(source_release.read_text())
    write(output/'manifest.json', dict(kind='FROZEN_NATIVE_VISIBILITY_SEMANTICS_CPU_V1', train_cameras=names,
        records=rows, native_source={k: selection[k] for k in ['step', 'checkpoint', 'checkpoint_sha256']},
        native_config=str(config), native_python_path=root_release['python_path'], native_data=str(native_data),
        bounds=ready['aabb'], native_world_to_box=ready['world_to_box'],
        native_to_floor_pose_exact_all63=True, intrinsics_and_raster_exact_all63=True,
        bound_inputs=bound, GPU_started=False, optimizer_started=False, EVAL_RGB_or_masks_read=False,
        RGB_images_read=False, class_targets_are_optical_alpha=False,
        sampling='Fixed seed42 stratified per camera;512 actor-interior,512 near-clear,128 far-clear,128 qualifiedbag. Balanced classifier objective; not an unbiased full-image RGB objective.',
        protected_policy='No negative from mask complement alone; qualified valid clear outside W64 support AND UNKNOWN, plus reviewed bag interiors. All hair/contact/mat UNKNOWN excluded.',
        limitation='No native weight cache captured yet. Positive-ray capacity and multi-layer identifiability remain mandatory technical gates.'))
    print(json.dumps(dict(output=str(output), train_cameras=63, rays=sum(sum(v['sampled'] for v in r['counts'].values()) for r in rows), GPU_started=False, optimizer_started=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.campaign, a.output)
