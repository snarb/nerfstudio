"""Seal a cold direct-PQ-code native control, without starting an optimizer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/fsx/tmp/marshawn_001660_hd')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def geometry_identity(original, current):
    """Exclude only image extension, checking all remaining fields exactly."""
    a, b = json.loads(original.read_text()), json.loads(current.read_text())
    def normalize(payload):
        p = copy.deepcopy(payload)
        for f in p['frames']:
            f['file_path'] = str(Path(f['file_path']).with_suffix(''))
        for key in ['train_filenames', 'val_filenames', 'test_filenames']:
            if key in p:
                p[key] = [str(Path(n).with_suffix('')) for n in p[key]]
        return p
    if normalize(a) != normalize(b):
        raise ValueError('EXR geometry differs beyond image extension')
    names = [f['physical_camera'] for f in b['frames']]
    if len(names) != 66 or len(set(names)) != 66:
        raise ValueError('Expected unique physical66')
    train = [f for f in b['frames'] if Path(f['file_path']).name.startswith('train_')]
    ev = [f['physical_camera'] for f in b['frames'] if Path(f['file_path']).name.startswith('eval_')]
    if len(train) != 63 or sorted(ev) != ['DC', 'HC', 'KA']:
        raise ValueError('Train/eval physical IDs changed')
    return dict(exact_except_image_extension=True, train=63, eval=ev,
                original_transforms_sha256=sha(original), current_transforms_sha256=sha(current),
                checked='All fields, all66: numeric K/c2w/distortion/dimensions/camera_model/physical IDs/splits/masks')


def stage(data, out):
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
    import numpy as np
    data, out = data.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    code = out/'code'; code.mkdir(exist_ok=True)
    # Copy new wrappers only; imported frozen final recipe keeps its original seal.
    for name in ['marshawn_direct_pq_pipeline.py', 'run_marshawn_direct_pq.py']:
        shutil.copyfile(Path(__file__).parent/name, code/name)
    identity = geometry_identity(ROOT/'data/transforms.json', data/'transforms.json')
    write(out/'geometry_identity.json', identity)
    source = ROOT/'free_space_candidates/manifest.json'
    manifest = json.loads(source.read_text())
    original_binding = copy.deepcopy(manifest['bindings'])
    manifest['bindings']['transforms_sha256'] = sha(data/'transforms.json')
    manifest['derived_colour_binding'] = dict(
        parent_manifest=str(source), parent_manifest_sha256=sha(source),
        parent_bindings=original_binding, geometry_identity_sha256=sha(out/'geometry_identity.json'),
        original_images_remain_triangulation_provenance=True,
        current_images_are_reencoding_of_same_physical_frame=True, no_new_geometry_or_eval_rgb=True)
    geo = out/'geometry'; geo.mkdir(exist_ok=True)
    for v in manifest['variants']:
        src = source.parent/v['file']
        if sha(src) != v['sha256']: raise ValueError('Original geometry archive changed')
        shutil.copyfile(src, geo/v['file'])
    write(geo/'manifest.json', manifest)
    frames = json.loads((data/'transforms.json').read_text())['frames']
    files = {}; stats = []
    for f in frames:
        image = cv2.imread(str(data/f['file_path']), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.float32 or image.shape != (f['h'], f['w'], 3):
            raise ValueError('Expected RGB numeric float32 EXR: '+f['physical_camera'])
        if not np.isfinite(image).all() or image.min() < 988/65535.-1e-7 or image.max() > 1:
            raise ValueError('Invalid PQ code range '+f['physical_camera'])
        stats.append(dict(camera=f['physical_camera'], min=float(image.min()), max=float(image.max())))
        for field in ['file_path', 'mask_path']:
            files[f[field]] = sha(data/f[field])
    contract = json.loads((data/'color_contract.json').read_text())
    if contract['eval_rgb_used_to_fit_gains'] is not False: raise ValueError('No eval gain fitting')
    for name in ['color_contract.json', 'transforms.json', 'complete.json']:
        files[name] = sha(data/name)
    gate = dict(approved=True, scope='Encoded-PQ numerical control only; not physical radiance',
                data=str(data), transforms_sha256=sha(data/'transforms.json'), files_sha256=files,
                geometry_identity_sha256=sha(out/'geometry_identity.json'),
                canonical_colour_audit_sha256=sha(ROOT/'report/canonical_colour_independent/audit.json'),
                numeric_exr_stats=stats, cache='float32', floor_code=988/65535.,
                uses_eval_rgb_to_fit=False, training_authorized=False)
    write(out/'color_gate.json', gate)
    for p in [ROOT/'code_native', ROOT/'code_native/LookCloser/scripts', ROOT/'code_sfm/LookCloser/scripts',
              ROOT/'final_pair/code', code]:
        sys.path.insert(0, str(p))
    import yaml
    from marshawn_direct_pq_pipeline import DirectPQNativePipeline
    cfg = yaml.load((ROOT/'final_pair/config_radius0.yml').read_text(), Loader=yaml.Loader)
    cfg.output_dir = out/'run'; cfg.experiment_name = 'marshawn_1660_canonical_direct_pq'
    cfg.pipeline._target = DirectPQNativePipeline
    cfg.pipeline.datamanager.data = data
    cfg.pipeline.datamanager.dataparser.data = data
    cfg.pipeline.datamanager.cache_images_type = 'float32'
    cfg.pipeline.final_color_gate = str(out/'color_gate.json')
    cfg.pipeline.final_color_gate_sha256 = sha(out/'color_gate.json')
    cfg.pipeline.final_geometry_manifest = str(geo/'manifest.json')
    cfg.pipeline.final_grid_artifact_dir = str(out/'grids')
    cfg.pipeline.direct_pq_source_sha256 = sha(code/'marshawn_direct_pq_pipeline.py')
    cfg.pipeline.direct_pq_floor_code = 988/65535.
    cfg.max_num_iterations = 22000
    assert cfg.steps_per_save == 2000 and cfg.steps_per_eval_all_images == 2000
    (out/'config_STAGED_NOT_LAUNCHED.yml').write_text(yaml.dump(cfg))
    write(out/'stage_receipt.json', dict(status='STAGED_NOT_LAUNCHED', optimizer_updates=0,
        data=str(data), data_gate_sha256=sha(out/'color_gate.json'), geometry_identity=identity,
        config_sha256=sha(out/'config_STAGED_NOT_LAUNCHED.yml'),
        source_sha256={p.name:sha(p) for p in code.iterdir() if p.is_file()},
        native_nn_unchanged=True, encoded_compositing_not_physical=True,
        fixed_background_code=988/65535., maximum_iterations=22000, first_visual_gate=2000))
    print(json.dumps(dict(staged=str(out), status='NO_TRAINING', images=66)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'color_variants/canonical_signed_ap1_pq_codes_exr')
    p.add_argument('--out', type=Path, default=ROOT/'native_direct_pq_STAGED')
    a = p.parse_args(); stage(a.data, a.out)
