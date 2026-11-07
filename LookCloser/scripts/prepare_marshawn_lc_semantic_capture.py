"""CPU-only sealing of a read-only best-LookCloser semantic capture proposal."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CAMPAIGN = Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(c, output):
    if output.exists():
        raise FileExistsError('New immutable proposal namespace required')
    query_path = c/'frozen_native_semantic_CPU_v2/manifest.json'
    qual_path = c/'frozen_native_semantic_head_technical_plan_v1/label_qualification.json'
    expected = {query_path: '1562bcfaf5ec5da00a840da210962ef136d521fbca0d1beb5bf30d4a9aeb30ba',
                qual_path: '944d88dbbfd24a0021c1166dbfe1fa949047b8d929b9b0cad65e6c84f07ff4f0'}
    for p, digest in expected.items():
        if sha(p) != digest:
            raise ValueError('Unqualified query/label generation')
    q = json.loads(query_path.read_text())
    qualification = json.loads(qual_path.read_text())
    floor = json.loads((c/'data/transforms.json').read_text())
    native = json.loads((Path(q['native_data'])/'transforms.json').read_text())
    floor_frames = {f['physical_camera']: f for f in floor['frames']}
    native_frames = {f['physical_camera']: f for f in native['frames']}
    frame = np.array(q['native_world_to_box'])
    if len(q['train_cameras']) != 63 or set(q['train_cameras']) & {'DC', 'HC', 'KA'}:
        raise ValueError('Qualified TRAIN63 only')
    total, usable, pose_error, overrides = 0, 0, 0., []
    records = []
    for row in q['records']:
        cam = row['camera']
        a, b = floor_frames[cam], native_frames[cam]
        pose_error = max(pose_error, float(abs(np.array(a['transform_matrix'])-frame@np.array(b['transform_matrix'])).max()))
        if any(a[k] != b[k] for k in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']):
            raise ValueError('Native query raster/intrinsics differ from LC floor camera')
        if sha(row['archive']) != row['archive_sha256']:
            raise ValueError('Fixed TRAIN queries changed')
        with np.load(row['archive'], allow_pickle=False) as src:
            pixels = src['pixel_xy']; target = src['semantic_target']; weight = src['label_weight']
            if len(pixels) and ((pixels < 0).any() or (pixels[:, 0] >= a['w']).any() or (pixels[:, 1] >= a['h']).any()):
                raise ValueError('Query outside native raster')
            for unknown in qualification['unknown_queries']:
                if unknown['camera'] != cam:
                    continue
                i = unknown['camera_ray_index']
                if total+i != unknown['global_query_index'] or pixels[i].tolist() != unknown['pixel_xy'] or target[i] != 1 or weight[i] != 1:
                    raise ValueError('Exact UNKNOWN override record mismatch')
                overrides.append(dict(unknown))
            records.append(dict(row, rays=len(pixels)))
            total += len(pixels); usable += int(len(pixels) > 0)
    if pose_error > 1e-12 or total != 69888 or usable != 60 or len(overrides) != 2:
        raise ValueError('Floor identity/query-count proof failed')
    archive = c/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin'
    branch = c/'semantic_quarter_distortion_candidate_20261107'
    source = json.loads((archive/'source_manifest.json').read_text())
    if sha(archive/'source_manifest.json') != 'ad32a034c492314162fac8d966be2215c834393a1bb984beb05f1eed8f9c43b3':
        raise ValueError('Wrong scientific generation')
    for name, digest in source['files_sha256'].items():
        if sha(branch/'code'/name) != digest:
            raise ValueError('Scientific source drift: '+name)
    original = json.loads((archive/'archive_receipt.json').read_text())
    archive_hashes = original['files_sha256']
    for name, digest in archive_hashes.items():
        if name != 'checkpoint.ckpt' and sha(archive/name) != digest:
            raise ValueError('Source archive drift: '+name)
    source_root = c/'independent_occupancy/source_mask_W32_counterfactual_v1'
    helpers = source_root/'helpers'
    bound = {str(p): sha(p) for p in [query_path, qual_path, c/'data/transforms.json', c/'data/ready.json',
        Path(q['native_data'])/'transforms.json', archive/'archive_receipt.json', archive/'source_manifest.json',
        source_root/'policy.json']}
    for p in helpers.glob('*.py'):
        bound[str(p)] = sha(p)
    semantic_source = branch/'source/manifest.json'
    bound[str(semantic_source)] = sha(semantic_source)
    for name, digest in json.loads(semantic_source.read_text())['files_sha256'].items():
        if sha(semantic_source.parent/name) != digest:
            raise ValueError('Frozen semantic support changed')
        bound[str(semantic_source.parent/name)] = digest
    output.mkdir(parents=True)
    (output/'code').mkdir()
    code = {}
    for name in ['marshawn_lc_semantic_capture.py', 'prepare_marshawn_lc_semantic_capture.py', 'run_marshawn_lc_semantic_capture.py']:
        p = output/'code'/name
        shutil.copyfile(Path(__file__).parent/name, p)
        code[name] = sha(p)
    proof = dict(LookCloser_world_to_floor=np.eye(4).tolist(),
        floor_vs_old_NGP_transformed_pose_max_error=pose_error,
        all63_intrinsics_and_rasters_exact=True, rays=total, usable_train_cameras=usable,
        declared_train_cameras=63, source_files_verified=len(source['files_sha256']),
        checkpoint_available_locally=(archive/'checkpoint.ckpt').exists(),
        GPU_started=False, RGB_images_read=False, masks_read=False,
        limitation='Actual parsed camera/ray identity and source RGB/alpha parity remain runtime gates.')
    write(output/'CPU_proof.json', proof)
    manifest = dict(kind='FROZEN_BEST_LC_NATIVE_SEMANTIC_CAPTURE_PLAN_V1', authorization=False,
        phase='capture_only', campaign=str(c), source_archive=str(archive), branch=str(branch),
        source_archive_sha256=archive_hashes, scientific_files_sha256=source['files_sha256'],
        helpers=str(helpers), bound_inputs=bound, code_sha256=code,
        python='/home/pavlo/.conda/envs/nerfstudio/bin/python', torch_version='2.7.1+cu128', cuda_version='12.8',
        source_FR=.3, source_step=126316, bounds=q['bounds'], world_to_box=np.eye(4).tolist(),
        train_cameras=q['train_cameras'], records=records, unknown_queries=overrides,
        chunk=128, allocator_cap_gib=8, max_process_mib=10240,
        max_seconds=1800, max_worker_RSS_gib=12, min_available_RAM_gib=8, min_disk_gib=20,
        GPU_policy='Exact exclusive physical UUID in CUDA_VISIBLE_DEVICES; global controller lease and parent-bound receipt mandatory; only logical cuda:0.',
        runtime_path_relocation='Only campaign-root path relocation, preserving all sealed bytes. Do not edit scientific source/config files.',
        limitations=['No semantic training/head transfer/export authorized.',
            'No opacity pruning during native eval (sigma callback None); occupancy skipping and legacy cap may omit geometry.',
            'Counts >=1024 flag saturation; actual discarded interval count is unavailable without instrumenting sampler and is not inferred.',
            'Native weights partition source occlusion only. Physical actor export needs independent full-path density sampling and recomputed transmittance.',
            'Sparse qualified TRAIN queries cannot certify perfect occupancy, hidden geometry, contour/hair/feet preservation or artifact-free novel views.'])
    write(output/'manifest.json', manifest)
    write(output/'release_TEMPLATE.json', dict(reviewer='root', capture_authorized=False,
        manifest_sha256=sha(output/'manifest.json'), code_sha256=code,
        note='Root must review concrete package and create separate release. This template never authorizes GPU.'))
    print(json.dumps(dict(output=str(output), manifest_sha256=sha(output/'manifest.json'), **proof)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=CAMPAIGN)
    p.add_argument('--output', type=Path, default=CAMPAIGN/'frozen_LC_semantic_capture_plan_v1')
    a = p.parse_args()
    prepare(a.campaign, a.output)
