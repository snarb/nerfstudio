"""Opt-in native 6K radiance sampling with frozen cinematic surface/source choices.

The remote action reads immutable PQ16 PNGs, decodes only the selected original
pixel taps, and returns linear Rec709 samples. It never downsizes an image.
The local action reuses verified HD graph/source labels and identical geometry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

BASE = Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
OUT = Path('/mnt/data/dec5_cinematic_wide_spiral_6k_v2')
REMOTE = 'ubuntu@dev3'
REMOTE_ROOT = '/fsx/tmp/lookcloser_cinematic_native6k_v2'
RAW = '/fsx/oregon/projects/Dec5Shoots/workspace/DEC5_5A_3/subpix_out/DEC5_5A_3/working/fullres_pq16'
REMOTE_PYTHON = '/home/ubuntu/anaconda3/envs/nerfstudio/bin/python'
SCALE = np.array([5461 / 1920, 3072 / 1080])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def call(args): subprocess.run([str(x) for x in args], check=True)


def bilinear(image, uv):
    """Sample zero-based pixel centers, with renderer-compatible border clamp."""
    h, w = image.shape[:2]; uv = np.clip(uv, [0, 0], [w-1, h-1])
    ix = np.floor(uv).astype(np.int64); frac = uv - ix
    out = np.zeros((len(uv), 3), np.float32)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        weight = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
        out += image[np.minimum(ix[:, 1]+dy, h-1), np.minimum(ix[:, 0]+dx, w-1)] * weight[:, None]
    return out


def remote_sample(job):
    import cv2
    import convert_dec5_5a3_pq16_to_exr as decoder
    cv2.setNumThreads(1)
    spec = read(job/'job.json')
    # NpzFile lazily seeks one shared ZipFile. Materialize before worker threads;
    # concurrent lazy loads can race in the remote Python3.8 zipfile reader.
    with np.load(job/'uv.npz') as archive:
        coordinates = {key:archive[key] for key in archive.files}
    assert spec['frame'].isdigit() and len(spec['frame']) == 6
    start = time.monotonic()
    def one(row):
        ci = row['index']; name = row['camera']; assert '/' not in name
        source = Path(RAW)/spec['frame']/'images'/f"{name}_{spec['frame']}.png"
        data = source.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assert bgr.shape == (3072, 6144, 3) and bgr.dtype == np.uint16
        # Invert both Pillow resizing lattices. Crop origin enters only here.
        uv = (coordinates[str(ci)].astype(np.float64)+.5)*SCALE-.5
        uv = np.clip(uv, [0, 0], [5460, 3071]); ix = np.floor(uv).astype(np.int64); frac = uv-ix
        sampled = np.zeros((len(uv), 3), np.float32)
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            raw = bgr[np.minimum(ix[:, 1]+dy, 3071), 341+np.minimum(ix[:, 0]+dx, 5460), ::-1]
            linear = decoder.pq_decode_array(raw.astype(np.float32)/np.float32(65535)) / np.float32(decoder.GAIN_TO_NITS)
            linear[raw == decoder.FLOOR_U16] = 0
            linear = linear @ decoder.AP1_TO_REC709.astype(np.float32).T
            weight = (frac[:, 0] if dx else 1-frac[:, 0])*(frac[:, 1] if dy else 1-frac[:, 1])
            sampled += linear * weight[:, None]
        assert np.isfinite(sampled).all()
        return ci, sampled, dict(index=ci, camera=name, path=str(source), sha256=digest,
            bytes=len(data), dimensions=[6144, 3072], selected_samples=len(uv))
    result = {}; provenance = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ci, samples, source in pool.map(one, spec['sources']):
            result[str(ci)] = samples; provenance.append(source)
    np.savez(job/'rgb.npz', **result)
    write(job/'result.json', dict(frame=spec['frame'], sources=provenance,
        uv_sha256=sha(job/'uv.npz'), rgb_sha256=sha(job/'rgb.npz'),
        decoder_sha256=sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')),
        worker_sha256=sha(__file__), seconds=time.monotonic()-start,
        resize=False, source_crop=[341, 0, 5802, 3072], gain_to_nits=decoder.GAIN_TO_NITS))


def initialize():
    from joint_temporal_texture import ROOT as COLOR, CALIBRATION
    q = read(BASE/'request.json')
    assert not q['recipe']['static_registration'] and not q['uses_heldout_rgb']
    for filename, key in [('parameters.npz', 'profiles_sha256'), ('exposure.json', 'exposure_sha256')]:
        assert sha(COLOR/filename) == q[key]
    assert sha(CALIBRATION) == q['calibration_sha256']
    config = dict(parent=str(BASE), parent_request_sha256=sha(BASE/'request.json'),
        script_sha256=sha(__file__), decoder_sha256=sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')),
        source_root=RAW, output_dimensions=[1080, 1920], fps=24, frame_count=150,
        camera_inventory_sha256=hashlib.sha256(json.dumps(q['inventory'], sort_keys=True).encode()).hexdigest(),
        source_scale_xy=SCALE.tolist(), pixel_mapping='u6=(uHD+0.5)*5461/1920-0.5; v6=(vHD+0.5)*3072/1080-0.5; raw_x=u6+341',
        fixed_profiles_sha256=q['profiles_sha256'], fixed_exposure_sha256=q['exposure_sha256'],
        reuse_baseline_source_labels=True, geometry_changed=False, heldout_rgb_used=False,
        native_linear_taps=True, synthetic_super_resolution=False)
    OUT.mkdir(exist_ok=True)
    if (OUT/'request.json').exists(): assert read(OUT/'request.json') == config
    else: write(OUT/'request.json', config)
    call(['ssh', REMOTE, 'mkdir', '-p', REMOTE_ROOT])
    call(['scp', '-q', __file__, Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py'), f'{REMOTE}:{REMOTE_ROOT}/'])
    return q


def native_samples(frame, rows, uv_by_source, folder):
    from joint_temporal_texture import HELD_CAMERAS
    job = OUT/'scratch'/frame; job.mkdir(parents=True, exist_ok=False)
    sources = [dict(index=int(ci), camera=rows[int(ci)]['physical_camera']) for ci in uv_by_source]
    assert not (set(r['camera'] for r in sources) & HELD_CAMERAS)
    write(job/'job.json', dict(frame=frame, sources=sources))
    np.savez(job/'uv.npz', **{str(ci):uv for ci, uv in uv_by_source.items()})
    remote_job = f'{REMOTE_ROOT}/{frame}'
    call(['ssh', REMOTE, 'mkdir', '-p', remote_job])
    call(['scp', '-q', job/'job.json', job/'uv.npz', f'{REMOTE}:{remote_job}/'])
    call(['ssh', REMOTE, f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 {REMOTE_PYTHON} {REMOTE_ROOT}/render_cinematic_6k_texture.py remote --job {remote_job}'])
    call(['scp', '-q', f'{REMOTE}:{remote_job}/rgb.npz', f'{REMOTE}:{remote_job}/result.json', str(job)])
    receipt = read(job/'result.json')
    assert receipt['uv_sha256'] == sha(job/'uv.npz') and receipt['rgb_sha256'] == sha(job/'rgb.npz')
    config = read(OUT/'request.json')
    assert receipt['decoder_sha256'] == config['decoder_sha256'] and receipt['worker_sha256'] == config['script_sha256']
    with np.load(job/'rgb.npz') as loaded: result = {int(k):loaded[k] for k in loaded.files}
    write(folder/'source_provenance.json', receipt)
    # Only this invocation's exact successful scratch is removed, after hashing.
    call(['ssh', REMOTE, 'rm', '-f', f'{remote_job}/uv.npz', f'{remote_job}/rgb.npz', f'{remote_job}/job.json', f'{remote_job}/result.json'])
    call(['ssh', REMOTE, 'rmdir', remote_job])
    shutil.rmtree(job)
    return result


def render_frame(q, record, hd_check=False):
    import open3d as o3d
    import torch
    from PIL import Image
    from joint_temporal_texture import cameras, ROOT as COLOR, display, project, exr
    from native_texture_footprint import sample_native, snap_centers
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from compose_cinematic_train_ending import dissolve_alpha, blend_display, write_png
    torch.set_num_threads(2)
    start = time.monotonic(); frame=record['frame_id']; index=record['index']
    folder=OUT/'frames'/frame; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        receipt=read(folder/'complete.json'); assert receipt['request_sha256']==sha(OUT/'request.json')
        for name, digest in receipt['hashes'].items(): assert sha(folder/name)==digest
        return
    rows, _, _ = cameras(frame)
    log_gain=np.load(COLOR/'parameters.npz')['log_gain']; gains=np.exp(log_gain-log_gain.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']; alpha=dissolve_alpha(index)
    write(OUT/'progress.json', dict(frame=frame,index=index,stage='surface_coordinates',pid=os.getpid()))
    uv_by_source={}; locations={}; checks={}; raw=None
    if alpha < 1:
        baseline=BASE/'frames'/frame; receipt=read(baseline/'complete.json')
        assert receipt['request_sha256']==sha(BASE/'request.json')
        for name, digest in receipt['hashes'].items(): assert sha(baseline/name)==digest
        old_result=read(baseline/'result.json'); assert old_result['camera']==record['camera']
        assert old_result['source_cameras']==[r['physical_camera'] for r in rows]
        for key, hashkey in [('mesh','mesh_sha256'),('metadata','metadata_sha256')]: assert sha(record[key])==record[hashkey]
        mesh=o3d.io.read_triangle_mesh(record['mesh']); vertices=np.asarray(mesh.vertices,np.float32)
        triangles=np.asarray(mesh.triangles,np.uint32); tv=vertices[triangles]
        depth, face_ids, barycentric=camera_depth(scene_for(vertices,triangles),record['camera'])
        saved=np.load(baseline/'target_depth.npz')['depth']
        np.testing.assert_array_equal(np.where(np.isfinite(depth),depth,0),saved)
        ids=np.asarray(Image.open(baseline/'source_ids.png')).ravel()
        pixels=np.flatnonzero(np.isfinite(depth)); face=face_ids.ravel()[pixels]
        bary=barycentric.reshape(-1,2)[pixels]; weights=np.column_stack((1-bary.sum(1),bary))
        points=(tv[face]*weights[:,:,None]).sum(1)
        # Project one selected camera at a time; preserve renderer HD center snap.
        for ci in np.unique(ids[ids!=255]):
            subset=ids[pixels]==ci; where=pixels[subset]; uv,_=project(points[subset],[rows[int(ci)]])
            uv=snap_centers(torch.from_numpy(uv)).numpy()[0]
            locations[int(ci)]=where; uv_by_source[int(ci)]=uv
        raw=np.zeros((1080*1920,3),np.float32)
        for name in ['source_ids.png','target_depth.npz','face_source_labels.npy']:
            shutil.copyfile(baseline/name,folder/name)
        checks.update(depth_byte_equal=True, source_ids_byte_equal=True, face_labels_byte_equal=True,
            parent_complete_sha256=sha(baseline/'complete.json'))
        if hd_check:
            control=np.zeros_like(raw)
            for ci,uv in uv_by_source.items():
                im=torch.from_numpy(exr(rows[ci]['file_path']).transpose(2,0,1)[None]).cuda()
                samples=sample_native(im,torch.from_numpy(uv[None,None]).cuda())[0,:,0].T.cpu().numpy()
                control[locations[ci]]=display(np.maximum(samples*gains[ci],0),exposure)
            control=np.rint(control.reshape(1080,1920,3)*255).clip(0,255).astype(np.uint8)
            old=np.asarray(Image.open(baseline/'prediction_native.png'));delta=np.abs(control.astype(int)-old.astype(int))
            checks['hd_reproduction']=dict(mae_code=float(delta.mean()),max_code=int(delta.max()),changed_fraction=float(np.any(delta,2).mean()))
            assert delta.mean()<.03 and np.quantile(delta,.999)<3, checks
            write_png(folder/'hd_reproduction.png',np.rot90(control))
    ending_ci=None; ending_uv=None; train_offset=0
    if alpha>0:
        ending_ci=next(i for i,r in enumerate(rows) if r['physical_camera']==q['camera_path_report']['endpoint_train_camera'])
        row=rows[ending_ci]; camera=record['camera']; yy,xx=np.mgrid[:1080,:1920]
        ending_uv=np.stack(((xx+.5-camera['cx'])/camera['fl_x']*row['fl_x']+row['cx']-.5,
            (yy+.5-camera['cy'])/camera['fl_y']*row['fl_y']+row['cy']-.5),-1).reshape(-1,2)
        assert ending_uv.min()>=0 and ending_uv[:,0].max()<1920 and ending_uv[:,1].max()<1080
        train_offset=len(uv_by_source.get(ending_ci,[]))
        uv_by_source[ending_ci]=np.concatenate((uv_by_source.get(ending_ci,np.empty((0,2))),ending_uv))
    write(OUT/'progress.json',dict(frame=frame,index=index,stage='remote_native_sampling',pid=os.getpid()))
    samples=native_samples(frame,rows,uv_by_source,folder)
    if raw is not None:
        for ci,where in locations.items():raw[where]=display(np.maximum(samples[ci][:len(where)]*gains[ci],0),exposure)
        raw=np.rot90(np.rint(raw.reshape(1080,1920,3)*255).clip(0,255).astype(np.uint8))
        write_png(folder/'render.png',raw)
    train=None
    if alpha>0:
        train=np.rot90(np.rint(display(samples[ending_ci][train_offset:]*gains[ending_ci],exposure).reshape(1080,1920,3)*255).clip(0,255).astype(np.uint8))
        write_png(folder/'train.png',train)
    image=raw if alpha==0 else train if alpha==1 else blend_display(raw,train,alpha)
    write_png(folder/'frame.png',image)
    write(folder/'result.json',dict(frame_id=frame,index=index,camera=record['camera'],train_alpha=alpha,
        kind='3d_render' if alpha==0 else 'real_train_rgb' if alpha==1 else 'explicit_3d_to_train_dissolve',
        fixed_exposure=exposure,seconds=time.monotonic()-start,checks=checks,visual_status='pending'))
    write(folder/'complete.json',dict(request_sha256=sha(OUT/'request.json'),
        hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(f'frame={frame} index={index} seconds={time.monotonic()-start:.1f} native_sources={len(samples)} checks={checks.get("hd_reproduction")}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['remote','render'])
    p.add_argument('--job',type=Path);p.add_argument('--frames',nargs='+');p.add_argument('--hd-check',action='store_true');a=p.parse_args()
    if a.action=='remote':remote_sample(a.job);return
    q=initialize()
    for row in q['inventory']:
        if a.frames is None or row['frame_id'] in a.frames:render_frame(q,row,a.hd_check)
    write(OUT/'progress.json',dict(stage='requested_frames_finished',complete=len(list((OUT/'frames').glob('*/complete.json')))))


if __name__=='__main__':main()
