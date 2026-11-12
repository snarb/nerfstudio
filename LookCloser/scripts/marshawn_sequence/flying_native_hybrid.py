"""Adapt the qualified native single-donor detail renderer to the 908 flight.

Uses original-size PNG training crops as RGB donors, same-instant frozen NeRF
quantiles for visibility, and the accepted union-interior composite as fallback.
The 14 photographic tail frames are linked unchanged.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

LEGACY = Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107/video/reference/legacy')


def native_png_sample(raw, uv, row):
    """Four taps in original pixel coordinates, no resized RGB intermediate."""
    scale = np.array([raw.shape[1] / row['w'], raw.shape[0] / row['h']])
    native = (uv + .5) * scale - .5
    valid = np.isfinite(native).all(-1) & (native >= 0).all(-1)
    valid &= (native < np.array([raw.shape[1]-1, raw.shape[0]-1])).all(-1)
    safe = np.where(valid[..., None], native, 0)
    ij = np.floor(safe).astype(np.int32)
    frac = safe - ij
    result = np.zeros((*uv.shape[:-1], 3), np.float32)
    # Match the PNG recipe's fixed exposure; interpolate decoded display-linear
    # samples, then re-encode sRGB. Input is the native PNG recipe, not RAW TIFF.
    for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:
        weight = (frac[...,0] if dx else 1-frac[...,0]) * (frac[...,1] if dy else 1-frac[...,1])
        v = raw[ij[...,1]+dy, ij[...,0]+dx].astype(np.float32) / 255
        linear = np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)
        result += linear * weight[...,None]
    return np.where(result <= .0031308, result*12.92, 1.055*result**(1/2.4)-.055), valid


class Sources:
    def __init__(self, native, rows, geometry, records):
        self.native, self.rows, self.geometry = native, rows, geometry
        self.allowed = set(rows)
        self.records = records
        self.inventory = {}

    def get(self, name):
        from common import sha
        row = self.rows[name]
        rec = self.records[name]
        p = self.native / rec['image']
        assert sha(p) == rec['rgb_sha256'], 'Changed donor RGB'
        raw = np.array(Image.open(p).convert('RGB'))
        assert [raw.shape[1], raw.shape[0]] == [rec['camera_row']['w'], rec['camera_row']['h']]
        sem_path = self.native / 'semantics' / (name+'.npz')
        assert sha(sem_path) == rec['semantic_sha256'], 'Changed donor semantics'
        with np.load(sem_path) as s:
            # Current sequence has reviewed production subject masks, not the
            # old take's per-person UNKNOWN fields. Fail closed outside core.
            support = s['opaque_core'] & s['valid'] & s['black']
            if 'unknown' in s: support &= ~s['unknown']
            if 'source_ineligible' in s: support &= ~s['source_ineligible']
        support = cv2.resize(support.astype('uint8'), (row['w'], row['h']), interpolation=cv2.INTER_NEAREST) > 0
        with np.load(self.geometry / (name+'.npz')) as z:
            g = {k:z[k].copy() for k in z.files}
        self.inventory[name] = dict(path=str(p), sha256=rec['rgb_sha256'], native_wh=[raw.shape[1],raw.shape[0]],
            rgb_resized=False, geometry_wh=[row['w'],row['h']], semantic_sha256=rec['semantic_sha256'])
        return row, g, raw, support


def run(root, index):
    root = Path(root)
    sys.path[:0] = [str(root/'tools'), str(LEGACY/'campaign'), str(LEGACY/'hybrid_reference')]
    from common import sha, atomic
    from inference import load_pipeline
    import misha50_hybrid as legacy
    legacy.native_sample = native_png_sample
    op = root/'operations'
    base = op/'flying_union_interior_20261112'
    out = op/'flying_native_hybrid_4k_20261112'
    out.mkdir(exist_ok=True)
    path = json.loads((base/'path.json').read_text())
    row = path['records'][index]
    dest = out/'neural'/f'{index:03}'
    dest.mkdir(parents=True, exist_ok=True)
    audit = json.loads((root/'sparse24_storage_audit.json').read_text())['results']
    saved = next(r for r in audit if r['frame'] == row['source_frame'])
    checkpoint = Path(saved['path'])
    assert sha(checkpoint) == saved['model_sha256']
    native = root/'frames'/f'{row["source_frame"]:06}'/'native'
    records = {r['camera']:r for r in json.loads((native/'native_ready.json').read_text())['records']}
    start = time.time()
    pipe, spec, step = load_pipeline(root, checkpoint.parent.parent/'train/spec.json', checkpoint, dest/'loader')
    pipe.model.config.adaptive_max_frequency_level = 10
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    # Bind calibrated, normalized poses to the actual loaded frame's dataset.
    datasets = [pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset]
    poses = {}
    for dataset in datasets:
        if dataset is None: continue
        for j,p in enumerate(dataset.image_filenames):
            pose = np.eye(4); pose[:3] = dataset.cameras.camera_to_worlds[j].cpu().numpy()
            poses[Path(p).stem] = pose.tolist()
    rows = {}
    for cid,(name,rec) in enumerate(sorted(records.items())):
        if name not in poses: continue
        r = copy.deepcopy(rec['camera_row']);r['transform_matrix'] = poses[name];r['camera_id'] = cid
        rows[name] = legacy.scaled(r, min(1920, r['w']))
    target = dict(w=3840,h=2160,fl_x=row['intrinsics'][0]*2,fl_y=row['intrinsics'][1]*2,
        cx=row['intrinsics'][2]*2,cy=row['intrinsics'][3]*2,transform_matrix=row['pose'])
    names = sorted(rows, key=lambda n:np.linalg.norm(np.array(rows[n]['transform_matrix'])[:3,3]-np.array(row['pose'])[:3,3]))[:16]
    geometry = dest/'geometry';geometry.mkdir(exist_ok=True)
    g = legacy.render_geometry(pipe.model, target, geometry/'target.npz', chunk=4096)
    for n in names:
        pilot = op/'flying_native_hybrid_20261112/neural'/f'{index:03}'/'geometry'
        for suffix in ['.npz', '.json']:
            cached = pilot/(n+suffix)
            if cached.exists() and not (geometry/(n+suffix)).exists():
                (geometry/(n+suffix)).symlink_to(cached)
        legacy.render_geometry(pipe.model, rows[n], geometry/(n+'.npz'), chunk=4096)
        print('GEOMETRY', index, n, round(time.time()-start,1), flush=True)
    # Geometry remains original field; photometric fallback is the accepted
    # cleanup. Limit transfer to its opaque interior; never alter alpha/edges.
    accepted = np.array(Image.open(base/'neural'/f'{index:03}'/'strict.png'))
    raw = np.array(Image.open(op/'flying_strict_low_20261112/neural'/f'{index:03}'/'raw.png'))
    # Reuse the accepted contour treatment, but integrate RGB and geometry
    # natively at 4K. Interior gain is exactly one wherever cleanup preserved
    # raw RGB; only contour suppression is inherited from the HD matte.
    ratio = np.minimum(accepted.astype(np.float32)/np.maximum(raw,1),1)
    ratio[(accepted==raw)] = 1
    ratio[np.all(accepted==0,axis=-1)] = 0
    gain4k = cv2.resize(ratio,(3840,2160),interpolation=cv2.INTER_LINEAR)
    clean4k = np.uint8(np.clip(g['rgb']*gain4k*255+.5,0,255))
    distance = ndi.distance_transform_edt(np.any(accepted > 8, axis=-1))
    eligible_hd = (distance >= 6) & (np.max(np.abs(accepted.astype(int)-raw.astype(int)),axis=-1) <= 2)
    eligible = cv2.resize(eligible_hd.astype('uint8'),(3840,2160),interpolation=cv2.INTER_NEAREST)>0
    accepted = clean4k
    Image.fromarray(accepted).save(dest/'baseline_4k.png')
    g = dict(g, rgb=accepted.astype(np.float32)/255, alpha=np.where(eligible, g['alpha'], 0))
    sources = Sources(native, {n:rows[n] for n in names}, geometry, records)
    hybrid, info, labels, weight = legacy.hybrid_image(root,sources,target,g,count=16,mode='detail',sigma=4.,gain=.75,pixel_scale=2.)
    result = np.uint8(np.clip(hybrid*255+.5,0,255))
    assert np.array_equal(result[weight==0],accepted[weight==0])
    assert np.array_equal(result[~eligible],accepted[~eligible])
    Image.fromarray(result).save(dest/'strict.png')
    Image.fromarray(np.uint8(weight.clip(0,1)*255)).save(dest/'transfer.png')
    np.save(dest/'source_labels.npy', labels)
    atomic(dest/'receipt.json',dict(index=index,source_frame=row['source_frame'],checkpoint_sha256=saved['model_sha256'],
        loaded_step=step,path_sha256=sha(base/'path.json'),code_sha256=sha(__file__),legacy_sha256=sha(LEGACY/'campaign/misha50_hybrid.py'),
        detail=info,donors=sources.inventory,seconds=time.time()-start,output_sha256=sha(dest/'strict.png'),
        geometry_cache_checkpoint=saved['model_sha256'],alpha_modified=False,unqualified_fallback_exact=True,render_wh=[3840,2160],contour_source='accepted HD suppression; native 4K field RGB',
        source_format='native full-resolution PNG recipe crops, display-linear four-tap sampling'))
    print('DONE', index, round(time.time()-start,1), info, flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--index',required=True,type=int);a=p.parse_args();run(a.root,a.index)
