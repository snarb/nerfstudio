"""Bounded source-registration counterfactual on one fixed overlap domain.

No image or geometry is corrected. All trial offsets use the intersection of
their native visibility domains, so favorable trials cannot discard errors.
"""
import itertools
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, exr, display
from diagnose_matched_forearm_radiometry import ROOT as INPUT, bilinear, paired_summary
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from review_jaw_repair_transfer import panel

ROOT = Path('/mnt/data/dec5_forearm_radiometry_shift')


def centered_ncc(a, b):
    x, y = a-a.mean(0), b-b.mean(0)
    denom = np.sqrt((x*x).sum()*(y*y).sum())
    return float((x*y).sum()/denom) if denom > 1e-12 else None


def run():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():
        raise ValueError('Requires new root')
    q = read(INPUT/'request.json');r = read(INPUT/'result.json')
    assert sha(INPUT/'request.json') == r['request_sha256']
    for path, digest in {**q['hashes'], **read(INPUT/'complete.json')['hashes']}.items():
        assert sha(path) == digest, path
    sample = np.load(INPUT/'samples.npz');hit = sample['hit'];sourceids = q['source_indices']
    rows = q['rows'];gains = np.asarray(q['source_gain']);exposure = q['fixed_exposure']
    offsets = list(itertools.product(range(-4,5), repeat=2))
    pairs = [(37,42), (33,42), (32,37), (31,42)]
    request = dict(input_request_sha256=sha(INPUT/'request.json'), input_complete_sha256=sha(INPUT/'complete.json'),
        script_sha256=sha(__file__), pairs=pairs, offsets=offsets, fixed_domain='intersection of all trial visibility domains',
        corrections_applied=False, diagnostic_not_quality_metrics=True, uses_eval_rgb=False)
    atomic_json(ROOT/'request.json',request)
    meshpath = next(p for p in q['hashes'] if p.endswith('/dec5_multiview_forearm_admission/mesh.ply'))
    mesh = o3d.io.read_triangle_mesh(meshpath);scene = scene_for(np.asarray(mesh.vertices,np.float32),np.asarray(mesh.triangles))
    maskpath = next(p for p in q['hashes'] if p.endswith('/masks.npz'))
    masknames = read(Path(maskpath).with_name('cameras.json'));masks = np.load(maskpath)['masks']
    records = []
    for source_a, source_b in pairs:
        a,b = sourceids.index(source_a), sourceids.index(source_b)
        raw = exr(rows[b]['file_path']);depth,_,_ = camera_depth(scene,rows[b])
        depth = np.where(np.isfinite(depth),depth,0);depth[masks[masknames.index(rows[b]['physical_camera'])]==0] = 0
        uv = sample['uv'][b];z = sample['z'][b]
        base = sample['profiled'][a][hit]
        domain = sample['valid'][a][hit] & sample['valid'][b][hit]
        # Exclude blue clothing in BOTH baseline sources, not candidate-dependent selection.
        domain &= (base[:,0]-base[:,2])*255 > 8
        other = sample['profiled'][b][hit]
        domain &= (other[:,0]-other[:,2])*255 > 8
        for dx,dy in offsets:
            p = uv+[dx,dy];d = bilinear(depth,p)
            good = (z>0)&(d>0)&(np.abs(d-z)<.0015*z)
            good &= (p[:,0]>2)&(p[:,0]<1917)&(p[:,1]>2)&(p[:,1]<1077)
            for tx,ty in [(0,0),(1,0),(0,1),(1,1)]:
                tap = bilinear(depth,np.floor(p)+[tx,ty])
                good &= (tap>0)&(np.abs(tap-z)<.003*z)
            domain &= good
        if domain.sum()<100:
            records.append(dict(a=source_a,b=source_b,count=int(domain.sum()),status='insufficient_common_domain'));continue
        fixed = base[domain];p = uv[domain];trials=[];colors=[]
        for dx,dy in offsets:
            color = display(bilinear(raw,p+[dx,dy]).clip(0)*gains[b],exposure)
            colors.append(color)
            delta = (color-fixed)*255
            trials.append(dict(offset=[dx,dy], median_absolute_rgb_difference=float(np.median(np.abs(delta))),
                median_rgb_delta=np.median(delta,axis=0).tolist(), centered_ncc=centered_ncc(fixed,color),
                median_abs_after_constant_offset=float(np.median(np.abs(delta-np.median(delta,axis=0))))))
        baseline = trials[offsets.index((0,0))]
        best = min(trials,key=lambda x:x['median_absolute_rgb_difference'])
        ncc = max((x for x in trials if x['centered_ncc'] is not None),key=lambda x:x['centered_ncc'])
        # Cross-spatial prediction of a constant RGB offset: split at median target y.
        yy = np.indices(hit.shape)[0][hit][domain];split = np.median(yy)
        color = colors[offsets.index((0,0))];delta = (color-fixed)*255
        fits=[]
        for upper in [True,False]:
            fit = (yy<=split) if upper else (yy>split);test = ~fit
            offset = np.median(delta[fit],axis=0)
            fits.append(dict(fit='upper' if upper else 'lower',fit_count=int(fit.sum()),test_count=int(test.sum()),
                offset_rgb=offset.tolist(), test_median_abs_before=float(np.median(np.abs(delta[test]))),
                test_median_abs_after=float(np.median(np.abs(delta[test]-offset)))))
        record=dict(a=source_a,b=source_b,count=int(domain.sum()),baseline=baseline,best_color_shift=best,
            best_centered_ncc_shift=ncc,spatial_constant_offset_controls=fits,trials=trials)
        records.append(record)
        images=[];labels=[]
        for label, color in [('reference '+str(source_a),fixed),('source '+str(source_b),colors[offsets.index((0,0))]),
            ('best color shift '+str(best['offset']),colors[offsets.index(tuple(best['offset']))]),
            ('best NCC shift '+str(ncc['offset']),colors[offsets.index(tuple(ncc['offset']))])]:
            canvas=np.zeros((*hit.shape,3),np.uint8);where=np.zeros_like(hit);where[hit]=domain
            canvas[where]=np.rint(color*255).clip(0,255).astype(np.uint8);images.append(canvas);labels.append(label)
        panel(ROOT/f'{source_a}_{source_b}.png',images,labels,(0,0,hit.shape[1],hit.shape[0]))
        print(source_a,source_b,'N',int(domain.sum()),'baseline',round(baseline['median_absolute_rgb_difference'],3),
            'best',best['offset'],round(best['median_absolute_rgb_difference'],3),'NCC',ncc['offset'],flush=True)
    atomic_json(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),records=records,
        visual_status='pending',production_updated=False,not_a_registration_estimate=True))
    atomic_json(ROOT/'complete.json',dict(hashes={str(p):sha(p) for p in ROOT.iterdir() if p.is_file()}))


if __name__ == '__main__':
    run()
