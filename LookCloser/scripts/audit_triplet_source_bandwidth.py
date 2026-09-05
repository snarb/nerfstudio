#!/usr/bin/env python3
"""Train-only three-camera instrumental bandwidth diagnostic, not a PSF claim.

For small blur differences, B ~= gain * (A + variance/2 * Laplacian(A)).
Moments against a third independently noisy camera avoid the two-camera noise
term under this model. Shared scene/registration and independent sensor noise
are assumptions, not verified facts about DEC5. No RGB prediction is changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def instrumental_moments(a,b,c,margin=6):
    if (a.shape!=b.shape or a.shape!=c.shape or a.ndim!=2 or min(a.shape)<=2*margin
            or not all(np.isfinite(x).all() for x in (a,b,c))):
        raise ValueError('Need three finite registered grayscale patches')
    def design(q):
        q=np.asarray(q,np.float64)
        lap=sum(np.roll(q,d,axis=k) for k in (0,1) for d in (-1,1))-4*q
        x=np.stack([q,lap],-1)[margin:-margin,margin:-margin].reshape(-1,2)
        return x-x.mean(0)
    x,z=design(a),design(c)
    y=b[margin:-margin,margin:-margin].astype(np.float64).ravel();y-=y.mean()
    return z.T@x,z.T@y


def solve_instrumental_variance(matrix,rhs):
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():raise ValueError('Nonfinite moments')
    condition=float(np.linalg.cond(matrix))
    if condition>100:return {'valid':False,'reason':'ill_conditioned','condition':condition}
    gain,lap=np.linalg.solve(matrix,rhs)
    if not .5<gain<2:return {'valid':False,'reason':'gain_out_of_bounds','gain':float(gain),'condition':condition}
    variance=float(2*lap/gain)
    return dict(valid=abs(variance)<=6.25,variance=variance,gain=float(gain),condition=condition)


def noise_canary():
    from audit_warped_source_registration import relative_blur_profile
    rows=[]
    for sigma in (0.,.6,1.):
        matrix=np.zeros((2,2));rhs=np.zeros(2);old=[]
        for seed in range(100):
            rng=np.random.default_rng(seed)
            shared=cv2.GaussianBlur(rng.normal(0,1,(64,64)).astype(np.float32),(0,0),1.2)*.1+.4
            a=shared+rng.normal(0,.015,shared.shape)
            b=(cv2.GaussianBlur(shared,(0,0),sigma) if sigma else shared)+rng.normal(0,.003,shared.shape)
            c=shared+rng.normal(0,.008,shared.shape)
            m,v=instrumental_moments(a,b,c);matrix+=m;rhs+=v
            if seed<30:old.append(relative_blur_profile(a,b))
        result=solve_instrumental_variance(matrix,rhs)
        rows.append(dict(true_added_variance=sigma*sigma,independent_noise_std=[.015,.003,.008],
            patches=100,legacy_first_30_nonzero_blur_above_gain_005=sum(p['ncc_gain']>=.005 and p['sigma_pixels']>0 for p in old),
            legacy_first_30_median_reported_sigma=float(np.median([p['sigma_pixels'] for p in old])),instrument=result))
    return rows


def main():
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous diagnostic')
    observations=json.loads(a.observations.read_text());hashes=dict(observations['input_hashes'])
    if observations['uses_eval_rgb'] is not False or observations['uses_semantic_masks'] is not False:
        raise ValueError('Only train-only unmasked observations')
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed input: {file}')
    render=a.observations.parent
    audit=json.loads((render/'reprojection_audit.json').read_text())
    data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    source_frames=[frames[str(Path(s['source_image']).resolve())] for s in audit['sources']]
    if [f['physical_camera'] for f in source_frames]!=observations['sources']:raise ValueError('Source order differs')
    if {'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}&set(observations['sources']):raise ValueError('Held-out RGB forbidden')
    centers=np.array([np.asarray(f['transform_matrix'])[:3,3] for f in source_frames])
    distances=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    images=[np.asarray(Image.open(render/'source_warps'/f'source_{i:02d}.png').convert('RGB'),np.float32)
            @np.array([.2126,.7152,.0722],np.float32)/255 for i in range(len(source_frames))]
    rows=observations['observations'];lookup={(r['primary_rank'],r['source_rank'],r['x'],r['y']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate camera-pair/patch observation')
    priorities={(i,j):[int(k) for k in np.argsort(distances[i]+distances[j]) if k not in (i,j)]
                for i in range(len(images)) for j in range(i+1,len(images))}
    def shift(i,j,x,y):
        row=lookup.get((min(i,j),max(i,j),x,y))
        return None if row is None else np.array([row['dx'],row['dy']])*(1 if i<j else -1)
    grouped={};selected=0;no_triplet=0;size=observations['patch_size']
    for r in rows:
        i,j,x,y=r['primary_rank'],r['source_rank'],r['x'],r['y'];ij=np.array([r['dx'],r['dy']])
        chosen=None
        for k in priorities[i,j]:
            ik=shift(i,k,x,y);jk=shift(j,k,x,y)
            if ik is not None and jk is not None and np.linalg.norm(ik-ij-jk)<=.5:
                chosen=k;break
        if chosen is None:no_triplet+=1;continue
        # Symmetric half-shift gives A/B equal bilinear interpolation variance.
        # The third camera is an instrument; its transfer function need not match.
        center=np.array([x-.5,y-.5])-ij/2
        patches=[cv2.getRectSubPix(images[index],(size,size),tuple(c.astype(float)))
                 for index,c in [(i,center),(j,center+ij),(chosen,center+ik)]]
        m,v=instrumental_moments(*patches);mr,vr=instrumental_moments(patches[1],patches[0],patches[2])
        key=(i,j,*r['block'])
        g=grouped.setdefault(key,dict(matrix=np.zeros((2,2)),rhs=np.zeros(2),reverse_matrix=np.zeros((2,2)),
            reverse_rhs=np.zeros(2),old=[],held=r['held'],third_counts={}))
        if g['held']!=r['held']:raise ValueError('Fit/held spatial block overlap')
        g['matrix']+=m;g['rhs']+=v;g['reverse_matrix']+=mr;g['reverse_rhs']+=vr
        g['old'].append(r['relative_blur_variance']);g['third_counts'][chosen]=g['third_counts'].get(chosen,0)+1;selected+=1
    output=[]
    for key,g in sorted(grouped.items()):
        if len(g['old'])<3:continue
        forward=solve_instrumental_variance(g['matrix'],g['rhs'])
        reverse=solve_instrumental_variance(g['reverse_matrix'],g['reverse_rhs'])
        valid=forward['valid'] and reverse['valid']
        output.append(dict(pair=list(key[:2]),block=list(key[2:]),held=g['held'],patches=len(g['old']),
            third_counts=g['third_counts'],legacy_variance=float(np.median(g['old'])),forward=forward,reverse=reverse,
            valid=valid,variance=(forward['variance']-reverse['variance'])/2 if valid else None,
            forward_reverse_disagreement=abs(forward['variance']+reverse['variance']) if valid else None))
    summary=[]
    for held in [False,True]:
        subset=[r for r in output if r['held']==held and r['valid']]
        summary.append(dict(held=held,valid_pair_blocks=len(subset),
            legacy_abs_variance_median=float(np.median([abs(r['legacy_variance']) for r in subset])) if subset else None,
            instrumental_abs_variance_median=float(np.median([abs(r['variance']) for r in subset])) if subset else None,
            median_absolute_estimator_difference=float(np.median([abs(r['variance']-r['legacy_variance']) for r in subset])) if subset else None,
            median_forward_reverse_disagreement=float(np.median([r['forward_reverse_disagreement'] for r in subset])) if subset else None))
    hashes[str(a.observations)]=sha256(a.observations)
    hashes[str(Path(__file__))]=sha256(Path(__file__))
    helper=Path(__file__).with_name('audit_warped_source_registration.py');hashes[str(helper)]=sha256(helper)
    result=dict(uses_eval_rgb=False,uses_semantic_masks=False,changes_prediction=False,sources=observations['sources'],
        method='third_camera_instrumental_small_blur_moments',interpretation=__doc__,
        third_camera_choice='geometry-ordered cameras with an existing cycle-consistent triplet; max cycle error .5 pixels',
        input_observations=len(rows),selected_triplet_observations=selected,no_triplet_observations=no_triplet,
        pair_blocks=len(output),summary=summary,blocks=output,noise_only_and_known_blur_canary=noise_canary(),input_hashes=hashes)
    atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','input_hashes']}))


if __name__=='__main__':main()
