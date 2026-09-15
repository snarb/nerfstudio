"""Local rectified NCC evidence for the remaining skin-to-skin depth conflict.

Read-only diagnostic at five previously frozen F/E veto samples. Fixed windows
and disparity search, no depth selection, no guard relaxation or held-out RGB.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_confidence_depth_prior import unproject
from study_foundation_lower_forearm import ROOT as LOWER,FRAME
from study_forearm_layer_qualified_guard import ROOT


def patch(im,uv,radius):
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    return np.column_stack([map_coordinates(im[...,c],[y.ravel()+uv[1],x.ravel()+uv[0]],order=1,mode='constant',cval=np.nan) for c in range(3)])


def ncc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():return np.nan
    a=a-a.mean(0);b=b-b.mean(0);norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/norm) if norm>1e-8 else np.nan


def run():
    source=LOWER/FRAME/'E004_E_F004_E';cal=np.load(source/'calibration.npz')
    stage=read(LOWER/FRAME/'request.json');pair=next(p for p in stage['pairs'] if Path(p['directory'])==source)
    for name in ['left.png','right.png','calibration.npz']:assert sha(source/name)==pair['hashes'][name]
    images=[np.array(Image.open(source/(side+'.png')),dtype=float) for side in ['left','right']]
    e1=cal['rectified_extrinsic'];e2=np.eye(4);e2[:3,:3]=cal['R2']@cal['E2'][:3,:3];e2[:3,3]=cal['R2']@cal['E2'][:3,3]
    row=next(r for r in cameras(FRAME)[0] if r['physical_camera']==pair['right'])
    group=next(r for r in read(LOWER/'veto_diagnosis/result.json')['records'] if r['camera']==pair['right'])
    dest=ROOT/'ambiguity';dest.mkdir(exist_ok=False);records=[];curves={}
    canvas=Image.new('RGB',(900,5*220),'white');draw=ImageDraw.Draw(canvas)
    for index,event in enumerate(group['samples']):
        x,y=event['native_xy'];points=unproject(row,np.array([x,x]),np.array([y,y]),np.array([event['proposed_depth'],event['observed_depth']]))
        projected=[]
        for e,k in [(e1,cal['cropped_intrinsic']),(e2,cal['P2'][:,:3])]:
            p=points@e[:3,:3].T+e[:3,3];uv=p@k.T;projected.append(uv[:,:2]/uv[:,2:])
        left,right=projected;np.testing.assert_allclose(right[0],right[1],atol=.01)
        np.testing.assert_allclose(left[:,1],right[:,1],atol=.02)
        hypotheses=left[:,0]-right[:,0]
        ds=np.arange(np.floor(hypotheses.min()-64),np.ceil(hypotheses.max()+64)+.5,.5)
        estimates=[]
        for radius,color in [(3,'red'),(7,'green'),(15,'blue')]:
            ref=patch(images[1],right[0],radius)
            scores=np.array([ncc(patch(images[0],right[0]+[d,0],radius),ref) for d in ds])
            proposed=ncc(patch(images[0],left[0],radius),ref);observed=ncc(patch(images[0],left[1],radius),ref)
            finite=np.isfinite(scores)
            best=float(np.max(scores[finite])) if finite.any() else None
            estimates.append(dict(radius=radius,query_rgb_std=float(np.std(ref,axis=0).mean()),
                prior_ncc=proposed,pm_ncc=observed,best_ncc=best,
                near_best_disparity_samples=int((scores>=best-.02).sum()) if best is not None else 0))
            curves[f'{index}_{radius}_disparity']=ds;curves[f'{index}_{radius}_ncc']=scores
            coords=[(50+800*(d-ds[0])/(ds[-1]-ds[0]),index*220+190-140*(s+1)/2) for d,s in zip(ds,scores) if np.isfinite(s)]
            if len(coords)>1:draw.line(coords,fill=color,width=2)
        draw.text((5,index*220+5),f'sample {index}; red 7x7, green 15x15, blue 31x31; vertical red prior / black PM',fill='black')
        for disparity,color in zip(hypotheses,['red','black']):
            px=50+800*(disparity-ds[0])/(ds[-1]-ds[0]);draw.line((px,index*220+35,px,index*220+195),fill=color,width=1)
        records.append(dict(event_index=index,prior_disparity=float(hypotheses[0]),pm_disparity=float(hypotheses[1]),estimates=estimates))
    canvas.save(dest/'ncc_curves.png');np.savez_compressed(dest/'curves.npz',**curves)
    atomic_json(dest/'result.json',dict(records=records,script_sha256=sha(__file__),
        source_hashes={str(source/n):sha(source/n) for n in ['left.png','right.png','calibration.npz']},
        event_source_sha256=sha(LOWER/'veto_diagnosis/result.json'),
        photometric_patch_diagnostic_not_colmap_internal_cost=True,
        slanted_plane_warp_not_modeled=True,guard_changed=False,geometry_changed=False,visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':run()
