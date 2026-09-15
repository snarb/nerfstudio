"""Can depth changes satisfy unchanged skin annotations? Diagnostic, no mesh edit.

Retains every discrete valid sample (not a convex interval hiding forbidden
depths). +/-0.03 is a diagnostic search range, not an accepted displacement.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import unproject,project_integer
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from ordered_forearm_admission import point_votes
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    prior.configure();v1=prior.v2.v1;folder=root/frame;request=read(folder/'request.json')
    residual=read(folder/'residual_attribution.json');codes=np.load(folder/'residual_attribution.npz')['codes']
    if sha(folder/'residual_attribution.npz')!=residual['arrays_sha256']:raise ValueError('Changed residual attribution')
    rows,depths,hashes=v1.load_real(frame)
    if hashes!=request['source_depth_sha256']:raise ValueError('Changed depth inputs')
    camera=request['reference_camera'];reference=request['fit_reference_camera'];masks=v1.masks(frame)
    fit=next(r for r in read(Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json')['fit'] if r['model']=='quadratic')
    analysis=read(prior.OUT/frame/'analysis.json');data=np.load(prior.OUT/frame/'diagnostic.npz')
    y,x=np.nonzero(np.isin(codes,[4,5,6]));center=np.asarray(camera['transform_matrix'])[:3,3]
    directions=unproject(camera,x+.5,y+.5,np.ones(len(x)))-center
    z,_=intersect_near_plane(center,directions,world_quadric(reference,fit),world_plane(reference,analysis['plane_inverse_coefficients']))
    offsets=np.linspace(-.03,.03,121);valid=[]
    for delta in offsets:
        zz=z+delta;points=unproject(camera,x+.5,y+.5,zz)
        support,negative,free=point_votes(points,rows,v1.NAMES,masks,data,depths,prior.v2.semantic_domain)
        valid.append((zz>0)&np.isfinite(zz)&(support>=2)&(negative==0)&(free==0))
    valid=np.stack(valid,1);distance=np.where(valid,np.abs(offsets)[None],np.inf)
    best=distance.argmin(1);solvable=valid.any(1);chosen=offsets[best];chosen[~solvable]=np.nan
    groups=[]
    for code in [4,5,6]:
        select=codes[y,x]==code
        groups.append(dict(attribution_code=code,points=int(select.sum()),
            valid_within_001=int((select&(distance.min(1)<=.01000001)).sum()),
            valid_within_002=int((select&(distance.min(1)<=.02000001)).sum()),
            valid_within_003=int((select&solvable).sum()),no_valid_tested_depth=int((select&~solvable).sum())))
    dest=folder/'mask_depth_intervals';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'samples.npz',xy=np.column_stack([x,y]),base_depth=z,offsets=offsets,valid=valid,chosen_offset=chosen,source_code=codes[y,x])
    records=[];points=unproject(camera,x[solvable]+.5,y[solvable]+.5,z[solvable]+chosen[solvable])
    for name in v1.NAMES:
        rgb=np.array(Image.open(prior.OUT/frame/'rgb'/(name+'.png')));overlay=rgb.copy();row=next(r for r in rows if r['physical_camera']==name)
        uv,zz=project_integer(row,points);xy=np.rint(uv).astype(int);inside=(zz>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        overlay[xy[inside,1],xy[inside,0]]=[0,255,120]
        panel=Image.new('RGB',(860,450));draw=ImageDraw.Draw(panel)
        for i,im in enumerate([rgb,overlay]):panel.paste(Image.fromarray(np.rot90(im)).crop((0,1500,430,1920)),(i*430,25))
        draw.text((3,3),name+' GT / closest feasible DEPTH SAMPLES (not mesh)',fill='white');p=dest/(name+'.png');panel.save(p)
        records.append(dict(camera=name,path=str(p),sha256=sha(p)))
    atomic_json(dest/'result.json',dict(frame=frame,script_sha256=sha(__file__),residual_attribution_sha256=sha(folder/'residual_attribution.json'),
        groups=groups,discrete_sample_count=121,search_range=[-.03,.03],sample_step=.0005,
        masks_unchanged=True,new_depth_displacement_policy_accepted=False,full_62_view_guard_not_applied=True,
        geometry_changed=False,heldout_used=False,arrays_sha256=sha(dest/'samples.npz'),panels=records,visual_status='requires_actual_review'))
    print(frame,groups,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();run(a.root,a.frame)
