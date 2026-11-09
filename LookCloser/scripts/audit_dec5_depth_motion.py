#!/usr/bin/env python3
"""Audit mixed-reference versus self-projected depth transport on a fixed domain."""
from pathlib import Path
import argparse,json
import numpy as np
from dec5_hair import project,unproject
from review_dec5_hair_path import remap,pair_metrics
from run_dec5_hair_campaign import atomic_json,sha


def audit(path,baseline,candidate,output):
    if output.exists():raise FileExistsError(output)
    frames=json.loads(path.read_text())['frames'];rows=[];shifts=[];previous=None
    totals={k:{} for k in ['baseline','candidate_mixed','candidate_self']}
    for i,camera in enumerate(frames):
        name=camera['physical_camera'];b=np.load(baseline/(name+'.npz'));c=np.load(candidate/(name+'.npz'))
        if previous is not None:
            pb,pc=previous;yy,xx=np.nonzero(pb['roi']);uv0=np.stack([xx+.5,yy+.5],-1)
            depth=pb['front_depth'][yy,xx];uv,z=project(unproject(uv0,depth,frames[i-1]),camera)
            next_depth=remap(b['front_depth'],uv,True);tol=np.maximum(2*z/min(camera['fl_x'],camera['fl_y']),.002)
            mask=((depth>0)&(z>0)&(uv[:,0]>=.5)&(uv[:,0]<=camera['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=camera['h']-.5)
                  &~((next_depth>0)&(next_depth<z-tol)))
            cd=pc['front_depth'][yy,xx];cu,cz=project(unproject(uv0,cd,frames[i-1]),camera)
            cz=np.where(cd>0,cz,0);cu[cd<=0]=-1000
            values={}
            for key,args in [('baseline',(pb,b,uv,z,z)),('candidate_mixed',(pc,c,uv,z,cz)),('candidate_self',(pc,c,cu,cz,cz))]:
                a,n,u,ref,pred=args;value,_=pair_metrics(a,n,yy,xx,u,ref,mask,pred)
                values[key]=value
                for field,v in value.items():totals[key][field]=totals[key].get(field,0)+v
            shift=np.linalg.norm(cu-uv,axis=-1)[mask&(cd>0)];shifts.append(shift)
            rows.append(dict(pair=[i-1,i],**values))
        previous=b,c
    for group in totals.values():
        for field in ['visibility_toggles','source_switches','depth_exposure_dropouts','present_pixels']:
            group[field+'_fraction']=group[field]/max(group['anchors'],1)
    shifts=np.concatenate(shifts)
    atomic_json(output,dict(aggregate=totals,per_pair=rows,
        candidate_vs_reference_flow_shift_pixels=dict(median=float(np.median(shifts)),p95=float(np.quantile(shifts,.95))),
        domain='identical frozen B0 anchor domain and occlusion veto; missing candidate points not excluded',
        mixed='candidate camera-z transported but next depth sampled at B0 image coordinates',
        self='candidate point projects to its own next-view coordinates and camera-z',
        metric_kind='geometry consistency proxies, not independent 3D ground truth',
        baseline_receipt_sha256=sha(baseline/'complete.json'),candidate_receipt_sha256=sha(candidate/'complete.json'),
        path_sha256=sha(path),implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['path','baseline','candidate','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();audit(a.path,a.baseline,a.candidate,a.output)


if __name__=='__main__':main()
