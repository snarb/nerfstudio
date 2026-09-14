"""Check whether geometric depth witnesses also see the query patch's color.

Diagnostic only. No depth deletion, veto override, texture edit or mesh change.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from joint_temporal_texture import read,sha,atomic_json,exr,display,cameras,ROOT
from study_confidence_depth_prior import project_integer,unproject
import study_forearm_plane_transfer_v3 as prior


def patch_chroma(image,xy):
    xy=np.asarray(xy,dtype=int);rgb=np.zeros((len(xy),3),np.float64)
    for dy in [-2,-1,0,1,2]:
        for dx in [-2,-1,0,1,2]:
            rgb+=image[np.clip(xy[:,1]+dy,0,image.shape[0]-1),np.clip(xy[:,0]+dx,0,image.shape[1]-1)]
    return rgb/np.maximum(rgb.sum(1,keepdims=True),1e-8)


def witness_errors(points,reference,rows,depths,images):
    refuv,_=project_integer(reference,points);refxy=np.rint(refuv).astype(int)
    refcolor=patch_chroma(images[reference['physical_camera']],refxy)
    error=np.full((len(rows),len(points)),np.nan,np.float32)
    for i,(row,d) in enumerate(zip(rows,depths)):
        if row['physical_camera']==reference['physical_camera']:continue
        uv,z=project_integer(row,points);xy=np.rint(uv).astype(int)
        available=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(available);observed=d[xy[ids,1],xy[ids,0]]
        ids=ids[(observed>0)&np.isfinite(observed)&(np.abs(observed-z[ids])<=.001)]
        if not len(ids):continue
        actual=unproject(row,xy[ids,0],xy[ids,1],d[xy[ids,1],xy[ids,0]])
        back,_=project_integer(reference,actual)
        a=points[ids]-np.asarray(reference['transform_matrix'])[:3,3]
        b=points[ids]-np.asarray(row['transform_matrix'])[:3,3]
        cosine=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        ids=ids[(np.linalg.norm(back-refuv[ids],axis=1)<=1.5)&(cosine<np.cos(np.deg2rad(1)))]
        other=patch_chroma(images[row['physical_camera']],xy[ids])
        error[i,ids]=np.abs(other-refcolor[ids]).mean(1)
    return error


def summary(errors):
    count=np.isfinite(errors).sum(0);valid=count>=3;e=errors[:,valid]
    if not e.size:return dict(points=int(errors.shape[1]),supported=0)
    return dict(points=int(errors.shape[1]),supported=int(valid.sum()),
        pair_error_quantiles=np.nanquantile(e,[.1,.5,.9,.95]).tolist(),
        minimum_error_quantiles=np.quantile(np.nanmin(e,axis=0),[.1,.5,.9,.95]).tolist(),
        thresholds=[dict(chroma_mean_abs_limit=threshold,zero_compatible_witnesses=int(((e<=threshold).sum(0)==0).sum()),
            fewer_than_three_compatible=int(((e<=threshold).sum(0)<3).sum())) for threshold in [.04,.06,.08,.1]])


def run(root,frame):
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    diag=root/'free_space_diagnosis'/frame;receipt=read(diag/'result.json')
    for p,h in receipt['hashes'].items():
        if sha(diag/p)!=h:raise ValueError('Changed veto diagnosis')
    actual,_,_=cameras(frame);profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair;rgb=display(exr(row['file_path'])*np.exp(profiles[i]),gain)
        return row['physical_camera'],np.rint(rgb*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(actual)))
    anchorroot=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame
    anchor_result=read(anchorroot/'result.json')
    if sha(anchorroot/'anchors.npz')!=anchor_result['anchors_sha256']:raise ValueError('Changed anchor control')
    anchors=np.load(anchorroot/'anchors.npz')['points'];records=[];arrays={}
    for record in receipt['records']:
        name=record['camera'];reference=next(r for r in rows if r['physical_camera']==name)
        inputs=np.load(diag/(name+'.npz'));points=inputs['observed']
        errors=witness_errors(points,reference,rows,depths,images)
        if not np.array_equal(np.isfinite(errors).sum(0),inputs['votes']):raise ValueError('Witness implementation differs from geometric guard')
        # Positive control: independently collected skin anchors that agree with
        # this query camera's measured depth, not candidate-mesh surface points.
        uv,z=project_integer(reference,anchors);xy=np.rint(uv).astype(int)
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(valid);d=depths[next(i for i,r in enumerate(rows) if r['physical_camera']==name)]
        ids=ids[np.abs(d[xy[ids,1],xy[ids,0]]-z[ids])<=.001]
        ids=ids[::max(1,int(np.ceil(len(ids)/1000)))];control=witness_errors(anchors[ids],reference,rows,depths,images)
        arrays[name+'_veto']=errors;arrays[name+'_control']=control
        records.append(dict(camera=name,veto=summary(errors),skin_anchor_control=summary(control)))
    dest=root/'color_witness_diagnosis'/frame;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'witness_errors.npz',**arrays)
    atomic_json(dest/'result.json',dict(frame=frame,script_sha256=sha(__file__),records=records,
        prior_veto_diagnosis_sha256=sha(diag/'result.json'),source_depth_hashes=hashes,
        source_rgb_hashes={r['file_path']:sha(r['file_path']) for r in actual},
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        errors_sha256=sha(dest/'witness_errors.npz'),protocol='5x5 display patch chromaticity, mean absolute channel error; four diagnostic thresholds',
        geometry_changed=False,guard_changed=False,threshold_selected=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',default='001037');a=p.parse_args();run(a.root,a.frame)
