#!/usr/bin/env python3
"""Train-source-only orientation/patch consensus around frozen LookCloser depth.

Target RGB is never loaded. Candidate acceptance requires stronger patch
agreement, a separated minimum, unchanged witness support and no worse 3D
orientation agreement. Uncertain candidates retain exact parent depth.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from dec5_hair import unproject,project,orientations,orientation_planes,orientation_cost
from study_dec5_hair_geometry import Scene
from run_dec5_hair_campaign import atomic_json,sha


def sample(array,uv):
    shape=uv.shape[:-1];flat=uv.reshape(-1,2);out=[]
    for begin in range(0,len(flat),16000):
        q=flat[begin:begin+16000]
        out.append(cv2.remap(array,(q[:,0]-.5).astype(np.float32)[:,None],
            (q[:,1]-.5).astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0])
    return np.concatenate(out).reshape(shape+array.shape[2:])


def patch_consensus(patches,valid,minimum_std=2/255):
    """Median pairwise ZNCC distance; neutral flat patches provide no evidence."""
    center=patches-patches.mean(-1,keepdims=True)
    std=np.sqrt(np.mean(center*center,-1));valid=valid&(std>=minimum_std)
    unit=center/np.maximum(np.linalg.norm(center,axis=-1,keepdims=True),1e-8)
    costs=[]
    for i in range(len(patches)):
        for j in range(i):
            agreement=np.clip(np.sum(unit[i]*unit[j],-1),-1,1)
            costs.append(np.where(valid[i]&valid[j],1-agreement,np.nan))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        cost=np.nanmedian(np.stack(costs),axis=0)
    count=valid.sum(0);cost=np.where((count>=4)&np.isfinite(cost),cost,np.inf)
    return cost,count


def choose_candidates(photo,orientation,support):
    """Pareto gate rather than tunable weighted appearance/orientation sum."""
    allowed=(support>=np.maximum(support[:,0,None],4))&np.isfinite(orientation)
    allowed &= orientation<=orientation[:,0,None]+1e-4
    score=np.where(allowed,photo,np.inf);best=score.argmin(-1);row=np.arange(len(best))
    ordered=np.sort(score,axis=-1);base=photo[:,0];selected=score[row,best]
    good=(best!=0)&np.isfinite(base)&np.isfinite(selected)&(base>1e-5)
    good &= selected<=.9*base
    gap=np.subtract(ordered[:,1],selected,out=np.zeros_like(selected),where=np.isfinite(selected))
    good &= gap>=.05*base
    return best,good


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','target-depth','source-depth','baseline','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    parent=json.load(open(a.target_depth/'complete.json'));sources=json.load(open(a.source_depth/'complete.json'))
    if parent['checkpoint_sha256']!=sources['checkpoint_sha256'] or parent['dataset_sha256']!=sources['dataset_sha256']:
        raise ValueError('Matching frozen target/source fields required')
    if not parent['frozen_state_unchanged'] or not sources['frozen_state_unchanged'] or sources['uses_validation_rgb'] or sources['uses_test_rgb']:
        raise ValueError('Frozen train-only source provenance required')
    scene=Scene(a.root,load_rgb=False,mvs=a.root/'pm1920_native_v2',mesh=a.root/'center_tsdf_base/mesh.ply')
    if parent['dataset_sha256']!=sha(scene.data/'transforms.json'):raise ValueError('Dataset mismatch')
    by={r['camera']:r for r in sources['records']};parent_by={r['camera']:r for r in parent['records']}
    centers=scene.centers;orders={};needed=set()
    from dec5_hair import cv_pose
    for f in scene.validation:
        order=np.argsort(np.linalg.norm(centers-cv_pose(f)[:3,3],axis=-1),kind='stable')[:8]
        orders[f['physical_camera']]=order;needed.update(order.tolist())
    maps={};rgb_sources=[]
    for i in sorted(needed):
        f=scene.frames[i];n=f['physical_camera'];path=a.source_depth/(n+'.npz')
        if sha(path)!=by[n]['sha256']:raise ValueError('Changed source depth')
        saved=np.load(path);depth=np.where(saved['roi'],saved['median'],scene.observed_depth[i])
        image=cv2.imread(str(scene.data/f['file_path']))[...,::-1].astype(np.float32)/255
        rgb_sources.append(dict(camera=n,sha256=sha(scene.data/f['file_path'])))
        angle,q=orientations(image);gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        maps[i]=(depth,gray,angle,q)
    offsets=np.array([0,-1,1,-2,2,-3,3,-4,4],np.float32)
    patch=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]],np.float32)
    records=[];baseline_by={r['camera']:r for r in json.load(open(a.baseline/'complete.json'))['records']}
    for f in scene.validation:
        n=f['physical_camera'];path=a.target_depth/(n+'.npz')
        if sha(path)!=parent_by[n]['sha256']:raise ValueError('Changed target depth')
        with np.load(path) as v:arrays={k:v[k] for k in v.files}
        old=arrays['median'];baseline_path=a.baseline/(n+'.npz')
        if sha(baseline_path)!=baseline_by[n]['npz_sha256']:raise ValueError('Changed frozen target ROI')
        base=np.load(baseline_path)
        domain=base['roi']&(base['hair_probability']>.5)&(old>0)
        yy,xx=np.nonzero(domain);proposal=old.copy();changed=np.zeros_like(domain);confidence=np.zeros_like(old)
        for begin in range(0,len(xx),1024):
            y,x=yy[begin:begin+1024],xx[begin:begin+1024];uv=np.stack((x+.5,y+.5),-1);d=old[y,x]
            delta=np.maximum(2*d/min(f['fl_x'],f['fl_y']),.002)
            candidates=d[:,None]+delta[:,None]*offsets
            xyz=unproject(uv[:,None,:],candidates,f)
            patch_xyz=unproject(uv[:,None,None,:]+patch[None,None],np.broadcast_to(candidates[...,None],candidates.shape+(9,)),f)
            textures=[];normals=[];weights=[];valids=[]
            for i in orders[n]:
                camera=scene.cameras[i];depth,gray,angle,q=maps[i]
                coords,z=project(xyz,camera);observed=sample(depth,coords);sem=sample(scene.semantic[i][...,0],coords)
                valid=(candidates>0)&(z>0)&(observed>0)&(sem>.5)&(np.abs(observed-z)<=.01*z+.002)
                valid &= (coords[...,0]>=1.5)&(coords[...,0]<camera['w']-1.5)&(coords[...,1]>=1.5)&(coords[...,1]<camera['h']-1.5)
                ori=sample(angle,coords);quality=sample(q,coords)
                normals.append(orientation_planes(coords,ori,camera));weights.append(quality*valid)
                pc,_=project(patch_xyz,camera);textures.append(sample(gray,pc));valids.append(valid)
            oc,_=orientation_cost(np.stack(normals),np.stack(weights));photo,support=patch_consensus(np.stack(textures),np.stack(valids))
            selected,good=choose_candidates(photo,oc,support);row=np.arange(len(y))
            proposal[y[good],x[good]]=candidates[row[good],selected[good]];changed[y[good],x[good]]=True
            confidence[y[good],x[good]]=support[row[good],selected[good]]/8
        arrays['median']=proposal;arrays['multiview_changed']=changed;arrays['multiview_confidence']=confidence
        np.testing.assert_array_equal(proposal[~domain],old[~domain])
        out=a.output/(n+'.npz');np.savez_compressed(out,**arrays)
        records.append(dict(camera=n,sha256=sha(out),domain_pixels=len(xx),changed_pixels=int(changed.sum())))
        atomic_json(a.output/'progress.json',dict(completed=len(records),total=3,records=records))
    receipt=dict(parent);receipt.update(records=records,derived_depth=True,
        method='LookCloser-prior novel-ray depth re-ranking by train-view patch/orientation consensus',
        parent_receipt_sha256=sha(a.target_depth/'complete.json'),source_receipt_sha256=sha(a.source_depth/'complete.json'),
        source_count=8,candidate_offsets=offsets.tolist(),depth_step='max(2*z/min(fx,fy), .002 calibration units)',
        patch_size=3,acceptance='>=4 nonflat witnesses; support not reduced; orientation no worse; >=10% ZNCC gain; runner-up gap >=5% baseline cost',
        rgb_sources=rgb_sources,baseline_roi_receipt_sha256=sha(a.baseline/'complete.json'),
        no_target_rgb_loaded=True,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__))
    atomic_json(a.output/'complete.json',receipt)

if __name__=='__main__':main()
