#!/usr/bin/env python3
"""Train-only, pose-diverse stereo/orientation-qualified hair depth anchors."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from dec5_hair import VALIDATION,TEST,cv_pose,project,unproject,orientations,orientation_planes,orientation_cost,diverse_witnesses
from import_colmap_mvs_depth_dataset import read_colmap_dense_array,load_binary_pinhole_calibration
from run_dec5_hair_campaign import atomic_json,sha


def prepare(root,mvs,prior,targets,output,per_camera=8192):
    cv2.setNumThreads(2)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    payload=json.loads((root/'dataset/transforms.json').read_text());dataset_sha=sha(root/'dataset/transforms.json')
    frames=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Train59 split required')
    prior_receipt=json.loads((prior/'complete.json').read_text());matte_receipt=json.loads((targets/'complete.json').read_text())
    if prior_receipt['dataset_sha256']!=dataset_sha or matte_receipt['dataset_sha256']!=dataset_sha:raise ValueError('Prior/matte dataset mismatch')
    by_prior={r['camera']:r for r in prior_receipt['records']};by_matte={r['camera']:r for r in matte_receipt['records']}
    calibration=load_binary_pinhole_calibration(mvs/'dense/sparse');cache=[];provenance=[];focus_points=[]
    for i,frame in enumerate(frames):
        name=frame['physical_camera'];cal=calibration[frame['file_path']]
        for k in ['w','h','fl_x','fl_y','cx','cy']:
            if not np.isclose(cal[k],frame[k],rtol=1e-9,atol=1e-7):raise ValueError('Native MVS calibration mismatch '+k)
        dp=mvs/'dense/stereo/depth_maps'/(frame['file_path']+'.geometric.bin');pp=Path(str(dp).replace('.geometric.bin','.photometric.bin'))
        depth=read_colmap_dense_array(dp)[...,0];photo=read_colmap_dense_array(pp)[...,0]
        sem=np.load(root/'semantics'/(name+'.npz'))['probability'][...,0].astype(np.float32)/65535
        mp=targets/(name+'.npz');npz=prior/(name+'.npz')
        if sha(mp)!=by_matte[name]['sha256'] or sha(npz)!=by_prior[name]['sha256']:raise ValueError('Changed prior/matte')
        matte=np.load(mp);h,w=depth.shape;x0,y0,x1,y1=matte['portrait_crop'].astype(int);portrait=np.zeros((w,h),np.float32)
        portrait[y0:y1,x0:x1]=matte['alpha'].astype(np.float32)/65535;alpha=np.ascontiguousarray(np.rot90(portrait,-1))
        image_path=root/'dataset'/frame['file_path'];image=cv2.imread(str(image_path))[...,::-1].astype(np.float32)/255
        tangent,q=orientations(image);nd=np.load(npz)['median']
        quality=((depth>0)&(photo>0)&np.isfinite(depth)&(np.abs(photo-depth)<=.001*np.maximum(depth,1e-8))
                 &(sem>.8)&(alpha>=.98)&(q>.35)&(nd>0)&(np.abs(depth-nd)<=.01*nd))
        yy,xx=np.nonzero(quality)
        if len(xx):focus_points.append(unproject(np.stack([xx[::100]+.5,yy[::100]+.5],-1),depth[yy[::100],xx[::100]],frame))
        cache.append(dict(depth=depth,hair=sem,tangent=tangent,q=q,quality=quality))
        provenance.append(dict(camera=name,geometric_sha256=sha(dp),photometric_sha256=sha(pp),rgb_sha256=sha(image_path),matte_sha256=sha(mp),prior_sha256=sha(npz)))
        atomic_json(output/'progress.json',dict(stage='maps',completed=i+1,total=59))
    centers=np.stack([cv_pose(f)[:3,3] for f in frames]);focus=np.median(np.concatenate(focus_points),axis=0)
    rng=np.random.default_rng(42);all_camera=[];all_uv=[];all_z=[];all_support=[];all_cost=[];rows=[]
    for index,(frame,entry) in enumerate(zip(frames,cache)):
        yy,xx=np.nonzero(entry['quality']);chosen=rng.choice(len(xx),min(per_camera,len(xx)),replace=False);yy,xx=yy[chosen],xx[chosen]
        uv=np.stack([xx+.5,yy+.5],-1);z=entry['depth'][yy,xx];xyz=unproject(uv,z,frame)
        witnesses=diverse_witnesses(centers,index,focus);normals=[orientation_planes(uv,entry['tangent'][yy,xx],frame)];weights=[entry['q'][yy,xx]];support=np.zeros(len(xx),np.int16)
        for j in witnesses:
            other=frames[j];source=cache[j];p,d=project(xyz,other);mx,my=(p[:,0]-.5).astype(np.float32)[:,None],(p[:,1]-.5).astype(np.float32)[:,None]
            sample=lambda a,mode=cv2.INTER_LINEAR:cv2.remap(a,mx,my,mode,borderMode=cv2.BORDER_CONSTANT)[:,0]
            observed=sample(source['depth'],cv2.INTER_NEAREST);qualified=sample(source['quality'].astype(np.float32),cv2.INTER_NEAREST)>.5
            backward,_=project(unproject(p,observed,other),frame);rt=np.linalg.norm(backward-uv,axis=-1)
            a,b=xyz-centers[index],xyz-centers[j];cosine=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).clip(1e-12)
            parallax=np.arccos(np.clip(cosine,-1,1))>=np.deg2rad(3.)
            valid=(qualified&(d>0)&(observed>0)&(rt<=1.5)&(np.abs(observed-d)<=.002*np.maximum(d,1e-8))&parallax)
            support+=valid;normals.append(orientation_planes(p,sample(source['tangent']),other));weights.append(sample(source['q'])*valid)
        cost,_=orientation_cost(np.stack(normals),np.stack(weights))
        take=(support>=3)&(cost<.03)
        all_camera.append(np.full(int(take.sum()),index,np.int16));all_uv.append(uv[take].astype(np.float32));all_z.append(z[take].astype(np.float32));all_support.append(support[take].astype(np.uint8));all_cost.append(cost[take].astype(np.float32))
        rows.append(dict(camera=frame['physical_camera'],qualified_pixels=int(entry['quality'].sum()),candidates=len(xx),stereo_supported=int((support>=3).sum()),orientation_supported=int(take.sum()),witnesses=[frames[j]['physical_camera'] for j in witnesses]))
        atomic_json(output/'progress.json',dict(stage='multiview',completed=index+1,total=59,accepted=sum(len(x) for x in all_z)))
    camera=np.concatenate(all_camera);uv=np.concatenate(all_uv);z=np.concatenate(all_z);validation=rng.random(len(z))<.05
    np.savez_compressed(output/'anchors.npz',camera_names=np.asarray([f['physical_camera'] for f in frames]),camera=camera,uv=uv,camera_z=z,support=np.concatenate(all_support),orientation_cost=np.concatenate(all_cost),validation=validation)
    atomic_json(output/'anchors.json',dict(dataset_sha256=dataset_sha,npz_sha256=sha(output/'anchors.npz'),anchors=len(z),fit_anchors=int((~validation).sum()),validation_anchors=int(validation.sum()),cameras_with_anchors=int(len(np.unique(camera))),
        provenance=provenance,records=rows,source='existing native PatchMatch; no new stereo fit',prior='frozen LookCloser median within 1%; no RoMa witness locations used for filtering',
        support='>=3 other pose-diverse stereo views; parallax>=3deg, source roundtrip<=1.5px, relative z<=.2%',orientation='multiscale structure tensors, modulo pi, normalized lambda_min/trace<.03',
        source_mask='semantic hair>.8, train person matte>=.98, orientation confidence>.35, photo/geometric agreement<=.1%',
        prior_receipt_sha256=sha(prior/'complete.json'),uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))
    atomic_json(output/'progress.json',dict(stage='complete',anchors=len(z),cameras_with_anchors=int(len(np.unique(camera)))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','mvs','prior','targets','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.mvs,a.prior,a.targets,a.output)


if __name__=='__main__':main()
