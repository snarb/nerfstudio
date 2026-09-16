"""Opt-in face-interior source recovery at within-face depth discontinuities.

Preserve baseline unless a newly admitted source has strictly higher existing
quality, an exact first mesh hit, a conservative face-only RGB footprint and
>=3 old-valid face-source witnesses. No target mask, RGB averaging or depth edit.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import numpy as np
import torch
import open3d as o3d
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from joint_temporal_texture import cameras,project,ROOT as COLOR,exr,display
from study_multiview_face_prior import read,save,sha
from study_temporal_source_retention import BASE
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from native_texture_footprint import sample_native,snap_centers,relevant_tap
from diagnose_gap_texture_admission import gates
from diagnose_nose_source_visibility import direct_visible
from calibrated_depth_witness import response_gains
from temporal_texture_view_prior import angle_weights
from view_consistent_source_quality import quality

ROOT=Path('/mnt/data/dec5_face_interior_visibility_001123')
SEM=Path('/mnt/data/dec5_train_face_support_001123')
FRAME='001123'


def proposals(old_valid,face_support,weights,chosen):
    n=old_valid.shape[1];j=np.arange(n);safe=np.clip(chosen,0,old_valid.shape[0]-1)
    anchor=(chosen<old_valid.shape[0])&old_valid[safe,j]&face_support[safe,j]
    votes=(old_valid&face_support).sum(0)
    old=weights[safe,j]
    candidate=(~old_valid)&face_support&(weights>old[None])&anchor[None]&(votes[None]>=3)
    return candidate,votes,old


def prepare():
    assert not ROOT.exists();ROOT.mkdir()
    sq=read(SEM/'request.json');complete=read(SEM/'complete.json');assert complete['request_sha256']==sha(SEM/'request.json')
    q=read(BASE/'request.json');record=next(r for r in q['inventory'] if r['frame_id']==FRAME)
    expected={r['physical_camera']:r['sha256'] for r in next(r for r in q['source_rows'] if Path(r['source_dataset']).name==FRAME)['source_images']}
    rows,_,_=cameras(FRAME);maskroot=Path(record['source_masks']['root'])
    names=read(maskroot/'cameras.json');original=np.load(maskroot/'masks.npz')['masks'];masks=[];bindings={}
    for n,k in [('masks.npz','masks_sha256'),('cameras.json','cameras_sha256'),('complete.json','complete_sha256')]:
        assert sha(maskroot/n)==record['source_masks'][k];bindings[str(maskroot/n)]=sha(maskroot/n)
    canvas=Image.new('RGB',(8*180,8*230));draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        name=row['physical_camera'];s=next(r for r in sq['records'] if r['camera']==name)
        assert s['source_sha256']==expected[name]
        r=next(r for r in complete['outputs'] if r['camera']==name);assert sha(r['path'])==r['sha256'];bindings[r['path']]=r['sha256']
        assert sha(s['input_path'])==s['input_sha256'];bindings[s['input_path']]=s['input_sha256']
        conf=np.load(r['path'])['confidence'];strict=distance_transform_edt(conf>=243)>=4
        full=np.zeros((1920,1080),bool);x0,y0,x1,y1=sq['crop'];full[y0:y1,x0:x1]=strict
        full=np.rot90(full,-1)&(original[names.index(name)]>0);masks.append(full)
        rgb=np.array(Image.open(s['input_path']));rgb[~strict]=(rgb[~strict]*.3).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb).resize((180,210)),((i%8)*180,(i//8)*230+20))
        draw.text(((i%8)*180+2,(i//8)*230+2),name[:9],fill='white')
        if name.startswith(('H004_C','I004_C','J004_C')):
            # Exact native nose context from the independently produced witness.
            witness=next(x for x in read('/mnt/data/dec5_nose_source_visibility_001123/result.json')['witnesses'] if x['camera']==name)
            b=witness['crop'];b=[b[0]-x0,b[1]-y0,b[2]-x0,b[3]-y0]
            Image.fromarray(rgb).crop(b).save(ROOT/(name+'_mask.png'))
    canvas.save(ROOT/'mask_overview.png');np.savez_compressed(ROOT/'face_masks.npz',masks=np.stack(masks))
    for p in [SEM/'request.json',SEM/'complete.json',BASE/'request.json',Path(__file__)]:bindings[str(p)]=sha(p)
    save(ROOT/'request.json',dict(frame=FRAME,input_hashes=bindings,face_masks_sha256=sha(ROOT/'face_masks.npz'),
        confidence_u8_min=243,interior_margin_pixels=4,minimum_old_valid_face_witnesses=3,
        exact_ray_tolerance=1e-5,quality_must_strictly_improve=True,geometry_changed=False,
        masks_from_train_only=True,model_probabilities_not_truth=True,script_sha256=sha(__file__)))


def render():
    assert not (ROOT/'result.json').exists();q=read(ROOT/'request.json')
    assert q['script_sha256']==sha(__file__)
    for p,h in q['input_hashes'].items():assert sha(p)==h,p
    assert sha(ROOT/'face_masks.npz')==q['face_masks_sha256']
    parent=read(BASE/'request.json');record=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    folder=BASE/'frames'/FRAME;r=read(folder/'result.json');receipt=read(folder/'complete.json')
    assert receipt['request_sha256']==sha(BASE/'request.json')
    for n,h in receipt['hashes'].items():assert sha(folder/n)==h,n
    assert sha(record['mesh'])==record['mesh_sha256']
    rows,_,_=cameras(FRAME);m=o3d.io.read_triangle_mesh(record['mesh']);v=np.asarray(m.vertices,np.float32);t=np.asarray(m.triangles,np.uint32)
    scene=scene_for(v,t);d,ids,b=camera_depth(scene,r['camera']);hit=np.isfinite(d)
    np.testing.assert_array_equal(np.where(hit,d,0),np.load(folder/'target_depth.npz')['depth'])
    rgb=np.array(Image.open(folder/'prediction_native.png'));oldrgb=rgb.copy();selected=np.array(Image.open(folder/'source_ids.png'));oldselected=selected.copy()
    spec=record['source_masks'];maskroot=Path(spec['root']);masknames=read(maskroot/'cameras.json');mm=np.load(maskroot/'masks.npz')['masks']
    with ThreadPoolExecutor(max_workers=4) as pool:depth=np.stack(list(pool.map(lambda row:camera_depth(scene,row)[0],rows)))
    for i,row in enumerate(rows):depth[i][mm[masknames.index(row['physical_camera'])]==0]=np.inf
    depth=np.where(np.isfinite(depth),depth,0)
    faces_tensor=torch.tensor(np.load(ROOT/'face_masks.npz')['masks'][:,None].astype(np.float32),device='cuda')
    centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    angle,_=angle_weights(rows,r['camera'],parent['recipe']['target_angle_sigma_degrees'])
    normal=np.cross(v[t][:,1]-v[t][:,0],v[t][:,2]-v[t][:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    pixels=np.flatnonzero(hit);records=[];proposed_count=0
    for start in range(0,len(pixels),30000):
        px=pixels[start:start+30000];f=ids.ravel()[px];bary=b.reshape(-1,2)[px];w=np.c_[1-bary.sum(1),bary]
        points=(v[t[f]]*w[:,:,None]).sum(1);uv,z=project(points,rows)
        qt=snap_centers(torch.tensor(uv[:,None],device='cuda'));support=torch.ones((len(rows),len(px)),dtype=torch.bool,device='cuda')
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            value=sample_native(faces_tensor,qt.floor()+qt.new_tensor([dx,dy]))[:,0,0]
            support&=(value>.5)|~relevant_tap(qt,dx,dy)
        support=support.cpu().numpy()
        # Avoid costly full visibility replay for batches with no face interior.
        if not support.any():continue
        oldg=gates(depth,uv,z)['final']
        direction=centers[:,None]-points;length=np.linalg.norm(direction,axis=2);direction/=length[...,None]
        weights=quality((direction*normal[f]).sum(-1),length,'incidence2')*angle[:,None]
        old=oldselected.ravel()[px];candidate,votes,oldquality=proposals(oldg,support,weights,old)
        # Keep the existing image border requirement for every new source.
        candidate&=(uv[...,0]>2)&(uv[...,0]<1917)&(uv[...,1]>2)&(uv[...,1]<1077)&(z>0)
        proposed_count+=int(candidate.sum());visible=np.zeros_like(candidate);direct=np.full(candidate.shape,np.nan,np.float32)
        for ci in np.flatnonzero(candidate.any(1)):
            j=np.flatnonzero(candidate[ci]);center=centers[ci]
            rays=np.c_[np.broadcast_to(center,points[j].shape),points[j]-center].astype(np.float32)
            values=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            direct[ci,j]=values;visible[ci,j]=direct_visible(values)
        new=np.argmax(weights*visible,axis=0);take=visible.any(0);js=np.flatnonzero(take)
        for j in js:records.append((int(px[j]),int(old[j]),int(new[j]),points[j],uv[new[j],j],float(oldquality[j]),float(weights[new[j],j]),int(votes[j]),float(direct[new[j],j])))
        print('pixels',start+len(px),'/',len(pixels),'new source points',len(records),flush=True)
    gains=response_gains(np.load(COLOR/'parameters.npz')['log_gain']);exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    expected={r['physical_camera']:r['sha256'] for r in next(r for r in parent['source_rows'] if Path(r['source_dataset']).name==FRAME)['source_images']}
    inputs={}
    for ci in sorted({r[2] for r in records}):
        group=[r for r in records if r[2]==ci];px=np.array([r[0] for r in group]);uv=np.array([r[4] for r in group],np.float32)
        path=rows[ci]['file_path'];assert sha(path)==expected[rows[ci]['physical_camera']];inputs[path]=sha(path)
        source=exr(path);qt=snap_centers(torch.tensor(uv[None,None],device='cuda'))
        values=sample_native(torch.tensor(source.transpose(2,0,1)[None],device='cuda'),qt)[0,:,0].T.cpu().numpy()*gains[ci]
        rgb.reshape(-1,3)[px]=np.rint(display(values.clip(0),exposure)*255).clip(0,255).astype(np.uint8);selected.ravel()[px]=ci
    changed_source=selected!=oldselected
    np.testing.assert_array_equal(rgb[~changed_source],oldrgb[~changed_source])
    Image.fromarray(rgb).save(ROOT/'prediction_native.png');Image.fromarray(np.rot90(rgb)).save(ROOT/'frame.png');Image.fromarray(selected).save(ROOT/'source_ids.png')
    np.savez_compressed(ROOT/'evidence.npz',pixels=np.array([r[0] for r in records],np.int64),old_sources=np.array([r[1] for r in records]),
        new_sources=np.array([r[2] for r in records]),points=np.array([r[3] for r in records]),uv=np.array([r[4] for r in records]),
        old_quality=np.array([r[5] for r in records]),new_quality=np.array([r[6] for r in records]),
        old_valid_face_votes=np.array([r[7] for r in records]),direct_t=np.array([r[8] for r in records]))
    save(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),baseline_complete_sha256=sha(folder/'complete.json'),
        baseline=str(folder),frame=FRAME,geometry_unchanged=True,depth_reference=str(folder/'target_depth.npz'),
        depth_sha256=sha(folder/'target_depth.npz'),new_source_points=len(records),candidate_pairs=proposed_count,
        changed_rgb=int(np.any(rgb!=oldrgb,2).sum()),new_black=int(((oldrgb.max(2)>0)&(rgb.max(2)==0)).sum()),
        source_rgb_hashes=inputs,visual_status='pending',production_promoted=False,
        hashes={n:sha(ROOT/n) for n in ['prediction_native.png','frame.png','source_ids.png','evidence.npz']}))
    print(read(ROOT/'result.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render']);a=p.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode():globals()[a.stage]()
