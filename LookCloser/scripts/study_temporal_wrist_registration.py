"""Train-only temporal flow + rigid palm registration; no production mesh edits."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from joint_temporal_texture import read, sha, atomic_json, cameras
from study_wrist_observations import OUTPUT as OBS, NAMES
from study_confidence_depth_prior import unproject, raycast_integer
from diffusion_mesh_repair import scene_for
from temporal_rigid_patch import transfer_points, project_native, warp_rigid, fit_rigid, compose_flow_fields

OUT = Path('/mnt/data/dec5_temporal_wrist_registration')
SOURCE = '001029'; TARGET = '001037'
PALM = [(245,1530),(308,1510),(343,1560),(361,1618),(348,1650),(281,1630),(218,1580)]
CROP = (0,1152,640,1920)


def sample(array, xy):
    return cv2.remap(array.astype(np.float32), xy[:, 0].astype(np.float32)[None],
                     xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]


def flow(output, chain=False):
    output.mkdir(parents=True, exist_ok=False)
    weights = Raft_Large_Weights.C_T_SKHT_V2
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
    if not checkpoint.exists(): raise ValueError('Expected cached RAFT weights')
    times=['001029','001031','001033','001035','001037'] if chain else [SOURCE,TARGET]
    request = dict(source_time=SOURCE, target_time=TARGET, cameras=NAMES, crop=CROP, flow_time_chain=times,
        source_palm_polygon=PALM, flow_weights=weights.name, flow_weights_sha256=sha(checkpoint),
        model_updates=12, forward_backward_pixels=2., heldout_used=False,
        source_observation_sha256=sha(OBS/SOURCE/'result.json'), target_observation_sha256=sha(OBS/TARGET/'result.json'),
        validation_camera=NAMES[-1], geometry_modified=False,
        observation_receipts={f:sha(OBS/f/'result.json') for f in times},
        scripts={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'temporal_rigid_patch.py','study_wrist_observations.py']})
    atomic_json(output/'request.json',request)
    torch.set_num_threads(2); model=raft_large(weights=weights,progress=False).cuda().eval(); records=[]
    for name in NAMES:
        paths=[OBS/f/(name+'.png') for f in times]
        for frame,path in zip(times,paths):
            expected=next(r for r in read(OBS/frame/'result.json')['records'] if r['camera']['physical_camera']==name)
            if sha(path)!=expected['image_sha256']:raise ValueError('Changed observation')
        images=[np.array(Image.open(p).crop(CROP)) for p in paths]
        fields=[]
        for i in range(len(times)-1):
            inputs=torch.from_numpy(np.stack(images[i:i+2]).transpose(0,3,1,2).copy()).float().cuda()/255
            a,b=weights.transforms()(inputs,inputs.flip(0))
            with torch.inference_mode(): prediction=model(a,b,num_flow_updates=12)[-1].cpu().numpy()
            fields.append(prediction.transpose(0,2,3,1))
        forward,fv=compose_flow_fields([p[0] for p in fields]);backward,bv=compose_flow_fields([p[1] for p in reversed(fields)])
        path=output/(name+'_flow.npz');np.savez_compressed(path,forward=forward,backward=backward,forward_valid=fv,backward_valid=bv,pair_flows=np.stack(fields))
        records.append(dict(camera=name,path=str(path),sha256=sha(path)))
        print('flow',name,flush=True)
    atomic_json(output/'flow_result.json',dict(records=records,request_sha256=sha(output/'request.json')))


def register(output):
    request=read(output/'request.json')
    for name,digest in request['scripts'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Changed registration code')
    sr,mesh_path,sm=cameras(SOURCE);tr,_,tm=cameras(TARGET)
    sr={r['physical_camera']:r for r in sr};tr={r['physical_camera']:r for r in tr}
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    reference=sr[NAMES[1]];md=raycast_integer(scene,reference)
    polygon=np.zeros((1920,1080),np.uint8);cv2.fillPoly(polygon,[np.array(PALM,np.int32)],1)
    mask=np.rot90(polygon,-1).astype(bool);yy,xx=np.mgrid[:1080,:1920]
    y,x=np.nonzero(mask&(md>0)&(xx%4==0)&(yy%4==0))
    source_points=unproject(reference,x,y,md[y,x]);points=transfer_points(source_points,read(sm),read(tm))
    pi=[];ci=[];observations=[];source_uv=[];fb_errors=[];diagnostics=[]
    for index,name in enumerate(NAMES):
        record=next(r for r in read(output/'flow_result.json')['records'] if r['camera']==name)
        if sha(record['path'])!=record['sha256']:raise ValueError('Changed flow')
        data=np.load(record['path']);uv,z=project_native(source_points,sr[name]);xy=np.column_stack([uv[:,1],1919-uv[:,0]])
        cropxy=xy-[CROP[0],CROP[1]]
        forward=sample(data['forward'],cropxy);targetxy=xy+forward;back=sample(data['backward'],targetxy-[CROP[0],CROP[1]])
        fb=np.linalg.norm(forward+back,axis=1)
        depth=raycast_integer(scene,sr[name]);observed=sample(depth,uv)
        inside=(cropxy[:,0]>3)&(cropxy[:,0]<636)&(cropxy[:,1]>3)&(cropxy[:,1]<764)
        dest=targetxy-[CROP[0],CROP[1]]
        visible=inside&(dest[:,0]>3)&(dest[:,0]<636)&(dest[:,1]>3)&(dest[:,1]<764)&(z>0)&(observed>0)&(abs(z-observed)<.002)&(fb<2)
        if 'forward_valid' in data:
            visible&=(sample(data['forward_valid'],cropxy)>.999)&(sample(data['backward_valid'],dest)>.999)
        ids=np.flatnonzero(visible);pi.extend(ids);ci.extend([index]*len(ids));observations.extend(np.column_stack([1919-targetxy[ids,1],targetxy[ids,0]]))
        source_uv.extend(xy[ids]);fb_errors.extend(fb[ids]);diagnostics.append(dict(camera=name,accepted=len(ids),sampled=len(points)))
    pi=np.array(pi);ci=np.array(ci);observations=np.asarray(observations);target_cameras=[tr[n] for n in NAMES]
    chosen=ci==1;camera=target_cameras[1]
    atomic_json(output/'correspondence_counts.json',dict(records=diagnostics,reference_accepted=int(chosen.sum()),
        request_sha256=sha(output/'request.json'),minimum_reference_points=30))
    if chosen.sum()<30:raise ValueError(f'Only {chosen.sum()} consistent reference tracks; no rigid fit')
    k=np.array([[camera['fl_x'],0,camera['cx']],[0,camera['fl_y'],camera['cy']],[0,0,1.]])
    cv2.setRNGSeed(0)
    okay,rv,tv,inliers=cv2.solvePnPRansac(points[pi[chosen]],observations[chosen],k,None,iterationsCount=500,reprojectionError=3.,flags=cv2.SOLVEPNP_EPNP)
    if not okay or inliers is None or len(inliers)<30:raise ValueError('PnP initialization failed')
    pose=np.asarray(camera['transform_matrix'])@np.diag([1.,-1.,-1.,1.]);motion=np.eye(4)
    motion[:3,:3]=cv2.Rodrigues(rv)[0];motion[:3,3]=tv[:,0];motion=pose@motion
    center=points.mean(0);initial=np.r_[Rotation.from_matrix(motion[:3,:3]).as_rotvec(),motion[:3,:3]@center+motion[:3,3]-center]
    parameters,center,errors=fit_rigid(points,target_cameras,pi,ci,observations,initial,list(range(5)))
    moved=warp_rigid(points,parameters,center);folder=output/'review';folder.mkdir(exist_ok=True)
    for index,name in enumerate(NAMES):
        hit=ci==index;predicted=project_native(moved[pi[hit]],target_cameras[index])[0]
        projected=np.column_stack([predicted[:,1],1919-predicted[:,0]])
        observed=np.column_stack([observations[hit,1],1919-observations[hit,0]])
        im=Image.open(OBS/TARGET/(name+'.png')).copy();draw=ImageDraw.Draw(im)
        for a,b in zip(observed[::5],projected[::5]):
            draw.line([tuple(a),tuple(b)],fill=(255,255,0),width=1)
            draw.ellipse((b[0]-2,b[1]-2,b[0]+2,b[1]+2),fill=(0,255,255))
        im.crop((0,1250,500,1920)).save(folder/(name+'_reprojection.png'))
        diagnostics[index].update(median_error=float(np.median(errors[hit])),p90_error=float(np.quantile(errors[hit],.9)),fit_input=index<5)
    np.savez_compressed(output/'correspondences.npz',source_points=source_points,common_gauge_points=points,
        point_indices=pi,camera_indices=ci,observations=observations,source_portrait_uv=np.array(source_uv),fb_errors=np.array(fb_errors),
        parameters=parameters,center=center,errors=errors)
    atomic_json(output/'registration_result.json',dict(source_mesh=str(mesh_path),source_mesh_sha256=sha(mesh_path),
        source_metadata=str(sm),source_metadata_sha256=sha(sm),target_metadata=str(tm),target_metadata_sha256=sha(tm),
        parameters=parameters.tolist(),center=center.tolist(),records=diagnostics,
        point_count=len(points),correspondence_count=len(pi),array_sha256=sha(output/'correspondences.npz'),
        request_sha256=sha(output/'request.json'),geometry_modified=False,visual_status='pending'))
    print(diagnostics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['flow','flow-chain','register']);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();{'flow':flow,'flow-chain':lambda p:flow(p,True),'register':register}[a.action](a.output)
