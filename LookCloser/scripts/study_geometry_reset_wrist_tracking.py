"""Re-anchor adjacent flow to shared 3D motion; preserve independent end tracks."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras
from temporal_rigid_patch import transfer_points,project_native,warp_rigid,fit_rigid
from study_temporal_wrist_registration import sample,CROP
from study_wrist_observations import OUTPUT as OBS,NAMES

PARENT=Path('/mnt/data/dec5_temporal_wrist_chain_registration')
OUT=Path('/mnt/data/dec5_geometry_reset_wrist_tracking')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    parent=read(PARENT/'registration_result.json');data=np.load(PARENT/'correspondences.npz');request=read(PARENT/'request.json')
    if sha(PARENT/'correspondences.npz')!=parent['array_sha256']:raise ValueError('Changed correspondence evidence')
    flow=[];hashes={}
    for r in read(PARENT/'flow_result.json')['records']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed flow')
        flow.append(np.load(r['path'])['pair_flows']);hashes[r['path']]=r['sha256']
    times=request['flow_time_chain'];points=data['source_points'].copy();previous_meta=read(parent['source_metadata']);records=[];arrays={}
    atomic_json(output/'request.json',dict(parent_result_sha256=sha(PARENT/'registration_result.json'),
        original_tracks_sha256=parent['array_sha256'],flow_hashes=hashes,times=times,
        shared_pose_seed_each_step=True,forward_backward_pixels=2.,fit_cameras=NAMES[:5],
        independent_end_validation_camera=NAMES[-1],heldout_rgb_used=False,geometry_modified=False,
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('temporal_rigid_patch.py'))))
    for step,frame in enumerate(times[1:]):
        previous_rows,_,_=cameras(times[step]);previous={r['physical_camera']:r for r in previous_rows}
        target_rows,_,meta=cameras(frame);target={r['physical_camera']:r for r in target_rows};current_meta=read(meta)
        canonical=transfer_points(points,previous_meta,current_meta);pi=[];ci=[];obs=[];fb_stats=[]
        for index,name in enumerate(NAMES):
            ids=data['point_indices'][data['camera_indices']==index]
            uv,z=project_native(points[ids],previous[name]);portrait=np.column_stack([uv[:,1],1919-uv[:,0]]);q=portrait-[CROP[0],CROP[1]]
            forward=sample(flow[index][step,0],q);end=q+forward;back=sample(flow[index][step,1],end)
            fb=np.linalg.norm(forward+back,axis=1)
            good=(z>0)&(q[:,0]>3)&(q[:,0]<636)&(q[:,1]>3)&(q[:,1]<764)&(end[:,0]>3)&(end[:,0]<636)&(end[:,1]>3)&(end[:,1]<764)&(fb<2)
            xy=portrait[good]+forward[good];pi.extend(ids[good]);ci.extend([index]*int(good.sum()));obs.extend(np.column_stack([1919-xy[:,1],xy[:,0]]))
            fb_stats.append(dict(camera=name,accepted=int(good.sum()),available=len(ids)))
        pi=np.array(pi);ci=np.array(ci);obs=np.asarray(obs);rows=[target[n] for n in NAMES]
        params,center,errors=fit_rigid(canonical,rows,pi,ci,obs,np.zeros(6),list(range(5)))
        points=warp_rigid(canonical,params,center);previous_meta=current_meta
        for index,record in enumerate(fb_stats):
            selected=ci==index
            record.update(local_median=float(np.median(errors[selected])),local_p90=float(np.quantile(errors[selected],.9)))
        records.append(dict(frame=frame,parameters=params.tolist(),center=center.tolist(),records=fb_stats,metadata=str(meta),metadata_sha256=sha(meta)))
        arrays[frame+'_points']=points.copy();arrays[frame+'_point_indices']=pi;arrays[frame+'_camera_indices']=ci;arrays[frame+'_observations']=obs
        print(frame,fb_stats,flush=True)
    final=[]
    for index,name in enumerate(NAMES):
        selected=data['camera_indices']==index;ids=data['point_indices'][selected];original=data['observations'][selected]
        predicted,z=project_native(points[ids],target[name])
        if not (z>0).all():raise ValueError('Moved points behind camera')
        error=np.linalg.norm(predicted-original,axis=1)
        final.append(dict(camera=name,count=len(ids),independent_validation=index==5,
            original_chain_median=float(np.median(data['errors'][selected])),new_median=float(np.median(error)),new_p90=float(np.quantile(error,.9))))
        image=Image.open(OBS/times[-1]/(name+'.png')).copy();draw=ImageDraw.Draw(image)
        a=np.column_stack([predicted[:,1],1919-predicted[:,0]]);b=np.column_stack([original[:,1],1919-original[:,0]])
        for p,q in zip(a[::5],b[::5]):
            draw.line([tuple(p),tuple(q)],fill=(255,255,0),width=1);draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=(0,255,255))
        image.crop((0,1250,500,1920)).save(output/(name+'_independent_reprojection.png'))
    np.savez_compressed(output/'evidence.npz',**arrays)
    atomic_json(output/'result.json',dict(stages=records,independent_final_comparison=final,
        request_sha256=sha(output/'request.json'),arrays_sha256=sha(output/'evidence.npz'),
        geometry_modified=False,visual_status='pending'))
    print('independent final',final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);run(p.parse_args().output)
