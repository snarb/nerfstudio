"""Locate when the temporal multiview palm discrepancy appears, on fixed tracks."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, cameras
from temporal_rigid_patch import compose_flow_fields, transfer_points, project_native, warp_rigid, fit_rigid
from study_temporal_wrist_registration import sample, CROP
from study_wrist_observations import OUTPUT as OBS, NAMES

PARENT=Path('/mnt/data/dec5_temporal_wrist_chain_registration')
OUT=Path('/mnt/data/dec5_temporal_wrist_stage_diagnosis')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    parent=read(PARENT/'registration_result.json');tracks=np.load(PARENT/'correspondences.npz')
    if sha(PARENT/'correspondences.npz')!=parent['array_sha256']:raise ValueError('Changed tracks')
    request=read(PARENT/'request.json');times=request['flow_time_chain'];fields=[];hashes={}
    for item in read(PARENT/'flow_result.json')['records']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed flow')
        fields.append(np.load(item['path'])['pair_flows']);hashes[item['path']]=item['sha256']
    pi=tracks['point_indices'];ci=tracks['camera_indices'];source_uv=tracks['source_portrait_uv'];records=[]
    atomic_json(output/'request.json',dict(parent_result_sha256=sha(PARENT/'registration_result.json'),
        source_arrays_sha256=parent['array_sha256'],flow_hashes=hashes,times=times,
        fixed_endpoint_track_population=True,heldout_rgb_used=False,production_changed=False,
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('temporal_rigid_patch.py'))))
    arrays={}
    for step,frame in enumerate(times[1:],1):
        rows,_,meta=cameras(frame);lookup={r['physical_camera']:r for r in rows};rows=[lookup[n] for n in NAMES]
        points=transfer_points(tracks['source_points'],read(parent['source_metadata']),read(meta))
        observations=np.empty_like(tracks['observations'])
        for index,name in enumerate(NAMES):
            selected=ci==index;flow,valid=compose_flow_fields([p[0] for p in fields[index][:step]])
            uv=source_uv[selected];target=uv+sample(flow,uv-[CROP[0],CROP[1]])
            if not (sample(valid,uv-[CROP[0],CROP[1]])>.999).all():raise ValueError('Fixed track leaves prefix crop')
            observations[selected]=np.column_stack([1919-target[:,1],target[:,0]])
        if step==len(times)-1:np.testing.assert_allclose(observations,tracks['observations'],atol=1e-3,rtol=0)
        full,center,error=fit_rigid(points,rows,pi,ci,observations,np.zeros(6),list(range(5)))
        minus,_,loo=fit_rigid(points,rows,pi,ci,observations,full,[0,1,2,4])
        moved=warp_rigid(points,full,center);folder=output/frame;folder.mkdir();stats=[]
        for index,name in enumerate(NAMES):
            selected=ci==index;pred=project_native(moved[pi[selected]],rows[index])[0];target=observations[selected]
            pred=np.column_stack([pred[:,1],1919-pred[:,0]]);target=np.column_stack([target[:,1],1919-target[:,0]])
            im=Image.open(OBS/frame/(name+'.png')).copy();draw=ImageDraw.Draw(im)
            for a,b in zip(pred[::5],target[::5]):
                draw.line([tuple(a),tuple(b)],fill=(255,255,0),width=1);draw.ellipse((a[0]-2,a[1]-2,a[0]+2,a[1]+2),fill=(0,255,255))
            im.crop((0,1250,500,1920)).save(folder/(name+'_reprojection.png'))
            stats.append(dict(camera=name,count=int(selected.sum()),median=float(np.median(error[selected])),p90=float(np.quantile(error[selected],.9)),
                without_H_A_median=float(np.median(loo[selected])),without_H_A_p90=float(np.quantile(loo[selected],.9))))
        records.append(dict(frame=frame,records=stats,parameters=full.tolist(),without_H_A_parameters=minus.tolist(),metadata=str(meta),metadata_sha256=sha(meta)))
        arrays[frame+'_observations']=observations;arrays[frame+'_errors']=error;arrays[frame+'_without_H_A_errors']=loo
        print(frame,stats,flush=True)
    np.savez_compressed(output/'stage_evidence.npz',**arrays)
    atomic_json(output/'result.json',dict(records=records,arrays_sha256=sha(output/'stage_evidence.npz'),
        request_sha256=sha(output/'request.json'),geometry_changed=False,visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);run(p.parse_args().output)
