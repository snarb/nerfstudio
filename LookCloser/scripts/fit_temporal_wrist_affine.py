"""Bounded affine versus rigid temporal palm prior; identical tracked evidence."""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw
from temporal_rigid_patch import project_native, warp_rigid
from joint_temporal_texture import cameras, read, sha, atomic_json
from study_wrist_observations import OUTPUT as OBS, NAMES

PARENT=Path('/mnt/data/dec5_temporal_wrist_chain_registration')
OUT=Path('/mnt/data/dec5_temporal_wrist_affine')


def fit(points, cameras, point_indices, camera_indices, observations):
    center=points.mean(0);pi=point_indices;ci=camera_indices;selected=ci<5
    def moved(parameters):
        return (points-center)@(np.eye(3)+parameters[:9].reshape(3,3)).T+center+parameters[9:]
    def residual(parameters):
        current=moved(parameters);prediction=np.empty_like(observations)
        for index,camera in enumerate(cameras):
            m=ci==index;prediction[m]=project_native(current[pi[m]],camera)[0]
        return prediction-observations
    # 10% coefficient bounds, with a preference for the already fitted rigid
    # shape; a separate post-fit bound rejects displacement over .003.
    limit=np.r_[np.full(9,.1),np.full(3,.003)]
    result=least_squares(lambda p:np.r_[residual(p)[selected].ravel(),100*p[:9],p[9:]/.0005],
                         np.zeros(12),bounds=(-limit,limit),loss='soft_l1',f_scale=2.,max_nfev=300)
    if not result.success:raise ValueError('Affine solve failed')
    new=moved(result.x);errors=np.linalg.norm(residual(result.x),axis=1)
    displacement=np.linalg.norm(new-points,axis=1)
    return new,errors,dict(parameters=result.x.tolist(),maximum_displacement=float(displacement.max()),
        within_displacement_bound=bool(displacement.max()<=.003),
        singular_values=np.linalg.svd(np.eye(3)+result.x[:9].reshape(3,3),compute_uv=False).tolist())


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    parent=read(PARENT/'registration_result.json');data=np.load(PARENT/'correspondences.npz')
    if sha(PARENT/'correspondences.npz')!=parent['array_sha256']:raise ValueError('Changed tracks')
    request=dict(parent_result_sha256=sha(PARENT/'registration_result.json'),correspondences_sha256=parent['array_sha256'],
        maximum_matrix_coefficient_change=.1,maximum_translation_component=.003,maximum_point_displacement=.003,
        fit_cameras=NAMES[:5],validation_camera=NAMES[-1],heldout_rgb_used=False,production_changed=False,
        script_sha256=sha(__file__),rigid_helper_sha256=sha(Path(__file__).with_name('temporal_rigid_patch.py')))
    atomic_json(output/'request.json',request)
    rows,_,_=cameras('001037');lookup={r['physical_camera']:r for r in rows};rows=[lookup[n] for n in NAMES]
    rigid=warp_rigid(data['common_gauge_points'],data['parameters'],data['center'])
    moved,errors,stats=fit(rigid,rows,data['point_indices'],data['camera_indices'],data['observations'])
    records=[]
    for index,name in enumerate(NAMES):
        mask=data['camera_indices']==index;ids=data['point_indices'][mask]
        a=project_native(rigid[ids],rows[index])[0];b=project_native(moved[ids],rows[index])[0]
        target=data['observations'][mask];panel=Image.new('RGB',(1000,694));draw=ImageDraw.Draw(panel)
        for col,(title,xy) in enumerate([('rigid',a),('bounded affine',b)]):
            im=Image.open(OBS/'001037'/(name+'.png')).copy();d=ImageDraw.Draw(im)
            q=np.column_stack([xy[:,1],1919-xy[:,0]]);gt=np.column_stack([target[:,1],1919-target[:,0]])
            for p,t in zip(q[::5],gt[::5]):
                d.line([tuple(p),tuple(t)],fill=(255,255,0),width=1);d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=(0,255,255))
            panel.paste(im.crop((0,1250,500,1920)),(500*col,24));draw.text((500*col+4,4),title,fill='white')
        panel.save(output/(name+'_comparison.png'))
        records.append(dict(camera=name,fit_input=index<5,count=int(mask.sum()),
            rigid_median=float(np.median(data['errors'][mask])),rigid_p90=float(np.quantile(data['errors'][mask],.9)),
            affine_median=float(np.median(errors[mask])),affine_p90=float(np.quantile(errors[mask],.9))))
    np.savez_compressed(output/'fit.npz',points=moved,errors=errors)
    atomic_json(output/'result.json',dict(stats=stats,records=records,request_sha256=sha(output/'request.json'),
        fit_sha256=sha(output/'fit.npz'),production_changed=False,visual_status='pending'))
    print(stats,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);run(p.parse_args().output)
