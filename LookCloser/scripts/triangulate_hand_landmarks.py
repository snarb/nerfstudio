"""Train-only consistency check; predicted joints are not measured surface points."""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import least_squares
from temporal_rigid_patch import project_native


def triangulate(cameras, pixels):
    pixels = np.asarray(pixels, float)
    if len(cameras) < 3 or pixels.shape != (len(cameras), 2) or not np.isfinite(pixels).all():
        raise ValueError('Need at least three finite observations')
    equations = []
    for camera, (u, v) in zip(cameras, pixels):
        pose = np.asarray(camera['transform_matrix']) @ np.diag([1., -1., -1., 1.])
        extrinsic = np.linalg.inv(pose)[:3]
        intrinsic = np.array([[camera['fl_x'], 0, camera['cx']],
                              [0, camera['fl_y'], camera['cy']], [0, 0, 1.]])
        p = intrinsic @ extrinsic
        equations.extend([u*p[2]-p[0], v*p[2]-p[1]])
    _, singular, vh = np.linalg.svd(equations)
    if singular[-2] < singular[0]*1e-10 or abs(vh[-1, 3]) < 1e-10:
        raise ValueError('Degenerate camera geometry')
    initial = vh[-1, :3]/vh[-1, 3]
    def residual(point):
        return np.array([project_native(point[None], c)[0][0] for c in cameras])-pixels
    fit = least_squares(lambda p: residual(p).ravel(), initial, loss='soft_l1', f_scale=2., max_nfev=300)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Triangulation failed')
    if any(project_native(fit.x[None], c)[1][0] <= 0 for c in cameras):
        raise ValueError('Joint behind a fitting camera')
    return fit.x, np.linalg.norm(residual(fit.x), axis=1)


def run(source, output):
    from joint_temporal_texture import read, sha, atomic_json
    from PIL import Image, ImageDraw
    obs = Path('/mnt/data/dec5_wrist_observations')
    output.mkdir(parents=True, exist_ok=False)
    prior = read(source/'result.json'); request = read(source/'request.json')
    if sha(source/'request.json') != prior['request_sha256']:
        raise ValueError('Changed inference request')
    validation = 'H004_C005_1210SZ'
    manifest = dict(inference_result_sha256=sha(source/'result.json'),
                    inference_request_sha256=sha(source/'request.json'),
                    validation_camera=validation, fit_camera_names=[n for n in request['cameras'] if n != validation],
                    minimum_fit_views=3, robust_pixel_scale=2., script_sha256=sha(__file__),
                    helper_sha256=sha(Path(__file__).with_name('temporal_rigid_patch.py')),
                    heldout_used=False, geometry_changed=False, surface_depth_prior=False,
                    scope='cross-view agreement of model predictions, not error against ground-truth joints',
                    observations={f:sha(obs/f/'result.json') for f in request['times']})
    atomic_json(output/'request.json', manifest)
    summaries=[]
    for frame in request['times']:
        rows = {r['camera']['physical_camera']:r['camera'] for r in read(obs/frame/'result.json')['records']}
        records = [r for r in prior['records'] if r['frame']==frame]
        coordinates={}
        for record in records:
            if sha(record['source']) != record['source_sha256']: raise ValueError('Changed input RGB')
            if record['detected'] != 1: continue
            xy=np.array(record['hands'][0]['portrait_xy'])
            coordinates[record['camera']] = np.stack([1919-xy[:,1], xy[:,0]], axis=1)
        points=[]; entries=[]
        for joint in range(21):
            available=[n for n in manifest['fit_camera_names'] if n in coordinates
                       and np.all(coordinates[n][joint]>=0) and np.all(coordinates[n][joint]<[1920,1080])]
            if len(available)<3:
                entries.append(dict(joint=joint,status='insufficient_in_image_fit_views')); points.append([0.,0.,0.]); continue
            point, errors=triangulate([rows[n] for n in available], [coordinates[n][joint] for n in available])
            entry=dict(joint=joint,status='triangulated',fit_views=available,fit_errors=errors.tolist())
            if validation in coordinates and np.all(coordinates[validation][joint]>=0) and np.all(coordinates[validation][joint]<[1920,1080]):
                uv, depth=project_native(point[None],rows[validation])
                entry['validation_error']=float(np.linalg.norm(uv[0]-coordinates[validation][joint]))
                entry['validation_positive_depth']=bool(depth[0]>0)
            entries.append(entry); points.append(point.tolist())
        points=np.array(points); good=np.array([e['status']=='triangulated' for e in entries])
        folder=output/frame;folder.mkdir(); np.savez_compressed(folder/'evidence.npz',points=points,good=good)
        for name, xy in coordinates.items():
            image=Image.open(obs/frame/(name+'.png')).convert('RGB'); draw=ImageDraw.Draw(image)
            projected, _=project_native(points[good], rows[name]); portrait=np.stack([projected[:,1],1919-projected[:,0]],axis=1)
            observed=np.stack([xy[good,1],1919-xy[good,0]],axis=1)
            for a,b in zip(portrait,observed):
                draw.line([tuple(a),tuple(b)],fill='yellow',width=2)
                draw.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill='cyan')
            image.crop((0,1250,500,1920)).save(folder/(name+'_reprojection.png'))
        fit_errors=[x for e in entries for x in e.get('fit_errors',[])]
        validation_errors=[e['validation_error'] for e in entries if 'validation_error' in e]
        summary=dict(frame=frame,joints=entries,fit_median=float(np.median(fit_errors)),
                     validation_count=len(validation_errors),validation_median=float(np.median(validation_errors)),
                     validation_p90=float(np.percentile(validation_errors,90)))
        summaries.append(summary); print(frame, {k:v for k,v in summary.items() if k!='joints'},flush=True)
    atomic_json(output/'result.json',dict(request_sha256=sha(output/'request.json'),frames=summaries,
        visual_status='pending',geometry_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/mnt/data/dec5_hand_landmark_prior'))
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_hand_landmark_triangulation'))
    a=p.parse_args();run(a.source,a.output)
