"""Opt-in presentation edit: dynamic 3D shot dissolves into actual train footage.

The user explicitly requested a real train-camera ending. This is not a mesh
repair or a reconstruction-quality result. The full original RGB background is
retained. No segmentation, generated imagery, temporal interpolation or new
radiometric fitting occurs. Existing raw 3D renders are never modified.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import cameras, exr, display, read, sha, atomic_json, ROOT as COLOR
from audit_cinematic_path_requests import raw_matrix, CALIBRATION, HELD
from review_jaw_repair_transfer import verified_image

FIRST_TRAIN = 118
PURE_TRAIN = 126
COUNT = 150


def dissolve_alpha(index):
    """Eight intermediate weights, then exactly 24 pure source-video frames."""
    t = np.clip((index-(FIRST_TRAIN-1))/(PURE_TRAIN-(FIRST_TRAIN-1)), 0., 1.)
    return float(t**3*(10-15*t+6*t*t))


def blend_display(render, train, alpha):
    if alpha == 0: return render.copy()
    if alpha == 1: return train.copy()
    return np.rint(render.astype(np.float32)*(1-alpha)+train.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)


def intrinsics_sample(linear, source, target):
    """Exact NumPy bilinear RGB sampling for coincident undistorted pinholes."""
    height, width = int(target['h']), int(target['w'])
    yy, xx = np.mgrid[:height, :width]
    u = (xx+.5-target['cx'])/target['fl_x']*source['fl_x']+source['cx']-.5
    v = (yy+.5-target['cy'])/target['fl_y']*source['fl_y']+source['cy']-.5
    h, w = linear.shape[:2]
    if not (u.min() >= -1e-9 and v.min() >= -1e-9 and u.max() <= w-1+1e-9 and v.max() <= h-1+1e-9):
        raise ValueError('Virtual ending lens asks for pixels outside the real train image')
    u, v = np.clip(u,0,w-1), np.clip(v,0,h-1)
    ix, iy = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u-ix, v-iy
    result = np.zeros((height,width,3),np.float32)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        weight=(fu if dx else 1-fu)*(fv if dy else 1-fv)
        result += linear[np.minimum(iy+dy,h-1),np.minimum(ix+dx,w-1)]*weight[...,None]
    return result


def write_png(path, image):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    Image.fromarray(image).save(temp,format='PNG');os.replace(temp,path)


def configuration(root):
    q=read(root/'request.json'); expected=[f'{899+2*i:06d}' for i in range(COUNT)]
    assert q['ordered_frame_ids']==expected and [r['frame_id'] for r in q['inventory']]==expected
    assert [r['index'] for r in q['inventory']]==list(range(COUNT))
    for name,key in [('parameters.npz','profiles_sha256'),('exposure.json','exposure_sha256')]:
        assert sha(COLOR/name)==q[key]
    assert sha(CALIBRATION)==q['calibration_sha256']
    cal=read(CALIBRATION); name=q['camera_path_report']['endpoint_train_camera']
    assert name not in HELD
    actual=next(r for r in cal['frames'] if r['physical_camera']==name)
    for record in q['inventory'][FIRST_TRAIN:]:
        assert sha(record['metadata'])==record['metadata_sha256']
        raw=raw_matrix(record['camera']['transform_matrix'],read(record['metadata']),cal)
        np.testing.assert_allclose(raw,actual['transform_matrix'],atol=1e-10,rtol=0)
        for key in ['fl_x','fl_y','cx','cy','w','h']:
            assert record['camera'][key]==q['inventory'][-1]['camera'][key]
        assert all(abs(record['camera'].get(k,0)) < 1e-12 for k in ['k1','k2','p1','p2'])
    return q,dict(raw_request_sha256=sha(root/'request.json'),script_sha256=sha(__file__),
        audit_helper_sha256=sha(Path(__file__).with_name('audit_cinematic_path_requests.py')),
        color_helper_sha256=sha(Path(__file__).with_name('joint_temporal_texture.py')),
        heldout_used=False,explicit_user_requested_live_action_ending=True,
        real_train_camera=name,dissolve_indices=list(range(FIRST_TRAIN,PURE_TRAIN)),
        pure_train_indices=list(range(PURE_TRAIN,COUNT)),camera_and_lens_hold_start_index=FIRST_TRAIN,
        original_source_background_preserved=True,semantic_mask=False,new_color_fit=False,
        mesh_repair=False,all_frames_are_3d_renders=False,quality_metrics=False,
        fixed_profiles_sha256=q['profiles_sha256'],fixed_exposure_sha256=q['exposure_sha256'])


def prepare(root):
    q,config=configuration(root);out=root/'train_ending';out.mkdir(exist_ok=True)
    if (out/'request.json').exists():assert read(out/'request.json')==config
    else:atomic_json(out/'request.json',config)
    log_gain=np.load(COLOR/'parameters.npz')['log_gain'];gains=np.exp(log_gain-log_gain.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    for record in q['inventory'][FIRST_TRAIN:]:
        frame=record['frame_id'];folder=out/'frames'/frame;done=folder/'complete.json'
        if done.exists():
            receipt=read(done);assert receipt['request_sha256']==sha(out/'request.json')
            assert sha(folder/'frame.png')==receipt['image_sha256']
            assert sha(receipt['source_exr'])==receipt['source_sha256'];continue
        rows,_,_=cameras(frame);ci=next(i for i,r in enumerate(rows) if r['physical_camera']==config['real_train_camera'])
        source=rows[ci];manifest=next(r for r in q['source_rows'] if Path(r['source_dataset']).name==frame)
        frozen=next(r for r in manifest['source_images'] if r['physical_camera']==config['real_train_camera'])
        assert 'frame_train_' in frozen['file_path']
        assert Path(source['file_path'])==(Path(manifest['source_dataset'])/frozen['file_path']).resolve()
        assert sha(source['file_path'])==frozen['sha256']
        linear=intrinsics_sample(exr(source['file_path']),source,record['camera'])
        rgb=np.rot90(np.rint(display(linear*gains[ci],exposure)*255).clip(0,255).astype(np.uint8))
        assert rgb.shape==(1920,1080,3)
        write_png(folder/'frame.png',rgb)
        atomic_json(done,dict(frame_id=frame,index=record['index'],request_sha256=sha(out/'request.json'),
            image_sha256=sha(folder/'frame.png'),source_exr=source['file_path'],source_sha256=frozen['sha256'],
            source_physical_camera=config['real_train_camera'],camera=record['camera'],
            real_train_rgb=True,mesh_used=False,generated_pixels=False,bilinear_virtual_lens=True))
    print('Prepared 32 hash-bound actual train RGB ending frames',root,flush=True)


def compose(root):
    q,config=configuration(root);ending=root/'train_ending';out=root/'presentation';out.mkdir(exist_ok=True)
    assert read(ending/'request.json')==config
    if (out/'request.json').exists():assert read(out/'request.json')==config
    else:atomic_json(out/'request.json',config)
    records=[]
    for entry in q['inventory']:
        frame=entry['frame_id'];index=entry['index'];alpha=dissolve_alpha(index)
        provenance=dict(frame_id=frame,index=index,train_alpha=alpha,raw_render=None,real_train=None)
        train=None;raw=None
        if alpha<1:
            raw,receipt=verified_image(root,frame)
            assert receipt['camera']==entry['camera']
            provenance['raw_render']=dict(path=str(root/'frames'/frame/'frame.png'),
                sha256=sha(root/'frames'/frame/'frame.png'),complete_sha256=sha(root/'frames'/frame/'complete.json'))
        if alpha>0:
            folder=ending/'frames'/frame;receipt=read(folder/'complete.json')
            assert receipt['request_sha256']==sha(ending/'request.json')
            assert sha(folder/'frame.png')==receipt['image_sha256']
            train=np.asarray(Image.open(folder/'frame.png'))
            provenance['real_train']=dict(path=str(folder/'frame.png'),sha256=receipt['image_sha256'],
                complete_sha256=sha(folder/'complete.json'),source_exr=receipt['source_exr'],source_sha256=receipt['source_sha256'])
        image=raw if alpha==0 else train if alpha==1 else blend_display(raw,train,alpha)
        assert image.shape==(1920,1080,3)
        target=out/'frames'/frame;write_png(target/'frame.png',image)
        provenance.update(kind='3d_render' if alpha==0 else 'real_train_rgb' if alpha==1 else 'explicit_3d_to_train_dissolve',
            image_sha256=sha(target/'frame.png'),request_sha256=sha(out/'request.json'))
        atomic_json(target/'complete.json',provenance);records.append(provenance)
    assert len(records)==150 and sum(r['kind']=='real_train_rgb' for r in records)==24
    for r in records:
        image=np.asarray(Image.open(out/'frames'/r['frame_id']/'frame.png'))
        if r['train_alpha']==1:
            np.testing.assert_array_equal(image,np.asarray(Image.open(r['real_train']['path'])))
        if r['train_alpha']==0:
            np.testing.assert_array_equal(image,np.asarray(Image.open(r['raw_render']['path'])))
    atomic_json(out/'complete.json',dict(request_sha256=sha(out/'request.json'),ordered_frame_ids=q['ordered_frame_ids'],
        records=records,frame_count=150,fps=24,duration_seconds=6.25,pure_train_count=24,dissolve_count=8,
        all_frames_are_3d_renders=False,visual_status='pending',last_second_is_actual_dynamic_train_video=True))
    print('Composed and checked150 presentation frames; 118 rendered +8 dissolve +24 real train',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-ending',action='store_true');action.add_argument('--compose',action='store_true')
    a=parser.parse_args();prepare(a.root) if a.prepare_ending else compose(a.root)
