"""Post-hoc additional held-out views; never calibration/geometry/texture inputs.

No extra face/campaign metrics. The original F/B campaign evaluation protocol
is unchanged. These two unused eval cameras test visual shape generalization.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,SOURCE,ROOT,exr,display,HELD_CAMERAS

OUT=Path('/mnt/data/dec5_constrained_forearm_additional_heldout')
NAMES=['J004_D005_1210TA','L004_B005_12106A']
BASE=Path('/mnt/data/dec5_forearm_admission_quadric_bounded')
NEW=Path('/mnt/data/dec5_constrained_forearm_surface')
PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')


def ground_truth(output,frame):
    source=read(SOURCE/frame/'transforms.json');gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    folder=output/frame;folder.mkdir(parents=True,exist_ok=True);records=[]
    for name in NAMES:
        row=next(r for r in source['frames'] if r['physical_camera']==name);path=SOURCE/frame/row['file_path']
        rgb=np.rint(display(exr(path),gain)*255).clip(0,255).astype(np.uint8)
        dest=folder/(name+'_gt.png');Image.fromarray(np.rot90(rgb)).save(dest)
        records.append(dict(camera=name,source=str(path),source_sha256=sha(path),gt=str(dest),gt_sha256=sha(dest)))
    atomic_json(folder/'gt_receipt.json',dict(records=records,exposure_sha256=sha(ROOT/'exposure.json'),fixed_exposure_gain=gain,
        per_camera_profile_applied_to_heldout=False,evaluation_only=True,geometry_or_source_selection_input=False,
        script_sha256=sha(__file__)))


def render(output,frame,name):
    from render_patchmatch_camera_path import normalize_frame
    import render_smooth_temporal_mesh_video as renderer
    from study_early_texture_prior import install
    install();renderer.torch.set_num_threads(2);parent=renderer.verify_request(PARENT)
    rows,_,metadata=cameras(frame)
    if name not in HELD_CAMERAS or any(r['physical_camera']==name for r in rows):raise ValueError('Target is not held out')
    cal=read(CALIBRATION);target=normalize_frame(next(r for r in cal['frames'] if r['physical_camera']==name),cal,read(metadata))
    for label,root in [('previous',BASE),('constrained',NEW)]:
        result=read(root/frame/'geometry_result.json')
        if not result['observed_guard_passed'] or sha(root/frame/'guarded.ply')!=result['hashes']['guarded.ply']:raise ValueError('Invalid geometry input')
        dest=output/'rgb'/frame/name/label;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
        request=deepcopy(parent);request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
        row=request['inventory'][0];row.update(camera=target,mesh=str(root/frame/'guarded.ply'),mesh_sha256=sha(root/frame/'guarded.ply'))
        request.update(partial_diagnostic_only=True,full_video_candidate=False,additional_heldout_visual_only=True,
            geometry_result_sha256=sha(root/frame/'geometry_result.json'),calibration_sha256=sha(CALIBRATION))
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Frozen diagnostic render mismatch')
        atomic_json(dest/'request.json',request);renderer.render(dest,[frame])


def panels(output,frame):
    records=[];receipt=read(output/frame/'gt_receipt.json')
    for name in NAMES:
        r=next(r for r in receipt['records'] if r['camera']==name);gt=Path(r['gt'])
        if sha(gt)!=r['gt_sha256']:raise ValueError('Changed GT')
        ims=[Image.open(gt).convert('RGB')];hashes={str(gt):sha(gt)}
        for label in ['previous','constrained']:
            root=output/'rgb'/frame/name/label;folder=root/'frames'/frame;c=read(folder/'complete.json')
            if c['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed render request')
            for p,h in c['hashes'].items():
                if sha(folder/p)!=h:raise ValueError('Changed prediction')
            result=read(folder/'result.json')
            if result['target_rgb_read'] or set(result['source_cameras'])&HELD_CAMERAS:raise ValueError('Heldout leakage')
            p=folder/'frame.png';ims.append(Image.open(p).convert('RGB'));hashes[str(p)]=sha(p)
        overview=Image.new('RGB',(1620,990));draw=ImageDraw.Draw(overview)
        for i,(im,label) in enumerate(zip(ims,['GT','previous','constrained'])):
            overview.paste(im.resize((540,960)),(i*540,30));draw.text((i*540+4,5),name+' '+label,fill='white')
        p=output/frame/(name+'_overview.png');overview.save(p);hashes[str(p)]=sha(p)
        records.append(dict(camera=name,hashes=hashes,visual_status='requires_actual_review'))
    atomic_json(output/frame/'comparison.json',dict(records=records,face_metrics_computed=False,campaign_csv_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['gt','render','panels']);p.add_argument('--frame',default='001037')
    p.add_argument('--camera',choices=NAMES);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.action=='gt':ground_truth(a.output,a.frame)
    elif a.action=='render':
        if a.camera is None:p.error('--camera required for render')
        render(a.output,a.frame,a.camera)
    else:panels(a.output,a.frame)
