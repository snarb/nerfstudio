#!/usr/bin/env python3
"""One held-out full-FOV4K test, native6K source RGB, HD field geometry.

Only training images feed source transfer/background. Held-out RGB and semantic
reference live in a separate metric dataset and never gate the rendered ROI.
"""
from pathlib import Path
import argparse,copy
import cv2,numpy as np,torch
from dec5_hair import cv_pose
from dec5_ray_background import RayBackgroundNetwork
from run_dec5_hair_campaign import atomic_json,sha
from run_dec5_sequence import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);a=p.parse_args();r=a.root
    torch.set_num_threads(2);cv2.setNumThreads(1)
    native=read(r/'native_rgb/data/transforms.json');meta=read(r/'frame/semantic_data/transforms.json')
    heldout=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(heldout)!=1 or len(meta['train_filenames'])!=64:raise ValueError('64+1 required')
    target=copy.deepcopy(heldout[0]);target.update(w=3840,h=1920)
    for key in ['fl_x','fl_y','cx','cy']:target[key]*=2
    center=cv_pose(target)[:3,3];train=[f for f in meta['frames'] if f['file_path'] in meta['train_filenames']]
    order=np.argsort([np.linalg.norm(cv_pose(f)[:3,3]-center) for f in train],kind='stable')[:8]
    sources=[train[int(i)] for i in order]
    for f in sources:
        image=r/'native_rgb/data'/f['file_path']
        if not image.is_file():raise ValueError('Missing native RGB source '+f['physical_camera'])
    atomic_json(r/'source8_hd.json',dict(frames=sources));atomic_json(r/'target4k.json',dict(frames=[target]))
    roi=r/'roi4k';roi.mkdir(exist_ok=True);path=roi/(target['physical_camera']+'.npz')
    np.savez_compressed(path,roi=np.ones((1920,3840),bool))
    atomic_json(roi/'complete.json',dict(records=[dict(camera=target['physical_camera'],npz_sha256=sha(path))],
        uses_validation_rgb=False,uses_test_rgb=False,method='entire target raster, no semantic output gate'))
    scene=r/'benchmark_scene';data=scene/'dataset';references=scene/'validation_reference'
    (data/'images').mkdir(parents=True,exist_ok=True);references.mkdir(exist_ok=True)
    gt_path=r/'native_rgb/data'/heldout[0]['file_path'];image=cv2.imread(str(gt_path))
    if image.shape[:2]!=(3072,6144):raise ValueError('Real native6K reference required')
    destination=data/target['file_path'];cv2.imwrite(str(destination),cv2.resize(image,(3840,1920),interpolation=cv2.INTER_AREA))
    evaluation=dict(camera_model='OPENCV',frame_id='000901',w=3840,h=1920,frames=[target],
        train_filenames=[],val_filenames=[target['file_path']],test_filenames=[])
    atomic_json(data/'transforms.json',evaluation)
    probability=np.load(a.reference)['probability'];probability=cv2.resize(probability,(3840,1920),interpolation=cv2.INTER_LINEAR)
    reference_path=references/(target['physical_camera']+'.npz');np.savez_compressed(reference_path,probability=probability)
    atomic_json(references/'complete.json',dict(reference_sha256=sha(reference_path),gt_sha256=sha(destination),
        source_reference_sha256=sha(a.reference),kind='fallible HD Sapiens reference resized to4K, metrics only',
        no_alpha_or_depth_ground_truth=True,used_for_fitting=False))
    checkpoint=r/'frame/ray_background/background.pt';request=read(checkpoint.parent/'request.json')
    if request['uses_validation_rgb'] or request['uses_test_rgb']:raise ValueError('Background leakage')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=RayBackgroundNetwork(state['origin_levels'],state['direction_levels'],state['width']).cuda().eval()
    model.load_state_dict(state['state_dict']);pose=torch.as_tensor(cv_pose(target),device='cuda',dtype=torch.float32)
    origin=(pose[:3,3]-torch.tensor(state['center'],device='cuda'))/state['radius']
    y,x=np.mgrid[:1920,:3840];x,y=x.reshape(-1),y.reshape(-1);rgb=np.empty((len(x),3),np.float32)
    with torch.no_grad():
        for start in range(0,len(x),65536):
            xx=torch.tensor(x[start:start+65536],device='cuda');yy=torch.tensor(y[start:start+65536],device='cuda')
            direction=torch.stack(((xx+.5-target['cx'])/target['fl_x'],(yy+.5-target['cy'])/target['fl_y'],torch.ones_like(xx)),1).float()@pose[:3,:3].T
            direction=torch.nn.functional.normalize(direction,dim=-1)
            rgb[start:start+len(xx)]=model(origin.expand(len(xx),3),direction).cpu().numpy()
    output=r/'fallback4k';output.mkdir(exist_ok=True);path=output/(target['physical_camera']+'.npz')
    np.savez_compressed(path,rgb_display=rgb.reshape(1920,3840,3),roi=np.ones((1920,3840),bool))
    atomic_json(output/'complete.json',dict(records=[dict(camera=target['physical_camera'],sha256=sha(path))],
        camera_manifest_sha256=sha(r/'target4k.json'),checkpoint_sha256=sha(checkpoint),
        uses_validation_rgb=False,uses_test_rgb=False,artifact_kind='common frozen train-only background fallback'))
    atomic_json(r/'benchmark_prepared.json',dict(source_cameras=[f['physical_camera'] for f in sources],heldout=target['physical_camera'],
        native_rgb_resolution=[6144,3072],field_training_resolution=[1920,960],evaluation_resolution=[3840,1920],
        reference_used_for_metrics_only=True,no_crop_no_upscaling=True))


if __name__=='__main__':main()
