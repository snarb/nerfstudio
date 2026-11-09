#!/usr/bin/env python3
"""Common GT transfer for the HD/native head-resolution control; no MVS needed.

Visibility uses calibrated HD ray rasters from the same field as target depth.
RGB always comes from identical native train crops. The HD field's native RGB
is the fixed fallback for both arms. Outside the fixed head domain is neutral;
this is a 4K head-ROI benchmark, not a whole-frame reconstruction claim.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import cv2,numpy as np,torch
from dec5_hair import cv_pose,VALIDATION,TEST
from study_dec5_hair_geometry import Renderer,display
from add_dec5_neural_fallback import inverse_display
from run_dec5_hair_campaign import atomic_json,sha


def validate_source_split(data,expected=59):
    train=[f for f in data['frames'] if f['file_path'] in data['train_filenames']]
    if len(train)!=expected:raise ValueError('Unexpected RGB training split')
    if expected==59:
        if any(f['physical_camera'] in VALIDATION+TEST for f in train):raise ValueError('Native RGB must be train59 only')
        return train
    if expected!=64 or not 901<=int(data.get('frame_id',-1))<=950:
        raise ValueError('64+1 RGB authorization applies only to the new901..950 sequence')
    heldout=[f for f in data['frames'] if f['file_path'] in data['val_filenames']]
    names={f['physical_camera'] for f in train}
    if len(data['frames'])!=65 or len(names)!=64 or len(heldout)!=1 or heldout[0]['physical_camera'] in names:
        raise ValueError('64+1 RGB split leaks held-out camera or has duplicate IDs')
    return train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source-rgb-data','source-cameras','source-depth','semantics','target-depth','fallback','cameras','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-reference-camera',type=Path)
    p.add_argument('--direct-source-training',type=Path);p.add_argument('--direct-source-checkpoint',type=Path);p.add_argument('--visibility-parent',type=Path)
    p.add_argument('--depth-key',default='median',help='Matching source/target native depth statistic; default unchanged');p.add_argument('--camera');p.add_argument('--max-direct-queries',type=int,default=2000000)
    p.add_argument('--expected-train-count',type=int,choices=[59,64],default=59)
    a=p.parse_args();torch.set_num_threads(2)
    if any([a.direct_source_training,a.direct_source_checkpoint,a.visibility_parent]) and not all([a.direct_source_training,a.direct_source_checkpoint,a.visibility_parent]):p.error('Direct visibility needs field training/checkpoint and frozen parent')
    torch.cuda.set_per_process_memory_fraction(18*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    sources=json.load(open(a.source_depth/'complete.json'));targets=json.load(open(a.target_depth/'complete.json'));fallback=json.load(open(a.fallback/'complete.json'))
    if sources['checkpoint_sha256']!=targets['checkpoint_sha256'] or sources['dataset_sha256']!=targets['dataset_sha256']:raise ValueError('Matching target/source field required')
    if sources.get('sigma_surface')!=targets.get('sigma_surface'):raise ValueError('Source/target density surfaces differ')
    if sources.get('empty_space_gate')!=targets.get('empty_space_gate'):raise ValueError('Source/target density gates differ')
    if not sources['frozen_state_unchanged'] or not targets['frozen_state_unchanged'] or not sources['source_visibility_only']:raise ValueError('Frozen source maps required')
    if sources['uses_validation_rgb'] or sources['uses_test_rgb'] or targets['uses_validation_rgb'] or targets['uses_test_rgb']:raise ValueError('No held-out RGB inputs permitted')
    if sources['camera_manifest_sha256']!=sha(a.source_cameras) or targets['camera_manifest_sha256']!=sha(a.cameras) or fallback['camera_manifest_sha256']!=sha(a.cameras):raise ValueError('Camera raster mismatch')
    d=json.load(open(a.source_rgb_data/'transforms.json'));train=validate_source_split(d,a.expected_train_count);by={f['physical_camera']:f for f in train}
    frames=json.load(open(a.source_cameras))['frames'];source_by={x['camera']:x for x in sources['records']}
    if set(source_by)!={f['physical_camera'] for f in frames} or not set(source_by)<=set(by):raise ValueError('Source inventory mismatch')
    renderer=object.__new__(Renderer);scene=SimpleNamespace(cameras=frames,frames=frames,centers=np.stack([cv_pose(f)[:3,3] for f in frames]),allowed_visibility_sources=set(source_by));renderer.scene=scene;renderer.visibility_depth_sampling='bilinear';renderer.depth=[];renderer.semantic=[];renderer.rgb=[];rgb_records=[]
    if a.source_reference_camera:scene.source_reference_center=cv_pose(json.load(open(a.source_reference_camera)))[:3,3]
    for f in frames:
        n=f['physical_camera'];rgb_frame=by[n];path=a.source_depth/(n+'.npz')
        if sha(path)!=source_by[n]['sha256']:raise ValueError('Changed source depth')
        np.testing.assert_allclose(f['transform_matrix'],rgb_frame['transform_matrix'],atol=1e-10,rtol=0)
        sx,sy=rgb_frame['w']/f['w'],rgb_frame['h']/f['h']
        np.testing.assert_allclose([f['fl_x']*sx,f['fl_y']*sy,f['cx']*sx,f['cy']*sy],
            [rgb_frame[k] for k in ['fl_x','fl_y','cx','cy']],atol=1e-5,rtol=0)
        depth=np.load(path)[a.depth_key];sem=np.load(a.semantics/(n+'.npz'))['probability'].astype(np.float32)/65535
        if depth.shape!=(f['h'],f['w']) or sem.shape[:2]!=depth.shape:raise ValueError('Source visibility/semantic raster mismatch')
        image_path=a.source_rgb_data/rgb_frame['file_path'];image=cv2.imread(str(image_path))[...,::-1].astype(np.float32)/255
        if image.shape[:2]!=(rgb_frame['h'],rgb_frame['w']):raise ValueError('Native RGB raster mismatch')
        renderer.depth.append(torch.as_tensor(depth,device='cuda')[None]);renderer.semantic.append(torch.as_tensor(sem,device='cuda').permute(2,0,1));renderer.rgb.append(torch.as_tensor(inverse_display(image),device='cuda').permute(2,0,1))
        rgb_records.append(dict(camera=n,sha256=sha(image_path)))
    direct=None;parent_by={}
    if a.direct_source_training:
        from dec5_direct_visibility import DirectHairVisibility
        direct=DirectHairVisibility(a.source_rgb_data,a.direct_source_training,a.direct_source_checkpoint,frames,targets['checkpoint_sha256'],a.max_direct_queries);renderer.visibility_depth_resampler=direct
        parent_by={r['camera']:r for r in json.load(open(a.visibility_parent/'complete.json'))['records']}
    selected_frames=json.load(open(a.cameras))['frames']
    if a.camera:
        selected_frames=[f for f in selected_frames if f['physical_camera']==a.camera]
        if len(selected_frames)!=1:raise ValueError('Unknown single-camera diagnostic')
    target_by={r['camera']:r for r in targets['records']};fallback_by={r['camera']:r for r in fallback['records']};all_centers=np.stack([cv_pose(f)[:3,3] for f in train]);records=[];a.output.mkdir(parents=True,exist_ok=False)
    for f in selected_frames:
        n=f['physical_camera'];path=a.target_depth/(n+'.npz');fp=a.fallback/(n+'.npz')
        if sha(path)!=target_by[n]['sha256'] or sha(fp)!=fallback_by[n]['sha256']:raise ValueError('Changed target or frozen fallback')
        data=np.load(path);base=np.load(fp);roi=data['roi'];np.testing.assert_array_equal(roi,base['roi'])
        center=getattr(scene,'source_reference_center',cv_pose(f)[:3,3]);expected=[train[i]['physical_camera'] for i in np.argsort(np.linalg.norm(all_centers-center,axis=-1),kind='stable')[:8]];selected=[frames[i]['physical_camera'] for i in renderer.source_order(f)]
        if selected!=expected:raise ValueError('Subset changed nearest-eight camera selection')
        if direct:
            parent_path=a.visibility_parent/(n+'.npz')
            if sha(parent_path)!=parent_by[n]['npz_sha256']:raise ValueError('Changed visibility parent')
            parent=np.load(parent_path);domain=roi&~((parent['skin_probability']>.6)&(parent['hair_probability']<.2));renderer.visibility_resampling_domain=torch.as_tensor(domain,device='cuda')
        depth=data[a.depth_key];result={k:v.cpu().numpy() for k,v in renderer.warp(f,torch.as_tensor(depth,device='cuda')).items()}
        if 'rgb_display' not in base:raise ValueError('Explicit HD-field fallback RGB required')
        valid=roi&(result['source_id']>=0);fallback_rgb=np.full((*roi.shape,3),.5,np.float32);fallback_rgb[roi]=base['rgb_display'][roi]
        image_rgb=inverse_display(fallback_rgb);image_rgb[valid]=result['rgb'][valid]
        if direct and not np.array_equal(image_rgb[~domain],parent['rgb'][~domain]):raise ValueError('Direct visibility changed protected skin/exterior RGB')
        result['rgb']=image_rgb;result.update(front_depth=depth,projected_hair_depth=np.where(result['hair_probability']>.25,depth,0),back_depth=np.zeros_like(depth),roi=roi,hair_coverage=result['hair_probability'],neural_fallback=roi&~valid)
        np.savez_compressed(a.output/(n+'.npz'),**result);image=np.rint(display(image_rgb)*255).astype(np.uint8);cv2.imwrite(str(a.output/(n+'.png')),image[...,::-1]);cv2.imwrite(str(a.output/(n+'_review.jpg')),cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA)[...,::-1])
        records.append(dict(camera=n,npz_sha256=sha(a.output/(n+'.npz')),png_sha256=sha(a.output/(n+'.png')),roi_pixels=int(roi.sum()),gt_supported_pixels=int(valid.sum())))
        atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(selected_frames)))
    receipt=dict(records=records,direct_visibility=direct.receipt() if direct else None,selected_cameras=[f['physical_camera'] for f in selected_frames],visibility_parent_sha256=sha(a.visibility_parent/'complete.json') if direct else None,sigma_surface=targets.get('sigma_surface'),source_reference_camera_sha256=sha(a.source_reference_camera) if a.source_reference_camera else None,empty_space_gate=targets.get('empty_space_gate'),method='native train-JPEG GT transfer; same-field depth visibility; common frozen RGB fallback',depth_key=a.depth_key,artifact_kind='head ROI only; neutral outside; no whole-frame geometry claim',frame=d['frame_id'],checkpoint_sha256=targets['checkpoint_sha256'],target_receipt_sha256=sha(a.target_depth/'complete.json'),source_receipt_sha256=sha(a.source_depth/'complete.json'),source_rgb_dataset_sha256=sha(a.source_rgb_data/'transforms.json'),fallback_receipt_sha256=sha(a.fallback/'complete.json'),source_rgb=rgb_records,source_count=8,uses_validation_rgb=False,uses_test_rgb=False,source_selection='same nearest-eight; first depth-valid; no RGB averaging',depth_convention='OpenCV camera-z in calibration units',source_rgb_representation='8-bit JPEG-derived pseudo-linear via inverse fixed display; not recovered HDR',implementation_sha256=sha(__file__))
    if a.expected_train_count==64:
        receipt.update(expected_train_count=64,heldout_cameras=[f['physical_camera'] for f in d['frames'] if f['file_path'] in d['val_filenames']],
            method='native train-PNG GT transfer; same-field depth visibility; common frozen background',
            artifact_kind='bounded foreground field in the explicit target ROI; separate background RGB, no background-depth claim',
            source_rgb_representation='native8-bit display PNG from Log3G10/RWG TIFF; inverse fixed display is pseudo-linear, not recovered HDR')
    atomic_json(a.output/'complete.json',receipt)

if __name__=='__main__':main()
