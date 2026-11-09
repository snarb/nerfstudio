#!/usr/bin/env python3
"""Frozen native extraction of separate head depth/opacity and background radiance.

A checkpoint's oriented head scope defines the density compartment. Original
full-ray weights/transmittance are retained; head depth is never alpha-averaged
with background depth. RGB remains the learned field's low-frequency backdrop.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch,yaml,nerfacc
from dec5_hair import ray_to_z,packed_layer_depth_quantile
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','training','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--source-semantics',type=Path);p.add_argument('--camera-manifest',type=Path)
    p.add_argument('--include-layer-rgb',action='store_true');a=p.parse_args()
    if bool(a.parent)==bool(a.source_semantics):p.error('Choose frozen target ROI or train source semantics')
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.field_components.field_heads import FieldHeadNames
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(20*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    request=json.loads((a.training/'request.json').read_text());dataset_sha=sha(a.data/'transforms.json')
    if request['data_sha256']!=dataset_sha:raise ValueError('Field/dataset mismatch')
    configs=list(a.training.rglob('config.yml'))
    if len(configs)!=1:raise ValueError('One frozen configuration required')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
    matrix=model.field.density_geometry_roi_transform
    if matrix is None:raise ValueError('A checkpoint with an explicit head scope is required')
    payload=json.loads((a.data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
    train=pipe.datamanager.train_dataset;raw=np.asarray([by[Path(p).name]['transform_matrix'] for p in train.image_filenames]);normal=train.cameras.camera_to_worlds.cpu().numpy()
    R=normal[0,:3,:3]@raw[0,:3,:3].T;s=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);t=normal[0,:3,3]-s*R@raw[0,:3,3]
    alignment=float(np.linalg.norm(s*raw[:,:3,3]@R.T+t-normal[:,:3,3],axis=-1).max())
    if alignment>1e-4:raise ValueError('Camera alignment mismatch')
    frames=json.loads(a.camera_manifest.read_text())['frames'] if a.camera_manifest else [f for f in payload['frames'] if f['file_path'] in payload['train_filenames' if a.source_semantics else 'val_filenames']]
    trace={}
    def capture(module,args,outputs):trace['rgb']=outputs[FieldHeadNames.RGB]
    hook=model.field.register_forward_hook(capture) if a.include_layer_rgb else None
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    try:
        for frame in frames:
            name=frame['physical_camera']
            if a.source_semantics:
                if frame['file_path'] not in payload['train_filenames']:raise ValueError('Source camera leak')
                sem=np.load(a.source_semantics/(name+'.npz'))['probability'].astype(np.float32)/65535
                roi=cv2.dilate(((sem[...,0]>.05)|(sem[...,1]>.05)).astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
            else:roi=np.load(a.parent/(name+'.npz'))['roi']
            yy,xx=np.nonzero(roi);uv=np.stack((xx+.5,yy+.5),-1);pose=np.asarray(frame['transform_matrix'])
            c2w=np.c_[R@pose[:3,:3],s*R@pose[:3,3]+t]
            camera=Cameras(camera_to_worlds=torch.as_tensor(c2w,dtype=torch.float32)[None],fx=frame['fl_x'],fy=frame['fl_y'],cx=frame['cx'],cy=frame['cy'],width=frame['w'],height=frame['h']).to('cuda')
            median=np.zeros(roi.shape,np.float32);opacity=np.zeros_like(median);global_depth=np.zeros_like(median);expected=np.zeros_like(median);total_opacity=np.zeros_like(median)
            backdrop=np.zeros(roi.shape+(3,),np.float32) if a.include_layer_rgb else None
            identity_error=0.
            for begin in range(0,len(uv),256):
                pixels=uv[begin:begin+256];trace.clear();rays=camera.generate_rays(0,coords=torch.as_tensor(pixels[:,::-1].copy(),device='cuda',dtype=torch.float32),keep_shape=False)
                with torch.no_grad():result=model(rays)
                ids=result['packed_ray_indices'].long();starts=result['packed_spacing_starts'].reshape(-1);ends=result['packed_spacing_ends'].reshape(-1)
                weights=result['packed_weights'].reshape(-1).float()
                middle=rays.nears.reshape(-1)[ids]+((starts+ends)/2)*(rays.fars-rays.nears).reshape(-1)[ids]
                position=rays.origins[ids]+rays.directions[ids]*middle[:,None]
                unit=(position.float()[:,None,:]*matrix[None,:,:3]).sum(-1)+matrix[:,3]
                inside=((unit>=0)&(unit<=1)).all(-1)
                values=[result[k].float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
                distance,mass=packed_layer_depth_quantile(*values,ids.cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy(),inside.cpu().numpy(),minimum_mass=.001)
                z=ray_to_z(distance/s,pixels,frame);valid=(distance>0)&np.isfinite(z)&(z>0)
                y,x=yy[begin:begin+len(pixels)],xx[begin:begin+len(pixels)]
                median[y,x]=np.where(valid,z,0);opacity[y,x]=np.clip(mass,0,1)
                total_opacity[y,x]=result['accumulation'].reshape(-1).float().cpu().numpy()
                moment=(nerfacc.accumulate_along_rays(weights*inside,values=middle[:,None],ray_indices=ids,n_rays=len(pixels)).reshape(-1).cpu().numpy()
                        if len(ids) else np.zeros(len(pixels),np.float32))
                mean=np.divide(moment,mass,out=np.zeros_like(moment),where=mass>=.001)
                expected[y,x]=ray_to_z(mean/s,pixels,frame)
                global_depth[y,x]=ray_to_z(result['depth'].reshape(-1).float().cpu().numpy()/s,pixels,frame)
                if a.include_layer_rgb:
                    if len(ids):
                        rgb=trace['rgb'].reshape(-1,3).float()
                        if len(rgb)!=len(ids):raise ValueError('Packed RGB binding mismatch')
                        full=nerfacc.accumulate_along_rays(weights,values=rgb,ray_indices=ids,n_rays=len(pixels))
                        head=nerfacc.accumulate_along_rays(weights*inside,values=rgb,ray_indices=ids,n_rays=len(pixels))
                        accum=result['accumulation'].reshape(-1).float();remainder=result['rgb'].reshape(-1,3).float()-full
                        if bool((remainder < -2e-4).any()) or bool((remainder>(1-accum)[:,None]+2e-4).any()):raise ValueError('Native RGB/volume binding identity failed')
                    else:head=torch.zeros((len(pixels),3),device='cuda')
                    alpha=torch.as_tensor(np.clip(mass,0,1),device='cuda',dtype=torch.float32)
                    residual=result['rgb'].reshape(-1,3).float()-head
                    background=residual/(1-alpha[:,None]).clamp_min(1e-6)
                    soft=alpha<.99
                    if bool(soft.any()):identity_error=max(identity_error,float((head[soft]+background[soft]*(1-alpha[soft,None])-result['rgb'].reshape(-1,3)[soft]).abs().max()))
                    backdrop[y,x]=background.clamp(0,1).cpu().numpy()
            arrays=dict(median=median,expected=expected,global_expected=global_depth,opacity=opacity,total_opacity=total_opacity,roi=roi)
            if backdrop is not None:arrays['background_display']=backdrop
            path=a.output/(name+'.npz');np.savez_compressed(path,**arrays)
            records.append(dict(camera=name,sha256=sha(path),roi_pixels=len(uv),valid_depth_pixels=int((median>0).sum()),background_recomposition_max_error=identity_error))
            atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(frames),camera=name))
    finally:
        if hook is not None:hook.remove()
    if fingerprint(model)!=before:raise ValueError('Layer extraction mutated field')
    atomic_json(a.output/'complete.json',dict(records=records,checkpoint_sha256=sha(a.checkpoint),checkpoint_path=str(a.checkpoint),step=state['step'],
        density_layer='oriented head compartment',head_roi_transform=matrix.cpu().tolist(),head_minimum_contribution=.001,
        background='conditional radiance outside head, including actual renderer residual' if a.include_layer_rgb else None,
        depth_convention='OpenCV camera-z',units='calibration units',expected_key='foreground conditional native sample-midpoint mean, never mixed with background',
        native_ray_sampling_mode=model.config.ray_sampling_mode,native_adaptive_enabled=model.config.enable_adaptive_ray_marching,
        frozen_state_unchanged=True,dataset_sha256=dataset_sha,normalized_per_calibration_unit=float(s),pose_alignment_max_error=alignment,
        source_visibility_only=bool(a.source_semantics),source_inventory='explicit train subset' if a.source_semantics and a.camera_manifest else 'all train cameras' if a.source_semantics else None,
        camera_manifest_sha256=sha(a.camera_manifest) if a.camera_manifest else None,
        target_roi_parent_receipt_sha256=sha(a.parent/'complete.json') if a.parent else None,
        opacity='visible head volume-weight contribution, not calibrated ground-truth alpha',uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
