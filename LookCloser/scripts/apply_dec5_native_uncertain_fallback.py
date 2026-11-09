#!/usr/bin/env python3
"""Use the same frozen LookCloser field where head GT transfer is unsupported.

This changes RGB fallback and honest GT-source provenance only. Geometry and
confident skin are retained. Native RGB must agree with saved layer opacity;
no held-out RGB or semantic reference is consumed.
"""
from pathlib import Path
import argparse,json
import numpy as np,torch,yaml,cv2
from nerfstudio.cameras.cameras import Cameras
from export_dec5_hair_volume import fingerprint
from add_dec5_neural_fallback import inverse_display
from study_dec5_hair_geometry import display
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','training','checkpoint','parent','layers','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--camera-manifest',type=Path)
    p.add_argument('--boundary-only',action='store_true',help='Protect opaque head volume AND opaque source alpha; fallback only when both are below .95')
    a=p.parse_args()
    parent=json.load(open(a.parent/'complete.json'));layer_receipt=json.load(open(a.layers/'complete.json'));checkpoint_sha=sha(a.checkpoint)
    if parent['checkpoint_sha256']!=checkpoint_sha or layer_receipt['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('Fallback must use the identical frozen head field')
    dataset_sha=sha(a.data/'transforms.json');request=json.load(open(a.training/'request.json'))
    if request['data_sha256']!=dataset_sha or layer_receipt['dataset_sha256']!=dataset_sha:raise ValueError('Dataset mismatch')
    if layer_receipt.get('camera_manifest_sha256')!=(sha(a.camera_manifest) if a.camera_manifest else None):raise ValueError('Target pose mismatch')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
    if bool(getattr(model.field,'use_view_directions',True))!=bool(request.get('use_view_directions',True)):raise ValueError('Color direction mode mismatch')
    payload=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in payload['frames']};train=pipe.datamanager.train_dataset
    raw=np.asarray([by[Path(p).name]['transform_matrix'] for p in train.image_filenames]);normal=train.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);t=normal[0,:3,3]-scale*R@raw[0,:3,3]
    if np.linalg.norm(scale*raw[:,:3,3]@R.T+t-normal[:,:3,3],axis=-1).max()>1e-4:raise ValueError('Camera similarity mismatch')
    frames=json.load(open(a.camera_manifest))['frames'] if a.camera_manifest else [f for f in payload['frames'] if f['file_path'] in payload['val_filenames']]
    records=[];a.output.mkdir(parents=True,exist_ok=False);parents={r['camera']:r for r in parent['records']};layers={r['camera']:r for r in layer_receipt['records']}
    for f in frames:
        n=f['physical_camera'];pp=a.parent/(n+'.npz');lp=a.layers/(n+'.npz')
        if sha(pp)!=parents[n]['npz_sha256'] or sha(lp)!=layers[n]['sha256']:raise ValueError('Changed parent or layer arrays')
        with np.load(pp) as v:arrays={k:v[k] for k in v.files}
        layer=np.load(lp);skin=arrays['skin_probability']>=.5
        domain=arrays['roi']&~arrays['head_layer_gate']&~skin&(layer['total_opacity']>=.95)
        domain &= (layer['opacity']>.01)|(arrays['hair_probability']>.1)
        opaque_head=(layer['opacity']>=.95)|(arrays['head_alpha']>=.95)
        if a.boundary_only:domain &= ~opaque_head
        y,x=np.nonzero(domain);pose=np.asarray(f['transform_matrix']);c2w=np.c_[R@pose[:3,:3],scale*R@pose[:3,3]+t]
        camera=Cameras(camera_to_worlds=torch.as_tensor(c2w,dtype=torch.float32)[None],fx=f['fl_x'],fy=f['fl_y'],cx=f['cx'],cy=f['cy'],width=f['w'],height=f['h']).to('cuda')
        native=np.zeros((len(y),3),np.float32);opacity_error=0.
        for begin in range(0,len(y),256):
            coords=np.stack((y[begin:begin+256]+.5,x[begin:begin+256]+.5),-1)
            rays=camera.generate_rays(0,coords=torch.as_tensor(coords,device='cuda',dtype=torch.float32),keep_shape=False)
            with torch.no_grad():out=model(rays)
            rgb=out['rgb'].reshape(-1,3).float().cpu().numpy();accum=out['accumulation'].reshape(-1).float().cpu().numpy()
            expected=layer['total_opacity'][y[begin:begin+len(coords)],x[begin:begin+len(coords)]]
            opacity_error=max(opacity_error,float(np.abs(accum-expected).max()))
            if opacity_error>2e-4:raise ValueError('Native field differs from saved layer opacity')
            native[begin:begin+len(coords)]=rgb
        valid=np.isfinite(native).all(-1)&(native>=0).all(-1)&(native.max(-1)<.995)
        yy,xx=y[valid],x[valid];gate=np.zeros_like(domain);gate[yy,xx]=True;original=arrays['rgb'].copy();lost=int((arrays['source_id'][gate]>=0).sum())
        arrays['rgb'][yy,xx]=inverse_display(native[valid]);arrays['source_id'][gate]=-1;arrays['support'][gate]=0;arrays['reprojection_error'][gate]=20
        if 'neural_fallback' in arrays:arrays['neural_fallback'][gate]=True
        arrays['native_uncertain_fallback']=gate
        np.testing.assert_array_equal(arrays['rgb'][~gate],original[~gate]);np.testing.assert_array_equal(arrays['rgb'][skin],original[skin])
        if a.boundary_only:np.testing.assert_array_equal(arrays['rgb'][opaque_head],original[opaque_head])
        for key in ['front_depth','projected_hair_depth','back_depth','roi']:np.testing.assert_array_equal(arrays[key],np.load(pp)[key])
        np.savez_compressed(a.output/(n+'.npz'),**arrays);image=np.rint(display(arrays['rgb'])*255).astype(np.uint8);cv2.imwrite(str(a.output/(n+'.png')),image[...,::-1]);cv2.imwrite(str(a.output/(n+'_review.jpg')),cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA)[...,::-1])
        records.append(dict(camera=n,npz_sha256=sha(a.output/(n+'.npz')),png_sha256=sha(a.output/(n+'.png')),native_fallback_pixels=int(gate.sum()),previously_accepted_gt_sources_removed=lost,native_opacity_binding_max_error=opacity_error))
        atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(frames)))
    if fingerprint(model)!=before:raise ValueError('Fallback mutated frozen field')
    receipt=dict(parent);receipt.update(records=records,parent_sha256=sha(a.parent/'complete.json'),method=parent['method']+'; native LookCloser RGB at unsupported head pixels',
        frozen_state_unchanged=True,native_fallback='unsupported head transfer; native volume opacity >= .95; preserve confident skin; source_id=-1 and support=0',
        geometry_unchanged_exact=True,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__))
    receipt['opaque_head_rgb_unchanged_exact']=a.boundary_only
    atomic_json(a.output/'complete.json',receipt)

if __name__=='__main__':main()
