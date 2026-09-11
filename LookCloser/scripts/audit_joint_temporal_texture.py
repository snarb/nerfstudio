"""Verify frozen calibration, immutable inputs, portable meshes, and face-only reports."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from joint_temporal_texture import ROOT,SOURCE,CALIBRATION,HELD_CAMERAS,read,sha,atomic_json,load_parameters,validate_cache,exr,display


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(root,frames):
    request=read(root/'fit_request.json');profiles=read(root/'camera_profiles.json');exposure=read(root/'exposure.json')
    fit_frames=request['fit_frames'];held=request['held_frames']
    require(not set(fit_frames)&set(held),'Temporal holdout leakage')
    names=profiles['physical_cameras']
    require(len(names)==len(set(names))==62 and not set(names)&HELD_CAMERAS,'Camera inventory/leakage')
    require(profiles['fixed_across_time'] and exposure['exposure_mode']=='fixed' and not exposure['time_varying_gain'],'Exposure must be temporally fixed')
    require(not request['uses_eval_rgb'] and request['geometry_fixed'] and request['poses_fixed'],'Fit contract changed')
    load_parameters(root,fit_frames[0],device='cpu')
    caches=fit_frames+held
    for frame in caches:
        dest=root/'cache'/frame;validate_cache(dest);inputs=read(dest/'request.json')
        require(sha(dest/'request.json')==request['input_hashes'][frame],'Fit cache request changed')
        require(sha(CALIBRATION)==inputs['calibration_sha256'],'Calibration template changed')
        require(sha(SOURCE/frame/'transforms.json')==inputs['source_transforms_sha256'],'Source transforms changed')
        require(sha(inputs['mesh'])==inputs['mesh_sha256'],'Source geometry changed')
        paths=[r['file_path'] for r in inputs['cameras']]
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual=list(pool.map(sha,paths))
        require(actual==[inputs['source_hashes'][r['physical_camera']] for r in inputs['cameras']],'Source EXR changed')
        print(f'audit cache={frame} sources=62 verified',flush=True)
    rows=[]
    for frame in frames:
        out=root/'frames'/frame;asset=read(out/'asset_manifest.json');bake=read(out/'bake_result.json')
        require(asset['geometry_changed'] is False and bake['geometry_changed'] is False,'Unrecorded geometry edit')
        require(bake['parameters_sha256']==sha(root/'parameters.npz') and bake['exposure_sha256']==sha(root/'exposure.json'),'Stale bake calibration')
        for filename,digest in bake['scripts_sha256'].items():
            require(sha(root/'config'/filename)==digest,f'Bake script snapshot mismatch: {filename}')
        for key in ['glb','obj_zip']:
            require(sha(asset[key])==asset[key+'_sha256'],f'{key} hash mismatch')
        mesh=next(iter(trimesh.load(asset['glb'],force='scene',process=False).geometry.values()))
        require(len(mesh.faces)==asset['triangles']>0 and np.isfinite(mesh.vertices).all(),'Invalid exported geometry')
        atlas=np.load(out/'atlas_geometry.npz')
        require(np.array_equal(atlas['mapping'][atlas['indices']],atlas['triangles']),'UV atlas changed topology')
        texture=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
        require(np.array_equal(texture,np.asarray(Image.open(out/'texture_joint.png'))),'Embedded texture differs')
        require(np.allclose(mesh.visual.uv,atlas['uv'],atol=1e-6),'Exported UVs differ')
        for variant,digest in bake['texture_sha256'].items():require(sha(out/f'texture_{variant}.png')==digest,'Texture hash mismatch')
        source=read(SOURCE/frame/'transforms.json')
        for view in read(out/'review'/'manifest.json')['views']:
            path=out/'review'/view['camera'];require(sha(path/'joint.png')==view['joint_sha256'],'Render hash mismatch')
            require(sha(path/'gt.png')==view['gt_sha256'],'GT hash mismatch')
            row=next(r for r in source['frames'] if r['physical_camera']==view['camera'])
            gt=np.rint(display(exr(SOURCE/frame/row['file_path']),exposure['fixed_exposure_gain'])*255).clip(0,255).astype(np.uint8)
            require(np.array_equal(np.asarray(Image.open(path/'gt.png')),gt),'GT is not exact frozen-exposure source replay')
            for variant in ['fixed_exposure','camera_profile','joint']:
                rgb=np.asarray(Image.open(path/f'{variant}.png'))
                require(rgb.shape==(1080,1920,3),'Non-native render')
        metrics=read(out/'review'/'F004_B005_1210O9'/'face_metrics.json')
        require(metrics['protocol']['no_full_frame_metrics'] and not metrics['protocol']['candidate_surface_mask'],'Forbidden metric protocol')
        require(metrics['roi_sha256']==sha(root/'config'/f'face_roi_{frame}.json'),'ROI changed after scoring')
        for variant,values in metrics['variants'].items():
            require(all(np.isfinite(values[k]) for k in ['face_psnr','face_ssim','face_lpips']),'Non-finite face metric')
            require(sha(out/'review'/'F004_B005_1210O9'/f'{variant}.png')==values['prediction_sha256'],'Stale face metrics')
        verdict=read(out/'visual_review.json')
        require(verdict['status'] in ['pass','fail'],'Missing visual verdict')
        rows.append({'frame':frame,'visual_status':verdict['status'],'triangles':asset['triangles'],
                     'glb_sha256':asset['glb_sha256'],'face_metrics':metrics['variants']})
        print(f'audit frame={frame} portable_mesh=verified visual={verdict["status"]}',flush=True)
    # Retained outputs only. Large input caches and explicitly rejected prototypes
    # stay outside this delivery inventory; their own manifests remain available.
    paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['cache','diagnostics'] and p.name!='audit.json']
    atomic_json(root/'audit.json',{'status':'verified_delivery_with_known_visual_defects' if any(r['visual_status']=='fail' for r in rows) else 'verified',
        'frames':rows,'source_frames_verified':caches,'frozen_exposure_gain':exposure['fixed_exposure_gain'],
        'native_uv_shift_bound_per_axis_px':2.,'native_uv_shift_norm_bound_px':float(2*np.sqrt(2)),
        'raw_tsdf_volume_serialized':False,'geometry_is_extracted_mesh':True,
        'retained_hashes':{str(p.relative_to(root)):sha(p) for p in sorted(paths)}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--frames',nargs='+',default=['000973','001059']);a=p.parse_args();audit(a.output,a.frames)


if __name__=='__main__':main()
