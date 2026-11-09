#!/usr/bin/env python3
"""Composite GT foreground detail with native LookCloser head coverage/background.

Uses a matching frozen field for source/target head depths. Background is the
conditional radiance outside the head, evaluated at the target pose. Outside
the fixed parent ROI and at unsupported foreground pixels, retain the parent.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
from study_dec5_hair_geometry import Scene,Renderer,cv_pose,display
from add_dec5_neural_fallback import inverse_display
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','parent','target-layers','source-layers','matting','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--camera-manifest',type=Path);p.add_argument('--source-reference-camera',type=Path)
    p.add_argument('--geometry-only',action='store_true',help='Causal control: conditional head depth and visibility, raw GT RGB, no opacity compositing')
    p.add_argument('--opaque-only',action='store_true',help='Geometry confidence control: require head opacity >= .95 in target and source')
    p.add_argument('--source-alpha',action='store_true',help='Transfer source premultiplied RGB and source alpha, rather than treating neural volume weight as alpha')
    a=p.parse_args()
    if a.opaque_only and not a.geometry_only:p.error('--opaque-only requires --geometry-only')
    if a.source_alpha and a.geometry_only:p.error('--source-alpha is a compositing control')
    source=json.loads((a.source_layers/'complete.json').read_text());target=json.loads((a.target_layers/'complete.json').read_text())
    if (source['checkpoint_sha256']!=target['checkpoint_sha256'] or source['dataset_sha256']!=target['dataset_sha256']
            or source.get('density_layer')!='oriented head compartment' or not source['source_visibility_only']
            or not source['frozen_state_unchanged'] or not target['frozen_state_unchanged']
            or source.get('uses_validation_rgb') is not False or source.get('uses_test_rgb') is not False
            or target.get('uses_validation_rgb') is not False or target.get('uses_test_rgb') is not False):raise ValueError('One frozen train-only head field is required')
    np.testing.assert_array_equal(source['head_roi_transform'],target['head_roi_transform'])
    if target.get('camera_manifest_sha256')!=(sha(a.camera_manifest) if a.camera_manifest else None):
        raise ValueError('Target head layers belong to different camera poses')
    scene=Scene(a.root,load_rgb=True,mvs=a.root/'pm1920_native_v2',mesh=a.root/'center_tsdf_base/mesh.ply')
    if source['dataset_sha256']!=sha(scene.data/'transforms.json'):raise ValueError('Dataset/calibration mismatch')
    frames=json.loads(a.camera_manifest.read_text())['frames'] if a.camera_manifest else scene.validation
    if a.source_reference_camera:
        ref=json.loads(a.source_reference_camera.read_text());scene.source_reference_center=cv_pose(ref)[:3,3]
    source_by={r['camera']:r for r in source['records']};train_names={f['physical_camera'] for f in scene.frames}
    if not set(source_by)<=train_names:raise ValueError('Source camera leak')
    scene.visibility_depth=[]
    for f,old in zip(scene.frames,scene.observed_depth):
        name=f['physical_camera']
        if name in source_by:
            path=a.source_layers/(name+'.npz')
            if sha(path)!=source_by[name]['sha256']:raise ValueError('Changed source layer')
            data=np.load(path);proposal=data['median']
            if a.opaque_only:proposal=np.where(data['opacity']>=.95,proposal,0)
            value=np.where(data['roi'],proposal,old).astype(np.float32)
        else:value=old
        scene.visibility_depth.append(value)
    renderer=Renderer(scene);renderer.load_mattes(a.matting)
    parent_receipt=json.loads((a.parent/'complete.json').read_text());parents={r['camera']:r for r in parent_receipt['records']};targets={r['camera']:r for r in target['records']}
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    for camera in frames:
        name=camera['physical_camera'];path=a.parent/(name+'.npz');tp=a.target_layers/(name+'.npz')
        if sha(path)!=parents[name]['npz_sha256'] or sha(tp)!=targets[name]['sha256']:raise ValueError('Changed parent/target layer')
        selected={scene.frames[i]['physical_camera'] for i in renderer.source_order(camera)}
        if not selected<=set(source_by):raise ValueError('Source layer subset does not cover requested camera source order')
        with np.load(path) as values:arrays={k:values[k] for k in values.files}
        layer=np.load(tp);roi=arrays['roi'];depth=layer['median'];opacity=layer['opacity'].clip(0,1)
        np.testing.assert_array_equal(layer['roi'],roi)
        if 'background_display' not in layer:raise ValueError('Export target layer RGB')
        foreground=renderer.warp_layer(camera,torch.as_tensor(depth,device='cuda'),foreground=True,decontaminate=True,premultiply=a.source_alpha)
        foreground={k:v.cpu().numpy() for k,v in foreground.items()}
        geometry={k:v.cpu().numpy() for k,v in renderer.warp(camera,torch.as_tensor(depth,device='cuda')).items()}
        alpha=np.where(opacity>=.995,1,np.where(opacity<=.001,0,opacity)).astype(np.float32)
        if a.source_alpha:
            alpha=foreground['alpha'].clip(0,1)
            alpha=np.where(opacity<=.001,0,alpha)
        background_mass=np.maximum(layer['total_opacity']-opacity,0)
        bg_confident=(background_mass>=.01)&(background_mass/np.maximum(1-opacity,1e-6)>=.9)
        bg_confident&=np.isfinite(layer['background_display']).all(-1)&(layer['background_display'].max(-1)<.995)
        fg_confident=(depth>0)&(foreground['source_id']>=0)&(foreground['support']>=2)
        eligible=roi&((alpha==0)&bg_confident | (alpha>0)&fg_confident&((alpha==1)|bg_confident))
        bg=inverse_display(layer['background_display'].astype(np.float32))
        # Opaque pixels do not use a numerically unsupported conditional BG.
        bg[alpha==1]=0
        premultiplied=np.where(alpha[...,None]>0,foreground['rgb'],0) if a.source_alpha else foreground['rgb']*alpha[...,None]
        mixed=premultiplied+bg*(1-alpha[...,None])
        if a.geometry_only:
            eligible=roi&(depth>0)&(geometry['source_id']>=0)&(geometry['support']>=2)
            if a.opaque_only:eligible &= opacity>=.95
            foreground=geometry;alpha=np.ones_like(alpha);mixed=geometry['rgb'];premultiplied=geometry['rgb']
        protected_skin=arrays['skin_probability']>=.5
        eligible &= ~protected_skin
        original=arrays['rgb'].copy();arrays['rgb'][eligible]=mixed[eligible]
        foreground_pixels=eligible&(alpha>0);background_pixels=eligible&(alpha==0)
        arrays['front_depth'][foreground_pixels]=depth[foreground_pixels]
        arrays['front_depth'][background_pixels]=layer['global_expected'][background_pixels]
        arrays['source_id'][foreground_pixels]=foreground['source_id'][foreground_pixels];arrays['source_id'][background_pixels]=-1
        arrays['support'][foreground_pixels]=foreground['support'][foreground_pixels];arrays['support'][background_pixels]=0
        arrays['reprojection_error'][foreground_pixels]=geometry['reprojection_error'][foreground_pixels];arrays['reprojection_error'][background_pixels]=20
        for key in ['hair_probability','skin_probability']:arrays[key][eligible]=geometry[key][eligible]
        hair=arrays.get('hair_coverage',arrays['hair_probability']).copy();hair[eligible]=geometry['hair_probability'][eligible]*alpha[eligible];arrays['hair_coverage']=hair
        arrays['head_layer_gate']=eligible;arrays['head_alpha']=alpha;arrays['neural_background']=eligible&(alpha<1)
        arrays['foreground_premultiplied']=premultiplied
        if 'neural_fallback' in arrays:arrays['neural_fallback'][eligible]=arrays['source_id'][eligible]<0
        np.testing.assert_array_equal(arrays['rgb'][~roi],original[~roi])
        np.testing.assert_array_equal(arrays['rgb'][protected_skin],original[protected_skin])
        image=np.rint(display(arrays['rgb'])*255).astype(np.uint8)
        np.savez_compressed(a.output/(name+'.npz'),**arrays);cv2.imwrite(str(a.output/(name+'.png')),image[...,::-1])
        preview=cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA);cv2.imwrite(str(a.output/(name+'_review.jpg')),preview[...,::-1])
        records.append(dict(camera=name,npz_sha256=sha(a.output/(name+'.npz')),png_sha256=sha(a.output/(name+'.png')),
            composite_pixels=int(eligible.sum()),foreground_pixels=int(foreground_pixels.sum()),background_pixels=int(background_pixels.sum()),
            unsupported_parent_retained_pixels=int((roi&~eligible).sum())))
        atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(frames)))
    receipt=dict(parent_receipt);receipt.update(records=records,parent_sha256=sha(a.parent/'complete.json'),
        source_layer_receipt_sha256=sha(a.source_layers/'complete.json'),target_layer_receipt_sha256=sha(a.target_layers/'complete.json'),
        checkpoint_sha256=source['checkpoint_sha256'],method=('conditional head geometry + raw GT RGB' if a.geometry_only else
            ('GT source premultiplied RGBA + conditional neural background' if a.source_alpha else 'GT decontaminated foreground + native head coverage + conditional neural background')),
        geometry_only=a.geometry_only,
        source_premultiplied_rgba=a.source_alpha,
        minimum_head_opacity=.95 if a.opaque_only else None,
        original_rgb_outside_roi_exact=True,low_confidence_fallback='exact parent',alpha_numeric_endpoints=None if a.source_alpha else [.001,.995],
        original_rgb_on_confident_parent_skin_exact=True,skin_protection='frozen parent source-voted skin probability >= .5; no target RGB mask',
        alpha_source='warped train-camera person matte on hair-valid source pixels' if a.source_alpha else 'native visible head contribution',
        background_conditional_coverage_minimum=.9,
        foreground_sources=8,background_source='same frozen field outside its oriented head compartment; no generative painter',
        uses_eval_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__))
    atomic_json(a.output/'complete.json',receipt)

if __name__=='__main__':main()
