"""Verify an opt-in hard-source bake and build native old/new/GT comparisons.

This is post-hoc only. No held-out image can feed source selection or baking.
Manual visual verdicts remain separate from checksum/metric validity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import atomic_json,read,sha,HELD_CAMERAS


def audit(root,frame,calibration_root):
    out=root/'frames'/frame;old=calibration_root/'frames'/frame
    complete=read(out/'hard_texture_complete.json')
    for name,digest in complete['hashes'].items():
        if sha(out/name)!=digest:raise ValueError(f'Bake checksum mismatch: {name}')
    request=read(out/'hard_texture_request.json')
    for name,digest in request['calibration_hashes'].items():
        if sha(calibration_root/name)!=digest:raise ValueError(f'Calibration changed: {name}')
    if sha(calibration_root/'cache'/frame/'complete.json')!=request['cache_complete_sha256']:
        raise ValueError('Source cache provenance changed')
    before=np.load(old/'atlas_geometry.npz');after=np.load(out/'atlas_geometry.npz')
    for key in ['vertices','triangles','uv','mapping','indices']:
        if not np.array_equal(before[key],after[key]):raise ValueError(f'Geometry/atlas changed: {key}')
    bake=read(out/'bake_result.json');labels=read(out/'source_label_manifest.json')
    cameras=labels['physical_cameras']
    if len(cameras)!=62 or len(set(cameras))!=62 or set(cameras)&HELD_CAMERAS:
        raise ValueError('Invalid train-only camera inventory')
    if bake['geometry_changed'] or bake['uses_eval_rgb'] or labels['averages_rgb']:
        raise ValueError('Hard-source contract violated')
    source=np.load(out/'texture_support.npz')['source_map'].ravel()[after['pixels']]
    preferred=np.load(out/'face_source_labels.npy')[after['face_ids']]
    if not np.isin(source,[*range(62),255]).all():raise ValueError('Invalid camera label')
    fallback=int(((source!=255)&(source!=preferred)).sum())
    if fallback!=bake['fallback_texels']:raise ValueError('Fallback count mismatch')
    comparisons=out/'comparisons';comparisons.mkdir(exist_ok=True)
    for camera in sorted(HELD_CAMERAS):
        target=out/'review'/camera;previous=old/'review'/camera
        if sha(target/'gt.png')!=sha(previous/'gt.png'):raise ValueError('Display GT mismatch')
        images=[Image.open(p).convert('RGB') for p in [target/'gt.png',previous/'joint.png',target/'joint.png']]
        if any(im.size!=(1920,1080) for im in images):raise ValueError('Non-native image dimensions')
        names=['GT','Previous: multi-camera average','New: one camera per mesh patch']
        overview=Image.new('RGB',(432*3,768+26));draw=ImageDraw.Draw(overview)
        for i,(name,im) in enumerate(zip(names,images)):
            overview.paste(im.transpose(Image.Transpose.ROTATE_90).resize((432,768)),(i*432,26))
            draw.text((i*432+4,5),name,fill='white')
        overview.save(comparisons/f'{camera}_overview.png')
        if camera=='F004_B005_1210O9':
            boxes={'ear_hair':(805,315,1040,480),'lipstick_hand':(440,485,760,705),
                   'neck':(500,560,800,800),'face':(700,425,1150,730)} if frame=='000973' else {
                   'ear_hair':(835,360,1060,520),'face':(741,493,1180,790),'neck':(500,560,800,800)}
            for label,box in boxes.items():
                ww,hh=box[2]-box[0],box[3]-box[1]
                panel=Image.new('RGB',(ww*3,hh+26));draw=ImageDraw.Draw(panel)
                for i,(name,im) in enumerate(zip(names,images)):
                    panel.paste(im.crop(box),(i*ww,26));draw.text((i*ww+3,5),name,fill='white')
                panel.save(comparisons/f'{label}_native.png')
    metrics=read(out/'review'/'F004_B005_1210O9'/'face_metrics.json')
    if not metrics['protocol']['no_full_frame_metrics']:raise ValueError('Wrong metric protocol')
    previous_metrics=read(old/'review'/'F004_B005_1210O9'/'face_metrics.json')
    if metrics['roi_sha256']!=previous_metrics['roi_sha256']:raise ValueError('Face ROI changed')
    for result in metrics['variants'].values():
        if not all(np.isfinite(result[name]) for name in ['face_psnr','face_ssim','face_lpips']):
            raise ValueError('Nonfinite face metric')
    verdict_path=out/'visual_review.json'
    verdict=read(verdict_path) if verdict_path.exists() else {'status':'pending'}
    payload={'frame':frame,'status':'checksums_and_contracts_valid','visual_status':verdict['status'],
             'geometry_changed':False,'calibration_changed':False,'train_camera_count':62,
             'uses_eval_rgb_for_prediction':False,'averages_rgb_across_sources':False,
             'fallback_texels':fallback,'fallback_fraction':fallback/len(source),
             'unobserved_texels':int((source==255).sum()),'triangles':len(after['triangles']),
             'previous_face_metrics':previous_metrics['variants']['joint'],
             'new_face_metrics':metrics['variants']['joint'],'temporal_geometry_repair':'deferred_by_user',
             'audit_script_sha256':sha(__file__),
             'retained_hashes':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                                if p.is_file() and p.name!='audit.json'}}
    atomic_json(out/'audit.json',payload)
    print(f'audit frame={frame} status={payload["status"]} visual={verdict["status"]} files={len(payload["retained_hashes"])}',flush=True)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration-root',type=Path,required=True)
    p.add_argument('--frame',default='000973');a=p.parse_args()
    audit(a.output,a.frame,a.calibration_root)


if __name__=='__main__':main()
