#!/usr/bin/env python3
"""Appearance-only soft transition for the target-opacity validity gate.

Requires matched hard .1 and target-only .01 renders. Confident pixels must be
bit-identical. Uncertain pixels interpolate in bounded linear-sRGB, avoiding
white-pixel blowups from inverse-Reinhard pseudo-HDR. No semantic output mask.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from add_dec5_neural_fallback import inverse_display
from run_dec5_hair_campaign import atomic_json,sha


def blend(base,candidate,opacity,low=.01,high=.1):
    if not 0<=low<high<=1 or not np.isfinite(opacity).all():raise ValueError('Invalid opacity confidence interval')
    t=np.clip((opacity-low)/(high-low),0,1);weight=t*t*(3-2*t)
    values=[]
    for image in [base,candidate]:
        x=image.astype(np.float32)/255
        values.append(np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4))
    value=values[0]*(1-weight[...,None])+values[1]*weight[...,None]
    display=np.where(value<=.0031308,12.92*value,1.055*np.power(value,1/2.4)-.055)
    output=np.rint(np.clip(display,0,1)*255).astype(np.uint8)
    output[weight<=0]=base[weight<=0];output[weight>=1]=candidate[weight>=1]
    return output,weight


def main():
    p=argparse.ArgumentParser()
    for key in ['baseline','candidate','target-depth','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((a.baseline/'complete.json').read_text());candidate=json.loads((a.candidate/'complete.json').read_text())
    for key in ['checkpoint_sha256','selected_cameras','source_receipt_sha256',
                'source_reference_camera_sha256','source_rgb_dataset_sha256','fallback_receipt_sha256']:
        if baseline[key]!=candidate[key]:raise ValueError('Target-only counterfactual mismatch: '+key)
    depth_receipt=json.loads((a.target_depth/'complete.json').read_text())
    if depth_receipt['checkpoint_sha256']!=candidate['checkpoint_sha256']:
        raise ValueError('Opacity evidence belongs to another field')
    before={r['camera']:r for r in baseline['records']};records=[]
    for record in candidate['records']:
        name=record['camera'];base_path=a.baseline/(name+'.npz');candidate_path=a.candidate/(name+'.npz')
        if sha(base_path)!=before[name]['npz_sha256'] or sha(candidate_path)!=record['npz_sha256']:raise ValueError('Unsealed input prediction')
        base=dict(np.load(base_path));arrays=dict(np.load(candidate_path));opacity=np.load(a.target_depth/(name+'.npz'))['opacity']
        np.testing.assert_array_equal(base['roi'],arrays['roi']);protected=(opacity>=.1)&base['roi']
        np.testing.assert_array_equal(base['rgb'][protected],arrays['rgb'][protected])
        images=[cv2.imread(str(d/(name+'.png')))[...,::-1] for d in [a.baseline,a.candidate]]
        image,weight=blend(*images,opacity)
        changed=(weight>0)&(weight<1)&arrays['roi']
        arrays['rgb'][weight<=0]=base['rgb'][weight<=0]
        arrays['rgb'][changed]=inverse_display(image.astype(np.float32)/255)[changed]
        arrays['appearance_gt_confidence']=weight
        np.testing.assert_array_equal(base['rgb'][protected],arrays['rgb'][protected])
        target=a.output/(name+'.npz');np.savez_compressed(target,**arrays)
        png=a.output/(name+'.png');cv2.imwrite(str(png),image[...,::-1])
        cv2.imwrite(str(a.output/(name+'_review.jpg')),cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA)[...,::-1])
        records.append(dict(record,npz_sha256=sha(target),png_sha256=sha(png),soft_transition_pixels=int(changed.sum())))
    candidate.update(records=records,appearance_soft_target_opacity_gate=True,confidence_interval=[.01,.1],
        mixing_domain='bounded linear-sRGB of rendered JPEG-derived display colors; not recovered HDR',
        confident_pixels_bit_identical_to_hard_baseline=True,geometry_and_source_ids_unchanged_from_low_floor_parent=True,
        baseline_receipt_sha256=sha(a.baseline/'complete.json'),low_floor_parent_sha256=sha(a.candidate/'complete.json'))
    atomic_json(a.output/'complete.json',candidate)


if __name__=='__main__':main()
