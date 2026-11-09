#!/usr/bin/env python3
"""Prepare one separate native-JPEG head-resolution benchmark, preserving exclusions.

All crop positions come from a train-derived 3D head center, not held-out RGB.
Native inputs are real 6144x3072 JPEGs; no HDR fidelity is claimed. The 4K
reference is separate from both training datasets and is used only for scoring.
"""
from pathlib import Path
import argparse,json,copy
import cv2,numpy as np
from dec5_hair import cv_pose,VALIDATION,TEST
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['reference-data','jpeg-root','rig','head-bounds','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--frame',default='001055');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    original=json.load(open(a.reference_data/'transforms.json'));rig=json.load(open(a.rig));bounds=np.array(json.load(open(a.head_bounds))['head_bounds']);center=bounds.mean(0)
    layouts={'crop_hd':(1260,1080),'crop_native':(3584,3072),'reference_4k':(3840,2160)};datasets={}
    for label,(w,h) in layouts.items():
        dest=a.output/label;(dest/'images').mkdir(parents=True)
        data=copy.deepcopy(original)
        for k in ['fl_x','fl_y','cx','cy','w','h','frames']:data.pop(k,None)
        data.update(w=w,h=h,frames=[],frame_id=a.frame,source_kind='native 6144x3072 tonemap-frozen JPEG, 8-bit; not HDR',camera_model='OPENCV')
        datasets[label]=data
    records=[]
    for f in original['frames']:
        n=f['physical_camera'];path=a.jpeg_root/n/(a.frame+'.jpg');image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(3072,6144):raise ValueError('Unreadable native JPEG: '+str(path))
        camera=rig['cameras'][n];fx,fy,cx,cy,*dist=camera['params_fx_fy_cx_cy_k1_k2_p1_p2']
        if any(abs(v)>1e-12 for v in dist):raise ValueError('This control requires the known zero-distortion rig model')
        pose=cv_pose(f);point=(center-pose[:3,3])@pose[:3,:3]
        if point[2]<=0:raise ValueError('Head behind camera')
        u=fx*point[0]/point[2]+cx;x0=int(np.clip(round(u-3584/2),0,6144-3584));crop=image[:,x0:x0+3584]
        values={'crop_native':crop,'crop_hd':cv2.resize(crop,(1260,1080),interpolation=cv2.INTER_AREA),
            'reference_4k':cv2.resize(image[:,341:341+5461],(3840,2160),interpolation=cv2.INTER_AREA)}
        row=dict(camera=n,source=str(path),source_sha256=sha(path),native_crop=[x0,0,x0+3584,3072],images={})
        for label,im in values.items():
            out=a.output/label/f['file_path'];out.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(out),im,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise OSError('PNG write failed')
            frame=copy.deepcopy(f);w,h=layouts[label]
            if label=='reference_4k':sx,sy=w/5461,h/3072;offset=341
            else:sx,sy=w/3584,h/3072;offset=x0
            frame.update(w=w,h=h,fl_x=fx*sx,fl_y=fy*sy,cx=(cx-offset)*sx,cy=cy*sy,k1=0.,k2=0.,p1=0.,p2=0.)
            datasets[label]['frames'].append(frame);row['images'][label]=dict(sha256=sha(out),shape=list(im.shape))
        records.append(row);atomic_json(a.output/'progress.json',dict(completed=len(records),total=len(original['frames'])))
    for label,data in datasets.items():
        train=[f for f in data['frames'] if f['file_path'] in data['train_filenames']]
        if len(train)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in train):raise ValueError('Train59 exclusion mismatch')
        atomic_json(a.output/label/'transforms.json',data)
    atomic_json(a.output/'complete.json',dict(frame=a.frame,records=records,layouts=layouts,reference_pose_dataset_sha256=sha(a.reference_data/'transforms.json'),
        rig_sha256=sha(a.rig),head_bounds_sha256=sha(a.head_bounds),crop_selection='projected train-only 001083 head center; no target RGB or masks',
        head_center=center.tolist(),native_supervision='real crop, no upscaling',reference='center crop 5461x3072 to 3840x2160; evaluation only',
        fitting_cameras=59,validation_cameras=3,prepared_camera_count=len(records),
        prepared_test_camera_count=sum(r['camera'] in TEST for r in records),reserved_test_camera_identifiers=list(TEST),
        source_jpeg_bit_depth=8,implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
