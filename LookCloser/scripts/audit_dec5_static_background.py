#!/usr/bin/env python3
"""Train-only fixed-camera background stability; moving occluders excluded."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    cv2.setNumThreads(2);base=r/'frame/semantic_data';meta=read(base/'transforms.json')
    train={f['physical_camera']:f for f in meta['frames'] if f['file_path'] in meta['train_filenames']}
    selected=read(r/'source8_hd.json')['frames'];rows=[]
    for frame in ['000917','000932']:
        directory=r/'static_background_audit'/frame;data=directory/'semantic_data'
        other=read(data/'transforms.json');other_train={f['physical_camera']:f for f in other['frames'] if f['file_path'] in other['train_filenames']}
        if set(train)!=set(other_train):raise ValueError('Source splits differ')
        for name,f in train.items():
            g=other_train[name]
            np.testing.assert_allclose(f['transform_matrix'],g['transform_matrix'],atol=1e-8,rtol=0)
            np.testing.assert_allclose([f[k] for k in ['w','h','fl_x','fl_y','cx','cy']],
                                      [g[k] for k in ['w','h','fl_x','fl_y','cx','cy']],atol=1e-8,rtol=0)
        for entry in selected:
            name=entry['physical_camera'];f=train[name];g=other_train[name]
            path=base/f['file_path'];path2=data/g['file_path']
            images=[cv2.imread(str(p)).astype(np.float32)/255 for p in [path,path2]]
            probabilities=[np.load(p)['probability'].astype(np.float32)/65535 for p in
                [r/'frame/semantics'/(name+'.npz'),directory/'semantics'/(name+'.npz')]]
            mask=cv2.erode(((probabilities[0][...,2]>.98)&(probabilities[1][...,2]>.98)).astype(np.uint8),np.ones((33,33),np.uint8))>0
            hair=(probabilities[0][...,0]>.5)|(probabilities[1][...,0]>.5)
            adjacent=mask&(cv2.dilate(hair.astype(np.uint8),np.ones((65,65),np.uint8))>0)
            scores={}
            for label,domain in [('common_background',mask),('hair_adjacent_background',adjacent)]:
                if int(domain.sum())<100:raise ValueError('Insufficient static-background evidence')
                difference=images[1][domain]-images[0][domain]
                scores[label]=dict(pixels=int(domain.sum()),psnr=float(-10*np.log10(max(float(np.square(difference).mean()),1e-12))),
                                  mean_absolute_change=float(np.abs(difference).mean()))
            rows.append(dict(source_frame='000901',frame=frame,camera=name,regions=scores,
                identical_full_images=bool(np.array_equal(*images)),source_sha256=sha(path),target_sha256=sha(path2)))
    write(r/'static_background_audit/metrics.json',dict(pairs=rows,all64_camera_poses_and_intrinsics_equal=True,
        uses_validation_rgb=False,uses_test_rgb=False,background_static_assumption_not_yet_promoted=True,
        protocol='fixed train camera; common confident BG eroded16 HD pixels; adjacent band32px around either train hair ROI'))
    print({region:dict(min_psnr=min(v['regions'][region]['psnr'] for v in rows),
                      mean_psnr=float(np.mean([v['regions'][region]['psnr'] for v in rows])))
           for region in ['common_background','hair_adjacent_background']})


if __name__=='__main__':main()
