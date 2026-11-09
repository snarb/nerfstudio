#!/usr/bin/env python3
"""Freeze a small static-time camera slide using train-voted head bounds only."""
from pathlib import Path
from copy import deepcopy
import argparse,shutil
import numpy as np
from run_dec5_sequence import read,write
from dec5_hair import cv_pose
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'path16_bundle';out.mkdir(exist_ok=True);data=r/'frame/semantic_data';meta=read(data/'transforms.json')
    f=next(f for f in meta['frames'] if f['file_path'] in meta['val_filenames']);anchor=deepcopy(f);anchor.pop('file_path',None)
    write(out/'anchor.json',anchor);shutil.copyfile(r/'source8_hd.json',out/'source8_hd.json')
    with np.load(r/'geometry/background_control25k/prediction/D004_A005_12103C.npz') as saved:
        domain=(saved['hair_probability']>.5)|(saved['skin_probability']>.8)
    y,x=np.nonzero(domain)
    if len(x)<100:raise ValueError('Insufficient train-voted head domain')
    # Existing benchmark has twice the HD raster. Preserve HD pixel density.
    x0=max(0,(int(x.min()//2)-64)//2*2);y0=max(0,(int(y.min()//2)-64)//2*2)
    x1=min(f['w'],(int(x.max()//2)+66)//2*2);y1=min(f['h'],(int(y.max()//2)+66)//2*2)
    base=np.asarray(f['transform_matrix']);sources=read(r/'source8_hd.json')['frames']
    distance=np.median([np.linalg.norm(cv_pose(s)[:3,3]-cv_pose(f)[:3,3]) for s in sources])
    extent=.1*float(distance);frames=[];(out/'roi').mkdir(exist_ok=True);(out/'fallback').mkdir(exist_ok=True);records=[]
    for index,shift in enumerate(np.linspace(-extent,extent,16)):
        row=deepcopy(anchor);matrix=base.copy();matrix[:3,3]+=shift*matrix[:3,0]
        name='novel_%04d'%index;row.update(physical_camera=name,transform_matrix=matrix.tolist(),
            w=x1-x0,h=y1-y0,cx=f['cx']-x0,cy=f['cy']-y0);frames.append(row)
        roi=np.ones((row['h'],row['w']),bool);np.savez_compressed(out/'roi'/(name+'.npz'),roi=roi)
        path=out/'fallback'/(name+'.npz');np.savez_compressed(path,roi=roi,rgb_display=np.full((*roi.shape,3),.5,np.float32))
        records.append(dict(camera=name,sha256=sha(path)))
    write(out/'path.json',dict(frames=frames,static_time=True,frame_id='000901',target_rgb_read=False,
        train_voted_head_crop_hd_xyxy=[x0,y0,x1,y1],translation_extent_calibration_units=extent,
        target_pixel_density='HD; native6K source RGB, no4K-output claim',
        fixed_source_order=True,diagnostic_background='neutral gray; reveals missing foreground, not a complete-scene rendering'))
    write(out/'fallback/complete.json',dict(records=records,camera_manifest_sha256=sha(out/'path.json'),
        checkpoint_sha256=None,method='same constant gray diagnostic fallback for both geometry arms',uses_validation_rgb=False,uses_test_rgb=False))
    write(out/'roi/complete.json',dict(camera_manifest_sha256=sha(out/'path.json'),roi='entire fixed cropped raster',
        target_rgb_used=False,target_semantics_used=False))
    rgb=out/'native_rgb';(rgb/'images').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(r/'native_rgb/data/transforms.json',rgb/'transforms.json')
    native=read(rgb/'transforms.json');by={f['physical_camera']:f for f in native['frames']}
    for frame in sources:
        image=by[frame['physical_camera']]['file_path'];destination=rgb/image
        if not destination.exists():destination.symlink_to((r/'native_rgb/data'/image).resolve())
    for arm in ['control','volume']:
        source=r/('foreground_background_'+arm)/'frames/000901'/('training_background_'+arm)
        target=out/arm
        for name in ['request.json','background_dependency.json']:shutil.copyfile(source/name,target/name)
        shutil.copyfile(next(source.rglob('config.yml')),target/'config.yml')
    write(out/'prepared.json',dict(source_rgb_cameras=len(sources),target_frames=16,no_heldout_rgb_copied=True,
        target_raster=[x1-x0,y1-y0],physical_time_stability_established=False))
    print(read(out/'prepared.json'))


if __name__=='__main__':main()
