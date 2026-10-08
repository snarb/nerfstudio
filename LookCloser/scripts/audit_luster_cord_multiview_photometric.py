#!/usr/bin/env python3
"""Read-only train-camera color agreement along candidate cord depths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from triangulate_luster_cord_landmarks import cameras_from_colmap, project


REFERENCE_ID=97
TARGET_IDS=(98,96,26,27)
FRACTIONS=(.10,.25,.40,.55,.70,.85)
PATCH_SIZES=(3,5,9)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def color_at(image: np.ndarray, xy: np.ndarray, size: int) -> np.ndarray | None:
    margin=size//2+1
    if not (margin <= xy[0] < image.shape[1]-margin
            and margin <= xy[1] < image.shape[0]-margin):
        return None
    return cv2.getRectSubPix(image,(size,size),(float(xy[0]),float(xy[1]))).mean((0,1))


def audit(fit_path: Path, sparse: Path, images_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Preserve prior depth audits')
    fit=json.loads(fit_path.read_text())
    train_ids=set().union(*(set(ids) for ids in fit['fit_camera_ids'].values()))
    if (fit.get('train_only') is not True or fit['validation_camera_id'] in TARGET_IDS
            or not {REFERENCE_ID,*TARGET_IDS}.issubset(train_ids)):
        raise ValueError('Only fitted training cameras may participate')
    cameras,poses=cameras_from_colmap(sparse)
    image_paths={camera_id:images_root/f'cam_{camera_id:03d}_{fit["frame"]}.jpg'
                 for camera_id in (REFERENCE_ID,*TARGET_IDS)}
    images={camera_id:cv2.imread(str(path),cv2.IMREAD_COLOR).astype(np.float32)/255
            for camera_id,path in image_paths.items()}
    for camera_id in image_paths:
        if digest(image_paths[camera_id])!=fit['source_image_sha256'][str(camera_id)]:
            raise ValueError(f'Original camera {camera_id} source changed')
    raw_start=np.asarray(fit['fits']['junction']['original_world_xyz'])
    raw_end=np.asarray(fit['fits']['first_bead']['original_world_xyz'])
    normalized_scale=float(fit['original_to_normalized']['scale'])
    rotation,translation,_=poses[REFERENCE_ID]
    origin=-rotation.T@translation
    raw_offsets=np.linspace(-.12,.12,121)
    normalized_offsets=raw_offsets*normalized_scale
    results=[]
    for fraction in FRACTIONS:
        point=raw_start+fraction*(raw_end-raw_start)
        reference_xy,_=project(REFERENCE_ID,point,cameras,poses)
        direction=point-origin
        nominal_depth=float(np.linalg.norm(direction))
        direction/=nominal_depth
        result=dict(fraction=fraction,reference_xy=reference_xy.tolist(),
                    nominal_raw_depth=nominal_depth,patches={})
        for size in PATCH_SIZES:
            reference=color_at(images[REFERENCE_ID],reference_xy,size)
            if reference is None:
                raise ValueError('Reference patch outside original image')
            costs={str(camera_id):[] for camera_id in TARGET_IDS}
            for offset in raw_offsets:
                candidate=origin+(nominal_depth+offset)*direction
                for camera_id in TARGET_IDS:
                    xy,z=project(camera_id,candidate,cameras,poses)
                    color=color_at(images[camera_id],xy,size) if z>0 else None
                    costs[str(camera_id)].append(None if color is None else
                        float(np.linalg.norm(reference-color)))
            array=np.array([[np.nan if value is None else value for value in costs[str(camera_id)]]
                            for camera_id in TARGET_IDS])
            valid=np.isfinite(array).sum(0)
            with np.errstate(all='ignore'):
                median=np.nanmedian(array,axis=0)
            median[valid<2]=np.nan
            if not np.isfinite(median).any():
                raise ValueError('No multi-view evidence for a candidate depth')
            index=int(np.nanargmin(median))
            fit_index=int(np.argmin(np.abs(raw_offsets)))
            individual={}
            for camera_id,row in zip(TARGET_IDS,array):
                if not np.isfinite(row).any():
                    continue
                best=int(np.nanargmin(row))
                individual[str(camera_id)]=dict(best_offset_normalized=float(normalized_offsets[best]),
                    best_cost=float(row[best]),fit_cost=(None if not np.isfinite(row[fit_index])
                                                     else float(row[fit_index])))
            result['patches'][str(size)]=dict(reference_bgr=reference.tolist(),
                median_cost=[None if not np.isfinite(value) else float(value) for value in median],
                best_offset_normalized=float(normalized_offsets[index]),
                best_cost=float(median[index]),fit_cost=float(median[fit_index]),
                individual=individual)
        results.append(result)
    output.mkdir(parents=True)
    receipt=dict(scope='Read-only color agreement; not certified ground-truth depth',
        frame=fit['frame'],train_only=True,heldout_camera_not_used=fit['validation_camera_id'],
        reference_camera=REFERENCE_ID,target_cameras=list(TARGET_IDS),
        fractions=list(FRACTIONS),patch_sizes=list(PATCH_SIZES),
        normalized_offsets=normalized_offsets.tolist(),fit_sha256=digest(fit_path),
        sparse_camera_sha256=digest(sparse/'cameras.txt'),
        sparse_image_sha256=digest(sparse/'images.txt'),
        source_image_sha256={str(k):digest(v) for k,v in image_paths.items()},results=results)
    (output/'result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
    for ax,row in zip(axes.flat,results):
        for size in PATCH_SIZES:
            values=np.array([np.nan if v is None else v for v in row['patches'][str(size)]['median_cost']])
            ax.plot(normalized_offsets,values,label=f'{size}×{size}')
        ax.axvline(0,color='black',lw=1)
        ax.axvspan(-.003,.003,color='gray',alpha=.12)
        ax.set_title(f'u={row["fraction"]:.2f}')
        ax.set_ylabel('median RGB difference')
        ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('depth offset, normalized scene units')
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(output/'cost_curves.png',dpi=160)
    plt.close(fig)
    print(json.dumps([dict(fraction=row['fraction'],best_offsets={size:row['patches'][str(size)]['best_offset_normalized']
                       for size in map(str,PATCH_SIZES)}) for row in results]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--sparse',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.fit,args.sparse,args.images,args.output)
