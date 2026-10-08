#!/usr/bin/env python3
"""Read-only, train-only stereo check near the fitted 000470 cord segment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from triangulate_luster_cord_landmarks import cameras_from_colmap, project


REFERENCE_ID = 97
TARGET_IDS = (25, 26, 27, 98)
PATCH_SIZES = (5, 9, 13)
FRACTIONS = np.linspace(.08, .92, 12)
OFFSETS = np.linspace(-.012, .012, 97)  # normalized scene depth


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def patch(image: np.ndarray, xy: np.ndarray, size: int) -> np.ndarray:
    margin=size//2+1
    if not (margin<=xy[0]<image.shape[1]-margin
            and margin<=xy[1]<image.shape[0]-margin):
        raise ValueError('Projected patch leaves original image')
    return cv2.getRectSubPix(image,(size,size),(float(xy[0]),float(xy[1])))


def ncc(left: np.ndarray, right: np.ndarray) -> float:
    a=left-left.mean()
    b=right-right.mean()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    # A flat patch carries no correspondence evidence. Penalize that candidate
    # rather than discarding an otherwise visible training camera.
    return float((a*b).sum()/denominator) if denominator>1e-6 else -1.


def curve_summary(curve: np.ndarray) -> dict:
    if not np.isfinite(curve).all():
        raise ValueError('Undefined patch correlation in candidate depth scan')
    best=int(np.argmax(curve))
    separated=np.abs(OFFSETS-OFFSETS[best])>=.002
    return dict(best_offset_normalized=float(OFFSETS[best]),best_ncc=float(curve[best]),
                line_ncc=float(curve[len(OFFSETS)//2]),
                separated_peak_margin=float(curve[best]-curve[separated].max()),
                at_search_boundary=best in (0,len(OFFSETS)-1))


def audit(fit_path: Path, sparse: Path, images_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fit=json.loads(fit_path.read_text())
    train_ids=set.intersection(*(set(fit['fit_camera_ids'][label])
                                 for label in ('junction','first_bead')))
    ids=(REFERENCE_ID,*TARGET_IDS)
    if (fit.get('train_only') is not True or fit['validation_camera_id'] in ids
            or not set(ids).issubset(train_ids)
            or sha(sparse/'cameras.txt')!=fit['sparse_camera_sha256']
            or sha(sparse/'images.txt')!=fit['sparse_image_sha256']):
        raise ValueError('Only pinned original train cameras may enter stereo')
    cameras,poses=cameras_from_colmap(sparse)
    source={camera_id:images_root/f'cam_{camera_id:03d}_{fit["frame"]}.jpg'
            for camera_id in ids}
    for camera_id,path in source.items():
        if sha(path)!=fit['source_image_sha256'][str(camera_id)]:
            raise ValueError(f'Original train camera {camera_id} RGB changed')
    gray={camera_id:cv2.imread(str(path),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
          for camera_id,path in source.items()}
    colour={camera_id:cv2.imread(str(path),cv2.IMREAD_COLOR).astype(np.float32)/255
            for camera_id,path in source.items()}
    start=np.asarray(fit['fits']['junction']['original_world_xyz'],dtype=float)
    end=np.asarray(fit['fits']['first_bead']['original_world_xyz'],dtype=float)
    scale=float(fit['original_to_normalized']['scale'])
    matrix=np.asarray(fit['original_to_normalized']['rotation'],dtype=float)
    translation=np.asarray(fit['original_to_normalized']['translation'],dtype=float)
    camera_rotation,camera_translation,_=poses[REFERENCE_ID]
    reference_origin=-camera_rotation.T@camera_translation
    rows=[]
    for fraction in FRACTIONS:
        point=start+fraction*(end-start)
        reference_xy,_=project(REFERENCE_ID,point,cameras,poses)
        reference_direction=(point-reference_origin)/np.linalg.norm(point-reference_origin)
        projected={camera_id:[] for camera_id in TARGET_IDS}
        for offset in OFFSETS:
            candidate=point+reference_direction*(offset/scale)
            for camera_id in TARGET_IDS:
                xy,z=project(camera_id,candidate,cameras,poses)
                if z<=0:
                    raise ValueError('Candidate stereo point lies behind a camera')
                projected[camera_id].append(xy)
        by_size={}
        for size in PATCH_SIZES:
            reference=patch(gray[REFERENCE_ID],reference_xy,size)
            curves={}
            for camera_id in TARGET_IDS:
                curves[str(camera_id)]=np.array([
                    ncc(reference,patch(gray[camera_id],xy,size))
                    for xy in projected[camera_id]],dtype=float)
            array=np.stack(list(curves.values()))
            median=np.median(array,axis=0)
            median_summary=curve_summary(median)
            by_size[str(size)]=dict(reference_patch_std=float(reference.std()),
                                   median=median_summary,median_curve=median.tolist(),
                                   cameras={str(camera_id):dict(**curve_summary(curves[str(camera_id)]),
                                                                curve=curves[str(camera_id)].tolist())
                                            for camera_id in TARGET_IDS})
        main=by_size['9']
        candidate_offset=main['median']['best_offset_normalized']
        leave_one_out=[]
        for omitted in TARGET_IDS:
            subset=np.stack([np.asarray(main['cameras'][str(camera_id)]['curve'])
                             for camera_id in TARGET_IDS if camera_id!=omitted])
            leave_one_out.append(dict(omitted_camera=omitted,
                                      best_offset_normalized=curve_summary(np.median(subset,axis=0))['best_offset_normalized']))
        size_offsets=[by_size[str(size)]['median']['best_offset_normalized'] for size in PATCH_SIZES]
        view_offsets=[entry['best_offset_normalized'] for entry in leave_one_out]
        offset_index=int(np.argmin(abs(OFFSETS-candidate_offset)))
        camera_support=sum(main['cameras'][str(camera_id)]['curve'][offset_index]>.5
                           and main['cameras'][str(camera_id)]['best_ncc']
                           -main['cameras'][str(camera_id)]['curve'][offset_index]<.15
                           for camera_id in TARGET_IDS)
        provisional=bool(main['median']['best_ncc']>=.65
                         and main['median']['separated_peak_margin']>=.03
                         and not main['median']['at_search_boundary']
                         and max(abs(value-candidate_offset) for value in size_offsets)<=.002
                         and max(abs(value-candidate_offset) for value in view_offsets)<=.003
                         and camera_support>=3)
        candidate=point+reference_direction*(candidate_offset/scale)
        colour_reference=patch(colour[REFERENCE_ID],reference_xy,5).mean((0,1))
        colour_errors={}
        projections={}
        for camera_id in ids:
            xy,_=project(camera_id,candidate,cameras,poses)
            projections[str(camera_id)]=xy.tolist()
            if camera_id!=REFERENCE_ID:
                colour_target=patch(colour[camera_id],xy,5).mean((0,1))
                colour_errors[str(camera_id)]=float(np.linalg.norm(colour_target-colour_reference))
        rows.append(dict(fraction=float(fraction),reference_xy=reference_xy.tolist(),
                         fitted_point_original_xyz=point.tolist(),
                         candidate_original_xyz=candidate.tolist(),
                         candidate_normalized_xyz=(scale*candidate@matrix+translation).tolist(),
                         candidate_offset_normalized=candidate_offset,
                         provisional_high_confidence=provisional,
                         camera_support=camera_support,
                         leave_one_camera_out=leave_one_out,
                         patch_size_best_offsets_normalized=size_offsets,
                         source_mean_bgr_difference=colour_errors,
                         train_projections_original_px=projections,
                         patch_scores=by_size))
    output.mkdir(parents=True)
    receipt=dict(scope='Read-only train-camera guided stereo near a two-landmark line; not certified surface depth',
                 frame=fit['frame'],train_only=True,masks_used=False,
                 reference_camera=REFERENCE_ID,target_cameras=list(TARGET_IDS),
                 heldout_camera_not_used=fit['validation_camera_id'],
                 fractions=FRACTIONS.tolist(),patch_sizes=list(PATCH_SIZES),
                 normalized_depth_offsets=OFFSETS.tolist(),
                 fit_sha256=sha(fit_path),transforms_sha256=fit['transforms_sha256'],
                 sparse_camera_sha256=sha(sparse/'cameras.txt'),
                 sparse_image_sha256=sha(sparse/'images.txt'),
                 source_image_sha256={str(camera_id):sha(path) for camera_id,path in source.items()},
                 rows=rows)
    (output/'result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(3,4,figsize=(14,8),sharex=True,sharey=True)
    for ax,row in zip(axes.flat,rows):
        main=row['patch_scores']['9']
        for camera_id in TARGET_IDS:
            ax.plot(OFFSETS,main['cameras'][str(camera_id)]['curve'],lw=.7,alpha=.6,label=str(camera_id))
        ax.plot(OFFSETS,main['median_curve'],color='black',lw=1.6,label='median')
        ax.axvline(0,color='gray',lw=.8)
        ax.axvline(row['candidate_offset_normalized'],color='magenta',lw=.8)
        ax.set_title(f'u={row["fraction"]:.2f}, {"pass" if row["provisional_high_confidence"] else "uncertain"}')
        ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('normalized depth offset')
    for ax in axes[:,0]:ax.set_ylabel('9×9 grayscale NCC')
    axes[0,0].legend(ncol=3,fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'ncc_curves.png',dpi=150)
    plt.close(fig)
    print(json.dumps([dict(fraction=round(row['fraction'],3),
                           offset=round(row['candidate_offset_normalized'],5),
                           ncc=round(row['patch_scores']['9']['median']['best_ncc'],3),
                           support=row['camera_support'],
                           provisional=row['provisional_high_confidence'])
                      for row in rows]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--sparse',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.fit,args.sparse,args.images,args.output)
