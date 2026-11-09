#!/usr/bin/env python3
"""Fixed validation RGB/semantic-reference metrics, never a fitting input.

Sapiens masks are fallible reference ROIs, not human alpha/depth ground truth.
Missing prediction pixels stay in every fixed ROI denominator. LPIPS uses a
fixed neutral exterior inside the reference crop; PSNR/SSIM are ROI-weighted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from run_dec5_hair_campaign import atomic_json, sha


def metrics(prediction, gt, mask, lpips):
    yy, xx = np.nonzero(mask)
    if len(xx) < 64:
        raise ValueError('Too few reference pixels')
    x0, x1 = max(0, int(xx.min())-8), min(gt.shape[1], int(xx.max())+9)
    y0, y1 = max(0, int(yy.min())-8), min(gt.shape[0], int(yy.max())+9)
    m = torch.from_numpy(mask[y0:y1, x0:x1].copy())
    p, g = [torch.from_numpy(np.ascontiguousarray(x[y0:y1, x0:x1])).permute(2, 0, 1)[None]
            for x in (prediction, gt)]
    mse = (p-g).square()[0, :, m].mean()
    _, ssim_map = structural_similarity_index_measure(p, g, data_range=1., return_full_image=True)
    neutral = torch.full_like(p, .5)
    with torch.inference_mode():
        perceptual = lpips(torch.where(m[None, None], p, neutral), torch.where(m[None, None], g, neutral))
    return dict(psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                ssim=float(ssim_map[0, :, m].mean()), lpips=float(perceptual),
                pixels=int(mask.sum()), crop=[x0, y0, x1, y1])


def validate_test_selection(path):
    if path is None or not path.is_file():
        raise ValueError('Test scoring requires locked selection')
    locked=json.loads(path.read_text())
    if locked.get('status')!='validation and visual review locked before test RGB' or locked.get('independent_test_rgb_used') is not False:
        raise ValueError('Test scoring requires completed visual selection before test RGB')
    chosen=Path(locked['chosen_metrics'])
    if not chosen.is_file() or sha(chosen)!=locked['chosen_metrics_sha256']:
        raise ValueError('Chosen validation metrics changed after selection')
    if str(chosen)!=locked.get('baseline'):
        candidate=next((r for r in locked.get('candidates',[]) if r.get('metrics')==str(chosen)),{})
        if candidate.get('passes') is not True or candidate.get('visual_decision',{}).get('accepted') is not True:
            raise ValueError('Chosen candidate lacks accepted visual evidence')
        review=Path(candidate.get('path_review',''))
        if not review.is_file() or sha(review)!=candidate.get('path_review_sha256'):
            raise ValueError('Full-path evidence changed after selection')
        for evidence in candidate.get('visual_evidence',[]):
            if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Visual evidence changed after selection')
    return sha(path)


def score(root, prediction_dir, destination, partition='validation', selection=None, boundary_radius=3,include_person_roi=False):
    if boundary_radius<1:raise ValueError('Boundary radius must be positive')
    if destination.exists():
        raise FileExistsError('Metric outputs are immutable')
    complete = json.loads((prediction_dir/'complete.json').read_text())
    selection_sha=validate_test_selection(selection) if partition=='test' else None
    dataset = root/('test_dataset' if partition == 'test' else 'dataset')
    reference_dir = root/('test_reference' if partition == 'test' else 'validation_reference')
    payload = json.loads((dataset/'transforms.json').read_text())
    by_camera = {f['physical_camera']:f for f in payload['frames']}
    torch.set_num_threads(2)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).eval()
    rows = []
    for record in complete['records']:
        name = record['camera']
        if name not in by_camera or by_camera[name]['file_path'] not in payload['test_filenames' if partition == 'test' else 'val_filenames']:
            raise ValueError('Scorer accepts only fixed validation cameras')
        path = prediction_dir/(name+'.png')
        if sha(path) != record['png_sha256']:
            raise ValueError('Prediction changed after sealing')
        pred = cv2.imread(str(path))[..., ::-1].astype(np.float32)/255
        gt = cv2.imread(str(dataset/by_camera[name]['file_path']))[..., ::-1].astype(np.float32)/255
        reference = np.load(reference_dir/(name+'.npz'))['probability'].astype(np.float32)/65535
        hair, face = reference[..., 0] > .5, reference[..., 1] > .5
        inside = cv2.erode(hair.astype(np.uint8), np.ones((2*boundary_radius+1, 2*boundary_radius+1), np.uint8)).astype(bool)
        outside = cv2.dilate(hair.astype(np.uint8), np.ones((2*boundary_radius+1, 2*boundary_radius+1), np.uint8)).astype(bool)
        boundary = outside & ~inside
        arrays = np.load(prediction_dir/(name+'.npz'))
        predicted_hair = ((arrays['hair_coverage'] if 'hair_coverage' in arrays else arrays['hair_probability']) > .5)
        predicted_hair &= arrays['source_id'] >= 0
        tp = int((hair & predicted_hair).sum())
        fp = int((~hair & predicted_hair).sum())
        fn = int((hair & ~predicted_hair).sum())
        recall = tp/max(tp+fn, 1)
        precision = tp/max(tp+fp, 1)
        holes = (arrays['source_id'] < 0) & hair
        row = dict(camera=name,
            hair=metrics(pred, gt, hair, lpips), interior=metrics(pred, gt, inside, lpips),
            boundary=metrics(pred, gt, boundary, lpips), face_neck=metrics(pred, gt, face, lpips),
            full_frame=metrics(pred, gt, np.ones(hair.shape, bool), lpips),
            silhouette=dict(precision=precision, recall=recall, f1=2*precision*recall/max(precision+recall, 1e-12),
                            false_pixels=fp, missed_pixels=fn),
            holes_fraction=float(holes.sum()/max(int(hair.sum()), 1)),
            witness_deficit_fraction=float(((arrays['support'] < 3) & hair).sum()/max(int(hair.sum()), 1)),
            halo_fraction=float((predicted_hair & (reference[..., 2] > .9) & boundary).sum()/max(int(boundary.sum()), 1)))
        row['reprojection_error_hair'] = (float(np.minimum(arrays['reprojection_error'][hair],20).mean())
                                           if 'reprojection_error' in arrays else None)
        if include_person_roi:
            body=(reference[...,2]<.5)&(reference[...,0]<.2)&(reference[...,1]<.2)
            row['clothing_hands_proxy']=metrics(pred,gt,body,lpips)
        rows.append(row)
        directory = destination.parent/(destination.stem+'_review')
        directory.mkdir(exist_ok=True)
        yy, xx = np.nonzero(outside | face)
        x0, x1 = max(int(xx.min())-16, 0), min(int(xx.max())+17, pred.shape[1])
        y0, y1 = max(int(yy.min())-16, 0), min(int(yy.max())+17, pred.shape[0])
        panel = np.concatenate((np.rot90(gt[y0:y1, x0:x1]), np.rot90(pred[y0:y1, x0:x1])), 1)
        factor = min(1., 1200/panel.shape[1])
        panel = cv2.resize(panel, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(directory/(name+'.jpg')), np.rint(panel[..., ::-1]*255).astype(np.uint8))
    aggregate = {region:{key:float(np.mean([r[region][key] for r in rows]))
                         for key in ('psnr', 'ssim', 'lpips')} for region in ('hair','interior','boundary','face_neck','full_frame')}
    if include_person_roi:aggregate['clothing_hands_proxy']={k:float(np.mean([r['clothing_hands_proxy'][k] for r in rows])) for k in ['psnr','ssim','lpips']}
    for key in ('holes_fraction', 'witness_deficit_fraction', 'halo_fraction'):
        aggregate[key] = float(np.mean([r[key] for r in rows]))
    aggregate['silhouette_recall'] = float(np.mean([r['silhouette']['recall'] for r in rows]))
    if all(r['reprojection_error_hair'] is not None for r in rows):
        aggregate['reprojection_error_hair'] = float(np.mean([r['reprojection_error_hair'] for r in rows]))
    atomic_json(destination, dict(aggregate=aggregate, per_view=rows,
        prediction_receipt_sha256=sha(prediction_dir/'complete.json'),
        semantic_reference_receipt_sha256=sha(reference_dir/'complete.json'), partition=partition,
        selection_receipt_sha256=selection_sha,
        reference_kind='GT RGB + fallible Sapiens semantic ROI; no depth/alpha GT',
        boundary_radius_pixels=boundary_radius,
        lpips_protocol='neutral 0.5 exterior inside fixed reference crop', missing_pixels_excluded=False))
    print(json.dumps(dict(method=prediction_dir.name, **aggregate)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partition', choices=('validation','test'), default='validation')
    p.add_argument('--selection-receipt', type=Path)
    p.add_argument('--boundary-radius', type=int, default=3)
    p.add_argument('--include-person-roi',action='store_true')
    args = p.parse_args()
    score(args.root, args.prediction, args.output, args.partition, args.selection_receipt, args.boundary_radius,args.include_person_roi)


if __name__ == '__main__':
    main()
