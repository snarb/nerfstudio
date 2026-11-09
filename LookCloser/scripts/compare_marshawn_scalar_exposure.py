#!/usr/bin/env python3
"""Score a scalar-normalized model in the unchanged gain4 reference domain.

Undo only its fixed per-camera exposure in linear space. No spatial masks,
compositing, blur or learned image correction are applied to predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from summarize_marshawn_campaign import RoiScorer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def undo_ev(image, ev):
    code = np.asarray(image, dtype=np.float64)/255
    mapped = np.where(code <= .04045, code/12.92, ((code+.055)/1.055)**2.4)
    linear = mapped/np.maximum(1-mapped, 1e-12)
    linear *= 2**(-ev)
    mapped = linear/(1+linear)
    code = np.where(mapped <= .0031308, 12.92*mapped, 1.055*mapped**(1/2.4)-.055)
    return Image.fromarray(np.rint(np.clip(code, 0, 1)*255).astype(np.uint8))


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads(a.contract.read_text())
    assert contract['eval_rgb_used_to_fit_gains'] is False
    evs = {r['camera']: r['exposure_correction_ev'] for r in contract['rows']}
    meta = json.loads((a.reference/'transforms.json').read_text())
    views = sorted((f for f in meta['frames'] if f['file_path'] in meta['val_filenames']), key=lambda f: f['file_path'])
    rois = json.loads(a.problem_rois.read_text())['rows']
    scorer = RoiScorer(a.device)
    sheet = Image.new('RGB', (1440, 834));draw = ImageDraw.Draw(sheet)
    rows = []
    for index, frame in enumerate(views):
        name = frame['physical_camera']
        original = Image.open(a.reference/frame['file_path']).convert('RGB')
        filename = f'eval_img_{index:04d}.png'
        base_path, candidate_path = a.baseline/filename, a.candidate/filename
        baseline = Image.open(base_path).convert('RGB')
        candidate = Image.open(candidate_path).convert('RGB')
        assert baseline.size == candidate.size == (3840, 1012)
        assert np.array_equal(np.asarray(baseline.crop((0, 0, 1920, 1012))), np.asarray(original))
        baseline_pred = baseline.crop((1920, 0, 3840, 1012))
        common_pred = undo_ev(candidate.crop((1920, 0, 3840, 1012)), evs[name])
        common_gt = undo_ev(candidate.crop((0, 0, 1920, 1012)), evs[name])
        gt_quantization = abs(np.array(common_gt).astype(int)-np.array(original).astype(int))
        common_pred.save(a.output/(name+'_candidate_common_gain4.png'))
        row = dict(camera=name, inverse_fixed_ev=-evs[name],
                   source_files={str(p): sha(p) for p in [base_path, candidate_path, a.reference/frame['file_path']]},
                   gt_roundtrip_codes=dict(mean=float(gt_quantization.mean()), max=int(gt_quantization.max())),
                   full=dict(baseline=scorer.score(original, baseline_pred), scalar_candidate=scorer.score(original, common_pred)),
                   regions={})
        for region in [r for r in rois if r['camera'] == name]:
            bbox = tuple(region['bbox_xyxy'])
            row['regions'][region['region']] = dict(baseline=scorer.score(original.crop(bbox), baseline_pred.crop(bbox)),
                                                   scalar_candidate=scorer.score(original.crop(bbox), common_pred.crop(bbox)))
            panel = Image.new('RGB', ((bbox[2]-bbox[0])*3, bbox[3]-bbox[1]+24))
            pd = ImageDraw.Draw(panel)
            for j, (im, label) in enumerate([(original, 'Original gain4 GT'), (baseline_pred, 'Native scale-reg baseline'), (common_pred, 'Scalar candidate: fixed EV undone')]):
                panel.paste(im.crop(bbox), (j*(bbox[2]-bbox[0]), 24));pd.text((j*(bbox[2]-bbox[0])+3, 3), label, fill='white')
            panel.save(a.output/(region['region']+'_common_comparison.jpg'), quality=95)
        for j, (im, label) in enumerate([(original, 'Original gain4 GT'), (baseline_pred, 'Baseline'), (common_pred, 'Scalar candidate, original exposure')]):
            tile = im.copy();tile.thumbnail((480, 253));sheet.paste(tile, (j*480, index*278+24));draw.text((j*480+3, index*278+3), name+' '+label, fill='white')
        rows.append(row)
    sheet.save(a.output/'all_common_gain4_comparison.jpg', quality=94)
    report = dict(kind='SAVED_PNG_common_gain4_reference_comparison', native_resolution=[1920, 1012],
                  transform='sRGB decode -> inverse Reinhard -> undo fixed scalar EV -> Reinhard -> sRGB -> uint8',
                  contract_sha256=sha(a.contract), script_sha256=sha(__file__),
                  no_spatial_processing=True, eval_RGB_used_to_fit_gains=False,
                  metric_selection_unchanged=True, rows=rows,
                  average={branch:{key:float(np.mean([r['full'][branch][key] for r in rows])) for key in ['psnr','ssim','lpips']}
                           for branch in ['baseline','scalar_candidate']},
                  caveat='PNG quantization and nonlinear inverse near white are measured by GT roundtrip; authoritative within-run selection remains native all-eval metrics.')
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['average']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['reference', 'contract', 'baseline', 'candidate', 'problem-rois', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
