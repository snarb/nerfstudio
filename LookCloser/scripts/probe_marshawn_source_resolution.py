"""TRAIN-only source-resolution diagnostic; never changes a training dataset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prepare_marshawn_color_variants import linear_hd, sha, srgb, vendor
from marshawn_radial_photometry import radial_ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', default='AA')
    args = parser.parse_args()
    meta = json.loads((args.data / 'transforms.json').read_text())
    row = next(r for r in meta['frames'] if r['physical_camera'] == args.camera)
    if (row['w'], row['h']) != (1920, 1012):
        raise ValueError('This comparison expects the frozen FHD campaign')
    if row['file_path'] not in meta['train_filenames']:
        raise ValueError('Source diagnostic must use a TRAIN camera')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    pq = vendor()
    # Reuse the frozen campaign's complete photometry, with no refitting.
    source = (args.data / row['file_path']).resolve()
    contract = json.loads((source.parents[1] / 'color_contract.json').read_text())
    if contract.get('radial_and_scalar_baked') is not True:
        raise ValueError('Expected current campaign radial/scalar contract')
    fit_path = Path(contract['fit'])
    if sha(fit_path) != contract['fit_sha256']:
        raise ValueError('Existing photometry fit changed')
    fit = json.loads(fit_path.read_text())
    if fit['uses_eval_RGB'] is not False:
        raise ValueError('Expected TRAIN-only photometry fit')
    gain = contract['global_base_gain']
    ev = next(r['scalar_correction_EV'] for r in fit['camera_rows'] if r['camera'] == args.camera)
    def corrected(linear, camera):
        base = (linear * gain).astype(np.float32)
        field = radial_ev(camera, fit['k']) - fit['radial_global_center_EV'] + ev
        return srgb((base.astype(np.float64) * np.exp2(field[..., None])).astype(np.float32))
    low_linear, raw_receipt = linear_hd(row, pq)
    low = corrected(low_linear, row)
    actual = np.asarray(Image.open(source).convert('RGB'))
    if not np.array_equal(low, actual):
        raise ValueError('Rebuilt FHD source differs; do not attribute changes to resolution')
    high_row = dict(row)
    for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
        high_row[key] = row[key] * 2
    high_linear, high_receipt = linear_hd(high_row, pq)
    if raw_receipt['raw_sha256'] != high_receipt['raw_sha256']:
        raise ValueError('Raw source changed during diagnostic')
    high = corrected(high_linear, high_row)
    Image.fromarray(high).save(args.output / f'{args.camera}_3840_source.png')
    ready = json.loads((args.data / 'ready.json').read_text())
    record = next(r for r in ready['records'] if r['camera'] == args.camera)
    boxes = list(record['crops'].values())
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    # Identical projected region; a display upscale never becomes training input.
    crop = (2*x0, 2*y0, 2*x1, 2*y1)
    up = Image.fromarray(low).resize((high_row['w'], high_row['h']), Image.Resampling.BICUBIC).crop(crop)
    detail = Image.fromarray(high).crop(crop)
    sheet = Image.new('RGB', (up.width*2, up.height+32), 'white')
    sheet.paste(up, (0, 32)); sheet.paste(detail, (up.width, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), 'FHD source, display upscale x2', fill='black')
    draw.text((up.width+5, 5), '3840 source from original raw', fill='black')
    sheet.save(args.output / 'actors_source_comparison.jpg', quality=95)
    # Remapping and prefiltering do not commute; quantify rather than hiding this.
    down = cv2.resize(high, (row['w'], row['h']), interpolation=cv2.INTER_AREA)
    delta = np.abs(down.astype(float)-low.astype(float))
    result = dict(camera=args.camera, split='TRAIN', training_started=False,
                  frozen_dataset_modified=False, fhd_rebuild_byte_identical=True,
                  raw=raw_receipt, global_gain=gain, source_sha256=sha(source),
                  inherited_photometry_fit_sha256=sha(fit_path),
                  inherited_camera_ev=ev, inherited_radial_k=fit['k'],
                  helper_sha256=sha(Path(__file__)),
                  transforms_sha256=sha(args.data/'transforms.json'),
                  high_raster=[high_row['w'], high_row['h']],
                  downsampled_high_vs_fhd_mean_abs_code=float(delta.mean()),
                  downsampled_high_vs_fhd_p99_abs_code=float(np.quantile(delta, .99)),
                  scope='Resolution feasibility only; no mask, frequency, training or metric protocol change')
    (args.output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
