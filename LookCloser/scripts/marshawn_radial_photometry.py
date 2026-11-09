"""Experimental cos^k correction in an undistorted image, before tone mapping.

This is an analytic image conversion, not a certified flat-field calibration.
Intrinsics use COLMAP pixel centers; NumPy raster index (x,y) means (x+.5,y+.5).
No radius clipping, image-dependent normalization, RGB white balance or fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def radial_ev(frame, k, uv=None):
    """Inverse cos^k attenuation in EV; uv, if supplied, is COLMAP centers."""
    if any(abs(float(frame.get(key, 0))) > 1e-12
           for key in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')):
        raise ValueError('Expected already-undistorted calibrated pixels')
    if not np.isfinite(k) or k < 0 or k > 6:
        raise ValueError('Diagnostic k must be in declared [0,6] range')
    if uv is None:
        y, x = np.ogrid[:frame['h'], :frame['w']]
        x, y = x + .5, y + .5
    else:
        uv = np.asarray(uv, dtype=np.float64)
        x, y = uv[..., 0], uv[..., 1]
    r2 = ((x-frame['cx'])/frame['fl_x'])**2 + ((y-frame['cy'])/frame['fl_y'])**2
    return (k/2)*np.log2(1+r2)


def encode(linear):
    x = np.maximum(linear, 0)
    x = x/(1+x)
    return np.rint(np.clip(np.where(x <= .0031308, 12.92*x,
                                  1.055*x**(1/2.4)-.055), 0, 1)*255).astype(np.uint8)


def build(args):
    fit = json.loads(args.fit.read_text())
    meta = json.loads((args.source/'transforms.json').read_text())
    if fit['uses_eval_RGB'] is not False or fit['k'] != 4:
        raise ValueError('Requires reviewed train-only fixed-k4 diagnostic')
    if sha(args.source/'transforms.json') != fit['source_transforms_sha256']:
        raise ValueError('Calibration source transforms mismatch')
    evs = {r['camera']: r['scalar_correction_EV'] for r in fit['camera_rows']}
    train = {f['physical_camera'] for f in meta['frames']
             if f['file_path'] in meta['train_filenames']}
    if set(evs) != train or len(train) != 63:
        raise ValueError('Expected exactly63TRAIN fitted gains')
    predicted = fit['eval_predictions']
    if set(predicted) != {'DC', 'HC', 'KA'}:
        raise ValueError('Expected three pose-only eval predictions')
    evs.update({n: r['ev'] for n, r in predicted.items()})
    center = float(fit['radial_global_center_EV'])
    if not np.isfinite([center, *evs.values()]).all():
        raise ValueError('Nonfinite photometry')
    source_contract = json.loads((args.source/'color_contract.json').read_text())
    if source_contract['global_gain'] != 4 or source_contract['encoding'] != 'scene-linearfloat32':
        raise ValueError('Expected unscalar gain4 linear master, avoid double EV')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    out_meta = json.loads(json.dumps(meta))
    for kind in ['sdr_png', 'linear_rec709_exr']:
        (args.output/kind/'images').mkdir(parents=True)
        (args.output/kind/'masks').symlink_to((args.source/'masks').resolve(), target_is_directory=True)
    sheet = Image.new('RGB', (960, 3*280))
    draw = ImageDraw.Draw(sheet)
    for frame in meta['frames']:
        name = frame['physical_camera']
        if args.preview and name not in {'DC', 'HC', 'KA'}:
            continue
        path = args.source/frame['file_path']
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.dtype != np.float32 or bgr.shape != (frame['h'], frame['w'], 3):
            raise ValueError(str(path))
        field = radial_ev(frame, fit['k']) - center + evs[name]
        corrected = (bgr.astype(np.float64)*np.exp2(field[..., None])).astype(np.float32)
        if not np.isfinite(corrected).all():
            raise ValueError('Nonfinite converted image')
        rel = Path(frame['file_path'])
        ep = args.output/'linear_rec709_exr'/rel
        pp = args.output/'sdr_png'/rel.with_suffix('.png')
        if not cv2.imwrite(str(ep), corrected, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
            raise OSError(str(ep))
        if not np.array_equal(cv2.imread(str(ep), cv2.IMREAD_UNCHANGED), corrected):
            raise ValueError('EXR roundtrip mismatch')
        Image.fromarray(encode(corrected[..., ::-1])).save(pp)
        rows.append(dict(camera=name, source=str(path), source_sha256=sha(path),
                         exr=str(ep), exr_sha256=sha(ep), png=str(pp), png_sha256=sha(pp),
                         total_correction_EV_quantiles=np.quantile(field, [0, .5, 1]).tolist()))
        if name in {'DC', 'HC', 'KA'}:
            i = ['DC', 'HC', 'KA'].index(name)
            for j, image in enumerate([bgr, corrected]):
                tile = Image.fromarray(encode(image[..., ::-1])); tile.thumbnail((480, 253))
                sheet.paste(tile, (480*j, 280*i+24))
            draw.text((3, 280*i+3), name+' gain4 unscalar | fixedk4 + joint TRAIN scalar (diagnostic)', fill='white')
    for kind, suffix in [('linear_rec709_exr', '.exr'), ('sdr_png', '.png')]:
        copied = json.loads(json.dumps(out_meta))
        if args.preview:
            copied = None  # A three-image preview must not masquerade as a full dataset.
        else:
            for f in copied['frames']:
                f['file_path'] = str(Path(f['file_path']).with_suffix(suffix))
            for key in ['train_filenames', 'val_filenames', 'test_filenames']:
                copied[key] = [str(Path(p).with_suffix(suffix)) for p in copied[key]]
            (args.output/kind/'transforms.json').write_text(json.dumps(copied, indent=2)+'\n')
    receipt = dict(kind='EXPERIMENTAL_FIXED_K4_RADIAL_PLUS_SCALAR_NOT_CERTIFIED_FLATFIELD',
                   preview_only=args.preview, uses_eval_RGB=False, NN_updates=0,
                   k=fit['k'], radial_global_center_EV=center,
                   formula='base_gain4_linear*2**(k/2*log2(1+r2)-TRAIN_global_center_EV+camera_EV)',
                   display='Reinhard then sRGB8, no additional exposure',
                   fit=str(args.fit), fit_sha256=sha(args.fit), helper_sha256=sha(__file__), rows=rows)
    for kind, encoding in [('sdr_png', 'sRGB8display'),
                           ('linear_rec709_exr', 'scene-linearfloat32')]:
        contract = dict(receipt, encoding=encoding, primaries='Rec709/D65',
                        global_base_gain=4, radial_and_scalar_baked=True,
                        eval_rgb_used_to_fit_gains=False,
                        source_transforms_sha256=sha(args.source/'transforms.json'),
                        caution='Shared fixedk4 optical hypothesis, not a measured flat-field for all66 cameras')
        (args.output/kind/'color_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
    sheet.save(args.output/'source_comparison.jpg', quality=94)
    print(json.dumps(dict(output=str(args.output), images=len(rows), preview=args.preview)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preview', action='store_true')
    build(p.parse_args())
