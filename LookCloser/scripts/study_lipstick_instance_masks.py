"""Private train-only lipstick instance-mask pilot; never edits geometry.

Prepare native RGB crops around the previously diagnosed 000995 fin. Prompts
are explicit reviewed train-image coordinates, not independent shape truth.
SAM candidates remain unaccepted until an actual visual review is recorded.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
from PIL import Image, ImageDraw

from joint_temporal_texture import atomic_json, cameras, project, read, sha

ROOT = Path('/mnt/data/dec5_lipstick_instance_mask_000995')
SAM = Path('/home/brans/lookcloser_temp/sam2_lipstick_20260915')
DEPS = Path('/home/brans/lookcloser_temp/sam2_lipstick_deps_20260915')
CHECKPOINT = SAM / 'checkpoints/sam2.1_hiera_large.pt'
COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
FRAME = '000995'
PREFIXES = ['H004_C', 'K004_B', 'I004_C', 'J004_C', 'J004_A']


def portrait_xy(uv, width):
    uv = np.asarray(uv)
    return np.stack((uv[..., 1], width - 1 - uv[..., 0]), axis=-1)


def binary_masks(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected thresholded SAM masks, not unthresholded logits')
    return values.astype(bool)


def prepare():
    from calibrated_depth_witness import load_images
    from diagnose_lipstick_fin_depth import ROOT as DIAG
    assert not ROOT.exists(), 'Keep any existing/failed pilot intact'
    evidence = read(DIAG / 'result.json')
    representative = evidence['representative_triangle']
    point = next(r['centroid'] for r in evidence['triangles'] if r['triangle'] == representative)
    rows, _, _ = cameras(FRAME)
    images, _, rgb_receipt = load_images(FRAME)
    ROOT.mkdir()
    shutil.copyfile(__file__, ROOT / 'prepared_script.py')
    records = []
    for prefix in PREFIXES:
        matches = [r for r in rows if r['physical_camera'].startswith(prefix)]
        assert len(matches) == 1
        row = matches[0]; name = row['physical_camera']
        uv, z = project(np.array(point)[None], [row]); assert z[0, 0] > 0
        xy = np.rint(portrait_xy(uv[0, 0], row['w'])).astype(int)
        box = [int(xy[0]-120), int(xy[1]-150), int(xy[0]+120), int(xy[1]+150)]
        source = Image.fromarray(np.rot90(images[name]))
        assert 0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height
        crop = source.crop(box); path = ROOT / (prefix + '.png'); crop.save(path)
        # Explicitly enlarged review only. Inference retains native crop pixels.
        crop.resize((480, 600), Image.Resampling.NEAREST).save(ROOT / (prefix + '_review.png'))
        records.append(dict(camera=name, camera_parameters=row, crop=box,
                            image=str(path), image_sha256=sha(path)))
    atomic_json(ROOT / 'request.json', dict(frame=FRAME, rgb_receipt=rgb_receipt,
        diagnostic_result_sha256=sha(DIAG / 'result.json'), diagnostic_center=point,
        views=records, heldout_used=False, geometry_changed=False,
        manually_prompted_semantic_prior=True, masks_independent_geometry_truth=False,
        script_sha256=sha(__file__)))
    print('Prepared', len(records), 'native train crops', flush=True)


def predict(prompts_path, output):
    assert not output.exists(), 'Use a new output for every prompt/model trial'
    request = read(ROOT / 'request.json'); prompts = read(prompts_path)
    assert request['script_sha256'] == sha(ROOT / 'prepared_script.py')
    assert subprocess.check_output(['git', '-C', str(SAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    sys.path[:0] = [str(DEPS), str(SAM)]
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(2); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
        str(CHECKPOINT), device='cuda', apply_postprocessing=False), max_hole_area=0, max_sprinkle_area=0)
    output.mkdir(parents=True)
    records = []
    assert set(prompts) == {r['camera'] for r in request['views']}
    for row in request['views']:
        path = Path(row['image']); assert sha(path) == row['image_sha256']
        image = np.array(Image.open(path).convert('RGB')); h, w = image.shape[:2]
        spec = prompts[row['camera']]
        points = np.asarray(spec['points'], np.float32); labels = np.asarray(spec['labels'], np.int32)
        assert points.shape == (len(labels), 2) and set(labels).issubset({0, 1}) and (labels == 1).any()
        assert np.isfinite(points).all() and ((points >= 0) & (points < [w, h])).all()
        box = np.asarray(spec['box'], np.float32)
        assert box.shape == (4,) and 0 <= box[0] < box[2] < w and 0 <= box[1] < box[3] < h
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels,
                box=box, multimask_output=True)
        assert masks.shape == (3, h, w) and np.isfinite(scores).all() and np.isfinite(logits).all()
        masks = binary_masks(masks)
        folder = output / row['camera']; folder.mkdir()
        canvas = Image.new('RGB', (w*4, h+25)); draw = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(image), (0, 25)); draw.text((3, 3), 'actual train RGB', fill='white')
        for i, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8)*255).save(folder / f'mask_{i}.png')
            overlay = image.copy(); overlay[mask] = (image[mask]*.55 + np.array([255, 0, 255])*.45).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay), ((i+1)*w, 25))
            draw.text(((i+1)*w+3, 3), f'{i}: SAM score {scores[i]:.4f}', fill='white')
        canvas.save(folder / 'review.png')
        np.savez_compressed(folder / 'candidates.npz', masks=masks, scores=scores, logits=logits)
        records.append(dict(camera=row['camera'], scores=scores.tolist(), areas=masks.sum((1, 2)).tolist(),
            outputs={str(p.relative_to(output)): sha(p) for p in folder.iterdir() if p.is_file()}))
        print(row['camera'], records[-1]['areas'], flush=True)
    atomic_json(output / 'result.json', dict(request_sha256=sha(ROOT / 'request.json'),
        prompts_path=str(prompts_path), prompts_sha256=sha(prompts_path), script_sha256=sha(__file__),
        sam_commit=COMMIT, checkpoint_path=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
        postprocessing=False, dtype='bfloat16', tf32=False, views=records,
        visual_status='pending', geometry_changed=False, heldout_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'predict'])
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    else:
        assert args.prompts and args.output
        predict(args.prompts.resolve(), args.output.resolve())
