"""Render fixed native diagnostic cameras from an exact owned checkpoint.

Set PYTHONPATH to the experiment overlay and frozen nerfstudio sources. This
utility performs inference only; it never changes a training checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def render_views(model, manifest_path, output):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from nerfstudio.cameras.cameras import Cameras, CameraType

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    views = manifest.get('views', manifest.get('frames'))
    if not isinstance(views, list) or not views:
        raise ValueError('Nonempty fixed-camera manifest required')
    output.mkdir(parents=True, exist_ok=True)
    receipts, thumbnails = [], []
    model.eval()
    for index, row in enumerate(views):
        name = row.get('name', row.get('camera', row.get('physical_camera', f'view_{index:02d}')))
        if not isinstance(name, str) or not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unsafe camera name')
        width, height = int(row['w']), int(row['h'])
        if not 32 <= min(width, height) or max(width, height) > 8192:
            raise ValueError('Unexpected diagnostic raster')
        pose = torch.tensor(row['transform_matrix'], dtype=torch.float32)[:3, :4]
        camera = Cameras(camera_to_worlds=pose[None], fx=float(row['fl_x']),
                         fy=float(row['fl_y']), cx=float(row['cx']), cy=float(row['cy']),
                         width=width, height=height, camera_type=CameraType.PERSPECTIVE).to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            result = model.get_outputs_for_camera(camera)
        rgb, alpha = result['rgb'].float(), result['accumulation'].float()
        if tuple(rgb.shape) != (height, width, 3) or not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(alpha).all()):
            raise ValueError('Invalid diagnostic output')
        image_path, alpha_path = output / f'{name}.png', output / f'{name}_alpha.png'
        pixels = (rgb.clamp(0, 1).cpu().numpy() * 255 + .5).astype(np.uint8)
        Image.fromarray(pixels).save(image_path)
        Image.fromarray((alpha[..., 0].clamp(0, 1).cpu().numpy() * 65535 + .5).astype(np.uint16)).save(alpha_path)
        thumb = Image.fromarray(pixels)
        thumb.thumbnail((480, 640), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (480, 670), (32, 32, 32))
        panel.paste(thumb, ((480-thumb.width)//2, 25))
        ImageDraw.Draw(panel).text((8, 8), name, fill='white')
        thumbnails.append(panel)
        receipts.append(dict(name=name, width=width, height=height,
                             seconds=time.monotonic()-started, rgb_sha256=sha(image_path),
                             alpha_sha256=sha(alpha_path), no_resizing=True))
    sheet = Image.new('RGB', (480*len(thumbnails), 670))
    for index, thumb in enumerate(thumbnails):
        sheet.paste(thumb, (480*index, 0))
    sheet.save(output / 'contact.jpg', quality=92)
    report = dict(status='rendered_requires_visual_review', manifest_sha256=sha(manifest_path),
                  no_novel_view_GT_metrics=True, views=receipts)
    atomic_json(output / 'renders.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=8192)
    args = parser.parse_args()
    runtime = json.loads((args.run / 'runtime.json').read_text())
    config = Path(runtime['config'])
    if args.checkpoint is None:
        latest = json.loads((args.run / 'latest.json').read_text())
        checkpoint = args.run / latest['checkpoint']
        if sha(checkpoint) != latest['checkpoint_sha256']:
            raise ValueError('Latest checkpoint changed or is corrupt')
    else:
        checkpoint = args.checkpoint.resolve()
    if checkpoint.parent.resolve() != (args.run / 'checkpoints').resolve():
        raise ValueError('Only this run\'s checkpoints may be rendered')
    from nerfstudio.utils.eval_utils import eval_setup
    def exact_checkpoint(cfg):
        cfg.eval_checkpoint = checkpoint
        return cfg
    _, pipeline, loaded, step = eval_setup(config, args.chunk, test_mode='inference',
                                          update_config_callback=exact_checkpoint)
    if loaded.resolve() != checkpoint.resolve():
        raise ValueError('Wrong checkpoint loaded')
    report = render_views(pipeline.model, args.manifest, args.output)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), step=int(step),
                  config_sha256=sha(config), script_sha256=sha(__file__))
    atomic_json(args.output / 'complete.json', report)


if __name__ == '__main__':
    main()
