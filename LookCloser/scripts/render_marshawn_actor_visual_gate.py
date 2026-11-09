"""Full-raster actor checkpoint review, including pixels outside semantic masks.

No optimization, target compositing, color transform, or semantic ray filtering.
Core/+1 are inference sensitivity variants of the SAME immutable checkpoint.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, PngImagePlugin


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_png(path, array):
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite output')
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b'sRGB', b'\0')
    metadata.add_text('Encoding', 'Encoded PNG model output; no extra exposure/PQ/tone transform')
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path, pnginfo=metadata)


def main(args):
    root = Path('/fsx/tmp/marshawn_001660_hd')
    paths = [args.config.parent.parent / 'code', root / 'code_native',
             root / 'code_sfm/LookCloser/scripts', root / 'final_pair/code', Path(__file__).parent]
    sys.path[:0] = [str(p) for p in paths]
    import torch
    import torch.nn.functional as functional
    from nerfstudio.utils.eval_utils import eval_setup
    from probe_marshawn_occupancy import digest_model
    from marshawn_aabb_overlay import overlay_box

    torch.set_num_threads(2)
    checkpoint_hash = sha(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')

    def configure(cfg):
        cfg.eval_checkpoint = args.checkpoint.resolve()
        cfg.load_checkpoint = cfg.load_dir = cfg.load_step = cfg.load_config = None
        return cfg

    cfg, pipeline, loaded, step = eval_setup(args.config, test_mode='val',
        eval_num_rays_per_chunk=args.chunk, update_config_callback=configure)
    if loaded.resolve() != args.checkpoint.resolve():
        raise ValueError('Loaded a different checkpoint')
    pipeline.eval()
    model = pipeline.model
    model.current_train_step = step
    if model.renderer_rgb.background_color != 'black':
        raise ValueError('This actor PNG review requires the fixed black background')
    dataset = pipeline.datamanager.eval_dataset
    names = [Path(p).stem.split('_')[-1] for p in dataset.image_filenames]
    if set(names) != {'DC', 'HC', 'KA'} or len(names) != 3:
        raise ValueError('Exactly the three actual eval cameras required')
    args.output.mkdir(parents=True, exist_ok=False)
    before = digest_model(model)
    core = model.occupancy_grid.binaries.clone()
    if core.ndim != 4 or core.dtype != torch.bool:
        raise ValueError('Expected native cascade XYZ binary grid')
    bounds = model.field.aabb.detach().cpu().double().numpy()
    configured = np.asarray(cfg.pipeline.datamanager.dataparser.marshawn_aabb, dtype=np.float64)
    grid_bounds = model.occupancy_grid.aabbs.detach().cpu().double().numpy().reshape(-1, 2, 3)
    if (not np.allclose(bounds, configured, rtol=0, atol=1e-6)
            or len(grid_bounds) != 1 or not np.allclose(grid_bounds[0], bounds, rtol=0, atol=1e-6)):
        raise ValueError('Configured actor, field and single bounded grid AABBs must agree')
    rows = []
    try:
        for variant in args.variants:
            active = core if variant == 'core' else functional.max_pool3d(
                core.float()[:, None], kernel_size=3, stride=1, padding=1)[:, 0].bool()
            model.occupancy_grid.binaries.copy_(active)
            folder = args.output / variant
            folder.mkdir()
            for index, name in enumerate(names):
                camera = copy.deepcopy(dataset.cameras[index:index + 1]).to(model.device)
                w, h = int(camera.width.item()), int(camera.height.item())
                if (w, h) != (1920, 1012):
                    raise ValueError('Expected the genuine current HD calibrated raster')
                started = time.monotonic()
                with torch.inference_mode():
                    outputs = model.get_outputs_for_camera(camera)
                rgb = outputs['rgb'].detach().float().cpu().numpy()
                alpha = outputs['accumulation'].detach().float().cpu().numpy()
                if rgb.shape != (h, w, 3) or alpha.shape != (h, w, 1):
                    raise ValueError('Full-raster output shape mismatch')
                save_png(folder / (name + '.png'), rgb)
                save_png(folder / (name + '_alpha.png'), alpha[..., 0])
                np.savez_compressed(folder / (name + '_raw.npz'), rgb=rgb, alpha=alpha,
                    depth=outputs['depth'].detach().float().cpu().numpy())
                source = Image.open(dataset.image_filenames[index]).convert('RGB')
                if source.size != (w, h):
                    raise ValueError('Source/calibrated raster mismatch')
                c = dict(w=w, h=h, fl_x=float(camera.fx.item()), fl_y=float(camera.fy.item()),
                    cx=float(camera.cx.item()), cy=float(camera.cy.item()),
                    transform_matrix=camera.camera_to_worlds[0].detach().cpu().tolist())
                annotated, projection = overlay_box(source, bounds, c,
                    label=name + ' full source; all 12 AABB edges visible')
                annotated.save(folder / (name + '_source_box.jpg'), quality=95)
                truth = source.copy()
                prediction = Image.open(folder / (name + '.png')).convert('RGB')
                truth.thumbnail((960, 506))
                prediction.thumbnail((960, 506))
                contact = Image.new('RGB', (1920, 506))
                contact.paste(truth, (0, 0))
                contact.paste(prediction, (960, 0))
                contact.save(folder / (name + '_source_prediction.jpg'), quality=95)
                row = dict(camera=name, variant=variant, seconds=time.monotonic() - started,
                    rendered_pixels=w*h, semantic_filter=False, source=str(dataset.image_filenames[index]),
                    source_sha256=sha(dataset.image_filenames[index]), png_sha256=sha(folder / (name + '.png')),
                    alpha_mean=float(alpha.mean()), alpha_nonzero_pixels=int((alpha > 0).sum()),
                    box_projection=projection)
                rows.append(row)
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(json.dumps({k: row[k] for k in ['camera', 'variant', 'seconds', 'rendered_pixels']}), flush=True)
    finally:
        model.occupancy_grid.binaries.copy_(core)
    if digest_model(model) != before or sha(args.checkpoint) != checkpoint_hash:
        raise ValueError('Protected model/checkpoint changed during inference')
    receipt = dict(kind='FULL_RASTER_ACTOR_VISUAL_GATE', checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash, step=step, config_sha256=sha(args.config),
        source_sha256=sha(__file__), rows=rows, aabb=bounds.tolist(), model_state_before_after=before,
        no_optimizer=True, no_color_transform=True, halo_is_same_checkpoint_inference_only=True,
        no_claim_of_occupancy_correctness=True)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'output']:
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--variants', nargs='+', choices=['core', 'halo1'], default=['core', 'halo1'])
    main(parser.parse_args())
