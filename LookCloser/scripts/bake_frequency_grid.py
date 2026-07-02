#!/usr/bin/env python
"""Bake a 3D FrequencyGridManager grid from a trained temporal-NGP checkpoint + the existing 2D
frequency maps, so ARM gets a REAL per-scene frequency level (not the constant fallback).

For every TRAIN image we: load its 2D frequency map (scalar_resolution [Hm,Wm]) + json sidecar
(patch_size/stride/image_shape); subsample pixels; render DEPTH from the trained model; compute the
world surface point; look up f_2d at the pixel's patch; and scatter-MAX the resulting frequency level
into the grid (union over all frames — cameras are static, so this captures the max frequency demand
per 3D cell). Mirrors FrequencyGridManager.update_step / the LookCloser pipeline math, but offline.

Usage (bake):
  python LookCloser/scripts/bake_frequency_grid.py \
    --config <run>/config.yml --output <grid.pt> \
    --resolution 128 --min-res 16 --max-res 8192 --num-levels 16 \
    --freq-map-dir lookcloser_frequencies --pixels-per-image 20000

CPU logic check (no GPU / no checkpoint):
  python LookCloser/scripts/bake_frequency_grid.py --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager


def self_test() -> None:
    """CPU-only sanity check of the grid update/query math."""
    sb = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    grid = FrequencyGridManager(scene_box=sb, resolution=8, num_levels=16, min_res=16.0, max_res=8192.0)
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.5, 0.25]])
    # A high scalar resolution -> a high level; push it in and read it back.
    f2d = torch.tensor([[8192.0], [512.0]])
    focal = torch.tensor([[1000.0], [1000.0]])
    depth = torch.tensor([[1.0], [1.0]])
    grid.update_step(step=0, positions=pts, rendered_depth=depth, focals=focal, patch_f2d=f2d)
    got = grid.query(pts).reshape(-1)
    exp = grid.freq_to_level(f2d * (focal / (depth + 1e-6))).reshape(-1)
    assert torch.allclose(got, exp), f"query {got} != expected {exp}"
    # Untouched voxel stays 0.
    assert float(grid.query(torch.tensor([[-0.99, -0.99, -0.99]]))[0, 0]) == 0.0
    print(f"self-test OK: levels {got.tolist()} (expected {exp.tolist()}), non-empty={int((grid.grid>0).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run the CPU logic check and exit.")
    ap.add_argument("--config", type=Path, default=None, help="Trained run config.yml (eval_setup).")
    ap.add_argument("--output", type=Path, default=None, help="Where to write the baked grid .pt.")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--min-res", type=float, default=16.0)
    ap.add_argument("--max-res", type=float, default=8192.0)
    ap.add_argument("--num-levels", type=int, default=16)
    ap.add_argument("--freq-map-dir", type=str, default="lookcloser_frequencies",
                    help="Sub-dir under the dataset holding frame_*.pt/.json maps.")
    ap.add_argument("--pixels-per-image", type=int, default=20000)
    ap.add_argument("--max-images", type=int, default=None, help="Limit #train images (debug).")
    ap.add_argument("--chunk", type=int, default=16384, help="Rays per depth-render chunk.")
    ap.add_argument("--patch-size", type=int, default=None, help="Override sidecar patch_size.")
    ap.add_argument("--stride", type=int, default=None, help="Override sidecar stride.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    assert args.config is not None and args.output is not None, "--config and --output are required for baking."
    torch.manual_seed(args.seed)

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.model_components.ray_generators import RayGenerator

    _config, pipeline, _ckpt, _step = eval_setup(args.config, test_mode="inference")
    device = pipeline.device
    model = pipeline.model
    model.eval()

    dataset = pipeline.datamanager.train_dataset
    cameras = dataset.cameras
    filenames = dataset.image_filenames
    data_dir = Path(_config.pipeline.datamanager.dataparser.data)
    map_dir = data_dir / args.freq_map_dir

    sb = SceneBox(aabb=model.scene_box.aabb.detach().cpu())
    grid = FrequencyGridManager(
        scene_box=sb, resolution=args.resolution, num_levels=args.num_levels,
        min_res=args.min_res, max_res=args.max_res, enabled=True,
    ).to(device)

    ray_gen = RayGenerator(cameras.to(device))

    n_images = len(filenames) if args.max_images is None else min(args.max_images, len(filenames))
    used, skipped = 0, 0
    for idx in range(n_images):
        stem = Path(filenames[idx]).stem
        map_path = map_dir / f"{stem}.pt"
        json_path = map_dir / f"{stem}.json"
        if not map_path.exists():
            skipped += 1
            continue
        fmap = torch.load(map_path, map_location="cpu").float()  # (Hm, Wm) scalar resolution
        patch_size = args.patch_size
        stride = args.stride
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            patch_size = patch_size or int(meta.get("patch_size", 8))
            stride = stride or int(meta.get("stride", meta.get("patch_size", 8)))
        patch_size = patch_size or 8
        stride = stride or patch_size

        img_h = int(cameras.image_height[idx].item())
        img_w = int(cameras.image_width[idx].item())
        hm, wm = fmap.shape
        n = min(args.pixels_per_image, img_h * img_w)
        rows = torch.randint(0, img_h, (n,))
        cols = torch.randint(0, img_w, (n,))
        # Pixel -> patch-map cell (matches the LookCloser pipeline lookup), clamped to map bounds.
        my = (rows // stride).clamp_(0, hm - 1)
        mx = (cols // stride).clamp_(0, wm - 1)
        patch_f2d = fmap[my, mx].to(device).view(-1, 1)  # (n, 1)

        indices = torch.stack([torch.full((n,), idx, dtype=torch.long), rows, cols], dim=-1)
        focal = ((cameras.fx[idx] + cameras.fy[idx]) * 0.5).to(device).view(1, 1)

        for c0 in range(0, n, args.chunk):
            sel = slice(c0, c0 + args.chunk)
            ray_bundle = ray_gen(indices[sel]).to(device)  # ray_gen indexes CPU image_coords
            with torch.no_grad():
                out = model.get_outputs(ray_bundle)
            depth = out["depth"].reshape(-1, 1)  # (m, 1), metric distance along (normalized) ray
            origins = ray_bundle.origins.reshape(-1, 3)
            dirs = ray_bundle.directions.reshape(-1, 3)
            positions = origins + dirs * depth  # (m, 3) world surface point
            grid.update_step(
                step=0, positions=positions, rendered_depth=depth,
                focals=focal.expand(depth.shape[0], 1), patch_f2d=patch_f2d[sel],
            )
        used += 1
        if used % 50 == 0:
            nz = int((grid.grid > 0).sum().item())
            print(f"[{used}/{n_images}] baked; non-empty voxels={nz} "
                  f"level[min,max]=[{float(grid.grid.min()):.1f},{float(grid.grid.max()):.1f}]", flush=True)

    nz = int((grid.grid > 0).sum().item())
    total = args.resolution ** 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "grid": grid.grid.detach().cpu(),
            "aabb_min_buf": grid.aabb_min_buf.detach().cpu(),
            "aabb_max_buf": grid.aabb_max_buf.detach().cpu(),
            "aabb_size_buf": grid.aabb_size_buf.detach().cpu(),
            "resolution": args.resolution,
            "num_levels": args.num_levels,
            "min_res": args.min_res,
            "max_res": args.max_res,
        },
        args.output,
    )
    print(f"DONE: baked grid -> {args.output}; images used={used} skipped={skipped}; "
          f"non-empty voxels={nz}/{total} ({100.0*nz/total:.2f}%); "
          f"level[min,max]=[{float(grid.grid.min()):.1f},{float(grid.grid.max()):.1f}]")


if __name__ == "__main__":
    main()
