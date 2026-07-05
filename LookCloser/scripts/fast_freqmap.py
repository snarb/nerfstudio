#!/usr/bin/env python3
"""Fast LookCloser 2D frequency-map preprocessing.

Additive fast path. Same map math as the reference
train_progressive_and_estimate_frequency_map:
  - fresh InstantNGP2D per image
  - per-level progressive training with render_masked (active prefix of levels)
  - SSIM-threshold patch assignment at each level, first level to cross wins
  - unresolved -> max level
  - freq_map = get_resolution_at_level(level)

Speedups that DO NOT change the math:
  - TF32 fast math on Blackwell
  - GT patches precomputed once via unfold (no per-batch python loops)
  - eval/assign fully vectorized on GPU with a large patch batch
  - level-start coords held as GPU tensors

Tunable (validate quality): --steps-per-level, --train-batch-size, --amp.

Usage:
  fast_freqmap.py --images-dir <dir> --glob 'frame_train_*.jpg' --out <dir> \
     [--steps-per-level 1000] [--train-batch-size 8192] [--eval-patch-batch 8192] \
     [--amp] [--limit N] [--max-res 8192]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import sys, os
# Allow running from either the brans repo or the orchestrator repo checkout.
for _p in ("/home/brans/repos/nerfstudio",
           os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))):
    if os.path.isdir(os.path.join(_p, "nerfstudio")) and _p not in sys.path:
        sys.path.insert(0, _p)
from nerfstudio.scripts.lookcloser_preprocess import (
    InstantNGP2D, load_image_as_tensor, compute_patch_starts, compute_ssim,
)

MIN_RES = 16
N_LEVELS = 16
N_FEAT = 2
LOG2_HASH = 23


def precompute_gt_patches(img, y_starts, x_starts, ps):
    """Return (hs, ws, 3, ps, ps) GT patches via unfold. Exact same pixels as
    extract_gt_patches (top-left aligned, stride=ps, no tail)."""
    h, w, _ = img.shape
    chw = img.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    # unfold gives patches at stride=ps starting at 0; matches compute_patch_starts
    patches = F.unfold(chw, kernel_size=ps, stride=ps)  # 1, 3*ps*ps, L
    L = patches.shape[-1]
    hs, ws = len(y_starts), len(x_starts)
    assert L == hs * ws, f"unfold L={L} != {hs*ws}"
    patches = patches.view(3, ps, ps, hs, ws).permute(3, 4, 0, 1, 2).contiguous()
    return patches  # hs, ws, 3, ps, ps


def make_patch_uv_vec(xs, ys, h, w, ps):
    device = xs.device
    lx = torch.arange(ps, device=device, dtype=torch.float32) + 0.5
    ly = torch.arange(ps, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ly, lx, indexing="ij")
    x = xx.unsqueeze(0) + xs.float().view(-1, 1, 1)
    y = yy.unsqueeze(0) + ys.float().view(-1, 1, 1)
    uv = torch.stack([x / float(w), y / float(h)], dim=-1)
    return uv.view(-1, 2)


def process_image(img, steps_per_level, batch_size, eval_patch_batch, max_res,
                  ps=8, ssim_thr=0.95, win=7, lr=1e-2, amp=False):
    dev = img.device
    h, w, _ = img.shape
    y_starts = compute_patch_starts(h, ps, ps)
    x_starts = compute_patch_starts(w, ps, ps)
    hs, ws = len(y_starts), len(x_starts)

    model = InstantNGP2D(N_LEVELS, N_FEAT, MIN_RES, max_res, LOG2_HASH).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-15)

    level_map = torch.full((hs, ws), -1, dtype=torch.int16, device=dev)

    gt_patches = precompute_gt_patches(img, y_starts, x_starts, ps)  # hs,ws,3,ps,ps
    ys_grid = torch.tensor(y_starts, device=dev, dtype=torch.long)
    xs_grid = torch.tensor(x_starts, device=dev, dtype=torch.long)

    def eval_assign(level):
        unresolved = (level_map < 0).nonzero(as_tuple=False)
        if unresolved.numel() == 0:
            return
        model.eval()
        with torch.no_grad():
            for s in range(0, unresolved.shape[0], eval_patch_batch):
                idxs = unresolved[s:s + eval_patch_batch]
                iy = idxs[:, 0]; ix = idxs[:, 1]
                ys = ys_grid[iy]; xs = xs_grid[ix]
                gt = gt_patches[iy, ix]  # B,3,ps,ps
                uv = make_patch_uv_vec(xs, ys, h, w, ps)
                pred = model.render_masked(uv, level).view(idxs.shape[0], ps, ps, 3).permute(0, 3, 1, 2).contiguous().float()
                scores = compute_ssim(gt.float(), pred, window_size=win, size_average=False)
                ok = scores >= ssim_thr
                if ok.any():
                    oi = idxs[ok]; level_map[oi[:, 0], oi[:, 1]] = level
        model.train()

    model.train()
    for level in range(N_LEVELS):
        for _ in range(steps_per_level):
            iy = torch.randint(0, h, (batch_size,), device=dev)
            ix = torch.randint(0, w, (batch_size,), device=dev)
            target = img[iy, ix]
            uv = torch.stack([(ix.float() + 0.5) / w, (iy.float() + 0.5) / h], dim=-1)
            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model.render_masked(uv, level)
                    loss = F.mse_loss(pred, target.to(pred.dtype))
            else:
                pred = model.render_masked(uv, level)
                loss = F.mse_loss(pred, target.to(pred.dtype))
            loss.backward()
            opt.step()
        eval_assign(level)
        if (level_map >= 0).all():
            break

    level_map[level_map < 0] = N_LEVELS - 1
    freq_map = torch.empty((hs, ws), dtype=torch.float32, device=dev)
    for lvl in range(N_LEVELS):
        freq_map[level_map == lvl] = model.get_resolution_at_level(lvl)
    del model, opt
    return freq_map.detach().cpu()


def save_metadata(path, image_name, image_shape, ps, max_res):
    b = float(np.exp((np.log(max_res) - np.log(MIN_RES)) / (N_LEVELS - 1)))
    data = {
        "image": image_name, "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "crop_coords_xywh": None, "value_type": "scalar_resolution",
        "patch_size": int(ps), "stride": int(ps), "min_res": int(MIN_RES),
        "max_res": int(max_res), "n_levels": N_LEVELS, "n_features": N_FEAT,
        "log2_hashmap_size": LOG2_HASH, "per_level_scale": b,
        "level_resolution_schedule": [float(MIN_RES * (b ** l)) for l in range(N_LEVELS)],
    }
    path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--file-list", type=Path, default=None,
                    help="Text file with one image path per line (overrides images-dir/glob).")
    ap.add_argument("--glob", default="frame_train_*.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps-per-level", type=int, default=1000)
    ap.add_argument("--train-batch-size", type=int, default=8192)
    ap.add_argument("--eval-patch-batch", type=int, default=8192)
    ap.add_argument("--max-res", type=int, default=8192)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--ssim-threshold", type=float, default=0.95)
    ap.add_argument("--ssim-window", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true", default=True)
    a = ap.parse_args()

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.file_list is not None:
        paths = [Path(l.strip()) for l in a.file_list.read_text().splitlines() if l.strip()]
    else:
        paths = sorted(a.images_dir.glob(a.glob))
    if a.limit:
        paths = paths[:a.limit]
    print(f"processing {len(paths)} images -> {a.out}", flush=True)

    t0 = time.time()
    for i, p in enumerate(paths):
        ts = time.time()
        img = load_image_as_tensor(p, dev)
        fm = process_image(img, a.steps_per_level, a.train_batch_size, a.eval_patch_batch,
                            a.max_res, a.patch_size, a.ssim_threshold, a.ssim_window, lr=a.lr, amp=a.amp)
        torch.save(fm, a.out / f"{p.stem}.pt")
        save_metadata((a.out / f"{p.stem}.json"), p.name, (img.shape[0], img.shape[1]), a.patch_size, a.max_res)
        del img
        torch.cuda.empty_cache()
        print(f"  [{i+1}/{len(paths)}] {p.stem} {time.time()-ts:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"DONE {len(paths)} imgs in {dt:.1f}s -> {dt/max(len(paths),1):.2f}s/img", flush=True)


if __name__ == "__main__":
    main()
