#!/usr/bin/env python3
"""
Detect serious structural artifacts (broken/dislocated thin structures, floaters,
holes) between a GT render and a candidate render, via local-SSIM connected
components.

Outputs, three sensitivity tiers
--------------------------------
1. BINARY CLASSIFIER (strict): serious=True iff a contiguous blob of severe error
   (local SSIM < SSIM_SEVERE) has area >= AREA_SERIOUS. Tuned on a real eval
   triptych: candidate (broken stand) -> True; instant_ngp (diffuse blur) -> False.
2. DEFECT BOXES (medium): severe blobs with area >= AREA_BOX get bounding boxes,
   so small holes in thin structures (~150px) are still localized. Blobs
   >= AREA_SERIOUS are "major" (red), smaller "minor" (orange).
3. SUSPICION MAP (soft): pixels with local SSIM < SSIM_SUSPECT in connected
   regions >= AREA_SUSPECT. More false positives by design; guaranteed visual
   superset of everything questionable.

Scalar metric for an LLM judge
------------------------------
artifact_score = 1000 * sum_i(area_i * mean_err_i) / frame_px
over QUALIFYING blobs only (area >= AREA_BOX at the severe pixel threshold).
One number, monotone in count AND size AND severity of significant artifacts;
0.0 when nothing passes both thresholds; resolution-independent. Compare runs
of the same scene: lower = fewer/smaller significant artifacts.

Why local SSIM, not L2/PSNR: candidate has a global tone shift (L2 fires
everywhere) and thin breaks are tiny in pixel count (PSNR barely moves).
Why largest connected component, not mean error: uniform blur spreads error
thinly (high mean, no big blob); a break/hole is a dense concentrated failure.
"""
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ----- tuned defaults (candidate -> True, instant_ngp -> False) --------------
SSIM_SEVERE = 0.40   # pixel is "severe" if local SSIM below this
AREA_SERIOUS = 250   # blob area (px) for the binary serious verdict
AREA_BOX = 120       # blob area (px) to get a bounding box / enter the score
SEV_MIN = 0.70       # min MEAN severity of a blob to count as significant:
                     # kills soft/diffuse blobs (blur-style degradation) while
                     # hard failures (breaks/holes, mean err ~0.73+) pass
SSIM_SUSPECT = 0.50  # softer pixel threshold for the suspicion map
AREA_SUSPECT = 60    # softer area threshold for the suspicion map

# region tuple layout: (area, x0, y0, x1, y1, mean_severity)


def structural_error(gt, cand, win=7):
    g, c = rgb2gray(gt), rgb2gray(cand)
    _, smap = ssim(g, c, win_size=win, data_range=1.0, full=True)
    return np.clip(1.0 - smap, 0, 1)


def _components(err, err_thresh, min_area, close_size):
    mask = err > err_thresh
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((close_size,) * 2))
    lbl, n = ndi.label(mask)
    regions, kept = [], np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = lbl == i
        area = int(comp.sum())
        if area >= min_area:
            ys, xs = np.where(comp)
            regions.append((area, int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max()),
                            float(err[comp].mean())))
            kept |= comp
    regions.sort(reverse=True)
    return regions, kept


def detect_defects(gt, cand, *, ssim_severe=SSIM_SEVERE,
                   area_serious=AREA_SERIOUS, area_box=AREA_BOX,
                   sev_min=SEV_MIN,
                   ssim_suspect=SSIM_SUSPECT, area_suspect=AREA_SUSPECT,
                   close_size=7, win=7):
    """Full analysis. Returns dict with: serious (bool), artifact_score (float),
    artifact_count (int), major/minor region lists, suspicion mask, error map."""
    err = structural_error(gt, cand, win=win)
    regions, _ = _components(err, 1.0 - ssim_severe, area_box, close_size)
    regions = [r for r in regions if r[5] >= sev_min]   # severity gate
    major = [r for r in regions if r[0] >= area_serious]
    minor = [r for r in regions if r[0] < area_serious]
    suspect_regions, suspect_mask = _components(
        err, 1.0 - ssim_suspect, area_suspect, close_size=5)
    score = 1000.0 * sum(a * sev for a, _, _, _, _, sev in regions) / err.size
    return dict(serious=bool(major),
                artifact_score=round(score, 3),
                artifact_count=len(regions),
                largest_area=regions[0][0] if regions else 0,
                major=major, minor=minor,
                suspect_mask=suspect_mask, suspect_regions=suspect_regions,
                error_map=err)


def artifact_score(gt, cand, **kw):
    """One number for an LLM judge. Severity-weighted, frame-normalized area of
    all SIGNIFICANT artifact blobs (the ones that would get a bounding box:
    area >= AREA_BOX at local SSIM < SSIM_SEVERE). 0.0 = no significant
    artifacts. Monotone in count, size and severity; compare across runs of
    the same scene -- lower is better."""
    return detect_defects(gt, cand, **kw)["artifact_score"]


# ----------------------------- io / viz --------------------------------------
def load_pair(args, gt_idx, cand_idx):
    if args.gt_file and args.cand_file:
        gt = np.asarray(Image.open(args.gt_file).convert("RGB"))
        cand = np.asarray(Image.open(args.cand_file).convert("RGB"))
    else:
        im = np.asarray(Image.open(args.image).convert("RGB"))
        pw = im.shape[1] // args.panels
        gt = im[:, gt_idx * pw:(gt_idx + 1) * pw]
        cand = im[:, cand_idx * pw:(cand_idx + 1) * pw]
    t, b = args.crop_top, args.crop_bottom
    gt = gt[t:gt.shape[0] - b]; cand = cand[t:cand.shape[0] - b]
    h = min(gt.shape[0], cand.shape[0]); w = min(gt.shape[1], cand.shape[1])
    return gt[:h, :w], cand[:h, :w]


def save_heatmap(gt, err, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(gt); ax[0].set_title("GT"); ax[0].axis("off")
    ax[1].imshow(gt)
    hm = ax[1].imshow(err, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax[1].set_title("structural error (1 - local SSIM)"); ax[1].axis("off")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_suspicion(cand, res, path, label):
    overlay = cand.copy().astype(float)
    m = res["suspect_mask"]
    overlay[m] = overlay[m] * 0.35 + np.array([255, 0, 0]) * 0.65
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay.astype(np.uint8)); ax.axis("off")
    for area, x0, y0, x1, y1, _ in res["suspect_regions"]:
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor="yellow", linewidth=1.0, linestyle="--"))
    ax.set_title(f"{label}: suspicion map (SSIM<{SSIM_SUSPECT}, area>={AREA_SUSPECT}px)"
                 f" - {len(res['suspect_regions'])} regions")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_boxes(cand, res, path, label):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cand); ax.axis("off")
    verdict = "SERIOUS ARTIFACT" if res["serious"] else "clean"
    ax.set_title(f"{label}: {verdict}  score={res['artifact_score']}  "
                 f"(largest blob = {res['largest_area']} px)",
                 color="red" if res["serious"] else "green")
    for rank, (area, x0, y0, x1, y1, _) in enumerate(res["major"], 1):
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor="red", linewidth=2.5))
        ax.text(x0, y0 - 4, f"#{rank}: {area}px", color="red", fontsize=10,
                weight="bold")
    for rank, (area, x0, y0, x1, y1, _) in enumerate(res["minor"],
                                                     len(res["major"]) + 1):
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor="orange", linewidth=2.0))
        ax.text(x0, y0 - 4, f"#{rank}: {area}px", color="orange", fontsize=9,
                weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--gt-file"); ap.add_argument("--cand-file")
    ap.add_argument("--panels", type=int, default=3)
    ap.add_argument("--gt", type=int, default=0)
    ap.add_argument("--cand", type=int, default=2)
    ap.add_argument("--also", type=int, default=None,
                    help="extra panel index to evaluate (e.g. instant_ngp=1)")
    ap.add_argument("--crop-top", type=int, default=0)
    ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--out", default="defect")
    args = ap.parse_args()

    gt, cand = load_pair(args, args.gt, args.cand)
    res = detect_defects(gt, cand)
    save_heatmap(gt, res["error_map"], f"{args.out}_heatmap.png")
    save_boxes(cand, res, f"{args.out}_boxes.png", "candidate")
    save_suspicion(cand, res, f"{args.out}_suspicion.png", "candidate")
    print(f"[candidate] serious={res['serious']}  "
          f"artifact_score={res['artifact_score']}  count={res['artifact_count']}  "
          f"largest={res['largest_area']}px")
    print(f"  major: {[r[:5] for r in res['major']]}")
    print(f"  minor: {[r[:5] for r in res['minor']]}")

    if args.also is not None:
        _, neg = load_pair(args, args.gt, args.also)
        rn = detect_defects(gt, neg)
        save_boxes(neg, rn, f"{args.out}_neg_boxes.png", "instant_ngp")
        save_suspicion(neg, rn, f"{args.out}_neg_suspicion.png", "instant_ngp")
        print(f"[panel {args.also}] serious={rn['serious']}  "
              f"artifact_score={rn['artifact_score']}  count={rn['artifact_count']}  "
              f"largest={rn['largest_area']}px")

    # sanity: identical images must score 0
    res0 = detect_defects(gt, gt)
    print(f"[gt vs gt sanity] serious={res0['serious']}  "
          f"artifact_score={res0['artifact_score']}")


if __name__ == "__main__":
    main()
