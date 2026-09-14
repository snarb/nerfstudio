# DEC5 forearm: dual-grid clipping and out-of-frame uncertainty

## What was tested

One separately frozen follow-up to the [negative v1 transfer](dec5_forearm_plane_transfer.md),
at the same three times: 001029, 001033, and 001037. This is an inferred local
boundary plane, not a learned body model or recovered anatomical ground truth.
No DA3, COLMAP, training, or full-video job was rerun for this follow-up.

Artifact root: `/mnt/data/dec5_forearm_plane_transfer_v2`.
Helper: `scripts/study_forearm_plane_transfer_v2.py`; immutable `protocol.json`
records all source v1 hashes, polygons, and constants before candidate generation.

Only two rules change, uniformly across all three times:

- An out-of-image projection is unknown rather than a negative skin label.
  At least two distinct in-frame train skin-mask supports are required; **every
  in-frame mask disagreement still vetoes the candidate**.
- Append-only clipping checks both the exact integer observed-depth rays and
  half-pixel renderer rays in the same moving/G_A/H_A/H_C cameras. Any appended
  triangle hiding original depth by more than `0.001`, supported by at least
  three other observed maps, is removed. The unchanged limit is four passes.

Plane coefficients, original trusted anchor maps, manual inset train polygons,
100-pixel anchor distance, `0.003` trusted free-space veto, `0.002` maximum
triangle extent, 1.5-pixel return reprojection, and one-degree parallax remain
unchanged. All original vertices and triangles remain intact. No interior
measured vote is required; most added points have zero such votes. Distinct
stereo camera estimates are not statistically independent ground truth.

Real train RGB, metadata, camera profiles/exposure, and complete 62-map depth
controls are hash checked. Held-out F_B is evaluation only and cannot affect
geometry, clipping, masks, texture sources, or selection. Its actual added
forearm patch is outside the frame at these times; visible upper-forearm metrics
cannot validate the inferred missing interior. At 001037 the region remains N/A.

Native clay was reviewed before RGB. RGB deliberately uses the same raw matched
local ablation renderer as v1, **without** the published video's source-mask and
temporal source-label wrappers. It is not production-equivalent color.

## Results

| Time | Accepted plane points, v1 → v2 | Newly admitted with two skin views + one unknown | Zero measured interior votes, v2 | Added triangles, before → after dual clip |
|---|---:|---:|---:|---:|
| 001029 | 1,136 → 1,136 | 0 | 804 | 2,456 → 2,456 |
| 001033 | 5,657 → 7,585 | 1,928 | 6,996 | 15,325 → 15,311 |
| 001037 | 4,216 → 8,689 | 4,473 | 8,358 | 16,833 → 16,052 |

Both sampling lattices report **zero supported-old occlusion flags** after
clipping in all four fixed cameras at all three times. This corrects the v1
integer-grid failure without moving or deleting any original triangle. It is
not a guarantee for all continuous rays or unreviewed cameras.

Integer visible/front footprints have zero pixels outside the reviewed skin
polygons at 001029/001037. 001033 retains five H_A boundary pixels at most
`sqrt(2)` pixels outside the inset region; none is more than two pixels outside.
These are manual inset regions, not exact ground-truth silhouettes.

![001033 native moving clay](</mnt/data/dec5_forearm_plane_transfer_v2/001033/review/moving/clay_comparison_native.png>)

![001037 native moving clay](</mnt/data/dec5_forearm_plane_transfer_v2/001037/review/moving/clay_comparison_native.png>)

001029 remains the same small-hole canary. At 001033 the lower continuation
no longer ends at H_C's image boundary. At 001037 the new continuation extends
the lower patch, but the surface remains conspicuously planar and a thin
horizontal seam splits it. Larger filled area alone is not a successful shape
recovery; neither the seam nor missing cuff-side/background geometry is hidden
by the confidence statistics.

### The horizontal split is an in-frame mask veto, not clipping

It exists in the **unclipped** plane. A read-only probe of the central moving
strip `(x=310..369, y=1862..1882)` finds 248 missing pixels: 151 map to rejected
in-frame semantic candidates and 97 to neighboring accepted reference points
without moving-view triangle coverage. None maps to original-reference-hit
exclusion or a measured free-space veto. Direct projection into H_C puts 150
semantic disagreements at native rows **1917–1919**; its frozen inset polygon
ends at row 1916. G_A and H_A have no disagreement there. The one-count difference
comes from probing the continuous plane versus its nearest reference sample.
Below the image H_C becomes unknown, so the continuation resumes. Thus the
fixed inset mask leaves a three-row negative band at the FOV transition, rather
than revealing an anatomical break. No smoothing or mask adjustment was made.
The remaining broad planarity is a separate limitation of the prior.

![Read-only seam reasons: orange is semantic rejection, cyan adjacent accepted points](</mnt/data/dec5_forearm_plane_transfer_v2/001037/seam_reason_native.png>)

![Three times, fixed physical H_A: GT / original / v1 / v2](</mnt/data/dec5_forearm_plane_transfer_v2/three_time_H_A_native.png>)

Full H_A forearm-skin metrics, using the same previously frozen train regions:

| Time | Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| 001029 | Original | 21.373 | 0.82678 | 0.23121 |
| 001029 | V2 | 27.708 | 0.86383 | 0.14286 |
| 001033 | Original | 13.602 | 0.56413 | 0.53149 |
| 001033 | V2 | 21.676 | 0.78673 | 0.27269 |
| 001037 | Original | 12.496 | 0.23608 | 0.71758 |
| 001037 | V2 | 19.208 | 0.59650 | 0.51651 |

The RGB confirms improved lower-skin coverage at 001033 and 001037, but the
001037 horizontal split remains obvious; hand distortion, texture seams, and
cuff-side holes are not repaired. This is a useful partial result, **not a
successful complete or production repair**. Moving-view new coverage is 1,219,
8,971, and 9,182 pixels, respectively. Coverage is not an accuracy metric.

Held-out F_B's visible upper-forearm regions remain unchanged at 001029
(`26.72005 / 0.91493 / 0.06025`) and 001033 (`23.47423 / 0.91948 / 0.04783`).
001037 remains **N/A** for the requested forearm. None sees newly visible patch
geometry. Whole F_B RGB changes are 0/180/2,431 pixels, respectively, from
recomputed raw-renderer source visibility/labels; global color preservation is
not claimed. Missing-only train metrics are retained as secondary records;
their zero-outside-mask boxes can inflate SSIM/LPIPS relative to full-skin scores.

All 18 native renders and 18 metric-region records completed (16 metric triplets,
two N/A records). Source hashes, immutable mesh prefixes, both visibility guards,
and render receipts pass final audits. Three focused admission-rule tests pass.
No GPU OOM occurred; at most two render workers used about 9.3 GiB combined.
Summed render time is 446.3 seconds (160.5/158.9/127.0 by time), not a serial
wall-time benchmark. No production default or full video was changed.

Two implementation failures were retained: filesystem metadata preservation
failed during an initial staging copy (separate `..._failed_copy_metadata` root),
and an integer-ray shortcut failed exact equivalence before clipping emitted a
mesh. The final integer cast uses the exact original audit ray construction;
all four cameras pass its bit-exact hit-mask / `1e-6` depth check. Path aliases
were resolved for source-map hash comparison. These fixes change no frozen
geometry threshold or semantic decision.

## Insights

The FOV diagnosis is testable: treating H_C's missing coverage as uncertainty
does extend the forearm while preserving two real skin-view bounds. It does
not supply anatomical curvature or make an unobserved interior measured truth.
The dual-lattice guard is stronger than v1's half-pixel-only guard, but sparse
times and manual masks still do not establish automatic temporal transfer.
V2 is retained unchanged with its visible split. A separately authorized follow-up
can test whether semantic evidence should be unknown throughout the renderer's
existing image-border footprint domain, without widening individual skin masks.

Reproduce into a fresh output root with `freeze`, then per frame `analyze`,
`clip_plane`, `geometry_review`, and `semantic_review`. Review native clay before
`evaluation_inputs`, `render`, and `score`; finish with `audit` and `summarize`.
Every command takes the same `--output`; use the parent `.venv/bin/python` with
EXR enabled and two OpenBLAS/OMP threads. GPU rendering requires a coordinated
handoff. Failed and v1 workspaces are not overwritten.
