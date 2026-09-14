# DEC5 forearm: renderer-footprint border control

## What was tested

A separately frozen, three-time follow-up to [v2](dec5_forearm_plane_transfer_v2.md).
The sole geometric-admission change is semantic availability: source skin-mask
evidence is available only inside the existing raw renderer's footprint domain,
`z > 0, u > 2, u < w-3, v > 2, v < h-3`. Its exact projection includes the
renderer pixel-center `-0.5` convention. Outside that domain, evidence is unknown.
This is one global rule, **not per-frame mask widening**.

At least two distinct available train skin masks must support every proposal;
every available mask disagreement still vetoes it. All original in-frame trusted
measured-depth free-space contradictions remain vetoes, including at the border.
V2's plane coefficients, anchor maps, depth/reprojection/parallax thresholds,
100-pixel anchor distance, maximum triangle extent, and dual-grid append-only
clipping are unchanged. All original vertices and triangles remain intact.
The additions are inferred local planes, not measured anatomy or a learned model.

Artifact root: `/mnt/data/dec5_forearm_plane_transfer_v3`.
Helper: `scripts/study_forearm_plane_transfer_v3.py`, sharing the audited v2
implementation with a default-preserving semantic-domain hook. `protocol.json`
was frozen before generation across 001029/001033/001037. Source depth controls,
train-only masks, camera poses, profiles/exposure, and original geometry are
unchanged. No held-out image enters geometry, clipping, selection, or RGB sources.
No new learned inference, COLMAP, training, or full-video job was launched.

RGB remains the matched **raw local ablation**, not the published video renderer
with its extra foreground-source masks and temporal source-label prior. Production
color integration is outside this pilot.

## Results

001029 and 001033 meshes are **byte-identical to v2**, including clipped meshes.
At 001037, 172 additional reference points fill the false image-border band:
8,689 → 8,861 accepted points; 8,530 have zero measured interior votes.
Unclipped appended triangles increase from 16,833 to 17,380; the same 781
triangles are removed by the dual-grid guard, leaving 16,599.

![001037 native clay: original / unclipped / clipped](</mnt/data/dec5_forearm_plane_transfer_v3/001037/review/moving/clay_comparison_native.png>)

The horizontal split disappears in native clay. Moving-view newly covered pixels
increase from 9,182 to 9,572; H_A from 8,115 to 8,389. Both exact integer and
half-pixel guards report **zero supported-old occlusion** in the fixed moving
and three train cameras at all three times. This is bounded visibility evidence,
not a guarantee for every continuous ray or other camera.

At 001037, G_A and H_A have zero changed pixels outside their skin regions.
H_C has 326 changed pixels outside its inset polygon, **all confined to the
unavailable image border**, and zero outside-mask pixels in the available
domain. Native train review shows skin at that border, not a cloth bridge.
001033 retains its previously reported five H_A pixels within `sqrt(2)` pixels
of the inset boundary; the mesh is unchanged from v2. Manual inset polygons
are not exact ground-truth silhouettes.

![Frozen skin bounds and new visibility, 001037](</mnt/data/dec5_forearm_plane_transfer_v3/001037/semantic_footprints_native.png>)

![Same physical H_A, three times: GT / original / v2 / v3](</mnt/data/dec5_forearm_plane_transfer_v3/v2_v3_three_time_H_A_native.png>)

Native RGB confirms removal of the split without visible cloth bridging in the
reviewed moving and train angles. The exact previously diagnosed central strip
has **248 → 0 missing pixels**. The lower skin remains broad and planar; this
is a useful bounded completion, not recovery of anatomical curvature.

Full train H_A forearm-skin regions, fixed across variants:

| Time | Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| 001029 | Original | 21.373 | 0.82678 | 0.23121 |
| 001029 | V3 (= V2) | 27.708 | 0.86383 | 0.14286 |
| 001033 | Original | 13.602 | 0.56413 | 0.53149 |
| 001033 | V3 (= V2) | 21.676 | 0.78673 | 0.27269 |
| 001037 | Original | 12.496 | 0.23608 | 0.71758 |
| 001037 | V2 | 19.208 | 0.59650 | 0.51651 |
| 001037 | V3 | 19.831 | 0.63992 | 0.40393 |

These are train reprojection checks, not held-out generalization. F_B's visible
upper-forearm regions remain unchanged at 001029 (`26.72005 / 0.91493 / 0.06025`)
and 001033 (`23.47423 / 0.91948 / 0.04783`); neither sees the added interior.
001037 stays **N/A**, not replaced by a hand/wrist region. Its full F_B image
changes 2,431 RGB pixels from recomputed raw-renderer source selection, despite
zero newly visible patch pixels. Global color preservation is not claimed.

All 18 matched RGB artifacts, 16 metric triplets plus two N/A region records,
source hashes, original mesh prefixes, render receipts, and both visibility
guards pass final audits. Native clay, RGB, and train-boundary images were
visually inspected. No production default or full video was changed.

Identical v2 mesh/camera/source render requests are reused only after explicit
old/new bindings and output hashes are checked; original requests and receipts
remain untouched. Only changed 001037 candidate views are newly rendered.
The renderer code and original fitted static parameters are also hash verified.
Fifteen renders are reused and three 001037 candidate views are new GPU work,
90.2 summed render seconds. Reused historical times are
not counted as new work. One study renderer ran alongside up to four parent
renderers, around 23 GiB combined GPU memory, without OOM; timing is contended,
not a throughput benchmark.

## Insights

The v2 strip was an inconsistent treatment of unavailable image-border evidence,
not an anatomical gap or a dual-clipping defect. A common renderer-footprint
rule removes it without changing the other two meshes or opening an available
semantic disagreement. The broad surface is still a local plane, not recovered
forearm curvature; cuff-side holes, hand distortion, and texture seams remain.
Three sparse times with manually traced pose-specific skin regions do not prove
automatic full-clip transfer, continuous temporal stability, or held-out interior
accuracy. The held-out camera does not see the filled interior, and its 001037
forearm region remains explicitly N/A.

Reproduction: use `freeze`, then per time `analyze`, `clip_plane`,
`geometry_review`, `semantic_review`, and `safeguards`; inspect native clay before
`evaluation_inputs`, `render`, `score`, and `audit`; finish with `summarize`.
`seam_diagnosis --frame 001037` is a read-only diagnostic. Use a fresh common
`--output` path, parent `.venv/bin/python`, EXR enabled, and two OpenBLAS/OMP
threads. GPU use is coordinated separately. Five focused domain/admission tests
pass with `pytest -o addopts=''` (the parent config otherwise requires missing
pytest-xdist). All v1/v2 negative artifacts are retained.
