# DEC5 jaw repair: four-time transfer and current-renderer validation

## What was tested

The [train-anchor/fractional-footprint repair](dec5_jaw_train_confidence.md)
previously helped two small jaw defects. `study_jaw_repair_transfer.py` now
applies that same complete recipe without spot-specific dependencies to 001083,
001123, 001193 and 001195. Native measured 62-camera depth sets are reused and
checksum-checked; no new PatchMatch, learned depth or color fit is performed.

Proposals still use camera-independent short 3D boundary arcs, the unchanged
0.003 extent / 0.0015 endpoint-gap limits, deterministic train depth anchors,
four-tap fractional free-space evidence and the original strict 124-ray final
guard. Existing vertices and triangle prefixes stay exact. Masks, camera poses,
fixed exposure/profiles and early-angle hard-source texture policy remain fixed.
No target camera/RGB constructs geometry or confidence. Nothing is promoted to
production defaults or substituted into the published movie.

Four times each receive baseline/repaired RGB in the actual phase+30 moving
camera and lower physical F004_E005_1210FP: 16 fresh renders. A seventeenth fresh
render checks repaired 001193 in the previously frozen held-out face protocol,
against its retained baseline. No parameters are selected using held-out scores.

## Results

| Frame | Raw proposal triangles | Initially admitted | Final additions | Changed moving / F/E RGB pixels |
|---|---:|---:|---:|---:|
| 001083 | 447 | 6 | 6 | 0 / 0 |
| 001123 | 596 | 0 | 0 | 0 / 0 |
| 001193 | 465 | 39 | 39 | 11 / 84 |
| 001195 | 418 | 45 | 44 | 18 / 89 |

Both historical canaries reproduce their previously accepted local PLYs
**byte-for-byte**. The extra times establish limited transfer: the few 001083
additions are invisible in both inspected cameras; 001123 changes no geometry.
Pixel-change counts above are localization diagnostics, not full-frame metrics.

All four fresh independent audits replay raw proposals and retained mapping,
check original geometry and source hashes, and perform all 124 final ray checks
with zero qualified free-space violations. No new nonmanifold edges or islands:
component counts remain 95, 78, 51 and 63 respectively. Sixteen focused tests pass.

Main-agent native review covered all four moving head pairs and all four lower
real-view jaw triplets, the held-out comparison, two full real-camera context
images and the additional diagnostic crops below. No new conspicuous artifact
attributable to the small additions was seen. Existing crown gaps/brown fringe,
skin seams and lower-view under-chin tears remain. **This is not a four-frame
artifact-free pass or a reason to rerender the full movie.**

![Current moving comparison](/mnt/data/dec5_jaw_repair_transfer/review/001195/moving_head.png)
![Lower real-camera comparison](/mnt/data/dec5_jaw_repair_transfer/review/001193/F004_E005_1210FP_jaw.png)

### Held-out face gate

001193 baseline and repaired PNGs are identical, including outside the face.
The fixed GT-only face polygon therefore gives identical display-domain PSNR
30.276545, SSIM 0.939692, LPIPS 0.076123. This is a non-regression result in one
view, **not** evidence that the mesh is better there. No full-frame quality
metrics or main campaign CSV edits were made.

### Why some confirmed skin misses remain

Post-hoc polygons were drawn on the real F/E skin/shadow region, separately from
geometry construction and held-out scoring. The first interior polygon excludes
the visible silhouette; a second explicitly extends towards the real neck edge.
Both are retained, rather than substituting a more favorable ROI. Native overlay
inspection confirms the selected regions lie on visible skin/shadow in these
train images. These are small diagnostic patches, not whole-face completeness.

| Frame / diagnostic ROI | Before / after depth misses | Before any gate: raw caps cover misses |
|---|---:|---:|
| 001193 interior skin | 30 / 30 | 30 |
| 001195 interior skin | 0 / 0 | 0 |
| 001193 including neck edge | 45 / 35 | 45 |
| 001195 including neck edge | 14 / 6 | 8 |

At 001193, all 30 interior missing pixels have raw cap proposals. Five such
triangles have 61 semantic supports and exactly one mask veto, from D004_D005_1210LZ.
Three also satisfy the unchanged measured-support gate; two fail its median
sample support. None has a fractional-sample free-space veto. The negative-mask
samples are 5.39–14.87 native pixels outside that mask, too far to explain by a
half-pixel convention alone. Three of twenty sampled entries (with duplicated
vertices) have available query-camera measured depth agreeing within 0.001;
the others are missing, not observed free space.

![F/E diagnostic, interior skin](/mnt/data/dec5_jaw_repair_transfer/support_diagnosis/001193/native.png)
![Actual veto camera RGB and mask](/mnt/data/dec5_jaw_repair_transfer/mask_disagreement/001193/D004_D005_1210LZ.png)

The veto-camera crop and full portrait were inspected. This is a difficult dark
chin/neck contour in that view; the crop alone does not establish whether the
mask cuts real shadowed skin or a flat cap protrudes past the true contour. The
single mask is **not** silently discarded on the strength of a 61:1 vote.
At 001195, six edge-region misses have no raw cap coverage at all; removing mask
vetoes cannot fix those. These are distinct remaining mechanisms.

Artifacts: `/mnt/data/dec5_jaw_repair_transfer`, with per-time geometry/evidence,
independent audits, `rgb/`, `review/`, `heldout/metrics.json`, `support_diagnosis/`,
`support_diagnosis_edge/`, `mask_disagreement/`, and a retained-output manifest.
The original interior diagnostic script is archived in `config/` before adding
the separate edge ROI. All workers finished normally; sources and published
videos are unchanged.

## Insights

This transfer check prevents mistaking a successful two-spot patch for a general
head reconstruction improvement. It also separates lack of a geometric proposal
from rejection of one that exists. Next investigate the actual D/D contour with
nearby train/depth evidence before changing semantic confidence; test a curved,
multi-view-constrained surface where a planar cap conflicts with the silhouette.
For uncovered edge rays, a larger supported proposal is needed, not merely lower
admission thresholds. Camera/actor motion is already real in the published shot;
the full artifact-free video and substantially improved mesh goal remains open.
