# DEC5 calibrated neural stereo: new local depth evidence, not a repaired video

## What was tested

After coarse TSDF and silhouette-envelope controls failed, test a different
source of local depth: pretrained [FoundationStereo](https://github.com/NVlabs/FoundationStereo),
official code commit `6e8806816b533e4d13ddbb95ffa907b797060a62`, large checkpoint
`23-51-11`. This is a stereo disparity model, not standalone monocular depth.
No fine-tuning occurs. Keep physical calibration, actual time **001037**,
fixed camera color response/exposure and real train RGB. F/B held-out GT is
never consumed. No renderer or original model default changes.

The upstream [license](https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE)
limits this code to research. This pilot does not establish commercial
deployment permission. The official Drive config download hit its quota.
Public mirrors at `yizhouzhao-nv/FoundationStereo-Backup`,
`pablovela5620/foundation-stereo` and `Felix-Zhenghao/FoundationStereo` publish
the same 3,298,527,334-byte checkpoint SHA-256:
`60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`.
Pinned revisions and the actual downloaded hash are retained in
`/mnt/data/dec5_foundation_model_mirror/receipt.json`. This is three-mirror
agreement, **not an author-signed hash attestation**. No quota bypass was used.

### Calibration is part of this experiment

The existing RGB convention uses principal points minus half a pixel. Rotate
native landscape images into portrait together with the correct camera axes
and intrinsics. Standard `alpha=1` whole-image rectification gives a negative
focal length for one narrow convergent pair; the attempted stage fails closed.
Its workspace/log is retained. Use positive-focal rectification rotations and
local **768x768**, unresized-focal-scale projections around the rough hand center.

Independently recenter the two principal points to keep the focus disparity at
128 pixels. This changes image projections, **not physical poses**. Recover depth
using `f * baseline / (disparity - (cx_left - cx_right))`; omitting that principal
offset would produce the wrong metric scale. Existing imprecise hand landmarks
only locate the crop and disparity range, not measured surface geometry.
Tests recover known 3D points exactly; the real-data audit checks both original
source rays through the saved remaps. All three rectification panels were viewed.

Three physical pairs:

- F/A–G/A and E/C–F/C use four distinct train cameras.
- G/A–H/A adds lower-wrist coverage but shares G/A with the first pair.

Run 32 refinement iterations, FP16, no hierarchy or image downsampling. Reverse
inference swaps and horizontally flips inputs; sample its disparity at the
forward correspondence to check left/right agreement. Black remap borders and
off-image correspondences are unknown, not reliable predictions.

Safe `weights_only=True` checkpoint loading is retained. Explicit trusted
NumPy scalar/dtype and OmegaConf training-metadata types are allowlisted;
NumPy's old `numpy.core` alias is mapped explicitly. Two initial safe-loader
failures and producer snapshots remain. Unrestricted pickle loading is not used.
The full checkpoint loads strictly with no missing/unexpected model keys.

## Results

| Pair | Local warm-object mask | Common source domain | LR <=2 px | Forward + reverse time |
|---|---:|---:|---:|---:|
| F/A–G/A | 49,355 | 48,675 | 45,117 | 1.74 s |
| E/C–F/C | 104,059 | 75,028 | 66,624 | 1.12 s |
| G/A–H/A | 57,599 | 55,966 | 23,751 | 1.64 s |

Times exclude model initialization, download and later geometric audits; these
are local 768px patches, not full-resolution 62-camera reconstructions. A small
count in the second common domain partly reflects source-image clipping, not
missing model depth. Warm masks are the previous imperfect train-only masks,
including lipstick; they never black out inference RGB.

Extract both raw-domain and LR<=2 local depth-grid meshes, requiring both image
silhouettes and a maximum 0.002 normalized triangle edge. This produces six
**open learned surface patches**, not watertight hands. The main agent viewed
all three disparity panels and all three matched geometry panels (real H/A,
real E/C and the unchanged movie pose). First two pairs produce much smoother
front finger/palm surfaces than PatchMatch, with distinguishable fingers, but
occlusion/confidence gaps and missing lower forearm remain. The third pair
produces a visibly poor fragmented hand after the confidence gate.

Cross-project every fourth accepted point into another pair's accepted depth.
Numbers below are **pair disagreement**, not error against independent ground
truth; occlusion can contribute. Depth is in the original normalized scene units.

| Query pair -> other pair | Overlap samples | Median absolute depth difference | P90 | Within 0.002 |
|---|---:|---:|---:|---:|
| F/A–G/A -> E/C–F/C | 7,240 | 0.001332 | 0.002269 | 86.1% |
| E/C–F/C -> F/A–G/A | 14,999 | 0.001685 | 0.003076 | 67.4% |
| E/C–F/C -> G/A–H/A | 3,299 | 0.008277 | 0.013784 | 7.6% |

All six directions are retained in `result.json`. Good within-pair LR agreement
does not certify shared cross-pair shape. Do not simply average these depths or
replace existing fine fingers. No new textured RGB prediction, held-out face
metrics or full-frame quality metrics are claimed. No aggregate metric overrides
the visible holes. The production meshes and 150-frame movie remain unchanged.

- [First calibrated pair](/mnt/data/dec5_foundation_hand_stereo/001037/F004_A_G004_A/rectification_review.png).
- [First pair disparity](/mnt/data/dec5_foundation_hand_stereo/001037/inference/F004_A_G004_A/disparity_review.png).
- [Moving-view patch geometry](/mnt/data/dec5_foundation_hand_geometry/review/moving.png).
- [Cross-pair evidence](/mnt/data/dec5_foundation_hand_geometry/result.json).
- [Explicit review verdict](/mnt/data/dec5_foundation_hand_geometry/visual_review.json).
- [Ray-calibration audit and hashes](/mnt/data/dec5_foundation_hand_geometry/artifact_manifest.json).

Six focused tests pass across the TSDF command control, calibrated portrait
rotation, disparity offset, left/right ordering, distortion rejection and
depth-grid discontinuity handling. The audit reconstructs native source rays
from the learned disparity without refitting camera poses. It does not rerun
neural inference or certify anatomy. All jobs finish; failed attempts are retained.

## Insights

This opens a useful new depth-prior route: the first two stereo pairs recover
smooth front surfaces where fine PatchMatch geometry is ragged, without a
monocular scale fit. It is not yet a usable full-hand replacement. Before
composition, establish which bias/occlusion explains the cross-pair differences,
anchor uncertain patches against independently supported native depth, and use
source pairs that actually see the missing lower wrist. Keep existing reliable
fingers and head geometry. Then test RGB reprojection and temporal transfer;
one smooth depth map or this one-frame canary cannot finish the full goal.
