# DEC5 001033: append-only forearm confidence-prior pilot

## What was tested

One time, three initial candidates and one explicitly authorized attachment-clipping
control, original unrepaired COLMAP/PatchMatch TSDF baseline.
This is a follow-up to the negative [head/hair prior pilot](dec5_confidence_depth_prior.md),
not another head rerun or a full-video reconstruction. The genuine lower-forearm
hole is visible as skin in six real train views. The diagnostic moving-view ray
at native portrait `(365,1850)` has at most one independent geometric-depth
agreement over the parent's broad depth bracket, despite many camera frustum
intersections. Its interior therefore cannot be described as measured geometry.

Artifacts: `/mnt/data/dec5_forearm_confidence_prior`.
Source control: `/mnt/data/dec5_forearm_depth_control_001033`; all 62 geometric
depth maps from the completed pinned reconstruction were loaded and hashed.
Raw calibration is normalized against the original TSDF metadata and checked
against the frozen train cameras. No source files, original triangles, renderer,
camera path, exposure/profile settings, or production defaults were changed.

The actual missing reference region is in `G004_A005_121071`, **not** the intact
`G004_B005_1210FG` forearm/palm regions used in the earlier depth-stage diagnosis.
Train-RGB-only skin polygons in G_A, H_A, and H_C constrain the patch. They are
inset from the cuff; silhouettes count as anatomical limits, never stereo support.
The reference polygon has 22,357 pixels, including 7,812 original ray misses and
10,533 trusted observed boundary pixels. H_A/H_C contain 7,721/8,856 original
misses and 6,611/12,742 trusted pixels respectively.

![Actual reference holes and trusted boundary](</mnt/data/dec5_forearm_confidence_prior/001033/G004_A005_121071_support_native.png>)

Red = original mesh miss; green = valid geometric depth agreeing with original
mesh and at least three other measured depth maps. Votes require camera-z
agreement within `0.001` normalized units, round-trip reprojection within 1.5
native pixels, and at least 1 degree of parallax. Camera/frustum counts are not
used as depth confidence.

Candidates were frozen before held-out RGB was opened:

- `plane`: robust inverse-depth plane fitted to the trusted skin boundary.
- `da3_local`: cached DA3-LARGE-1.1, 16 train cameras, upright input with consistent
  pixel/intrinsic/extrinsic rotation. Robust whole-scene metric scale/shift uses
  independently supported anchors; a local residual plane aligns the forearm
  boundary while retaining the model's higher-order shape.
- `da3_consistent`: the same learned surface, additionally requiring agreement
  with both other locally aligned learned skin-view depths (`0.003` depth units,
  3-pixel return reprojection). These are model consistency votes, not measurements.

All candidates require three reviewed skin-region projections, distance at most
100 pixels from a trusted reference anchor, no free-space contradiction from the
trusted observed skin maps, and depth within `0.004` of the boundary plane.
Learned local boundary RMSE must be at most `0.0015`. There is deliberately **no
minimum interior measured-depth count**: entirely unsupported interiors may be
proposed, clearly labeled as inference. Triangles are added only at original
reference misses, with a one-pixel attachment ring and maximum triangle extent
`0.002`; every original vertex and triangle remains exactly intact. A separate
audit checks whether new surfaces nevertheless hide independently supported old
surfaces in another view. Array preservation alone is not visibility preservation.

The exact original, plane, two learned meshes, and clipped plane were rendered at the requested
moving camera, a distinct H_A train camera, and held-out F_B. F_B RGB was never
inference, alignment, mask-selection, or texture input. H_A was a DA3 input and
one of the semantic gates, so its RGB metrics are only train-view reprojection
checks. A single time cannot establish temporal stability.

## Results

| Candidate | Reference pixels added | Added triangles | Added pixels with zero measured votes | Moving-view newly visible pixels |
|---|---:|---:|---:|---:|
| Plane | 5,657 | 11,459 | 5,222 | 6,714 |
| DA3 local | 5,658 | 11,438 | 4,787 | 5,959 |
| DA3 consistent | 1,693 | 2,691 | 1,207 | 1,338 |

All three cover the actual `(365,1850)` ray. Its inferred camera-z is `0.6122703`
for the plane and `0.6093614` for both learned candidates; neither value is ground
truth. Median measured interior support is zero for every candidate. The strict
learned gate substantially reduces coverage without converting the surviving
points into measured geometry.

![Matched native moving-camera geometry](</mnt/data/dec5_forearm_confidence_prior/001033/review/moving/clay_comparison_native.png>)

![Matched native train H_A geometry](</mnt/data/dec5_forearm_confidence_prior/001033/review/train_H_A/clay_comparison_native.png>)

The plane fills much of the actual hole with a smooth but flat surface. Its
inset semantic limits leave a conspicuously straight lower edge and a residual
gap toward the cuff. DA3 local adds visible grid-like corrugation and a large
notch; the stricter arm leaves a fragmented patch and most of the hole. Coverage
alone is not an acceptance criterion.

The original plane's visibility guard flags four moving-view pixels at portrait
`(401–403,1863–1865)` and one H_A pixel at `(172,1802)`, all at the lower-right
skin/cuff attachment. Depth differences are `0.001251–0.001785`, exceeding the
unchanged `0.001` guard; this is not floating-point roundoff. This conservative
visibility test is a warning about hiding supported old surfaces, not proof of
the unknown true interior shape.

The bounded follow-up used one common rule in the moving camera and the same
three train skin cameras: remove appended triangles hiding old pixels supported
by at least three other measured views beyond that guard, then recheck. No
held-out input or selection was involved, and the original protocol is retained
alongside a separate `clip_request.json`. Two removal passes deleted five of
11,459 appended triangles; the next pass found zero flags in all four cameras.
The resulting `plane_clipped` keeps 11,454 triangles and all 6,714 moving-view
newly visible pixels; H_A new coverage changes from 4,959 to 4,953 pixels.

| Candidate | Supported old pixels hidden: moving | H_A | Fixed-view guard |
|---|---:|---:|---|
| Plane | 4 | 1 | Flagged attachment |
| DA3 local | 49 | 19 | Fail |
| DA3 consistent | 18 | 9 | Fail |
| Clipped plane | 0 | 0 | Pass in tested views |

The clipped plane still hides 88 moving-view and four H_A old pixels beyond
`0.001` that do **not** meet the three-vote trust threshold. Thus the statement
is preservation of the specified trusted subset, not universal visibility
preservation or proof of correctness from every camera.

![Matched native moving-camera RGB](</mnt/data/dec5_forearm_confidence_prior/001033/evaluation/moving_rgb_comparison_native.png>)

![Matched H_A train-reference RGB](</mnt/data/dec5_forearm_confidence_prior/001033/evaluation/train_H_A_rgb_comparison_native.png>)

H_A full manually traced forearm-skin region, 17,994 pixels:

| Candidate | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Original | 13.602 | 0.56413 | 0.53149 |
| Plane | 18.647 | 0.73064 | 0.35212 |
| DA3 local | 19.232 | 0.74151 | 0.35345 |
| DA3 consistent | 14.509 | 0.58613 | 0.50737 |
| Clipped plane | 18.638 | 0.73043 | 0.35233 |

On the fixed subset of 7,663 originally missing skin pixels, original → clipped
plane is `9.98495 → 15.21428 dB`, `0.23731 → 0.63039` SSIM, and
`0.80918 → 0.29793` LPIPS. The slight count difference from integer-ray diagnostic
misses is the renderer's half-pixel ray convention; depth votes use the explicit
matching convention. Metrics are masked RGB PSNR and tight-box, zero-outside-mask
SSIM/AlexNet LPIPS. **These are train-view checks, not held-out generalization.**
DA3 local has higher H_A PSNR/SSIM than the plane, but worse LPIPS, visibly
corrugated clay geometry, and more supported-surface occlusion. Its score gain
does not override those failures.

Held-out F_B cannot see the actual lower patch: all candidates change zero depth
pixels there. Its visible upper-forearm ROI is exactly unchanged, with
`23.47423 dB / 0.91948 / 0.04783` PSNR/SSIM/LPIPS. This is a preservation check,
**not validation of the filled hole**. Full F_B RGB is not identical: the frozen
hard-source renderer changes 180/185/149/180 pixels outside that ROI for
plane/DA3/strict/clipped respectively when source visibility/labels are recomputed.
No exposure or camera profile was refitted. We therefore do not claim globally
unchanged RGB merely because original geometry arrays are unchanged.

On 2,107 withheld trusted reference boundary anchors, plane MAE is `0.0009986`
and locally aligned DA3 MAE is `0.0006800` normalized units. This validates local
alignment, not the completely unobserved interior. DA3's whole-scene affine
alignment used 1,188/1,185/1,201 supported anchors in G_A/H_A/H_C. Its local
boundary RMSE was `0.0007912/0.0006419/0.0007152`.

Cached DA3 inference took 1.88 seconds through the model API and peaked at
12.80 GiB allocated GPU memory; image staging/loading/output compression are
additional. Fifteen full native RGB renders completed serially at roughly
22–24 seconds each. All source/depth hashes, original vertex/triangle prefixes,
render completion hashes, and 15 metric records are validated in `audit.json`
and `visual_review.json`. Raw failed controls, logs, meshes, depths and images
remain retained.

## Insights

This bounded study distinguishes absence of stereo evidence from an anatomical
gap: the former permits a labeled prior experiment, but not a claim of recovered
truth. It also distinguishes a smooth local geometric prior from a learned one.
The clipped plane is a **positive but limited single-time canary**: substantial
hole reduction and a passing fixed-view trust guard, without the learned ripple.
It is still an inferred flat surface with a straight inset lower edge, a residual
cuff gap, and pre-existing texture seams. The learned variants are rejected;
learned coverage and better train PSNR are not sufficient evidence of better
shape. No candidate is a production repair, and no 150-frame rerender was started.

A sensible next test, if authorized, is the same fixed clipping/anchor rule on
separate times and camera angles. Temporal stability and genuinely unseen-view
appearance of the lower patch remain untested. Do not generalize this forearm
result to face skin or hair: those separate two-time tests remain documented in
the earlier report, with no learned candidate promoted.

Reproduction uses `scripts/study_forearm_confidence_prior.py` with commands
`stage`, `infer`, `diagnose`, `analyze`, `geometry_review`, `render`,
`evaluation_inputs`, `score`, and `audit`, in that order. For the recorded bounded
follow-up, run `conflict_evidence`, `clip_plane`, then rerun `geometry_review`,
`render` (completed originals resume by hash), `score`, `audit`, and `finalize`.
For a new reproduction, pass the same fresh `--output /absolute/new/root` to
every command; `stage` refuses an existing staged input. Use the parent repo
`.venv/bin/python`, `OPENCV_IO_ENABLE_OPENEXR=1`, two OMP/OpenBLAS threads, and the
existing `/home/brans/deps/Depth-Anything-3/src` PYTHONPATH with offline model
cache for inference. Input, protocol, model, source, mesh, depth-map, and render
hashes are retained in the artifact root. The initial diagnostic visualization
failed on a non-contiguous rotated OpenCV array; its log is preserved, the array
was copied, and the diagnostic reran successfully. The first score pass likewise
hit negative NumPy strides on a rotated mask; the mask was copied and all scores
reran, with the failed log retained. No experiment data were
deleted or replaced by a passing result.
