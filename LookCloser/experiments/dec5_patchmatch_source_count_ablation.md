# DEC5 PatchMatch source-count ablation

## What was tested

Completed all six arms on 2026-09-14, 15:54–18:34 UTC; native visual review and
validation completed afterward. Hypothesis: increasing source neighbors from
12 to 24 or 36 repairs crown/cheek holes on frames `001083` and `001123`.

All **62 train cameras remain depth references**. Each reference receives 12,
24, or 36 of the other 61 cameras, selected by stable Euclidean camera-center
distance; the lists are nested. This does not mean only 12 cameras reconstruct
the scene. The separate angular-16 texture pool remains fixed.

Frozen original campaign scripts and ingest: per-image 70th-percentile exposure,
middle gray 0.18, Reinhard/sRGB, JPEG98; user-fixed GLOMAP calibration. No new
color calibration, masks, geometry repairs, or video processing enter the study.
Unchanged recipe: image 1920, three iterations per pass, photometric then
geometric, depth 4.5–20, NCC 0.1, geometric gates 6/2, minimum two consistent
views, angle 1 degree, TSDF voxel/truncation 0.0005/0.004, extraction weight 2,
crop ±0.15, component threshold max(100, 0.002 × largest component). Pinned
COLMAP defaults, including 15 samples and window radius 5, remain unchanged.

Used dev3 `/usr/local/bin/colmap`, CUDA 3.13.0.dev0 commit `5509fffe`, exclusively
and sequentially. Binary SHA256:
`27bfbe22c358062444495c2b105991d5847e1eab92dfaa8c9cae4b46f5c9c66e`.
Calibration SHA256:
`79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900`.
Old full depth workspaces were not retained, so both 12-source baselines were
rerun. Source EXRs and existing campaign/video outputs were not modified.

## Results

**Neither 24 nor 36 sources demonstrated a convincing crown-hole repair on
these two frames. Keep 12 as the current default.** Native matched train and
virtual clay views retain the crown-edge notches at every count. Small openings
move or change shape, but the larger source sets do not restore a continuous,
credible crown. No gross new head bridge is apparent in the reviewed views;
this does not establish that every added surface is correct.

One fixed held-out camera per frame (`F004_B005_1210O9`), display-referred,
face-only metrics. Higher PSNR/SSIM and lower LPIPS are better. GPU peak is the
sampled whole-pipeline memory, not PatchMatch alone.

| Frame | Sources/ref | Face PSNR | Face SSIM | Face LPIPS | PatchMatch minutes | GPU peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| 001083 | 12 | 27.2396 | 0.874968 | 0.056650 | 21.71 | 7743 |
| 001083 | 24 | 27.3256 | 0.874417 | 0.057855 | 25.40 | 7711 |
| 001083 | 36 | 26.4257 | 0.871747 | 0.061256 | 29.10 | 7743 |
| 001123 | 12 | 27.2900 | 0.870609 | 0.068737 | 21.70 | 7807 |
| 001123 | 24 | 27.1560 | 0.869581 | 0.069116 | 25.36 | 7775 |
| 001123 | 36 | 27.0978 | 0.868149 | 0.070048 | 29.02 | 7807 |

At 24 sources, `001083` gains only 0.086 dB while SSIM/LPIPS worsen; `001123`
worsens on all three metrics. At 36, both frames worsen on all three: PSNR
changes −0.814 and −0.192 dB versus 12. PatchMatch costs about 17% more at 24
and 34% more at 36. Photometric/geometric times are respectively about
462/840, 539/984, and 614/1129 seconds; exact stage times are in `summary.json`.
All meshes retain two components; triangle counts range from 128,296 to 130,620.

Fixed train camera `G004_A005_121071`, GT-drawn crown-band diagnostic regions:

| Frame | Sources/ref | Geometric valid | Photo-valid → geo-invalid pixels | Mean consistent observations on valid depth | Mesh hits |
|---|---:|---:|---:|---:|---:|
| 001083 | 12 | 63.15% | 3869 / 10500 | 3.81 | 93.10% |
| 001083 | 24 | 66.64% | 3503 / 10500 | 5.27 | 94.50% |
| 001083 | 36 | 63.14% | 3870 / 10500 | 6.54 | 94.87% |
| 001123 | 12 | 85.90% | 1098 / 7789 | 5.59 | 100.00% |
| 001123 | 24 | 93.70% | 491 / 7789 | 7.89 | 100.00% |
| 001123 | 36 | 91.87% | 633 / 7789 | 9.73 | 100.00% |

The photometric maps are valid throughout these probes. The lost-support count
includes both geometric refinement and filtering; it is not a specific rejection
reason. More available sources increase the absolute observation count, but
validity is not monotonic. In `001083`, 36 gains 637 valid-depth pixels and loses
638 versus 12. Its mesh gains 343 hits and loses 157, net +186/10,500 (+1.77 pp).
At 24 the mesh gains 320 and loses 173, net +147 (+1.40 pp). These net increases
do not close the visible notch.

The `001123` crown probe already has complete mesh hits and misses the important
edge slit visible in the native crops: its 100% is not a success criterion.
Interior-crown and cheek mesh probes have 100% hits in every arm. Cheek depth
is fully valid at 12/24; at 36 only three pixels lose validity in each frame.
Shared-hit cheek depth changes have p95 below 0.027% versus 12. The cheek's dark
region includes real cast shadow; some chin/shoulder gaps are true background.
Missing ray hits alone must not be labeled missing anatomy.

Final matched visual evidence (same camera and crop within each comparison):

| Frame | Native train crown | Native virtual crown | Held-out RGB head | Whole-head mesh review |
|---|---|---|---|---|
| 001083 | [GT + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001083/train_crown_native.png) | [historical + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001083/virtual_crown_native.png) | [GT + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001083/heldout_head_native.png) | [matched clay](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001083/matched_train_clay_native.png) |
| 001123 | [GT + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001123/train_crown_native.png) | [historical + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001123/virtual_crown_native.png) | [GT + 12/24/36](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001123/heldout_head_native.png) | [matched clay](/mnt/data/dec5_patchmatch_source_count_ablation/comparisons/001123/matched_train_clay_native.png) |

Validation: all 372 geometric maps are nonempty, unique within their arm, and
1080×1920. All six selected consistency graphs agree exactly with their valid
depth bitmap; observation histograms reconcile and respect the source-count
bound. The pinned graph stores column then row, protected by a regression test.
Hash-pinned PNG region masks avoid cross-host Pillow rasterization differences.
The frozen code/input hashes, original source-EXR hashes, nested lists, held-out
exclusion, and command equivalence checks pass. Both 12 reruns reproduce exact
historical aggregate depth coverage and triangle counts; `001083` has one extra
vertex (66,755 vs 66,754), while `001123` matches its old vertex count.

GT matches the retained renderer GT pixel-for-pixel in all arms. Independent
float64 PSNR recomputation agrees within 0.00001 dB, and local valid/lost-support
and mesh-hit counts reconcile. Face polygons were drawn on GT before ablation
predictions: exact selected RGB pixels for PSNR, tight bbox with both images
zeroed outside the mask for SSIM/LPIPS-Alex. The frozen established scorer is
reused; SSIM/LPIPS were not independently reimplemented. These new ROIs are not
numerically comparable to the old surface-masked experiment table.

Artifacts: [summary](/mnt/data/dec5_patchmatch_source_count_ablation/summary.json),
[validation](/mnt/data/dec5_patchmatch_source_count_ablation/validation.json),
[input/design audit](/mnt/data/dec5_patchmatch_source_count_ablation/design_audit.json),
[visual review](/mnt/data/dec5_patchmatch_source_count_ablation/visual_review.json).
Local root `/mnt/data/dec5_patchmatch_source_count_ablation`; full dense workspaces
remain on dev3 `/fsx/oregon/dec5_patchmatch_source_count_ablation`. Minute checks,
explicit supervising-agent checks, and two-second GPU samples are retained.
The final check found normal completion, idle GPU, and no OOM/CUDA/traceback evidence.

Reproduce artifact checks with `scripts/review_patchmatch_source_count_ablation.py`
actions `verify_design`, `summarize`, and `validate_results`; experiment launch
and collection are in the two new `run_...`/`review_...` scripts. Tests:
`python -m pytest -q -o addopts='' tests/test_patchmatch_source_count_depth.py`.
Final result: three tests passed; all three new scripts passed `py_compile`.
`record_visual_review` seals the separately authored manual review after image
inspection; `summarize` deliberately resets its verdict to pending when rerun.

## Insights

Do not promote 24 or 36 as a hole repair under this recipe. Extra observations
and slightly larger mesh coverage do not establish correct local shape, and the
added cost has no consistent image-quality benefit here. Current defaults are unchanged.

Limits: two selected frames, one held-out face camera per frame, one train camera
and one virtual geometry view; no statistical repeat series, full-rig/temporal
benchmark, or 3D ground truth. Train polygons are diagnostic probes, not perfect
hair mattes. Full-image depth coverage is integrity QC only, not a quality metric.
The unchanged mask-free hard-source renderer retains brown background fringe
around hair and its inherited depth-hole fill; raw mesh clay is inspected
separately so RGB filling cannot hide mesh notches. Hard source selection can
amplify small depth/visibility changes, so RGB-score changes measure the fixed
end-to-end recipe, not isolated geometric accuracy. Face metrics exclude the
hair boundary and cannot establish crown quality. Validation is ready within
this bounded scope, with these caveats.
