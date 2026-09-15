# DEC5 001195: frozen MHR completion on the actual production base

## What was tested

Bounded one-time production-base transfer of the previously sealed [001195 method](dec5_mhr_transfer_001195.md), analogous to the [001193 production check](dec5_mhr_production_patch_control.md). No new fitting, pose copying, parameter change, mask relaxation, source-policy change or rollout. **The local hole improvement transfers, with small contour regressions; nothing is promoted.**

The original raw 001195 prior is reused exactly: fresh-per-time MHR head/neck fit, smooth100 measured conformance and the capped 100-step silhouette fit, not either rejected zero-margin/freeze experiment. Only the candidate/admission base becomes the actual current cinematic production mesh. Its SHA-256 is `3223ce6785dc3ddd6524fd42180439040d9c39f031f2cd7a48d56bb8e26a4e87`; **61,703 vertices / 119,830 triangles** are retained as exact prefixes. The metadata path and calibration normalization match the original raw reconstruction; the source geometry hash also matches the independently measured-mask control's actual production source. Prior protocol, final topology, prior seal and original transfer seal are pinned.

Candidates are regenerated against that base; old added triangles are not copied. The unchanged domain is the neutral 135..153 band, excluding all crossing/reversed parent facets, subdivision edge ≤`.00075`, all vertices within `.002` of original surface / `.003` of original boundary vertices, normal dot ≥`.25`, and centroid gap ≥`.00002`. There is no nearest-open-edge requirement or target-driven construction.

The unchanged admission requires train foreground support and no in-frame mask disagreement, actual independent multiview geometric-depth support or the same observed-seed interpolation certificate, no trustworthy measured free-space contradiction, and iterative all-62-camera × integer/half-pixel native veto. Original and inferred geometry remain distinguishable. All three actual held-out views are excluded from source RGB and geometry; the eight fit-reserved train cameras were excluded from prior fitting, but admission itself uses all62 train cameras. No independent unseen-view metric is claimed.

The 12 serial CPU2 renders use the actual cinematic request's **incidence power2, unwarped/static-registration false, target-angle-before-incidence clipping and pixel-fallback angle prior**, with the exact current source-quality installer, texture masks, camera profiles and exposure. The old moving pose is a posthoc stress camera. F/E, M/B and C/E are differing calibrated train poses. D_D's geometry-only measured-mask override does not replace texture masks. This is a matched current-policy CPU ablation, not a CPU/CUDA equivalence or full-6K-delivery claim.

## Results

Private root: [/mnt/data/dec5_mhr_production_patch_001195](/mnt/data/dec5_mhr_production_patch_001195).

| Stage | Count |
|---|---:|
| Unsafe parent facets excluded in band | 88 |
| Subdivided facets | 2,949,120 |
| All-vertex locality pass | 695,448 |
| Centroid-gap rejection among local facets | 26,360 |
| Raw proposals | 669,088 |
| Semantic-admitted proposals | 342,282 |
| Strict initial → final | 215,598 → 215,322 |
| Certified-interpolation initial → final | 222,239 → 221,945 |
| Observed seeds: candidate → independently validated | 4,349 → 3,110 |
| Certified vertices / queries | 114,238 / 180,069 |

Admission took **197.30 s**. Native removal rounds were strict `268,7,1,0`, interpolated `286,7,1,0`, within the unchanged eight-round cap. The final zero-removal round covers all124 unique `(camera, offset)` pairs per branch.

Actually inspected all four [native clay panels](/mnt/data/dec5_mhr_production_patch_001195/admission/native_clay_review): old moving, G/B, M/B and E/B. The fixed moving-view miss count is **73→1 in both branches**, with **72 newly colored fills** and no fallback source IDs among them. Strict/interpolated are bit-identical in that fixed ROI. The puncture is largely closed; original face detail and the natural shadow remain. The one remaining miss is not silently relaxed away.

Also inspected all four current-policy RGB triplets: [old moving](/mnt/data/dec5_mhr_production_patch_001195/admission/rgb_review/old_moving_native.png), [F/E](/mnt/data/dec5_mhr_production_patch_001195/admission/rgb_review/F004_E_native.png), [M/B](/mnt/data/dec5_mhr_production_patch_001195/admission/rgb_review/M004_B_native.png), [C/E](/mnt/data/dec5_mhr_production_patch_001195/admission/rgb_review/C004_E_native.png). No broad new face fold, doubled surface or patch-local source seam is apparent in these views. Existing front-underchin gaps and lace-like neck fringe remain. This is not whole-neck repair.

| Native view | New hits strict / interpolation | Nearer >.003 strict / interpolation | Maximum nearer change | Newly black RGB, either branch | New untextured geometry, either branch |
|---|---:|---:|---:|---:|---:|
| Old moving | 112 / 115 | 1 / 1 | .010039 | 2 | 7 |
| F/E | 97 / 104 | 95 / 98 | .020131 | 0 | 0 |
| M/B | 140 / 141 | 2 / 3 | .011710 | 0 | 1 |
| C/E | 84 / 92 | 9 / 9 | .007742 | 0 | 0 |

These are full-native-frame counts, not just the primary ROI. No lost geometry or farther common-hit change >.003 occurs in these four views. Nearer changes are real, not raycast roundoff: F/E's 82-pixel cluster and maximum .020131 occur at existing underchin/fringe rims. Inspected [localized occlusion panels](/mnt/data/dec5_mhr_production_patch_001195/admission/occlusion_review) show no obvious broad new floating layer, but do not prove global absence of occlusion defects.

The two newly black moving pixels are portrait `(539,1075)` and `(538,1076)`, outside the primary `[505,1086,515,1098]` inclusive ROI, on the jaw/hair rim. Seven new untextured moving pixels and one M/B pixel lie at collar/fringe boundaries. These are retained regressions, not counted as successful colored fills. See [black-pixel evidence](/mnt/data/dec5_mhr_production_patch_001195/admission/black_pixel_review). The exact 22 panels inspected by this agent and 19 explicitly confirmed parent-review panels are hash-bound separately in [visual_review.json](/mnt/data/dec5_mhr_production_patch_001195/visual_review.json); neither claims every retained component was visually inspected.

Exact replay passed: raw candidate arrays and PLY, **3,422,820 support samples, 180,069 interpolation certificates, and 248 final native checks**, including all62 depth-map hashes. Final meshes preserve the original 61,703-vertex / 119,830-triangle prefixes exactly: strict 335,152 triangles; interpolation 341,775. The common 407,080-vertex arrays retain unused candidate suffix vertices. All 12 CPU renders and all audit workers are terminal.

### Reproduction

Use `/home/brans/repos/nerfstudio/.venv/bin/python` with `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1`:

```text
scripts/run_mhr_production_transfer_001195.py build
scripts/run_mhr_production_transfer_001195.py admit
scripts/run_mhr_production_transfer_001195.py clay
scripts/run_mhr_production_transfer_001195.py render
scripts/run_mhr_production_transfer_001195.py audit
scripts/run_mhr_production_transfer_001195.py rgb_review --views old_moving
scripts/run_mhr_production_transfer_001195.py rgb_review --views F004_E
scripts/run_mhr_production_transfer_001195.py rgb_review --views M004_B
scripts/run_mhr_production_transfer_001195.py rgb_review --views C004_E
scripts/run_mhr_production_transfer_001195.py side_effects
-m pytest tests/test_mhr_production_transfer_001195.py -q -o addopts=''
scripts/audit_mhr_production_transfer_001195.py
```

Three focused tests pass. Before the final sealer, perform actual image inspection and create the hash-bound `visual_review.json`; it is not an automatic visual pass. [audit.json](/mnt/data/dec5_mhr_production_patch_001195/audit.json) records full replay; [final_seal.json](/mnt/data/dec5_mhr_production_patch_001195/final_seal.json) binds source inputs, frozen helpers, review images, final render receipts, tests and this report. PSNR/SSIM/LPIPS are N/A: no independent held-out image-quality reference is evaluated. The report skill is used to separate measured geometry, actual visual review, and inferred-prior limitations.

Audit caveat: the frozen runner's common tail overwrote its richer side-effect adapter receipt with the generic stage receipt. The first sealer correctly failed on missing adapter fields (retained `_seal.log`). The final sealer reconstructs and binds the two exact source transformations from the pinned runner/helpers, without executing them or rewriting the original receipt. This supplementary reconstruction is explicitly not claimed as an original execution receipt; geometry and native admission replay are unaffected.

## Insights

The frozen method's main local benefit survives transfer from raw to actual production geometry at a second time, using freshly fitted 001195 anatomy. Interpolation has no additional primary-ROI benefit over strict admission here. Both retain one primary miss, pre-existing neck defects and small new contour/untextured-pixel defects. This supports a bounded local improvement, not artifact-free geometry, whole-prior replacement, temporal acceptance or broader rollout. Added surfaces remain inferred geometry, not direct measurements. Nothing is promoted to current production or the active 6K render.
