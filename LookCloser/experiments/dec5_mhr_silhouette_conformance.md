# Silhouette-aware neck fitting improves globally but remains locally invalid

## What was tested

One fixed, train-only control on frame 001193, starting from the frozen `smooth100` MHR measured-conformance prior. Additive displacement is allowed only at the 1,566 vertices with neutral-model `135 < y < 153` cm. Upper face and body are exactly fixed to that prior; the original COLMAP mesh is separately unchanged. This is a **prior-only fit**, not replacement geometry or an admitted patch.

The silhouette term produces genuine cross-view improvement, but the ten-step bounded fit remains insufficient at the requested jaw region. On a fixed reserved-camera sample cohort, silhouette P90 excess drops **55.31→36.09 pixels**, while requested first-hit points still violate a median of nine train masks. No patch or production promotion.

Root: [/mnt/data/dec5_mhr_silhouette_conformance](/mnt/data/dec5_mhr_silhouette_conformance/protocol.json).

The protocol was saved before fitting, with one weight and no search:

- 54 train cameras provide person-mask constraints and existing measured anchors. Eight named reserved train cameras are excluded from fitting, including silhouette constraints; the fitter receives only the selected fitting arrays. Dataset-held-out cameras and target RGB are never inputs.
- Native signed Euclidean distance is positive outside foreground. Exact bilinear sampling and analytic world-to-pixel Jacobians use the renderer's half-pixel convention. Outside the native footprint is unknown. A uniform two-pixel hinge tolerance applies to every camera; masks are not widened or edited.
- Silhouette coefficient 4, pixel sigma 2, robust residual scale 8 pixels. Weights divide by the fixed product of 54 cameras and 1,566 active vertices. This is a dimensionless average penalty alongside measured-distance residuals, not an assertion that pixels equal world units.
- Inherited depth association limits remain 0.006 scene distance and normal dot ≥0.25; equal mean weighting of associated face/neck groups, data sigma 0.001. Additive uniform-Laplacian sigma 0.0005 and displacement sigma 0.006 retain local smoothness and the starting shape.
- Joint XYZ sparse LSMR, at most 600 inner iterations; ten outer steps, each globally scaled to maximum vertex motion 0.001. Last iterate only; no reserved-view model selection, target-ray constraints, weight search or extra iterations.

Scene distances are in the calibrated normalized COLMAP gauge, not metres. The original geometry and inherited independently measured D_D foreground override have all-62-train provenance, so reserved cameras are **excluded from direct fitting**, not completely unseen from every baseline artifact. Existing anchor construction excludes them from fitting depth votes. The old repaired mask-control geometry receipt is not inherited; actual original SHA is `316dc2b6…`.

## Results

### Silhouette validation on identical available samples

Point-camera visibility changes as vertices move into the image. The table uses only samples available both before and after, preventing a denominator change from concealing actual improvement. Excess is the positive distance beyond the fixed two-pixel allowance.

| Fixed cohort | Samples | Mean excess, before→after px | P90 excess, before→after px | Outside samples, before→after |
|---|---:|---:|---:|---:|
| 54 fitting cameras, all active vertices | 78,340 | 11.45→7.33 | 40.92→24.86 | 16,125→15,146 |
| Eight reserved cameras, all active vertices | 11,486 | 14.35→9.29 | 55.31→36.09 | 2,300→2,188 |
| Reserved, neutral front (`z ≥ 0`) | 7,696 | 9.62→6.07 | 22.49→15.07 | 1,285→1,191 |
| Reserved, neutral back (`z < 0`) | 3,790 | 23.96→15.83 | 109.58→73.15 | 1,015→997 |

These front/back labels describe the canonical model, not hand-traced image anatomy. Errors remain much larger than the two-pixel allowance. On all currently available samples, reserved P90 is 55.29→55.30 px and outside fraction 20.02→21.62%; the available count grows 11,490→11,864. Those unpaired aggregates must not be interpreted as proof that the fit had no effect.

### Measured anchors and shape safety

| Measurement | Baseline | Silhouette fit |
|---|---:|---:|
| Reserved face-anchor surface P90 | 0.0001692 | 0.0001692 |
| Reserved neck-candidate surface P90 | 0.0001341 | 0.0001297 |
| Reserved anchors touching active triangles, P90 | 0.0001342 | 0.0001300 |
| Strict nonadjacent crossing pairs | 163 | 163 |
| Strict crossing pairs touching active band | 0 | 0 |
| New >90° triangle-normal changes | — | 0 |

There are 5,644 fitting and 828 reserved anchor associations touching active triangles; respectively 5,529 and 808 have interpolated neutral coordinates actually within the lower band. Thus the reported anchor preservation is not credited solely to untouched upper-face samples. Nearest-surface residuals still permit association sliding and do not establish correct depth ordering.

Active displacement median/P90/max is 0.000359 / 0.006363 / 0.009609. Minimum triangle area ratio to the starting prior is 0.07599: some facets compress substantially, although no new inversion-normal or transverse-crossing witness is found. All inherited mouth/nose crossings remain invalid for replacement use; this experiment does not excuse them.

LSMR converged internally in 234–239 iterations, but **the outer fit is not converged**. All ten steps hit the global 0.001 trust cap; maximum unconstrained step falls only 0.02051→0.01577. This is a bounded partial optimization result, not evidence that the chosen objective or anatomical model cannot ever satisfy silhouettes. Global step scaling can let the largest unconstrained motions throttle smaller local updates.

### Native review and requested-region check

Independent native inspection of C_E clay/projection, G_B clay and the requested prior-only crop shows real neck-bulge retraction, but a substantial under-chin cluster still projects into real background. The original COLMAP retains much better facial detail. Upper-face differences versus original are inherited from the starting MHR prior, not introduced by this lower-band fit.

![Native C_E RGB projection, baseline and silhouette fit; red means outside by over two pixels](/mnt/data/dec5_mhr_silhouette_conformance/review_v2/C004_E005_1210X7_projection.png)

[Native clay/RGB comparison](/mnt/data/dec5_mhr_silhouette_conformance/review_v2/C004_E005_1210X7_clay.png), [reserved G_B comparison](/mnt/data/dec5_mhr_silhouette_conformance/review_v2/G004_B005_1210FG_clay.png).

The fixed requested rays and original 24-point rim are used **only after fitting**. Both priors intersect all 44 original miss rays, which is not itself successful repair.

| Posthoc local check | Baseline | Silhouette fit |
|---|---:|---:|
| First-hit points within 0.002 of original surface | 44 / 44 | 44 / 44 |
| First-hit points passing every available train silhouette (2 px allowance) | 3 / 44 | 1 / 44 |
| Disagreeing cameras per first hit, median | 9 | 9 |
| Original-rim distance to prior, median / P90 | 0.000614 / 0.000900 | 0.000539 / 0.000750 |
| Signed original-normal rim offset, median / P90 | +0.000515 / +0.000824 | +0.000478 / +0.000655 |

First-hit camera-z changes only +0.000167…+0.000884 (median +0.000610). Local proximity and improved rim offsets still leave the front intersection incompatible with multiple real silhouettes. No local patch is constructed. [Posthoc locality evidence](/mnt/data/dec5_mhr_silhouette_conformance/locality/result.json).

No novel RGB texture render or reconstruction metrics are claimed; PSNR/SSIM/LPIPS are N/A for this prior-fit geometric gate. The native RGB panels are real train references, not synthesized colors.

## Insights

Silhouette-aware fitting is a real missing constraint: it reduces error in cameras excluded from direct fitting while preserving measured anchors and the fixed upper/body regions. However, this specific bounded result is **not precise enough for jaw completion**. It must not be promoted simply because global error decreases or the prior intersects all missing rays.

The remaining error is not established as a mask-shadow bug, absence of anatomical model support, or impossibility of silhouette fitting. The solver stops at its declared bound before convergence, and many first-hit points still represent the wrong silhouette/depth layer. Further optimization or a changed method would require a separately declared experiment; none is launched here.

### Reproduction, audit and retained failure

Use reconstruction Python, `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. Producer `scripts/fit_mhr_silhouette_conformance.py` requires a fresh fixed root; fitting took 7.68 seconds, CPU only. `scripts/audit_mhr_silhouette_conformance.py` independently reconstructs the 54-camera inputs and exactly replays vertices/history, with no reserved arrays passed to the fitter. All 33 checked bindings pass; the original mesh hash and every inactive vertex are exact. Three tests verify analytic world Jacobians, bilinear SDF gradients and the renderer half-pixel convention.

`scripts/review_mhr_silhouette_conformance.py` creates native RGB/clay/projection and topology evidence; `scripts/probe_mhr_silhouette_locality.py` supplies posthoc ray/rim checks. Initial review serialization rejected infinite original miss depths. Its failed producer and already-generated panels remain in `review/`; `review_v2/` represents missing depth as null. The fit was not rerun or changed to fix this reporting error.

The [final seal](/mnt/data/dec5_mhr_silhouette_conformance/final_seal.json) binds protocol, producer/helpers, input receipts, numerical replay, native evidence, failed-review snapshot, report and runtime versions. No production source, geometry, texture, defaults, architecture, temporal run or active 6K artifact was changed.
