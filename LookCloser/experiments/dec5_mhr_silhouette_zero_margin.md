# DEC5 001193: zero-offset silhouette fit

## What was tested

**Rejected fit-only control. No patch admission or production change.** The prior's soft two-pixel fitting margin disagreed with the existing zero-tolerance binary-mask admission. This single opt-in control changes only `max(SDF−2,0)` to `max(SDF−0,0)`. The denominator remains **2 pixels**, silhouette weight **4**, robust scale **8 pixels**, and all measured-anchor, Laplacian, magnitude, association and trust-step settings remain unchanged. It is not a weight sweep or mask relaxation.

Both the fresh ten-step check and the 100-step run start from the original `/mnt/data/dec5_mhr_measured_conformance/smooth100/fit.npz`. Magnitude/Laplacian references never reset to a fitted intermediate. Only neutral `135<y<153` vertices may move; upper face/body remain exactly fixed. The same 54 train cameras supply fitting silhouettes/anchors, with eight reserved train cameras excluded from fitting and anchor depth voting. Original COLMAP and the inherited independent D_D foreground override have all-62 provenance: reserved-camera checks are **not unseen-from-baseline evaluation**. The three true held-out RGB views are never used. Target rays and residual rectangles enter posthoc diagnostics only.

The unchanged stop rule allows at most 100 outer steps, requires unconstrained maximum step ≤`1e-5` for three consecutive steps for convergence, and fails on nonfinite/collapsed or unstable updates. Original COLMAP triangles/vertices and the current production baseline are untouched.

## Results

Artifacts: [/mnt/data/dec5_mhr_silhouette_zero_margin](/mnt/data/dec5_mhr_silhouette_zero_margin). Comparison: [result.json](/mnt/data/dec5_mhr_silhouette_zero_margin/comparison/result.json); exact replay: [audit.json](/mnt/data/dec5_mhr_silhouette_zero_margin/fit100/audit.json); [final seal](/mnt/data/dec5_mhr_silhouette_zero_margin/final_seal.json).

All silhouette rows below use identical availability intersections across the original base, margin-2 control and margin-0 control. “Outside” uses bilinear signed-distance >0, not binary-mask admission.

| Check | Margin 2 | Margin 0 |
|---|---:|---:|
| Fit outside / fixed 78,370 samples | 1,225 | 470 |
| Fit mean positive SDF, px | .032651 | .011213 |
| Reserved outside / fixed 11,490 samples | 172 | 50 |
| Reserved mean positive SDF, px | .024386 | .005215 |
| Reserved >2 px outside, same denominator | 50 | 7 |
| Primary 44 first hits: actual binary-mask pass | 44 | 44 |
| Residual 30 first hits: actual binary-mask pass | 0 | 14 |
| Residual first hits: bilinear SDF≤0 in all cameras | 0 | 16 |
| Strict transverse crossing pairs, global | 182 | 245 |
| New crossing pairs vs original conformance base | 19 | 82 |
| >90° normal changes vs original base | 82 | 92 |

The binary/SDF distinction matters: **14/30**, not 16/30, is the actual residual point gate. Point agreement is still not triangle admission. No whole-prior replacement, raw candidate extraction, strict/interpolated admission, or completed-hole RGB claim is made.

The 100-step run took **20.00 s**, reproduced its new ten-step prefix exactly, and stopped **hard-cap-not-converged**: maximum unconstrained step `.00063205`, maximum displacement `.03327509` scene units. Every iterate and diagnostic history replayed exactly. No nonfinite or collapsed triangle was detected; minimum relative triangle area `.001579` nevertheless worsened from `.004177`. This cap is not evidence of optimizer convergence or impossibility of the objective.

Reserved measured-neck-candidate nearest-surface P90 is `.00012550` (original conformance base `.00013407`); face P90 `.00016921` is essentially unchanged. These nearest-surface statistics can hide sliding associations. The primary rim's actual 3D distance P90 worsens slightly `.00039308→.00045844`, still below the unchanged `.002` locality limit; signed-normal median shifts `−.00014554→−.00019230`. These are inferred-prior distances, not newly measured geometry.

### Native and topology gate

Actually inspected: [residual under-chin clay](/mnt/data/dec5_mhr_silhouette_zero_margin/comparison/residual_prior_clay.png), [primary hole prior](/mnt/data/dec5_mhr_silhouette_zero_margin/comparison/requested_hole_prior_only.png), [C/E native clay](/mnt/data/dec5_mhr_silhouette_zero_margin/comparison/C004_E005_1210X7_clay.png), [G/B native clay](/mnt/data/dec5_mhr_silhouette_zero_margin/comparison/G004_B005_1210FG_clay.png), the C/E train-RGB landmark-free projection panel, both C/E and G/B anatomical wire overlays, and both visibility-filtered unsafe-region crops. The inherited projection helper explicitly colors **>2 px** red; the new quantitative tables and binary checks separately evaluate zero tolerance.

The jaw/neck contour moves inward slightly, the primary local surface remains coherent, and pre-existing coarse upper-face folds remain. However, **57 facets participate in newly appearing strict crossing pairs relative to margin 2**. All 57 lie within the lower-head/neck candidate band and have anterior neutral centroids (median neutral y `143.84 cm`). Minimum distance to primary rim `.02114`, to residual hits `.02836`; none is a residual first-hit facet. Twenty-one newly >90°-changed facets lie ≥`.00572` from the primary rim. A normal change is not by itself proof of a triangle inversion; crossing counts use the independent strict transverse-intersection test.

The actual [G/B safety crop](/mnt/data/dec5_mhr_silhouette_zero_margin/fit100/safety_review/G004_B005_1210FG.png) shows visible collar/shoulder-region folds: 561 visible new-crossing pixels and 1,432 reversal pixels. The [C/E safety crop](/mnt/data/dec5_mhr_silhouette_zero_margin/fit100/safety_review/C004_E005_1210X7.png) has no visible unsafe pixels. Wire overlays are explicitly not visibility-filtered; the safety crops are raycast first-hit evidence. Thus improved local silhouette agreement does **not** pass the requested global “not worse” gate. Production admission was not launched.

### Reproduction and validation

Use `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1` with `/home/brans/repos/nerfstudio/.venv/bin/python`:

```text
scripts/study_mhr_zero_margin.py fit10
scripts/study_mhr_zero_margin.py fit100
scripts/study_mhr_zero_margin.py review
scripts/audit_mhr_zero_margin.py compare
scripts/audit_mhr_zero_margin.py audit
scripts/inspect_mhr_zero_margin_topology.py
scripts/review_mhr_zero_margin_safety.py
-m pytest tests/test_mhr_zero_margin.py -q -o addopts=''
scripts/seal_mhr_zero_margin.py
```

Three focused tests pass; the repository default pytest `-n=4` failed because this environment lacks that plugin, so the explicit `-o addopts=''` invocation was used without changing configuration. All fit/review/audit workers terminated cleanly. Protocols pin the generated offset-only source adapters, unchanged helpers, source/base arrays, depth receipts, masks and validation split; the seal rehashes all 62 depth maps and original geometry. The report-skill workflow was used to separate reproducible numerical evidence, actual image review, and the admission decision. PSNR/SSIM/LPIPS are N/A: no new RGB patch render or independent target reference is evaluated.

## Insights

The fitting/admission margin mismatch contributes to the residual silhouette error, but changing the offset alone is insufficient: it improves real binary-mask support while increasing global folded geometry. Preserve the previous margin-2 admitted control and this negative margin-0 workspace. No unsafe facets, whole-prior geometry, or new production asset are promoted. A topology-preserving fitting mechanism would need its own separately declared and audited control; it was not run here.
