# DEC5 001193: a static vertex freeze removes crossings but fails the fit gate

## What was tested

**Rejected fit-only control; no patch admission or production change.** One fixed exclusion set was derived from the failed [zero-margin fit](dec5_mhr_silhouette_zero_margin.md): vertices incident to its new strict crossing pairs or >90° normal changes relative to the original measured-conformance base. There are 133 implicated parent facets and **120 originally active vertices**. Those vertices are fixed at the **original smooth100 base**, not at the failed fit. The remaining **1,446** vertices are refitted from that same base. No target ROI, residual ray, camera exception, iterative exclusion enlargement, or parameter sweep selects the set.

The zero-offset objective, 54-fit/8-reserved split, anchors, masks, association rules, step cap and 100-iteration stopping protocol are unchanged. Normalization remains **1,566**, the original active count. The Laplacian retains all original 1,566 rows and only restricts its columns; all available original silhouette rows remain, including fixed-vertex constant residuals with zero derivative. Magnitude/Laplacian reference never resets. Original geometry and all vertices outside `135<neutral_y<153` remain exact.

The original COLMAP reconstruction and independent D_D measured-mask override have all-62-train provenance. Reserved cameras are excluded from fitting/anchor voting, but are not unseen-from-baseline evaluation. The three true held-out RGB views remain unused. This is a human-shape inference control, not new measured surface or a full-prior replacement.

## Results

Root: [/mnt/data/dec5_mhr_static_unsafe_freeze](/mnt/data/dec5_mhr_static_unsafe_freeze). [Comparison](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/result.json), [exact replay](/mnt/data/dec5_mhr_static_unsafe_freeze/fit100/audit.json), [restriction attribution](/mnt/data/dec5_mhr_static_unsafe_freeze/restriction_tradeoff.json), [final seal](/mnt/data/dec5_mhr_static_unsafe_freeze/final_seal.json).

All silhouette comparisons below use the identical availability intersection across original base, margin-2, unrestricted zero-margin and frozen-vertex fits. SDF counts are bilinear signed distance >0; point-mask counts use the actual unchanged binary-mask helper.

| Check | Margin 2 | Unrestricted zero | Static freeze |
|---|---:|---:|---:|
| Fit SDF outside / 78,370 | 1,225 | 470 | 4,081 |
| Fit mean positive SDF, pixels | .032651 | .011213 | 3.802567 |
| Reserved SDF outside / 11,490 | 172 | 50 | 539 |
| Reserved mean positive SDF, pixels | .024386 | .005215 | 4.515988 |
| Primary actual binary-mask pass / 44 hits | 44 | 44 | 44 |
| Residual actual binary-mask pass / 30 hits | 0 | 14 | 15 |
| Global strict crossing pairs | 182 | 245 | 163 |
| New strict pairs vs original smooth100 | 19 | 82 | **0** |
| >90° normal changes vs original smooth100 | 82 | 92 | **100** |

The new-crossing half of the gate passes, but the normal-change half fails. This was explicitly a **zero-new-crossings AND zero-new->90°-normals** gate; it was not changed after seeing the result. A >90° normal change alone does not prove an actual self-intersection or intrinsic triangle inversion. The independently replayed strict intersection test returns exactly the original 163 inherited pairs; those inherited folds are not accepted whole-prior geometry.

The 100-step fit took **19.27 s** and remains **hard-cap-not-converged**: final unconstrained step `.00072117`, maximum displacement `.02801134` scene units. Minimum relative triangle area `.07930`; no nonfinite/collapsed/instability stop. Every saved iterate replayed exactly, including the fresh ten-step prefix, and all 120 fixed vertices were checked exact at every step. The free-variable solve has 4,338 columns; the Laplacian has 4,698 rows, exactly the original 1,566×3. An independent random-displacement check verifies the restricted and full operators have identical energy when frozen coordinates are zero.

Reserved neck-candidate nearest-surface P90 changes `.00012550→.00013190` from unrestricted zero; reserved face P90 remains `.00016921`. Nearest-surface residuals may hide sliding associations. Primary rim distance P90 is `.00044793`, signed-normal median `−.00021595`, still within the unchanged `.002` locality limit. Neither the primary nor residual first-hit facets belongs to the remaining unsafe sets; none of these point checks constitutes admitted triangle geometry.

### Why the restriction fails

Frozen-vertex projections are **bit-exact to the wrong original-base silhouettes**. On the fixed evaluation cohort, 3,324/4,115 fitting samples and 445/506 reserved samples on the fixed vertices remain outside. They contribute **98.72%** and **98.99%** of total positive fitting/reserved SDF respectively. The free vertices are still substantially better aligned (reserved mean positive SDF `.04760`), but cannot move the locked original bulge. Restoring the fixed vertices moves some positions up to `.03163` scene units away from the failed zero fit.

Of the 100 normal-changed faces, 31 have one frozen vertex, 30 have two, and 39 have none; no face has all three frozen. Thus freezing previous unsafe sites does not prevent new orientation problems elsewhere or at fixed/free boundaries. Remaining normal-changed facets are at least `.004002` from the primary rim; their neutral centroids split 45 anterior / 55 posterior. No further exclusion set was tried.

### Native visual gate

Actually inspected [C/E](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/C004_E005_1210X7_clay.png), [G/B](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/G004_B005_1210FG_clay.png), [M/B](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/M004_B005_12109O_clay.png), [primary crop](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/requested_hole_prior_only.png), [residual crop](/mnt/data/dec5_mhr_static_unsafe_freeze/comparison/residual_prior_clay.png), and both visibility-filtered safety panels. The local under-jaw surface is similar, with only one additional residual point passing. The larger [G/B safety panel](/mnt/data/dec5_mhr_static_unsafe_freeze/fit100/safety_review_empty_safe/G004_B005_1210FG.png) exposes a pronounced coarse protruding neck/shoulder surface against real train RGB; the [C/E panel](/mnt/data/dec5_mhr_static_unsafe_freeze/fit100/safety_review_empty_safe/C004_E005_1210X7.png) also has an incorrect lower contour. This is not an acceptable full prior.

Neither safety view exposes a first-hit normal-change-colored pixel, so the report does **not** claim the 100 normal changes are visually proven crossings. Crops are chosen from those facets' projected locations, not from target error. The geometric audit and markedly worse calibrated silhouettes establish rejection independently of their visibility.

### Reproduction and audit

Use `/home/brans/repos/nerfstudio/.venv/bin/python` with `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1`:

```text
scripts/study_mhr_static_unsafe_freeze.py fit10
scripts/study_mhr_static_unsafe_freeze.py fit100
scripts/study_mhr_static_unsafe_freeze.py review
scripts/audit_mhr_static_unsafe_freeze.py
scripts/review_mhr_static_unsafe_freeze.py
scripts/review_mhr_static_freeze_safety.py
scripts/explain_mhr_static_freeze.py
-m pytest tests/test_mhr_static_unsafe_freeze.py -q -o addopts=''
scripts/seal_mhr_static_unsafe_freeze.py
```

Four tests pass. The first reused audit and safety helper assumed a nonempty new-crossing set and failed during empty-set reporting **after fitting**, not during optimization. Their outputs/logs remain preserved. New, hash-bound adapters handle only that empty diagnostic set and put retries in separate paths; frozen helpers and fitted artifacts were not changed. The complete 100-step audit then passed, and the sealer independently recomputes global intersections/normals. All input/source/mask/depth bindings from the sealed zero control are revalidated. Report-skill guidance separates measured evidence, inference and the decision. PSNR/SSIM/LPIPS are N/A: no new textured patch or independent image-quality reference was evaluated.

## Insights

A static unsafe-site freeze can eliminate new transverse crossings, but it pins the original silhouette error and is not a topology-preserving fitting mechanism. The agreed combined gate fails; no candidate extraction, depth admission, production mutation or video rerender was launched. Preserve this negative control and the earlier admitted margin-2 baseline. This bounded test does not establish that a constrained deformation objective is impossible.
