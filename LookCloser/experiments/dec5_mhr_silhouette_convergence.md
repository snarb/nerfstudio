# Longer silhouette fitting reaches a plausible local jaw surface, with folds elsewhere

## What was tested

One continuation-budget control on frame 001193: the [previous silhouette-aware fit](dec5_mhr_silhouette_conformance.md), unchanged objective, weights, masks, measured anchors, calibration and global 0.001 step cap, now permitted up to 100 outer iterations. The original `smooth100` mesh remains the displacement/Laplacian reference throughout; the ten-step mesh is **not** used as a new regularization origin.

**Result:** the requested 44 first-hit points now pass the unchanged binary foreground masks in all 62 train cameras and lie close to original measured geometry. However, the full prior develops 19 new strict crossing pairs away from that region and still does not meet the declared convergence test. This is promising **local prior evidence**, not an accepted patch or whole-prior replacement.

Root: [/mnt/data/dec5_mhr_silhouette_convergence](/mnt/data/dec5_mhr_silhouette_convergence/protocol.json).

The sole optimizer-body modification is a post-step diagnostic/termination observer. Both original and instrumented source text/hashes are stored. First-ten vertices and history are bit-exact against the previous control; the independent audit also replays all 100 saved iterates exactly. Only the stopping budget changes the optimization.

Predeclared stopping: unconstrained maximum proposed step ≤1e-5 for three consecutive iterations, hard cap 100. A step is not considered converged merely because it was capped. Failure guards cover nonfinite states, triangle-area ratio ≤1e-6, increase in a fixed-IRLS quadratic after its own step, or three consecutive >25% increases in the re-evaluated nonlinear objective. No parameter or iteration selection uses reserved views.

The 1,566 active vertices remain `135 < neutral-y < 153` cm; all upper-face/body vertices are exactly fixed. Fit uses 54 train cameras and excludes the same eight reserved cameras from direct fitting and silhouette constraints. The original geometry and independently measured D_D mask override retain all-62 baseline provenance, so these are reserved-from-direct-fit checks, not wholly unseen-baseline evaluation. Dataset-held-out cameras and target RGB are never fit inputs. No original COLMAP vertex/triangle, texture, source, default or active 6K artifact changes.

## Results

### Convergence and objective trend

| Iteration | Re-evaluated nonlinear objective | Maximum unconstrained step |
|---|---:|---:|
| 1 | 148.3231 | 0.020510 |
| 10 | 124.2990 | 0.015774 |
| 20 | 76.3934 | 0.011237 |
| 40 | 4.2584 | 0.002934 |
| 60 | 3.4157 | 0.001338 |
| 80 | 3.4155 | 0.000598 |
| 100 | 3.4180 | 0.000690 |

The dimensionless diagnostic objective sums the pseudo-Huber data and silhouette terms corresponding to the frozen IRLS weights, plus squared Laplacian/magnitude terms. Association and available-FOV sets are recomputed and explicitly logged; this is not an image-quality metric. The quadratic decreases after each of its own trust-scaled steps. The nonlinear objective drops markedly, then has small late oscillations (minimum 3.4006). No predeclared numerical-instability failure triggers.

**Hard cap reached, not converged:** final unconstrained step 0.000690 remains well above 1e-5. Some steps remain capped as late as iteration 87. Final-iterate-only reporting preserves this failure rather than selecting an earlier favorable result. Runtime 19.52 seconds, CPU only.

### Reserved-camera silhouette and measured depth

The main comparison uses point-camera pairs available at both baseline and final iteration, avoiding changing FOV denominators. Excess means positive signed-distance violation beyond the unchanged two-pixel fitting allowance.

| Fixed cohort | Samples | Mean excess px, baseline→100 | P90 excess px, baseline→100 | Outside samples, baseline→100 |
|---|---:|---:|---:|---:|
| Fit cameras | 78,370 | 11.4505→0.0101 | 40.8594→0 | 16,127→481 |
| Eight reserved cameras | 11,490 | 14.3487→0.0040 | 55.2899→0 | 2,300→50 |
| Reserved canonical-front vertices | 7,699 | 9.6175→0.0051 | 22.4149→0 | 1,285→45 |

P90 zero does **not** mean zero violations: the remaining 50 reserved common-cohort outside samples have median/P90 excess 0.607/2.236 px. On all final available reserved samples, including newly in-frame vertices, 104/12,409 violate the allowance, with nonzero-tail P90 15.75 px. This tail and denominator difference remain explicit.

Reserved face-anchor surface P90 stays 0.0001692; reserved neck-candidate P90 improves 0.0001341→0.0001207. Surface-distance agreement can still involve sliding associations and is not an interior-depth certificate. No patch admission is performed here.

### Local requested-ray feasibility improves substantially

The same original 44 missing rays and 24-point rim are checked only after fitting; they do not select fitted vertices or constraints.

| Posthoc local check | Starting prior | 100-step prior |
|---|---:|---:|
| Prior first hits at original missing rays | 44 / 44 | 44 / 44 |
| First hits passing all available masks, 2 px allowance | 3 / 44 | 44 / 44 |
| First hits passing **unchanged binary masks**, no allowance | not evaluated here | **44 / 44** |
| First-hit distance to original surface, median / P90 | 0.000443 / 0.000655 | 0.000178 / 0.000300 |
| Rim distance to prior, median / P90 | 0.000614 / 0.000900 | 0.000149 / 0.000393 |
| Signed original-normal rim offset, median / P90 | +0.000515 / +0.000824 | −0.000146 / −0.000103 |

All 44 final points have **62 supporting binary-mask cameras and zero disagreement** under the frozen admission helper. Their worst-view signed-distance margins are at least 6.99 pixels inside foreground, so this is not a benefit obtained by allowing two-pixel spill. Their first-hit camera-z shifts +0.008661…+0.010393 (median +0.009621), consistent with the separately diagnosed deeper silhouette-feasible region. Distance values use the normalized calibrated scene gauge, not metres.

![Native train projection: the misplaced under-chin cluster retracts inside silhouette](/mnt/data/dec5_mhr_silhouette_convergence/review_v2/C004_E005_1210X7_projection.png)

Main inspected the requested prior-only crop, C_E clay and G_B projection. This agent inspected C_E projection/clay and the requested crop: the neck now moves deeper and closer to the real silhouette. The coarse nose/eyes remain inherited defects; no whole-prior replacement is warranted. [Requested prior-only comparison](/mnt/data/dec5_mhr_silhouette_convergence/review_v2/requested_hole_prior_only.png).

### New topology failures remain unacceptable outside a tightly guarded local use

| Geometry check | Starting prior | 100-step prior |
|---|---:|---:|
| Strict nonadjacent crossing pairs | 163 | 182 |
| New crossing pairs | — | 19 |
| >90° normal changes versus starting prior | — | 82 |
| Minimum triangle-area ratio to starting prior | 1 | 0.00418 |

Normal reversal alone is not proof of self-intersection: the additional 19 pairs have strict transverse segment-through-triangle witnesses. They involve 20 anterior lower-neck facets with neutral vertex-y 137.70…144.05 cm, and lie at least 0.025934 from the requested original rim. Reversed-normal facets are at least 0.017325 from that rim. None of the 44 requested first-hit facets belongs to either unsafe set.

The new crossings project near the lower-neck/shoulder junction. Native G_B review exposes 250 red crossing pixels and 1,418 amber reversal pixels; these are actual visible failures. C_E sees none of those unsafe facets in its corresponding crop, not evidence that the folds are absent. The inherited 163 upper-face crossings also remain invalid; they are not excused by prior existence.

![Visible unsafe lower-neck/shoulder facets in G_B](/mnt/data/dec5_mhr_silhouette_convergence/safety_review/G004_B005_1210FG.png)

Active displacement median/P90/max is 0.001096 / 0.015461 / 0.033089. The smallest relative triangle shrinks to about 0.42% of its initial area: above the declared numerical-collapse threshold but still substantial distortion. The first >90° normal changes arise at iteration 11. Original COLMAP remains exact and separate; these failures belong to the inferred prior only.

No synthesized RGB reconstruction or PSNR/SSIM/LPIPS evaluation is claimed. These metrics are N/A for this prior-only feasibility study; RGB panels are calibrated real train references.

## Insights

The previous ten-step limit was a material cause of insufficient neck retraction. With exactly the same objective and reference, longer fitting reaches a substantially more plausible local underside without sacrificing measured anchor accuracy. Thus the earlier bounded result did not refute silhouette-aware fitting.

More iterations alone do not solve everything: the optimizer remains unconverged and the unconstrained surface develops folds away from the requested region. This is **not** a globally safe human-shape replacement. The useful evidence is narrowly local and still requires separate generic patch extraction, exclusion of every unsafe facet, unchanged measured-depth/free-space admission, and native review. No patch or additional experiment is launched here.

### Reproduction and audit

Run `scripts/continue_mhr_silhouette_convergence.py` with reconstruction Python and `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`; output root must be fresh. The wrapper verifies the frozen previous producer and helpers, stores the exact optimizer-source instrumentation and all 100 iterates, and preserves the original base reference.

`scripts/review_mhr_silhouette_convergence.py` rebinds the two unchanged native/locality diagnostic producers. `scripts/audit_mhr_silhouette_convergence.py` reconstructs the train-only inputs, exactly replays all 100 iterates and diagnostics, verifies first-ten identity and fixed inactive geometry, evaluates common-availability cohorts, and localizes unsafe triangles. `scripts/review_mhr_silhouette_continuation_safety.py` replays the unchanged zero-allowance binary-mask guard and records calibrated native unsafe-facet crops.

Two continuation tests verify the optimizer-body-only observer insertion and the bounded convergence settings; the original three projection/SDF tests remain applicable. The [final seal](/mnt/data/dec5_mhr_silhouette_convergence/final_seal.json) binds producer/config/helpers, all receipts, both full iterate histories, tests, report and native review evidence. All failed geometry is retained. Nothing is promoted to production or the current 6K baseline.
