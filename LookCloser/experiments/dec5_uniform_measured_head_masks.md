# DEC5: uniform measured head-mask refinement

## What was tested

2026-09-15. Follow-up to [close-boundary transfer](dec5_close_boundary_transfer.md).
Hypothesis: conservative foreground masks reject genuine hair-boundary measurements;
a uniform, train-only multiview refinement could admit more useful completion without
relaxing the depth/free-space checks. This is an opt-in two-frame diagnostic, not a
new production recipe. The user now permits semantic/generated masks. This experiment
uses measured refinement of existing foreground masks, **not** newly generated masks
or separate hair/skin segmentation. No held-out view or target image is used to build it.

First, raycast the old mesh, unfiltered Poisson proposal, and previously admitted mesh
at moving and real train views for `001083` and `001123`. Re-evaluate ten-point depth
support even for proposals previously rejected before the depth stage. Raw proposal
coverage is diagnostic: these rectangles include actual background, and a raw triangle
covering a black pixel is not proof of correct anatomy.

Then refine all 62 original camera masks with one identical rule:

- Query in the original mask's 24-native-pixel exterior band, with finite measured
  PatchMatch depth and normalized head-band coordinate `x > -0.03`.
- At least three **other** physical cameras agree with measured depth within 0.001,
  roundtrip within 1.5 pixels, parallax at least 1 degree, calibrated 5×5 patch
  chromaticity error at most 0.04 and RGB error at most 0.12.
- Witnesses must lie in **original** foreground masks. No iterative bootstrap from
  newly added pixels. Preserve all original mask pixels; dilate certified seeds by
  four pixels, clipped to the exterior band.
- Use fixed exposure and the renderer's mean-centered camera profiles. Source texture
  masks remain unchanged. The added dilation is an uncertain boundary, not a precise matte.

Apply these masks only to semantic admission of the **exact same raw Poisson proposal**
as the previous study. Keep depth support, observed-seed local certificates, distance,
edge, normal and 62-camera final free-space gates unchanged. Do not regenerate Poisson.
The original mesh prefix is preserved; additions are inferred, not directly measured
or guaranteed watertight.

Render six fresh controls with current incidence-2, zero-RGB-warp, hard-source RGB.
Reuse the two production moving baselines. Native controls use exact real calibration
(`E004_B005_1210I7` for 083, `G004_A005_121071` for 123), but alias the **target name**
so the source-mask wrapper does not inadvertently mask the target raycast. Both native
baseline and candidate use this alias; source cameras/masks remain unchanged.

## Results

### Rejection diagnosis

| Frame / view | Raw proposal covering old missing-depth pixels | Pixels on raw faces rejected by masks | Pixels with depth-support gate satisfied, no free-space veto, but mask rejected |
|---|---:|---:|---:|
| 001083 moving | 7,719 | 7,719 | 952 |
| 001083 native train | 4,237 | 4,184 | 674 |
| 001123 moving | 6,137 | 6,135 | 1,402 |
| 001123 native train | 2,978 | 2,978 | 450 |

Masks dominate this proposal's rejection, but many rejected faces contradict dozens
of masks. For example, 083 moving depth-supported candidates receive 12–53 mask vetoes.
This does **not** justify simply allowing a majority vote or bypassing masks. Raw
completion visually includes both plausible hair bridges and questionable outer caps.

### Uniform refinement and guarded completion

| Frame | Native queries replayed | Certified seed pixels | Added mask pixels across 62 cameras | Added mesh triangles, original masks → refined masks |
|---|---:|---:|---:|---:|
| 001083 | 1,028,811 | 230,558 | 636,888 | 540 → 1,134 |
| 001123 | 973,366 | 222,889 | 618,314 | 472 → 2,358 |

Counts across camera masks are not unique 3D area or quality metrics. An independent
audit recomputed every query, witness count, seed, dilation and final mask for all 124
frame/camera combinations. New masks never become witnesses. Each final mesh also
passes 124 native free-space ray checks and independent observed-seed certificate replay.

### Matched RGB/depth comparison against production

| Frame / view | New depth pixels | Lost depth pixels | Changed RGB pixels | Black removed / introduced |
|---|---:|---:|---:|---:|
| 001083 moving | 5 | 0 | 18 | 3 / 0 |
| 001083 native unmasked | 83 | 0 | 104 | 84 / 1 |
| 001123 moving | 32 | 0 | 78 | 36 / 0 |
| 001123 native unmasked | 26 | 0 | 96 | 26 / 0 |

These are pixel-change diagnostics, **not full-frame PSNR/SSIM/LPIPS or anatomical
coverage metrics**. No new quality metric is computed in this diagnostic. No claim
that the small number of covered pixels establishes a successful hair repair.

Visual review: four raw-proposal panels, eight native mask panels, four matched crown
panels and two native jaw panels. Refined masks recover plausible curls cut by the old
contour, but the four-pixel dilation includes mixed background boundary pixels. The
conspicuous crown notches remain in all four final crown comparisons. Jaw comparisons
show no meaningful visible repair. Verdict: **not promoted; insufficient visible repair**.

Artifacts:

- [Raw rejection diagnostics](/mnt/data/dec5_crown_completion_rejection)
- [Saved per-camera masks and evidence](/mnt/data/dec5_uniform_measured_head_masks)
- [Independent mask replay](/mnt/data/dec5_uniform_measured_head_masks/audit.json)
- [Geometry, matched RGB and review](/mnt/data/dec5_measured_head_mask_completion)
- [083 crown vs train GT](/mnt/data/dec5_measured_head_mask_completion/review/001083/native_unmasked_crown.png)
- [123 crown vs train GT](/mnt/data/dec5_measured_head_mask_completion/review/001123/native_unmasked_crown.png)
- [Visual verdict](/mnt/data/dec5_measured_head_mask_completion/visual_review.json)
- [Combined audit](/mnt/data/dec5_measured_head_mask_completion/audit.json)

The sealed artifact manifest contains 544 hashes. Five focused tests pass. All study
workers are terminal. Production geometry, RGB defaults and running trajectory variants
were not changed by this experiment.

Reproduce the final verification:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/freeze_measured_head_mask_completion.py --check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python -m pytest -q -o addopts='' tests/test_uniform_measured_head_masks.py tests/test_measured_foreground_override.py
```

## Insights

1. Mask errors are real, but repairing a narrow measured boundary band is not enough
   to reconstruct the missing crown. More admitted triangles do not imply visible repair.
2. Keep semantic evidence separate from geometry and texture evidence. A semantic mask
   can authorize a region without proving its depth; dilation must not silently authorize
   background RGB sampling. The saved masks can support a later semantic-mask comparison.
3. Generated/learned masks are now permitted but must still be checked against native
   RGB, measured depth and multiple physical cameras. This experiment neither evaluates
   nor rules out those methods, or a genuinely new surface proposal.
4. Preserve the native-target wrapper fix in future diagnostics: otherwise real-camera
   target masking can hide the very geometry a mask experiment is trying to evaluate.
5. Do not promote this arm into 150-frame rendering. The separate trajectory work remains
   a visibility workaround, not evidence that these geometric defects have been repaired.
