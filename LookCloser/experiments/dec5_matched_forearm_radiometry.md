# Same-surface forearm radiometry and bounded-shift diagnosis

## What was tested

After [hard ownership failed](dec5_forearm_texture_ownership.md), test whether
the remaining `001037` wrist patch is caused by frozen camera gains or small
registration errors. Keep the multiview-admission mesh and moving target camera
fixed. Reproject actual triangle-barycentric surface hits to six physical train
cameras (indices 26,31,32,33,37,42). Compare bilinearly sampled **linear EXR** with
one fixed exposure, before/after the existing fixed camera RGB profiles. No new
profile, geometry, prediction, or exposure is written; no held-out RGB is loaded.

Source visibility uses the existing source masks, mesh depths, and four-tap
footprint gate. Pair comparisons use warm-color overlap in both cameras, not
candidate-dependent quality ROIs. These are diagnostic differences, not face
PSNR/SSIM/LPIPS. Same-point projection on an inferred mesh is not proof of true
anatomical correspondence.

For four source pairs, test all 81 integer native-image shifts in `[-4,4]^2`.
Every trial uses **one common intersection of all trial visibility domains**.
Also subtract a constant RGB offset fitted on the upper spatial half and test
it on the lower half, then reverse; this is an explanatory control, not a new
camera calibration or deployed correction.

## Results

**Turning off the fixed camera profiles does not remove the principal wrist
brightness disagreement. Small registration shifts do not explain it either.**

Median absolute RGB difference, in display levels out of 255; overlap is fixed
within each row. Pair names abbreviate physical camera columns/rows.

| Pair | Shared pixels | Fixed exposure only | Existing profiles |
|---|---:|---:|---:|
| I/B–J/B (37–42), wrist | 6,944 | 23.66 | 23.11 |
| H/C–J/B (33–42), all | 13,515 | 16.60 | 12.61 |
| H/B–I/B (32–37), wrist | 5,605 | 3.38 | 9.31 |
| H/A–J/B (31–42), all | 15,039 | 11.71 | 10.28 |

Profiles help some pairs and hurt H/B–I/B; they are not perfectly calibrated
for this region. But I/B and J/B have very similar frozen gains, and their large
disagreement exists without them. It is not a newly introduced per-frame gain.

The stricter common-domain shift control reduces sample counts:

| Pair | Pixels | Zero-shift difference | Lowest difference in ±4 px | Offset (native x,y) |
|---|---:|---:|---:|---|
| I/B–J/B | 5,626 | 23.14 | 18.44 | −4,+3 |
| H/C–J/B | 10,810 | 12.33 | 10.96 | −2,−4 |
| H/B–I/B | 3,417 | 9.11 | 4.99 | +4,−4 |
| H/A–J/B | 12,288 | 10.11 | 9.40 | −4,−4 |

All color-error optima reach a search boundary; they must **not** be treated as
recovered registration. I/B–J/B has centered RGB correlation 0.98794 already at
zero shift. Its best-correlation offset (+2,−1) increases color disagreement to
25.97 levels, despite correlation rising to 0.99019. Matching smooth structure
and matching absolute brightness are distinct objectives.

For I/B–J/B a constant offset fitted on one spatial half reduces the other
half's difference from 23.96 to 3.50 levels, or 22.78 to 3.85 in reverse.
This supports a locally broad radiometric mismatch. It does **not** demonstrate
a camera-wide fixed response: H/C–J/B varies spatially, and fitting its lower
half worsens the upper half from 7.14 to 8.93 levels. Smooth shading differences,
view-dependent reflection, geometric correspondence error and spatial sensor
response are not individually separated by this one-time experiment.

Visual inspection covered twelve images: six native source crops with matched
point markers, two six-source same-surface sheets, and four shift comparisons.
The broad skin brightness discrepancy remains visible. I/B's photographed
image ends near the upper wrist; it cannot texture the whole forearm. No panel
is an accepted repaired image or an artifact-free full-frame result.

- [Current RGB and profiled source projections](/mnt/data/dec5_matched_forearm_radiometry/profiled_same_surface.png)
- [Same projections without camera profiles](/mnt/data/dec5_matched_forearm_radiometry/fixed_only_same_surface.png)
- [I/B native photograph](/mnt/data/dec5_matched_forearm_radiometry/I004_B005_1210T5_native.png)
- [J/B native photograph](/mnt/data/dec5_matched_forearm_radiometry/J004_B005_1210GR_native.png)
- [I/B–J/B shift control](/mnt/data/dec5_forearm_radiometry_shift/37_42.png)
- [Manual diagnostic verdict](/mnt/data/dec5_forearm_radiometry_shift/visual_review.json)

The replay of selected source RGB agrees within **one uint8 level** with the
saved renderer. Three pixels differ at the four-tap visibility threshold under
CPU float64 versus CUDA sampling; exact visibility parity is not claimed.
Three tests cover bilinear sampling/borders against Torch, fixed-domain
statistics/abstention, and correlation's invariance to constant color offsets.
Both CPU workers completed; there was no new RGB video render in this study.
Audit: `scripts/freeze_matched_forearm_radiometry.py --check`.

## Insights

Do not disable all profiles or apply best-color registration shifts as a fix.
A useful next **experimental** correction is spatially regularized, mesh-space
low-frequency seam leveling, leaving the selected camera's high-frequency
texture intact. It must be fitted on train overlaps, tested on other views and
times, and kept distinct from fixed exposure/calibration; this diagnosis alone
does not establish that it will succeed. No correction is deployed here.

Forearm surface/hand defects and cheek holes remain separate geometry problems.
The production video is unchanged. In parallel, an independent agent is testing
user-requested trajectory workarounds; hiding a defect is not mesh recovery.
