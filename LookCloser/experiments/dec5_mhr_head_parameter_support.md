# MHR: head-parameter anatomical support before scene fitting

## What was tested

2026-09-15. Follow-up to the [public-model preflight](dec5_mhr_head_prior_preflight.md).
The neutral template contains an underside and neck, but this does not imply
that its head identity parameters can fit both. We vary coefficients 20–39
individually by ±1, keeping all body, pose and expression values zero. Forty
CPU model evaluations plus the neutral control use the pinned official asset.
No DEC5 images, camera poses, landmarks, depth maps or meshes enter this probe.

For each coefficient we retain the centered derivative `(plus-minus)/2` and
the symmetric residual `(plus+minus)/2-neutral`, in model centimeters. Review
bands are fixed neutral-coordinate ranges, **not anatomical GT segmentations**:
head `y>=145`, neck `135<=y<145`, lower-front `145<=y<=153,z>=0`, and body below
`y=135`. The lower-front band is a subset of the head band.

## Results

| Neutral spatial band | Vertices | Min / max coefficient RMS displacement, cm |
|---|---:|---:|
| Head | 5,919 | 0.04369 / 0.35466 |
| Lower front | 598 | 0.03399 / 0.61123 |
| Neck | 700 | 0.000878 / 0.02517 |
| Below neck | 11,820 | 0 / 0 |

The symmetric residual is at most 0.0000306 cm in these tests, consistent with
nearly linear identity deformation at the tested neutral pose. Below-neck
vertices are exactly unchanged for all forty tested perturbations. Singular
values and full per-coefficient values are retained; tiny neck singular values
should not be treated as meaningful anatomical fitting capacity.

Main LLM inspected the low-oblique ±1/neutral panels for coefficients 20, 21
and 23, selected by largest lower-front RMS. They show coherent lower-face/chin
shape changes, while the broad neck and shoulder remain nearly fixed. No scene
alignment or repair quality follows from these generic-template images.

![Lower-jaw basis variations](/mnt/data/dec5_mhr_head_parameter_support/lower_jaw_parameter_variants.png)

Root: `/mnt/data/dec5_mhr_head_parameter_support`. Full vertices, derivatives,
spatial bands, model/script hashes and the native review panel are retained.
Two focused tests check centered linear/quadratic separation and invalid inputs.
The independent audit recomputes differences, band statistics, the spectrum and
artifact hashes. This probe modifies no production state or model defaults.

## Insights

A head-only fit is a reasonable first control for the lower jaw, but is nearly
a fixed-neck fit. If head alignment succeeds while neck residuals remain biased,
this is a possible parameter-domain limitation, not necessarily an optimization
failure or proof that all human priors are ineffective. Conversely, head
expressiveness alone is not evidence that the requested missing surface can be
reconstructed correctly.

The parallel bounded scene fit retains its frozen similarity/head20 controls.
It must report head/underside/neck support separately. Do not expand tolerances
to conceal a boundary mismatch or insert a front surface at the wrong depth.
