# DEC5 001193: positive contact clearance and silhouette priority

## What was tested

Continuation of [guarded anatomical correction](dec5_mhr_guarded_correction.md).
The remaining under-chin hole is **not** repaired by the earlier prior: only
6/30 residual rays survive its semantic-only local proposal probe, before any
measured-depth admission. No production mesh or delivered 6K video is changed.

`run_mhr_clearance_correction.py` retains the anatomical model-space domain,
54 fitting / eight reserved train cameras, original warm reference, 40 outer
iterations, .001 maximum displacement and all-pair endpoint guard. It changes
contact proposals in three explicit ways:

- positive separation `1e-8` normalized scene units instead of zero;
- additional in-plane axes for coplanar pairs;
- detect all reported pairs when generating constraints, not only transverse ones.

This is a combined contact-proposal correction, not an ablation attributing all
effects to the margin alone. It reads no target pixels or residual-ray offsets.
The margin applies to the **full proposed endpoint**; subsequent trust-step
scaling/backtracking may reduce it. Final acceptance still rejects every new
reported all-pair contact. No universal final minimum gap or continuous-collision
certificate is claimed.

`run_mhr_silhouette_weight_control.py` adds one distinct, frozen control:
silhouette weight4→16, with the same clearance/domain/geometry bounds. This
changes the objective deliberately; it does not modify masks, cameras, source
data, any admission threshold or existing model/runner defaults. The executed
optimizer and recorded recipe both explicitly contain weight16. Posthoc render
and ray tests never enter fitting.

## Results

### Clearance control

The fit completes 40 iterations in **246.77 s**, versus 349.32 s for the anatomical
zero-gap control in the previous run. These are single wall-clock observations,
not a repeated speed benchmark. Both stop at the iteration cap, not convergence.

Final all/strict pair inventory remains 182/182, with no new pair or original
normal reversal. Minimum relative area is .63547 and normal cosine .90255;
maximum displacement .000734483. The independent guard audit replays all 40
iterates. Numerical movement continues late in the run, so the iteration cap
must not be relabelled a convergence certificate.

The [shared-cohort review](/mnt/data/dec5_mhr_clearance_review/result.json) uses
identical canonical vertex IDs and availability: 54,193 fitting and 8,048
reserved camera/vertex samples. These are geometric fit diagnostics, not
image-quality metrics.

| Fit | Fitting outside / mean positive SDF (px) | Reserved outside / mean positive SDF (px) | Residual first-hit mask passes /30 |
|---|---:|---:|---:|
| Anatomical zero-gap reference | 337 / .005812 | 15 / .002289 | 9 |
| Positive-clearance control | 336 / .005818 | 15 / .002286 | 9 |

No complete coarse parent triangle passes the binary multi-camera mask test in
either fit. Parent viewed the actual residual and C/E, G/B, M/B comparison PNGs
in [/mnt/data/dec5_mhr_clearance_review](/mnt/data/dec5_mhr_clearance_review).
They show no useful visible under-chin change; inherited full-prior eye/lip
folds remain. This control improves proposal handling, **not the hole**. It is
not promoted and was not rendered as a new RGB/video candidate.

### Precision qualification of the earlier contact diagnosis

The [independent review](dec5_mhr_clearance_independent_review.md) verifies the
constraint equations on 108 seeded cases, four noncollapsing rigid controls,
and twelve specified actual triangle pairs. Three new parent unit tests pass.
The review is not actor-quality approval.

The eleven previously reported new conic pairs have tiny positive separating
gaps in stored float64; so does the last rejected anatomical pair11510/11596.
Thus an Open3D pair report alone is **not exact proof of geometric intersection**.
However, seven of those twelve pairs lose their positive separating witness
when coordinates are rounded to float32, as used by the raycaster. Coordinate
rounding is larger than their original gaps. A missing positive-axis witness
is also not asserted to prove penetration. These results qualify the previous
near-contact interpretation and do not justify relaxing the final guard.

### Silhouette weight16 control

The fit finishes 40 iterations in **247.06 s**, again at the iteration cap,
not demonstrated convergence. Its maximum displacement reaches .000999939,
close to the unchanged .001 bound. Final minimum relative area is .46591 and
normal cosine .81937, with no new original normal reversal or reported pair.

The [weight16 shared-cohort review](/mnt/data/dec5_mhr_weight16_review/result.json)
uses exactly 54,193 fitting and 8,048 reserved samples across all three arms.
One extra available fitting sample in the candidate's own result is excluded
from this comparison rather than changing the denominator.

| Fit | Fitting outside / mean positive SDF (px) | Reserved outside / mean positive SDF (px) | Residual first-hit mask passes /30 |
|---|---:|---:|---:|
| Positive clearance, weight4 | 336 / .005818 | 15 / .002286 | 9 |
| Positive clearance, weight16 | 299 / .002314 | 13 / .001783 | 14 |

The reserved maximum positive SDF worsens from 3.06483 to 3.19653 pixels despite
the smaller mean: the improvement is not uniform. All 30 rays hit the fitted
prior, but no complete original parent triangle passes semantic admission.
These 14 point passes do **not** establish 14 filled production pixels.

Parent directly inspected `residual_clay.png`, `C004_E005_1210X7.png`,
`G004_B005_1210FG.png`, and `M004_B005_12109O.png` in the review root. The
under-jaw surface shifts modestly; no obvious new large fold is visible in these
four comparisons. Inherited eye/lip folds and faceting remain. This is a local
prior control, not an accepted complete human surface or artifact-free render.

All 40 saved iterates pass the independent geometry replay (81 bound files);
the subsequent geometry/provenance-only seal records the four actually viewed
comparisons. It is not anatomical or production approval.

The [subdivided candidate probe](/mnt/data/dec5_mhr_weight16_candidates/posthoc_probe/result.json)
retains the production mesh's exact 61,860-vertex / 120,073-triangle prefix.
Of 698,998 local proposals, 490,895 pass semantic-only gates. On the same 30
residual rays, the original mesh hits zero, the raw appended proposal hits all
30, and the semantic-only proposal hits **10**, leaving **20 misses**. The old
anatomical control hit six. This is a modest partial improvement, not a complete
repair: neither measured-depth admission nor new RGB/video rendering was run.

### Residual attribution and the next algorithmic control

The [posthoc attribution](/mnt/data/dec5_mhr_weight16_residual_diagnosis/result.json)
replays the saved first-hit point veto counts exactly. All remaining 16 point
vetoes come from fitting cameras; none comes from reserved cameras. A/B vetoes
one point, A/C ten, and B/B ten (sets overlap). Their maximum positive bilinear
SDFs are .09813, .50118 and 1.29748 pixels respectively. Binary-vs-bilinear signs
disagree on only one B/B point, so sampling-rounding alone does not explain the
remaining hole.

The affected parent triangles are 8113, 8226, 8229 and 8316, using vertices
3408, 3409, 3495, 3496, 3596, 3597. Their largest displacement is only .00018283;
none reaches 99% of the .001 bound. Three unrelated active vertices do reach
that fraction. Therefore the global maximum-displacement statistic must not be
used to claim that the local hole is limited by the displacement cap.

A direct read-only `project_jacobian` / `sample_sdf` check on these six vertices
in B004_B005_1210Z3 returns signed distances
`[-6.37820, -3.83951, -1.99455, -1.44082, -.59436, -4.54509]` pixels: **every
vertex is inside**, although eleven fitted triangle-interior ray points have
positive SDF (ten binary vetoes). A/C's largest vertex SDF is .01690 pixels,
whereas its interior reaches .50118. This is direct evidence of an objective
sampling blind spot: vertex-only silhouette penalties cannot represent the
outside excursion between vertices along this nonconvex projected boundary.
It is not evidence that those masks are wrong.

The next justified control is a train-only, barycentric surface-sampling term
whose Jacobian distributes each interior sample's gradient to its three mesh
vertices. Sample locations must be selected generically from mesh geometry,
not from these diagnostic target rays. Keep the current masks, displacement
and topology safeguards and the measured-depth admission contract. This tests
the objective mismatch instead of another unmotivated bound relaxation.

Thirty focused tests pass across the geometry solvers/guards, positive clearance
and new weight adapter. Adapter tests cover executed math and saved metadata,
disjoint output roots, wrapper binding and fail-closed parent-source changes.
The scientific diagnostic itself is checked against the actual saved 30-point
veto array; these tests do not establish final visual quality.

## Insights

The corrected contact proposal removes much of the late backtracking overhead
but reaches essentially the same silhouette fit and residual mask agreement.
Consequently, the remaining repair limit is not explained by that numerical
bottleneck alone. The next control increases training-silhouette priority while
retaining displacement, area and contact safeguards. That control improves the
mean fit and partial ray coverage, but exposes a vertex-only objective's missed
surface-interior violations. The next surface-sampled fit must demonstrate
measured-depth-admitted RGB improvement, not just more semantic-only coverage.

PSNR/SSIM/LPIPS are N/A here: no RGB candidate has been rendered. Fit SDF values
must not be substituted for face-only image metrics or for an actual hole repair.
