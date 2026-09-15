# DEC5: measured-surface conformance of the anatomical head prior

## What was tested

2026-09-15, one-time `001193` control. The preceding
[MHR parametric fit](dec5_mhr_local_head_prior.md) has the required jaw/neck
topology, but its limited shape/pose family stays offset from the observed rim.
This experiment smoothly deforms the prior toward trusted COLMAP surface
observations; it does not replace or modify the original reconstruction.

Three predeclared Laplacian weights, 0.25 / 1 / 4, share the same articulated
prior, 12,400 measured skin anchors, four correspondence/solve rounds and eight
reserved fitting cameras. Train associations require distance <=0.006 and
normal cosine >=0.25. Robust point-distance constraints balance the two inherited
face/neck candidate groups. Uniform displacement-Laplacian and magnitude
regularizers use scales 0.0005 and 0.006 respectively; the latter is a soft prior,
not a hard displacement bound. Only neutral-model vertices above `y=135` cm may
move. Remaining prior vertices are exactly fixed. All 36 coordinate solves
converged with LSMR stop code 2, before the 600-iteration cap.

All distances below are normalized scene coordinates, not meters. Source RGB,
poses, exposure, original mesh and the parallel 6K movie are unchanged. This
stage produces only untextured prior controls, not accepted hole patches.

## Results

| Smoothing | Reserved face point-plane P90 | Reserved neck candidate-group P90 | Rim distance P90 | Rim points within 0.002 | Missing rays passing prior locality |
|---:|---:|---:|---:|---:|---:|
| Parametric head20 + neck6 input | 0.001515 | 0.003967 | 0.004128 | 6/24 | 0/44 |
| 0.25 | 0.000137 | 0.000113 | 0.000814 | 24/24 | 44/44 |
| 1 | 0.000151 | 0.000132 | 0.000900 | 24/24 | 44/44 |
| 4 | 0.000171 | 0.000183 | 0.000826 | 24/24 | 40/44 |

The fixed hole and its rim are post-hoc probes, never fitting constraints.
Locality means distance <=0.002 to the original surface and <=0.003 to its
boundary vertices. Passing these tests is not a measured free-space certificate.
Maximum prior displacements are 0.006566 / 0.006064 / 0.005622 respectively.

The main LLM inspected all six native train RGB/clay comparison panels and the
native requested-hole comparison. The lower-jaw/neck surface now follows the
measured contour much more closely. However, the full priors have artificial
folds around eyes, nose and mouth and coarse stretched neck polygons. Stronger
smoothing reduces these defects but does not make the whole prior acceptable.
**Reject all three as whole-head replacements.**

![Requested-hole prior comparison](/mnt/data/dec5_mhr_measured_conformance/probe_smooth025_smooth100_smooth400/requested_hole_clay_native.png)
![Native train comparison](/mnt/data/dec5_mhr_measured_conformance/review_smooth025_smooth100_smooth400/G004_B005_1210FG.png)

### Independent checks and interpretation limits

An independent agent removed all 1,600 reserved-camera rows before any
correspondence or solve and replayed weight 4: resulting vertices are exactly
identical. This verifies exclusion from this fit, not complete independence
from baseline reconstruction: the fixed original COLMAP mesh used all 62 train
cameras, including the eight reserved here. The three true held-out RGB cameras
remain unused. Closest-surface residuals can also improve through correspondence
sliding, so they cannot replace native views, boundary checks or visibility tests.

There are 32 / 32 / 15 triangles whose normals change by more than 90 degrees
relative to the input prior. That alone is not proof of self-intersection.
Independent nonadjacent triangle-intersection checks find 109 pairs already in
the input prior, then 183 / 163 / 114 in the conformed variants. Strict
noncoplanar edge/interior witnesses confirm actual crossings, not just contacts.
Input crossings are in the mouth and are not excused because they pre-existed.
The conformed crossings are in mouth/nose/eye/ear regions; all implicated vertex
heights are at least 155.077, outside the proposed `135<=y<153` lower-head/neck
band. No normal reversal centroid falls below `y=153`; the closest such surface
at weight 4 is 0.0101 from the requested rim, and the closest crossing surface
is 0.006727 away. These findings support inspecting a restricted local patch,
not declaring the full mesh valid. Pair-ID changes alone do not identify newly
affected anatomical regions.

Evidence: [independent review report](dec5_mhr_conformance_independent_review.md),
[replay result](/mnt/data/dec5_mhr_conformance_independent_review/result.json).
Two focused tests cover pinned-boundary Laplacian behavior, barycentric assembly
and invalid inputs. The retained fit, input and native-review hashes are checked
by `audit_mhr_measured_conformance.py`; no virtual-view PSNR/SSIM/LPIPS is invented.

## Insights

The parametric family was too restrictive for this actor, while a regularized
deformation anchored to real multiview geometry materially improves local
boundary accuracy. This still does not establish safe missing-surface geometry.

The next stage must construct only camera-independent local candidates from
the lower jaw/neck prior, exclude altered eye/nose/mouth regions, verify genuine
original boundary-edge proximity, and retain strict measured-depth/free-space
checks before compositing any mesh. Original measured geometry stays intact.
No parameter was selected solely to maximize coverage in the diagnostic box;
all three fixed strengths and failures remain available. Temporal transfer and
textured validation have not been performed, and the full goal remains open.
