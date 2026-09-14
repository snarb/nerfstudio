# DEC5 secondary-reference forearm completion

## What was tested

Can missing skin outside the old G/A reference be recovered by sampling the
same inferred surface from H/A and H/C? `forearm_quadric_rays.py` converts the
existing inverse-depth polynomial into one world-space implicit quadric. It
intersects camera rays near the unchanged reference plane; there are no three
independent shape refits, new RGB gains, learned priors or held-out inputs.

`probe_forearm_reference_coverage.py` tests all three train references at
001029/001033/001037 against the best local shape-first control. The strict
policy retains two positive skin annotations, no available disagreement, no
trusted free-space veto, <=100 px from trusted samples and <=0.01 displacement
in the original reference depth. Positive-only counts are a diagnostic upper
bound, **not an accepted mask policy**. These are inferred candidate points,
not new measured observations or recovered RGB pixels.

`study_forearm_secondary_reference.py` assembles H/A then H/C patches on the
001037 canary, querying only still-missing rays at each stage. The old mesh is
preserved exactly. All final vertices obey the existing semantic/extent gates;
the same contrastive color-qualified 62-view depth guard is applied at integer
and half-pixel rays until no new-face contradiction remains. No default changes.

## Results

Probe: `/mnt/data/dec5_forearm_reference_coverage_probe`.
Canary: `/mnt/data/dec5_forearm_secondary_reference`.
Review: `/mnt/data/dec5_forearm_secondary_reference_review`.

| Time | G/A strict / positive-only | H/A strict / positive-only | H/C strict / positive-only |
|---|---:|---:|---:|
| 001029 | 14 / 14 | 19 / 20 | 46 / 46 |
| 001033 | 474 / 475 | 948 / 1053 | 878 / 1062 |
| 001037 | 1040 / 1145 | 547 / 796 | 418 / 851 |

All nine native GT/overlay probe crops were inspected. Most proposals lie on
the side/cuff margins or isolated interior specks; they are not a complete
wrist/hand repair. The existing annotation polygons have deliberately limited
domains (including flat upper cuts), so an annotation disagreement must not
automatically be interpreted as an anatomical silhouette.

001037 H/A generated 1,029 triangles. After that patch, only 101 H/C rays remained
eligible, generating 203 triangles; the independent probe's 418 H/C points are
not additive. The final guard retained **588 of 1,232** new triangles after
four pruning rounds and a clean fifth pass. Fresh independent replay passes
124 ray checks, exact old-mesh prefix and final semantic/extent tests.

Same fixed **train forearm ROI**, same hard-source texture renderer:

| 001037 variant | PSNR | SSIM | LPIPS | Black skin pixels | Missing-depth pixels |
|---|---:|---:|---:|---:|---:|
| Previous local quadric | 20.16230 | 0.692228 | 0.426160 | 1775 | 1737 |
| Secondary references | 20.57280 | 0.702146 | 0.389540 | 1642 | 1599 |

These are **not held-out face scores**, not full-frame metrics, and do not enter
the campaign face CSV. Two fresh native RGB renders were inspected: moving
camera and fixed H/A versus GT. Side coverage improves slightly, but the broad
wrist/forearm voids, broken hand surface and planar color seams remain.

![Moving camera, previous / secondary](/mnt/data/dec5_forearm_secondary_reference_review/001037/moving_comparison.png)
![Train GT, previous / secondary](/mnt/data/dec5_forearm_secondary_reference_review/001037/H004_A005_1210M6_comparison.png)

Index-connected mesh components increase from 101 to 294; components with fewer
than 100 triangles increase from 97 to 289. Nonmanifold edges remain zero.
This is combinatorial connectivity, **not proof of 193 physically detached
floating pieces**: grid rings use duplicated sampled vertices rather than welded
old-mesh indices. Nonetheless this construction does not deliver a coherent
completed surface, and the visible major holes persist. Canary verdict:
**partial local gain, not production accepted**. No meshes/renders on 001029 or
001033 were assembled after this visual gate; their results are probes only.

## Insights

Reference-camera coverage is one real limitation, but merely adding another
screen-grid tessellation does not solve the defect. Iterative depth carving and
patch connectivity still limit the retained surface. The next shape test should
address a coherent bounded surface with explicitly measured boundaries, rather
than pile additional grids onto the same partial mesh or silently discard
contradicting evidence. The quadric also extrapolates a fit collected near G/A;
sampling it from H/C is not independent evidence for anatomy.

The user's shot-level workaround remains separate: the existing
[phase-shifted smooth path](dec5_temporal_camera_phase.md) avoids the most
conspicuous late cheek cavity without shrinking camera travel or freezing the
actor. Its published 150-frame movie is unchanged; it still has known forearm,
hair and lipstick defects. Neither that workaround nor this canary constitutes
an artifact-free reconstruction.
