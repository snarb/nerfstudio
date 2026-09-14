# DEC5: foreground skin rejected by background depth witnesses

## What was tested

Following the [production forearm delta study](dec5_forearm_production_delta.md),
two separate hypotheses were tested on real times **001029 / 001033 / 001037**.
Neither changes the published dynamic movie, original meshes, COLMAP depths,
camera calibration, exposure, texture algorithm or model defaults.

1. **Boundary conditions:** keep all referenced appended old-depth ring vertices
   exactly fixed, introduce quadratic depth displacement with a 10-pixel interior
   smoothstep, and use the original strict per-axis triangle extent <0.002.
   The previous quadratic moved this ring and used a different Euclidean gate.
   Original production vertices and triangles remain exact prefixes.
2. **Depth confidence:** test whether a supposedly trustworthy far-depth
   observation is also consistent with the RGB seen by its geometric witnesses.
   A background point visible in several other cameras does not establish that
   the query camera really sees that point through the foreground arm.

The boundary-only root is `/mnt/data/dec5_forearm_production_boundary_curved`.
The color-qualified root is `/mnt/data/dec5_forearm_color_qualified_curved`.
Combined audit and reviews are in `/mnt/data/dec5_forearm_boundary_color_review`.

`diagnose_forearm_free_space.py` reproduces the largest initial pruning causes:
three integer-grid cameras at 001033/001037; 001029 has no initial integer veto,
so its two positive half-grid cameras are examined. Eight native panels show the
query RGB, actual veto rays and the candidate/observed 3D points reprojected into
a separate real train camera. Selection is diagnostic, not a per-frame exception.

`diagnose_forearm_color_witnesses.py` reproduces **exactly the same geometric
witness counts** as the old guard, then compares 5x5 display-RGB patch
chromaticity. RGB uses the existing fixed physical-camera profiles and exposure;
no new color fit is performed. Four mean absolute chromaticity limits
(.04/.06/.08/.10) are explored. Positive controls are separately collected
multiview skin anchors agreeing with the query camera's measured depth—not
points selected from the proposed mesh or held-out images.

`photometric_forearm_depth_guard.py` tests the fixed .04 limit, uniformly on all
three times: carving requires **three other witnesses satisfying both the old
depth/round-trip/parallax rules and RGB compatibility**. Threshold choice began
with 001037 and was checked on the other two times. This guard is intentionally
not equivalent to the previous depth-only veto. It does not edit input depth
maps and does not establish true anatomy wherever the observations are rejected.
Per-pixel witness evidence is cached across pruning passes; it is independent
of candidate geometry. New inference still comes from the bounded quadratic.

## Results

### The rejected depth often belongs to clothing, not the query skin

At 001037, the largest three integer-grid veto cameras are B004_E005_1210VE,
E004_E005_1210WX and F004_E005_1210FP. Their vetoes contain 2871 / 2029 / 1513
pixels. Their median observed-minus-candidate depth is 0.02194 / 0.02407 /
0.02480 **normalized units**, far beyond the 0.003 guard separation.

Native inspection shows many veto pixels lying on real skin in those query
images. The allegedly observed points reproject onto blue clothing in other
train cameras. For B004_E, 2861/2871 observed points fall outside the reference
forearm mask, while all 2871 candidate points lie inside. This is stronger
evidence than simply counting which depth maps agree.

![Skin query with deeper clothing witnesses](/mnt/data/dec5_forearm_production_boundary_curved/free_space_diagnosis/001037/B004_E005_1210VE.png)

The median **minimum** chromaticity error over geometric witnesses at 001037 is
.1298 / .1281 / .1105 for veto points, versus .00207 / .00243 / .00184 for the
supported skin-anchor controls. With the fixed .04 rule:

| Time / query camera | Depth-qualified veto samples | Fewer than 3 color-compatible witnesses | Supported skin controls losing 3-witness qualification |
|---|---:|---:|---:|
| 001029 B/E | 357 | 348 | 4 / 945 |
| 001029 A/D | 26 | 26 | No supported positive controls |
| 001033 B/E | 2930 | 2386 | 3 / 954 |
| 001033 C/E | 769 | 566 | 0 / 979 |
| 001033 D/E | 479 | 360 | 0 / 986 |
| 001037 B/E | 2871 | 2618 | 0 / 874 |
| 001037 E/E | 2029 | 1913 | 0 / 976 |
| 001037 F/E | 1513 | 1237 | 0 / 962 |

Controls lose qualification in **7 / 6676** supported samples (0.105%). These
are overlapping multiview samples, not 6676 independent ground-truth points.
The empty A/D control is disclosed, not treated as a passed test. Specular
objects, hair and different scenes have not been validated by this color gate.

### Boundary correction alone is insufficient; color qualification helps

All values below use the same fixed manual **real-train forearm-skin ROI**,
including holes: display PSNR / SSIM / AlexNet LPIPS. They are not held-out face
metrics. There are no full-frame metrics, loss values or changes to metrics.csv.

| Time | Production baseline | Boundary-only quadratic | Same quadratic + color-qualified guard |
|---|---|---|---|
| 001029 | 21.507 / .8184 / .2722 | 25.317 / .8408 / .2475 | **28.272 / .8514 / .2054** |
| 001033 | 13.622 / .5407 / .5550 | 17.286 / .6683 / .4599 | **20.618 / .7236 / .3646** |
| 001037 | 12.498 / .2324 / .7207 | 15.249 / .4223 / .6703 | **19.053 / .5608 / .5583** |

| Time | RGB holes: baseline / boundary-only / color-qualified | Added triangles: boundary-only / color-qualified | Components: boundary-only / color-qualified |
|---|---|---|---|
| 001029 | 1110 / 258 / 18 | 1983 / 2445 | 114 / 64 |
| 001033 | 7673 / 2874 / 1433 | 11389 / 14340 | 209 / 91 |
| 001037 | 10229 / 4530 / 2203 | 11866 / 16224 | 259 / 121 |

The color-qualified result also improves all three metrics over the earlier
production plane control, not just over the unsuccessful quadratic. The change
to the confidence rule is isolated from the boundary-only result: same proposed
surface, textures, camera poses and RGB. No new averaging is introduced.

![001029 train reference and color-qualified repair](/mnt/data/dec5_forearm_color_qualified_curved/001029/train_reference_native.png)
![001037 remaining defects](/mnt/data/dec5_forearm_color_qualified_curved/001037/train_reference_native.png)

Visual outcome: the selected 001029 forearm hole is almost closed; 001033 and
001037 recover much of the central missing skin. Black side cuts, small horizontal
cracks, skin seams and coarse original wrist/hand geometry remain. All six new
whole-crop verdicts remain **fail for artifact-free production acceptance**, with
the three color-qualified results explicitly recorded as local improvements.
The existing defects are not excused by improved average scores.

### Verification and operational notes

`audit_color_qualified_forearm.py` freshly raycasts all 62 cameras on both pixel
grids for each final color-qualified mesh: all 372 checks pass the **new** rule.
The old depth-only rule would still count 726 / 5172 / 12843 veto pixels, summed
over cameras and grids; these are not unique spatial points. The audit explicitly
records this rather than calling the old guard passed. Saved production vertex
and triangle prefixes, all render receipts, source hashes and baseline replay
equality are checked. No head edits arise from the local forearm experiment.

Nineteen focused tests pass, including fixed-boundary/ray preservation, matching
triangle extent, depth-only witnesses with incompatible color, exclusion of the
query camera, missing-depth rejection and scalar-brightness chromaticity behavior.
Native review covers the new clay/RGB/GT panels and eight correspondence panels;
the combined manifest also retains the prior inspected comparisons.

Geometry jobs ran on separate CPU processes, then disjoint RGB workers shared
the GPU. All workers terminated normally; no new PatchMatch jobs were launched.
An initial 001029 diagnostic stopped because it assumed integer-grid vetoes
existed; the half-grid fallback fixes that diagnostic, not the reconstruction.
An early scoring invocation correctly refused a missing render completion receipt;
it was rerun after both workers terminated. Failure logs and code snapshots remain.
No source, reference or failed workspace was deleted.

## Insights

1. **Geometric corroboration of a background point is not sufficient evidence
   of free space along a foreground query ray.** In these canaries, matching
   colors exposes that missing distinction. Motion blur and low skin texture are
   plausible contributors to the original stereo mistake, not experimentally
   isolated causes. A global exposure explanation is not needed for these panels.
2. Fixed boundary anchors remove a real implementation confound but cannot
   repair incorrectly trusted observations. Combining the checks makes progress
   without changing old geometry or weakening all depth consistency thresholds.
3. More work is required before video promotion: compare a matched color-qualified
   plane against the quadratic to separate remaining shape errors; investigate
   the residual side cuts with the same witness audit. Filtering unreliable
   observations before TSDF fusion is another testable hypothesis, not an already
   demonstrated fix. Do not infer missing anatomy merely from rejected evidence.

Replay the new arm in a fresh output root:

```bash
python scripts/study_forearm_production_delta.py prepare --frame 001037 \
  --root NEW_ROOT --curved-anchor-root /mnt/data/dec5_forearm_multiview_anchors \
  --boundary-conditioned --photometric-free-space
python scripts/study_forearm_production_delta.py render --frame 001037 --root NEW_ROOT
python scripts/audit_color_qualified_forearm.py --frame 001037 --root NEW_ROOT
```

Repeat uniformly on 001029/001033 before scoring or promotion. The published
150-time dynamic video remains unchanged and is **not artifact-free**.
