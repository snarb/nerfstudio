# DEC5 skin seams: geometry and fly-through repair

## What was tested

The requested outcome is skin without conspicuous seams or patches during free
camera movement, including large temporal changes. The completed 50-frame campaign
is a reference, not proof that a repair works. Geometry uses fixed train calibration;
held-out RGB is reserved for review and face metrics. No semantic mask or RGB
averaging is introduced.

Study root: `/mnt/data/lookcloser_dec5_5a3_surface_repair`.
Remote scratch: `/fsx/oregon/lookcloser_dec5_5a3_surface_repair` on `dev3`.
Temporal confirmation uses every 40th available dataset directory (there are 180,
spaced by two source IDs): `000899, 000979, 001059, 001139, 001219`, plus the final
`001257`. Frame `000973` is the initial failed diagnostic case. These confirmation
frames must be evaluated with one common recipe and several camera positions.

First isolated canary: PatchMatch NCC window radius 5 -> 3 in both photometric and
geometric passes. Other dense-stereo/TSDF parameters retain the pinned recipe.
The opt-in runner argument leaves its existing default command unchanged.

## Results

Work is in progress. No repaired recipe has passed the temporal or fly-through gate.
The original failure attribution is being rechecked against raw train depth, mesh
depth and visibility. The previous min-consistency=1 experiment partially restored
the hand but did not isolate why the second correspondence was absent.

### Isolated controls on 000973 (2026-09-05)

| Control | Measured/visual result | Decision |
|---|---|---|
| PatchMatch window radius 3, other geometry unchanged | 62 geometric maps; chin/hand patch remains, extra holes under jaw | Reject |
| Fine TSDF .00025 voxel / .0015 truncation / weight 2 | 310,948 vertices, 603,027 triangles, 2 components; no hand repair | Reject |
| DA3 Large 1.1, 16 calibrated train cameras, resolution 1008, standalone TSDF | Large face/hand holes and displaced tube; not a usable geometry replacement | Reject standalone; hybrid remains untested |
| Mesh visibility log tolerance .001 | More cracks/seams, no geometric recovery | Reject |
| 62 texture-camera pool, 8 nearest, hard seam cut | Most large neck patches reduced; brightness shift, residual boundary defects | Diagnostic only |
| Same pool, angular-16 primary retained, hard seam cut, rank penalty .001 | Face detail retained, some smaller seams removed; conspicuous neck/hand seam remains | Not passing |
| Hard seam cut with zero rank penalty | Neck patch smaller, but perceptual face detail worsens | Reject as a final recipe |

Artifacts: [study diagnostics](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973),
[angular-primary comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_cut8_angular_primary/review/hand.png),
[three held-out camera positions](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_cut_three_views/review/overview.png).
Three still views are not a completed fly-through gate; interpolated cameras are
being rendered separately. Source RGB is never averaged, and held-out RGB is not
read by the camera-path renderer.

### Face ROI audit: new study protocol, original campaign untouched

The inherited 000973 polygon was visibly offset: it excluded an eye/cheek region
and included hair/background. Its declared anatomy did not match its actual raster.
A new GT-only outline was drawn and visually checked at native resolution. It
includes both eyes, forehead, cheeks, nose, mouth and chin; it has no internal
candidate-defined exclusions. Ear/hair/hand remain separate visual-review regions.

[GT outline](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/gt_roi_review/baseline_metrics/face_mask_overlay.png)
and [polygon](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/gt_roi_review/face_polygon.json).
Every row below uses exactly this same outline and the existing display-domain
masked scorer. These numbers are **not comparable to the old campaign ROI numbers**.

| Same-frame candidate | Face PSNR | Face SSIM | Face LPIPS |
|---|---:|---:|---:|
| Unchanged published baseline | 31.642197 | 0.890242 | 0.052993 |
| Seam cut, angular primary, rank .001 | 31.292252 | 0.890760 | 0.047924 |
| Seam cut, nearest 8, rank .0001 | 24.455322 | 0.903054 | 0.052974 |
| Seam cut, nearest 8, rank 0 | 29.010620 | 0.911719 | 0.107517 |

The unchanged baseline was 0.107042 LPIPS under the inherited outline. This is an
evaluation-region error, not a reconstruction improvement. Skin seams on the neck
remain real and cannot be dismissed by excluding them from a face-only score.

### Large-motion baseline 001059 and gate status

The frozen baseline completed on dev3: 62/62 geometric maps, all 1080x1920;
mean/min depth coverage 0.397052 / 0.261861; 68,264 vertices, 131,740 triangles,
4 retained components. Photometric/geometric passes took 462.2 / 844.2 seconds.
Mesh, metadata, request, manifest and eval PNG/EXR were returned with matching
remote/local SHA-256. No GPU worker remained after the recorded completion check.

GT-only face ROI scoring: **26.455898 PSNR / 0.887777 SSIM / 0.060391 LPIPS**.
[GT/prediction overview](/mnt/data/lookcloser_dec5_5a3_surface_repair/frames/001059/review/overview.png).
The visible face and neck show no conspicuous skin patch in this single eval
view; the hand/tube is now out of view. This does not certify repaired geometry,
the full temporal subset, or other camera positions.

The 000973 zero-rank source-cut canary was also rendered at nine calibration-only
camera-path positions. Native GT comparisons at the three real held-out anchors
failed: under-jaw holes, softer detail, and a horizontal neck seam in the third
anchor. [Verdict](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_cut_zero_path9/visual_review.json)
and [coarse nine-sample diagnostic video](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_cut_zero_path9/coarse_path_diagnostic.mp4).
This coarse diagnostic is not a passed smooth fly-through test.

Read-only train RGB audit sampled 50,000 mesh vertices and 295 neighboring-camera
pairs, accepting only mutually visible non-clipped observations. E004_B005_1210I7
versus E004_C005_1210YM differed by median log RGB -0.0766/-0.0823/-0.0744
(about 7-8%); versus G004_B005_1210FG, about 17%. This supports a radiometric
contribution to hard-source seams. Pairwise global offsets fit consistently, but
within-pair residuals remain: global correction is an untested candidate, not a
proven fix. [Full audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/texture_consistency.json).
No color correction has been applied. Permission was requested before departing
from the earlier explicit prohibition on additional color correction. No further
temporal reconstruction was launched after the failed multi-view gate.

Current tests: 29 passed for the single-frame runner, renderer, graph-cut helper,
and camera normalization/path interpolation. The implementation remains an opt-in
diagnostic, not a completed repair. Source EXRs and the published 50-frame campaign
remain unchanged; failed scratch is retained for diagnosis.

### Local DA3 hybrid versus a plane control (2026-09-05)

An additional allowed geometry-only canary tested the proposed local hybrid before
any radiometric change. Existing DA3 depths had camera-dependent normalized depth
bias: medians +0.014169 in D004_A005_12103C and +0.005888 in E004_C005_1210YM
(28.3 and 11.8 voxels at .0005). The hybrid therefore did not directly mix raw
DA3 depth with PatchMatch. It fitted a local plane to PatchMatch-minus-DA3 on a
six-pixel hole boundary and added the corrected DA3 shape only inside enclosed
holes of at most 1,000 pixels. Boundary normalized RMSE had to be <= .001, and
the result could not create a new depth layer beyond the observed boundary range.
Every originally valid PatchMatch depth was preserved exactly. RGB was not read.
Derived values are not represented as independent stereo observations.

The plane control used identical holes, gates and 16 cameras, but no DA3 shape.
Both outputs retained all 62 original train cameras for frozen-parameter TSDF
fusion. They filled 78,248 / 90,912 depth pixels respectively; both meshes remained
one connected component. Rendering used the same nearest-fill16 options in all
three arms, including a fresh unchanged-mesh control. The published baseline's
extra source-continuity/primary-continuation options were deliberately not mixed
into this causal comparison.

| Matched renderer, same GT-only face ROI | Vertices | Triangles | Face PSNR | Face SSIM | Face LPIPS |
|---|---:|---:|---:|---:|---:|
| Unchanged mesh | 80,068 | 154,804 | 31.052523 | 0.889798 | 0.053505 |
| Local DA3 hybrid | 80,085 | 154,949 | 31.057217 | 0.889733 | 0.053560 |
| Local plane control | 80,067 | 154,928 | 31.057024 | 0.889719 | 0.053560 |

Native GT comparisons showed the same conspicuous chin and neck/hand source
patches in all arms: **both completion methods fail the requested repair gate**.
[Matched hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_completion_review_matched/hand.png),
[face comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_completion_review_matched/face.png),
[verdicts, metrics and hashes](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_completion_review_matched/result.json).
No new temporal runs were launched for these failed candidates.

A deterministic pseudo-holdout test also removed 32 small 12x12 patches of known
PatchMatch depth in each of four train cameras. This is a depth-completion
consistency test, not independent geometric ground truth. Median normalized
absolute error for DA3 / plane was 0.0000894 / 0.0000359 (D004_A005_12103C),
0.0000840 / 0.0000286 (E004_C005_1210YM), 0.0001000 / 0.0000372
(H004_C005_1210SZ), and 0.0000899 / 0.0000450 (L004_C005_1210QQ). DA3 did
not justify its added complexity for these small holes. Five new tests verify
exact preservation of measured depths, removal of local prior bias, preservation
of prior curvature, rejection of discontinuities/open background/oversized holes,
and rejection of nonfinite input. The focused suite now has **34 passing tests**.

The geometry-only alternatives tested so far do not resolve the visible RGB
seams. A train-only radiometric consistency canary is still awaiting permission
because the original instructions explicitly prohibited new color correction.
The goal remains incomplete, and no repair is being claimed as accepted.

### Authorized train-only color calibration (2026-09-05)

The user subsequently authorized additional color correction, resolving the
previous permission blocker. Geometry, held-out separation, source EXRs and the
published campaign remain unchanged. An opt-in renderer correction reverses the
known sRGB/Reinhard mapping, multiplies exposed-linear RGB, and reapplies the same
display mapping. It never averages source images or blurs their detail.

Calibration fits a connected graph of mutually visible train-camera observations
on the fixed mesh. Twenty percent of hashed .008-normalized-unit spatial blocks
are excluded from fitting. Neither eval RGB nor semantic masks are read. Global
brightness is fixed by geometric-mean train gain one, not by matching held-out GT.

| Held-out train-pair display L1 median (not face quality) | 000973 | 001059 |
|---|---:|---:|
| Uncorrected | .03574248 | .03504415 |
| Cancel per-image ingest exposure only | .02129486 | .02165866 |
| Fit one exposure per camera | .01889360 | .01913620 |
| Fit diagonal RGB per camera | .01877861 | .01897592 |
| Exposure plus smooth 8x5 achromatic field | .01792601 | .01810730 |

Much of the inconsistency is introduced by per-image JPEG ingest exposure. For
000973, E004_C005_1210YM and G004_B005_1210FG have ingest gains 6.4352 and
4.9749 (29% difference), while inferred pre-ingest relative responses are close.
RGB correction adds less than 1% residual reduction beyond scalar exposure.
Between the two frames, median absolute log variation is .00361 for chromatic
factors and .01910 for inferred pre-ingest RGB response. These observations cannot
identify hardware sensitivity separately from camera exposure, processing,
white balance, specularity and view-dependent illumination.

| 000973, common GT-only face polygon | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Matched uncorrected nearest-fill16 | 31.052523 | .889798 | .053505 |
| Exposure nearest-fill16 | 28.248308 | .888588 | .051774 |
| RGB nearest-fill16 | 28.248753 | .888523 | .051887 |
| Exposure angular-primary hard graph-cut8 | 28.275328 | .889492 | .048025 |

The train-only brightness gauge changes overall appearance relative to GT:
PSNR drops about 3 dB, despite reduced inter-camera disagreement. This is not
hidden as an improvement in all image metrics. Native visual review finds a
substantial reduction of the chin patch, but a remaining conspicuous neck/hand
seam. Smooth spatial exposure also fails the three-anchor novel-view gate:
under-jaw geometry holes and a horizontal neck band remain. The 001059 face is
coherent but brighter; neither this single view nor lower calibration residuals
establishes a successful repair.

[000973 fits](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_calibration/calibration_spatial.json),
[001059 fits](/mnt/data/lookcloser_dec5_5a3_surface_repair/frames/001059/color_calibration/calibration_spatial.json),
[scalar/RGB hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_calibration/review/hand.png),
[spatial three-anchor renders](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_calibration/spatial_cut_three_views),
[001059 comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/frames/001059/color_calibration/review/face.png).

Global correction alone is not accepted. No new temporal PatchMatch jobs are
being launched while this multi-view gate fails.

### Fixed mesh-space texturing control and finer color field

The official [mvs-texturing source](https://github.com/nmoehrle/mvs-texturing)
was built at `f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1`. A separate opt-in
exporter supplies exactly 62 fixed train cameras and the unchanged TSDF mesh.
Each triangle receives one source image. Poisson/local seam blending and texture
hole filling are explicitly disabled. Global seam leveling adds smoothly varying
per-vertex RGB offsets to each selected patch; no source detail is averaged.

An identical-label control isolates leveling: large chin/neck patches clearly
decrease with leveling enabled. However, sharpness-based mesh source selection
transfers inconsistent highlights and contours to the hand/tube and brightens
the face. Train exposure normalization plus photometric outlier clamping helps,
but does not make this a usable replacement. Native comparisons at all three
held-out camera anchors fail the gate. The input/output oriented triangle
inventories are identical (154,804); OBJ coordinate rounding is <= 5.002e-7
normalized units. This is a texturing failure, not a changed-mesh comparison.

| 000973, same GT-only face polygon | PSNR | SSIM | LPIPS | Decision |
|---|---:|---:|---:|---|
| Fixed mesh atlas, global leveling | 16.664492 | .766281 | .193125 | Reject |
| Fixed atlas, exposure + outlier clamp + global leveling | 23.028111 | .814143 | .122213 | Reject |
| Existing hard-source renderer, 16x9 exposure field | 28.967068 | .889862 | .047471 | Partial improvement; fails multi-view gate |

The 16x9 field uses smoothness 1, zero-field weight .5 and maximum multiplier 2,
selected as a train-only calibration canary. Its excluded-block pair L1 median
is .01693079 (52.63% below uncorrected), p90 .05669708. It does not blur source
RGB. The main face and chin improve, but a hand-adjacent neck boundary remains;
the other anchors show a neck band and soft/saturated tube detail. Thus better
face LPIPS does not override the visual failure.

The same 16x9 calibration on 001059 gives pair L1 .01744385, versus .03504415
uncorrected (50.22% reduction). This independently timed frame corroborates the
radiometric inconsistency, not a completed temporal reconstruction gate.

An additional train-only projected-overlap field (16x9) retained the same geometry
and source pool. On 000973 it reduced excluded-block pair L1 only from .01732351
to .01694092; face PSNR/SSIM/LPIPS are 29.002506/.889776/.047714.
Native face/hand review still shows the neck/hand seam; it is not
accepted. This correction is explicitly view-dependent, so it cannot be described
as a validated static appearance model for fly-through. Both source-camera and
projected fields leave high-frequency RGB in the selected source unchanged apart
from pointwise exposure mapping.

For causal interpretation, the closer uncorrected **angular-primary graph-cut8**
control scores 31.292252/.890760/.047924, versus 28.967068/.889862/.047471 with
the 16x9 camera field. Color calibration alone therefore gives only a small
face-LPIPS change and reduces PSNR under the train-only brightness gauge. The
larger change from nearest-fill16 also includes the change of hard source labels.

[Identical-label leveling comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mvs_texture_global/review_matched/hand.png),
[rejected exposure/clamp atlas](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mvs_texture_exposure_clamp_v2/review/hand.png),
[16x9 correction comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_calibration/review_spatial16/hand.png),
[third-anchor neck band](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_calibration/spatial16_cut_three_views/review_anchor2/hand.png).

The focused suite has 49 passing tests with the optional graph-cut dependency,
including exact ingest inversion, gain-only application, connected-graph gauge,
spatial fitting's exclusion of held-out samples, camera coordinate conversion,
atlas sampling convention and geometry-inventory preservation. No corrected
recipe has yet passed the full temporal/fly-through acceptance gate.

The staged release was also tested in a separate checkout-index snapshot, without
unrelated dirty-worktree edits: 44 tests passed (the additional five local DA3
tests are not part of this color release). `audit_patchmatch_color_canaries.py`
re-hashed six train-only fits, 12 three-anchor renders, four failed-candidate
verdicts, fixed-atlas outputs and face-only metric inputs. The artifact audit
passes; the repair gate explicitly does not. [Audit receipt](/mnt/data/lookcloser_dec5_5a3_surface_repair/color_canary_audit.json).

## Insights

The published render correction can use the primary train camera despite its failed
mesh visibility test on small components. This can improve one still image, but is
not sufficient evidence for correct novel-view rendering. New controls must separate
true geometric recovery from an occluder painted onto another surface.

The previous broad explanation, “the whole hand lacks two consistent PatchMatch
views,” is not supported by the new raw-depth audit. Most hand depth is present.
For a concrete chin point at eval pixel (730,570), the nearer E004_B005_1210I7
camera has projected depth 6.61095, raycast depth 6.61078 and raw PatchMatch depth
6.61117. The original angular primary E004_C005_1210YM sees projected depth
7.72321 but has foreground hand depth 7.47462: rejecting that source is correct.
The visible patch there comes from switching RGB sources across an occlusion;
it does not demonstrate absent chin geometry. The original primary is only the
fifth-nearest source in the full 62-camera pool. Boundary holes are a distinct
geometry issue and remain unresolved.
