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
**Localization correction:** the conspicuous lower bright strip at the left side
of the hand is not a source-camera seam. Both sides use the primary camera, while
the mesh switches between hand and neck depth. The earlier source-switch evidence
is valid for neighboring chin/upper-neck pixels, not for this entire defect.
See the localized free-space controls below; none is an accepted repair yet.
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

### Trace of the remaining lipstick-adjacent seam

The source-warp audit reproduces the downloaded spatial16 image byte-for-byte:
SHA-256 `3dafc697f09b0e3639230a5d5ef6dc7163c6e6a5f98c4b9d10e5e6b4199de69d`.
The jagged patch boundaries coincide with hard source labels: E004_C005_1210YM
is the main source, E004_B005_1210I7 supplies the left neck disocclusion strip,
and G004_B005_1210FG supplies the right strip. Native source patches show neck
skin in the sampled strip, not synthesized room/background RGB. Individual
unmasked source-warp previews deliberately include invalid projections; their
duplicate hands/tubes must not be mistaken for the final hard-selected output.

At output pixel (700,550), the primary source projects depth .782475, but its
mesh first hit is .754037 (foreground hand), a log mismatch .03702. Its rejection
is correct. The selected alternative E004_B005_1210I7 projects .670608 versus
raycast .674093, log mismatch .005183; it passes the current .01 tolerance.
Thus remaining boundary appearance involves source switching and geometric
uncertainty as well as radiometry. Sharpness/blur differences versus geometric
misregistration still require separation; the audit does not prove that sensor
defocus alone causes the perceived soft patch. Raw-stereo bilinear samples in
`trace.json` can mix invalid zero or discontinuous depths and are not independent
visibility evidence at a contour.

[Prediction/source labels](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_source_audit/review_labels/hand.png),
[three train-source warps](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_source_audit/review_sources/hand.png),
[native train patches at the neck point](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_source_audit/trace/point_02_train_patches.png),
[pixel trace](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_source_audit/trace/trace.json).

The earlier `per-image` exposure setting was already part of JPEG ingest, not
new manual camera color profiles: one scalar RGB multiplier is derived from each
image's 70th-percentile luminance before the shared Reinhard/sRGB curve. Different
framing and moving contents can therefore change gain with a physically fixed
camera. The newly authorized train-overlap calibration is a separate correction.

### Causal controls for the lipstick-adjacent skin patch

The downloaded best-current image is a hard-source mosaic, **not RGB averaging**.
All 18,061 pixels labelled E004_B005_1210I7 equal that single camera's corrected,
reprojected PNG exactly; the 25,945 G004_B005_1210FG pixels do too. Only 11 primary
pixels differ by one 8-bit level from their saved source warp, from rounding.
The native source-label comparison visibly aligns the jagged skin boundary with
the change of camera. In the hand/neck diagnostic rectangle, 95.95% of 889
primary/first-alternative seam adjacencies have absolute log mesh-depth jump below
.001. This describes the mesh's continuity, **not independent proof of true depth**.

Two actual visibility issues were isolated without changing the mesh or cameras:

- Open3D pinhole raycasts use half-pixel centres, whereas legacy depth unprojection
  and source-array lookup use integer centres. At target (700,550), the old point
  is about 4.70e-5 normalized units off the mesh; matching half-pixel centres reduces
  this to 1.25e-8. A slanted-plane unit test independently reproduces the bug.
- Bilinear depth-map samples near a silhouette can interpolate separate depth
  layers. The opt-in direct first-hit visibility control tests each source camera
  against the same mesh, with 2.5e-5 normalized tolerance, avoiding that comparison.
  Allowed plane-filled target holes may have no own triangle; a no-hit ray is only
  treated as unoccluded, never as independent depth evidence.

Neither control removes the patch. With matching half-pixel centres, direct rays
at (700,550) confirm that the primary E004_C005_1210YM is occluded by foreground
hand, while E004_B005_1210I7 and G004_B005_1210FG hit the target neck surface.
Native source patches show skin there, not synthesized or sampled room background.
The artificial boundary is a **source-camera disocclusion boundary on one continuous
skin surface**, not a semantic boundary in the target view. A person/skin mask alone
cannot provide the RGB hidden from the primary camera; none was introduced.

Train-only, fully overlapping 48x48 patch comparisons find residual small relative
shifts: E004_B005_1210I7 has median 1.41 pixels (37 hand/neck patches, NCC .9312 before
translation, .9646 after); G004_B005_1210FG has median 1 pixel (39 patches, .9226 to
.9476). This does not measure the primary-occluded pixels themselves, and does not
separate residual mesh/calibration error from capture/sharpness or view-dependent
appearance. Sensor defocus alone is **not established** as the cause of softness.

Two further train-only controls retain hard source RGB and the same mesh. Fitting
local RGB gains on mutually visible bands around primary disocclusions barely
changes the patch. Weakening the camera-rank preference only near large same-depth
visibility holes moves/softens part of the seam, but the remaining hand-adjacent strip
is still conspicuous. Neither is accepted; no new default or temporal recipe is set.

| Same 000973 study GT-only face polygon | PSNR | SSIM | LPIPS | Native neck/hand verdict |
|---|---:|---:|---:|---|
| Spatial16 downloaded control | 28.967068 | .889862 | .047471 | Fail |
| Matching half-pixel centres | 28.973961 | .888892 | .047530 | Fail |
| Plus direct mesh visibility | 29.059521 | .889487 | .047385 | Fail |
| Plus local disocclusion RGB gain matching | 29.034229 | .889548 | .047410 | Fail |
| Direct visibility + depth-aware rank prior, radius 64 | 29.063141 | .889683 | .047981 | Fail |

These face metrics do not replace inspection of the neck/hand and must not be
mixed with the old campaign's different face polygon. The four new canaries have
native GT comparison verdicts and retained hashes; they fail on the first anchor
and are not promoted to the every-40th-frame or fly-through confirmation stage.

[Exact per-source identity and legacy depth statistics](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_source_audit/causal_trace_v2/trace.json),
[direct visibility trace](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/exact_visibility/causal_trace_v2/trace.json),
[visible portions of three train warps](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/exact_visibility/causal_trace_v2/visible_train_sources.png),
[GT / exact visibility / relaxed source preference](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/visibility_rank64/review/hand.png),
[registration audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/source_registration_audit/audit.json),
[findings and failed-control receipts](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/seam_cause_findings.json).
Black regions in the individual visible-source previews mean that that camera
cannot see the surface; they are not holes in the final mesh or final prediction.

The focused causal-control suite passes 44 tests, including weighted graph-cut
energy against exhaustive binary labels, preservation of default labels, depth-layer
separation of the spatial prior, single-source identity, half-pixel geometry and
first-hit visibility. These checks validate diagnostic code, not visual acceptance.
The expanded release suite, including path/atlas regressions, passes 51 tests both
in the working tree and in an isolated checkout-index snapshot excluding unrelated
dirty changes. The new trace input hashes, four prediction/GT/ROI hash sets and
four review/audit hash sets were independently rechecked successfully.

### Further hard-source controls: no accepted repair

All controls below use the same 000973 mesh, fixed cameras, matching half-pixel
centres and direct mesh visibility. Correction/selection uses train RGB only.
Output RGB remains one reprojected source per pixel; consensus RGB is a selection
cost, not an averaged output. Native hand/neck, face and ear comparisons reject
every candidate despite small face-metric improvements. No temporal recipe or
existing renderer default has changed.

| Control, same study GT-only face polygon | PSNR | SSIM | LPIPS | Verdict |
|---|---:|---:|---:|---|
| Smooth depth-separated screen-space log-gain field | 29.045528 | .889479 | .047877 | Fail |
| Prefer secondary sources supported by raw stereo depth | 29.028446 | .889260 | .048181 | Fail |
| Matched global camera RGB control | 28.345644 | .889197 | .047981 | Fail |
| Mesh-attached angular RGB correction | 27.512985 | .888258 | .047963 | Fail |
| Hard-source boundary gain leveling | 29.071888 | .889788 | .047046 | Fail |
| Train RGB consensus selection, 8 sources | 29.036377 | .889352 | .048101 | Fail |
| Consensus selection, 16 sources | 29.052736 | .889415 | .047976 | Fail |
| Consensus selection, 32 sources | 29.088245 | .889435 | .047708 | Fail |
| Consensus selection, 62 sources | 29.073856 | .889404 | .047782 | Fail |

The mesh-attached angular model estimates per-vertex first-order view-direction
log-RGB gains using 62 train cameras, mesh-Laplacian regularization and spatially
held-out vertices. Held-out train pair L1 improves only .018839 to .018259 (3.08%);
the neck patch remains. One-sided boundary gain leveling lowers the median RGB
jump over 967 same-depth primary/secondary seam adjacencies from .014379 to
.010458, but the visible patch and some large local discontinuities persist.
These outcomes reject a simple exposure-only explanation or camera-pool shortage;
they do not establish whether local mesh error, residual calibration, capture
differences or view-dependent appearance dominates.

A nearest-neighbor raw-depth check at target (700,550) finds a useful distinction:
E004_B005_1210I7 has projected normalized depth .670613 but nearest raw depth
1.30216; only 1 of 25 neighboring pixels is within 1% of the mesh. G004_B005_1210FG
has a zero nearest sample, but 23 of 25 neighbors are valid and consistent near
.6966 versus projected .696943. Interpolated depth alone is therefore misleading
near occlusions. Preferring raw-stereo-supported fallback still fails visually.

The first smooth-field attempt failed its numerical convergence gate and emitted
no accepted prediction. Increasing only the solver iteration budget achieved
convergence, but the resulting image above still fails the visual gate.

[16/32/62-camera native hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/consensus_all62/review/hand.png),
[boundary-leveling comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/hard_source_seam_leveling/review/hand.png),
[angular-correction comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/angular_surface_render20/review/hand.png).
Each rendered candidate has a failed visual-review JSON with prediction/GT/ROI
hashes and its own face-only metric receipt.

### Fixed-camera photometric mesh refinement control

An opt-in OpenMVS 2.4.0 control now tests deformation of the existing mesh vertices,
not new SfM, camera optimization, segmentation or RGB synthesis. It receives only
the 62 training JPEGs and a private copy of the normalized TSDF mesh. Decimation,
subdivision, hole closing and planar vertex removal are disabled; topology and
all calibrated cameras are checked before accepting the output. The artifact is
a **TSDF-initialized photometrically refined mesh**, not a raw TSDF volume or an
unchanged TSDF extraction.

The official Ubuntu release archive SHA-256 is
`7104ae1ddd6ca38fbca9e0e4a70b20af59e21e0b497eb7181c864fbf38ca8d00`;
the wrapper pins both executable hashes. Source tag v2.4.0 resolves to
`58117204c86bbb11a0b25b26a8987676cf11274d`.
The first attempt stopped before refinement: OpenMVS text export rounds camera
parameters to six significant digits, producing up to .049995-pixel apparent
change. Binary export preserves the cameras to 1.92e-15. The audit handles the
release's `--no-points` binary image-count-header omission explicitly, with
truncation/inventory checks; the fixed-camera tolerance was not relaxed.

The 66-test expanded suite passes in both the working tree and an isolated
checkout-index snapshot excluding unrelated edits, including default renderer regressions,
single-source behavior, graph-cut energy, gain-field/depth-layer tests, binary
camera precision, nonfinite rejection and truncated-export rejection.
Native Scene archives must also be converted to interchange before InterfaceCOLMAP
can read them. The wrapper now pins `TransformScene`, supplies an explicit identity
transform, and audits the resulting cameras. The complete clean 45/22-iteration
wrapper run succeeds with all 62 cameras unchanged to 1.82e-12, one component,
80,068 vertices and exactly the original 154,804 oriented triangles. Median vertex
movement is .00014588 normalized units (p99 .00031265, maximum .00061942).

| 000973 matched geometry control | Face PSNR | Face SSIM | Face LPIPS | Verdict |
|---|---:|---:|---:|---|
| Original mesh + exact visibility | 29.059521 | .889487 | .047385 | Fail |
| Photometrically refined mesh, same colors/selection | 29.111637 | .890642 | .047190 | Fail, partial improvement |
| Refined mesh + one-sided gain leveling | 29.096230 | .890749 | .046996 | Fail |

The F-camera neck patch visibly shrinks when **only mesh geometry changes**.
A fresh original-mesh control reproduces the previous F PNG byte-for-byte; all
three camera anchors are rerendered with identical settings for comparison.
The narrow hand-adjacent bright/ragged edge remains, and the J-camera lipstick
and hand are softer than GT. Color leveling on the refined mesh does not remove
the remaining edge. This is evidence for a geometry contribution, not proof that
all residual softness or all seams are geometric. No candidate is promoted yet.

[Matched F native hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_three_views/review_F004/hand.png),
[matched J hand/neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_three_views/matched_J004/hand_neck.png),
[matched L neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_three_views/matched_L004/neck.png),
[complete clean wrapper receipt](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_fixed_verified/refinement_result.json).
The reviewed mesh came from the preceding identical computation, whose native
archive audit was recovered with identity conversion. Its producer request is
retained unchanged and the recovery is explicit in its receipt. A clean repeat
has only floating-point-scale vertex differences and its own separate hashes;
render receipts always reference the actual mesh used. The measured maximum
vertex-position difference between these repeats is 8.44e-7 normalized units.

The refined-mesh train overlap audit raises E004_B005_1210I7's median zero-shift
hand/neck NCC from .9312 to .9565, although reliable best translations still have
median length 1.41 pixels. Patch counts differ slightly with visibility (37 versus
36), so this is descriptive, not an identical-sample paired estimate.
The opt-in registered relative-bandwidth diagnostic uses no held-out RGB and
applies no filter to prediction. Among 42 reliable G004_B005_1210FG hand/neck
patches, 32 gain more than .005 NCC when the primary is low-pass filtered (median
Gaussian sigma .6 pixels); only two favor filtering the secondary. On the face,
135 of 147 patches favor filtering the primary with gain above .005 (median sigma
1 pixel), none favor filtering the secondary. This supports a relative bandwidth
difference but does not isolate optical defocus from resampling/noise/residual
geometry. A simple exposure explanation is inadequate for both the displacement
and bandwidth evidence.
[Train-only registration and bandwidth audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_relative_blur/audit.json).

Doubling refinement to 90/45 iterations preserves cameras/topology and gives
29.118050 / .890884 / .047197 face PSNR/SSIM/LPIPS. Native F/J/L comparisons show
little further change; the remaining F seam and J softness persist. Median vertex
movement reaches .00015471, p99 .00034882. This is not an accepted repair and does
not justify another iteration-count increase without a new hypothesis.
[45 versus 90 iterations, F hand](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine90_three_views/review_F004/hand.png).

The train-only subpixel epipolar audit retains NCC > .9 patch matches on locally
smooth surface. Relative to primary E004_C005_1210YM, E004_B005_1210I7 hand/neck
matches have .059-pixel median absolute epipolar distance (25 patches), whereas
G004_B005_1210FG has .647 pixels (25 patches). E004_B005_1210I7 face matches have
.951 pixels (136 patches). An exact correspondence cannot be moved across an
epipolar line by changing depth alone. However these are patch-centre estimates,
not independent feature ground truth: local appearance, residual registration,
motion or calibration may account for the residual. **Bad fixed calibration is
not yet established.** The next diagnostic should separate these possibilities
before touching the frozen camera template. No pose or intrinsics change was made.
[Epipolar audit and caveats](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/openmvs_refine_epipolar_audit_v2.json).

The final isolated release suite passes 72/72 tests, including known relative
blur recovery, equal-image/no-blur behavior, epipolar invariance under depth-only
motion and subpixel peak recovery. All three original/refine45/refine90 path
inventories contain three verified hashes; face prediction/GT/ROI hashes match.
Original versus refined path requests differ only in mesh and mesh-metadata hashes
and the derived request hash. No accepted repair or completed temporal/fly-through
validation is claimed; the original campaign and frozen calibration are unchanged.

### Independent source-image geometry and temporal camera stability

`audit_fixed_camera_feature_geometry.py` tests direct train-JPEG SIFT matches,
without a mesh, pose optimization or held-out-camera RGB. Mutual ratio-test
matches are split by 128-pixel spatial blocks; a diagnostic fundamental matrix
is fitted on the training subset and evaluated on held matches. The diagnostic
matrix never becomes a new calibration or a prediction input. Independent-model
inlier filtering is not independent ground truth; repetitive texture and a
dominant foreground can still produce misleading correspondences.

| Primary E004_C005_1210YM paired with | 000899 frozen / fitted median pixels | 000973 | 001059 |
|---|---:|---:|---:|
| E004_B005_1210I7 | .312 / .156 | .447 / .181 | .395 / .160 |
| G004_B005_1210FG | .266 / .224 | .397 / .143 | .404 / .195 |
| F004_A005_12103K | .245 / .231 | .448 / .198 | .453 / .166 |
| F004_C005_121059 | .650 / .198 | .740 / .267 | .842 / .277 |

Native inspection confirms some large residuals on recognizably corresponding
eyebrow/eyelid details (up to 2.43 pixels for E004_B005_1210I7 at 000973). A separate
same-camera temporal match audit compares 000899 with 000973/001059. Its near-
identity feature cluster has typical displacements .13–.24 pixels at 000973 and
.14–.33 at 001059 across eight sources. Spatially spread native crops for the
primary and E004_B005_1210I7 show genuinely static room fixtures. There is no
evidence here of multi-pixel rig drift. Too few cross-camera static-room matches
survive to attribute the foreground residual uniquely to calibration or capture
timing. The frozen calibration is therefore still unchanged.

[Direct 000973 feature audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/independent_feature_geometry/audit.json),
[native eyebrow correspondences](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/independent_feature_geometry/E004_C005_1210YM__E004_B005_1210I7_held_crops.png),
[000973 temporal stability](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/temporal_camera_stability/audit.json),
[001059 temporal stability](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/001059/temporal_camera_stability/audit.json).

An opt-in texture-coordinate micro-registration control now addresses residual
train-image misregistration without altering mesh or camera matrices. Shared
train patches constrain a depth-layer-separated displacement field; separate
spatial blocks check its prediction of train correspondences. The output still
samples one native train RGB source once, but **texture UV coordinates deliberately
change**. This is an appearance-alignment extension, not pure unchanged calibrated
projection and not a claim of improved physical geometry. The primary is unchanged,
visibility is never expanded, and unsafe adjustments revert to original sampling.
Displacements, helper hashes and train-only fit checks are retained. Visual and
novel-view acceptance remain required before any temporal promotion.

The first micro-registration canary **fails the native F visual gate**. For the
two main fallback cameras, median held-train shift residual decreases from
1.073 to .363 pixels and .688 to .242 pixels. All seven secondary cameras pass
that train check, yet the visible neck patch, thin bright hand-adjacent seam and
ragged skin/hair silhouette remain. Face PSNR/SSIM/LPIPS are
29.110649 / .890400 / .047048 using the unchanged study polygon. Better train
alignment is not equivalent to a seam-free render; no temporal promotion follows.
[GT / refined mesh / UV registration, native hand crop](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/surface_texture_registration/review/hand.png),
[failed verdict and retained hashes](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/surface_texture_registration/visual_review.json).

### Native RGB interpolation footprint at the lipstick/hand boundary

The exact first-hit visibility test concerns the sample centre, not all four
native pixels used by bilinear RGB interpolation. The unchanged refined-mesh
trace at target (650,535) gives E004_C005_1210YM projected depth .755406, but one
native tap has depth .781321 and contributes 9.50% of the sampled RGB. At target
(700,550), 18.17% of E004_B005_1210I7 and 47.16% of G004_B005_1210FG interpolation
weight lies on a different mesh depth layer (absolute log-depth tolerance .005).
At (680,540), both fallback sources have fully same-layer footprints, although the
primary is occluded. Thus mixed-layer RGB sampling is a concrete boundary issue,
but cannot explain the entire wider neck patch. This is interpolation **within one
camera**, not averaging two cameras.

Two opt-in controls keep mesh, cameras, UV coordinates and train-only color
calibration fixed. The strict footprint guard rejects a source if any contributing
tap crosses depth layers. The depth-aware sampler instead renormalizes the weights
of same-layer native pixels from that one source camera. Both use unfilled mesh
depth and retain the centre-ray visibility test; neither introduces segmentation
or fills missing color from held-out RGB.

| Same 000973 study face polygon | PSNR | SSIM | LPIPS | Mesh-supported pixels with RGB | Native verdict |
|---|---:|---:|---:|---:|---|
| Refined mesh, ordinary bilinear RGB | 29.111637 | .890642 | .047190 | 99.99694% | Fail |
| Reject mixed-depth RGB footprints | 29.076572 | .889319 | .047726 | 99.81440% | Fail: neck patch plus black edge slits |
| Renormalize same-depth native taps | 29.097021 | .890146 | .047661 | 99.96711% | Fail: patch persists; fewer but still visible holes |

The primary camera has 7,106 centre-visible samples with an unsafe full footprint;
renormalization recovers same-layer RGB for 6,534 of them, leaving 572 without any
matching native tap. The strict guard changes part of the bright hand-adjacent
edge, but creates black slits. Renormalization reduces that damage without removing
the conspicuous patch. Neither is an accepted repair, and neither is promoted to
temporal/fly-through confirmation. The original campaign and defaults remain
untouched. Native source crops also visibly show the lower bandwidth of G004_B005
relative to several other sources; attributing this uniquely to optical defocus
still requires separate evidence.

[Native tap trace and RGB identity check](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/rgb_footprint_trace/trace.json),
[native train sources at the neck point](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/rgb_footprint_trace/point_01_train_patches.png),
[strict guard hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/rgb_footprint_visibility/review/hand.png),
[depth-aware sampling hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_aware_rgb/review/hand.png).

### Localized bright-strip diagnosis and free-space controls

The native crop `[490,425,610,545]` shows the visible bright strip inside a region
labelled entirely with E004_C005_1210YM. At target (530,472), the refined mesh
gives depth .671826, whereas neighboring neck is near .705. The projected point
has camera depths .759481 / .648617 in E004_C005 / E004_B005, but their raw stereo
depth medians are .79430 / .68465: those cameras measure a farther surface.
For original triangle 12238, six train cameras supply locally consistent farther
depth, while eight supply near-depth evidence. Triangle 78593 has 15 farther-view
votes versus one near-depth vote. These are contradictions in measured geometry,
not a unique proof that camera calibration, capture timing or stereo is at fault.
The exact shared-mesh visibility test alone cannot detect them.

[GT / prediction / source label, localized strip](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/bright_neck_strip_trace/target_review/bright_strip.png),
[native train-camera patches](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/bright_neck_strip_trace/point_00_train_patches.png),
[ray and raw-depth trace](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/bright_neck_strip_trace/trace.json).

Before this localization, a train-only bandwidth prior independently demoted the
blurrier G004_B005 source: its selected fraction fell from 4.85% to .25%, primarily
replaced by E004_A005. The strip remained because it belongs to the unchanged
primary camera. No eval RGB or anatomical region was used to fit this prior;
spatially separated train patches checked relative bandwidth after symmetric
subpixel registration. Filtering measures bandwidth only and never filters output.

A separate numerical bug was fixed in the opt-in harmonic seam-leveling solver.
On a 512x512 constant-color test with a 256x256 differently exposed central patch,
the old float32/1e-4 stopping criterion reported convergence while leaving .04436
maximum RGB error. Float64/1e-9 with a strict true-residual gate reduces this to
.000206. The real strip changes little, which is consistent with its primary-source
label and depth discontinuity. Existing model and renderer defaults remain unchanged.

`carve_patchmatch_mesh_free_space.py` tests geometric deletion rather than RGB
selection. A triangle centroid needs three train cameras with at least 80% of a
5x5 native-depth footprint supporting a compact farther layer. The gap must exceed
both .005 normalized units and 1% of projected depth. Missing depth is not evidence.
On original 000973 it carves 5,995 triangles and removes 938 small-component
triangles, retaining 77,300 vertices / 147,871 triangles. Target (530,472) changes
from .671624 to .705561, exposing the reconstructed neck behind it. However new
black slits and ragged hand/lipstick boundaries reject this as a final repair.

The opt-in `--tensor-full-block-integration` instead discovers the bounded union
of surface blocks before fusing all 62 depths. Each view updates that whole union,
including positive free-space TSDF observations. A two-plane Open3D regression
test demonstrates the distinction: an early foreground voxel receives weight 2
under per-view block activation, but weight 6 and positive TSDF under union updates.
The upstream kernel accepts positive signed distances beyond truncation, clamped
to one; the block inventory determines which voxels are updated.
[Open3D 0.19 integration kernel](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h).
The real frame activates 2,222 bounded blocks and extracts 80,221 vertices /
155,052 triangles / one component, but the strip remains. Thus omitted block updates
are possible, yet are not established as the dominant cause here. Only the extracted
mesh and metadata are retained, not a serialized raw TSDF volume.

Finally `--source-observed-free-space-veto` rejects source RGB, including primary
RGB, when raw train depth supplies the same robust farther-surface contradiction.
Invalid raw depth remains unknown. This avoids relying only on visibility against
the candidate mesh, but the first control replaces parts of the bright strip with
wrong blue texture and small holes. Geometry inconsistencies cannot safely be
resolved just by selecting another source camera. This control also fails.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Decision |
|---|---:|---:|---:|---|
| Train bandwidth prior, refined mesh | 29.111843 | .890666 | .047378 | Fail |
| Accurate seam gain solver, refined mesh | 29.096813 | .890745 | .047002 | Fail |
| Original mesh + free-space triangle carving | 29.080326 | .889463 | .047544 | Fail |
| Full-block TSDF integration | 29.052221 | .889452 | .047512 | Fail |
| Original mesh + measured free-space RGB veto | 28.973825 | .888947 | .048794 | Fail |

[Triangle-carving F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/free_space_original_three_views/review_F/hand.png),
[triangle-carving J comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/free_space_original_three_views/review_J/hand_neck.png),
[full-block F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_three_views/review_F/hand.png),
[measured-visibility RGB comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/observed_free_space_rgb_verified/review/hand.png).
F/J/L were rendered for both geometric controls and native hand/neck comparisons
inspected. F defects and J hand/lipstick softness prevent acceptance; an apparently
clean L neck crop is not a full-view or fly-through pass. No temporal promotion.
The next bounded geometric hypothesis is enforcing robust free-space contradictions
in the volumetric scalar field before surface extraction, instead of deleting
finished triangles or merely averaging their contradictory observations.

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
