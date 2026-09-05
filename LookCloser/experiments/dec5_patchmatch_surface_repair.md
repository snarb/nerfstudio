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
**Measured partial repair:** rechecking the identical native pixels establishes
that full-block TSDF integration already removes this particular false fragment.
The earlier statement that it leaves the whole strip unchanged was too broad;
neighboring seams and hand defects still prevent overall acceptance.
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
155,052 triangles / one component. The initial review conflated the localized
false fragment with adjacent residual seams: the point-by-point recheck below
shows that full-block updates remove the tested false fragment. Only the extracted
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
The bounded geometric hypothesis of enforcing robust free-space contradictions
in the scalar field before surface extraction was tested next, as documented below.

### Pre-extraction veto and localized full-block recheck

`--tensor-free-space-min-views 3` is a new opt-in diagnostic, disabled by default.
It requires the bounded full-block mode and exactly 62 unique explicit train
cameras. Every active voxel is projected into native train depths. Three cameras
must support the same robust farther-layer criterion used by triangle carving.
Contradicted observed voxels become positive TSDF before marching cubes; unknown
voxels stay unknown and no integration weight is manufactured. No RGB, semantic
mask, eval image or target-image coordinate participates in this operation.

The 000973 CUDA run checks 9,101,312 active voxels in 2,222 blocks. It constrains
1,915,405 already observed voxels, of which only 978 had negative TSDF. The
extracted mesh has 80,063 vertices / 154,759 triangles / one component.
The native F crop shows an additional notch on the hand and no decisive repair
of adjacent neck seams; the J hand/lipstick remains soft/distorted. Reject this
as a final repair. In particular, extra hard vetoing must not be credited with
the false-fragment removal already achieved by full-block integration alone.

| Native F pixel | Original mesh depth | Full-block TSDF | Full-block + voxel veto |
|---|---:|---:|---:|
| (530,472) | .671624 | .705561 | .705561 |
| (540,464) | .672702 | .705842 | .705842 |
| (520,480) | .671367 | .671367 | .671367 |
| (550,475) | .706396 | .706396 | .706396 |

These are unfilled first-hit depths from the same held-out camera calibration.
At the first two points the false hand-depth fragment gives way to neck depth.
The matched original/full-block render crop confirms that the conspicuous bright
leaf disappears without changing source RGB or adding a segmentation mask.
This is evidence for missed free-space block updates contributing to this
localized artifact, not proof that every hand/neck defect has the same cause.
The older full-block visual verdict remains a historical overall fail; its phrase
"strip persists" is superseded by this localized depth/image comparison.

[GT / original / full-block / voxel-veto native comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/volumetric_free_space_three_views/review_localized_full_control/bright_strip.png),
[exact depth and mesh-hash audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/volumetric_free_space_three_views/localized_depth_audit.json).

A second control combines the full-block mesh with the already fixed accurate
harmonic gain solver. This tests remaining RGB seams after the localized geometry
repair, without the extra voxel veto. F/J/L are all rendered again from calibration
only. Adjacent F neck patches and J hand/lipstick softness remain, so this combined
control also fails the overall visual gate.

| Same 000973 study face polygon | PSNR | SSIM | LPIPS | Overall visual gate |
|---|---:|---:|---:|---|
| Full-block TSDF reference | 29.052221 | .889452 | .047512 | Fail; localized fragment repaired |
| Full-block + pre-extraction veto | 29.057016 | .889385 | .047595 | Fail; extra hand notch |
| Full-block + accurate gain solve | 29.066286 | .889747 | .047069 | Fail; residual seams/softness |

[Voxel-veto F hand](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/volumetric_free_space_three_views/review_F/hand.png),
[gain-solver F hand](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_accurate_leveling_three_views/review_F/hand.png),
[gain-solver J hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_accurate_leveling_three_views/review_J/hand_neck.png).
Native F face/ear/hand and J hand/neck plus L neck context were inspected for both
controls. No temporal or fly-through promotion. Tests compare CPU/CUDA native
depth evidence, preserve unqualified/unknown voxels and all integration weights,
and remove a synthetic contradictory foreground slab while keeping its supported
farther plane. Defaults of existing runners and models are unchanged.
The focused suite passes **96 tests** both in the working tree and in an isolated
index snapshot that excludes unrelated uncommitted changes. The new controls'
retained render/metric/review hashes and all 62 raw depth hashes are verified.

Further controls below separate stale color fitting, TSDF discretization and the
implicit assumption that the angular primary is the sharpest camera. None yet
passes the complete requested skin/hand/fly-through gate.

### Refit color after the localized geometry repair

Two fresh 16x9 train-only spatial color fits use the full-block mesh and its own
62 raycasts. The first preserves the old fitting convention; the second opts into
the same .5-pixel sampling and exact mesh visibility used by the newer renderer.
No camera pose or intrinsics is changed. `audit_camera_color_fits_common_samples.py`
compares all three fits on **identical** held train-surface samples. Comparing
their original per-fit summaries directly would mix different visibility sets.

| Same 1,698,892 held train-pair observations | Display L1 median | p90 |
|---|---:|---:|
| Existing fit on original mesh | .017347845 | .061298664 |
| Refit on full-block mesh, legacy sampling | .017345031 | .061293081 |
| Refit on full-block mesh, native/exact sampling | .017415279 | .061318270 |

These are radiometric consistency diagnostics, not face-quality or full-frame
reconstruction metrics. The matched comparison provides no practical evidence
that stale fitting was the dominant residual seam cause. Native F/J/L renders
retain F neck patches and J hand/lipstick softness. Both refits are rejected as
repairs. Defaults remain unchanged; new sampling modes are explicit opt-ins.
[Common-sample audit, replayed from the isolated index](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_refit_full_block/common_sample_audit_verified.json),
[native/exact refit F hand](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_refit_exact_three_views/review_F/hand.png).

### Minimal renderer control and finer full-block TSDF

A matched pair returns to the original geometry-selected **same 16 cameras**,
hard nearest-fill, no additional color correction, no source continuation and no
graph cut. Only the original/full-block mesh changes. Full-block integration
removes the localized false fragment but leaves large chin/neck source-color
patches. Thus that geometry change alone is insufficient, independently of the
more elaborate renderer controls.

A separate geometry control halves the full-block voxel to .00025 while keeping
truncation .004, extraction weight 2 and all other fusion parameters fixed. This
differs from the earlier rejected fine-TSDF experiment, which lacked full-block
updates and changed truncation to .0015. The new run allocates 6,261 blocks and
extracts 348,809 vertices / 681,257 triangles / one component. At the original
focal/depth scale .0005 is roughly seven image pixels per voxel, motivating this
discretization check. However F retains its neck seam and gains a hand notch;
J remains soft/distorted. Reject finer resolution as a sufficient repair.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Full-block mesh, legacy color refit | 28.965292 | .889463 | .047677 | Fail |
| Full-block mesh, native/exact color refit | 28.724285 | .889098 | .047989 | Fail |
| Original mesh, minimal angular16 nearest-fill | 31.052477 | .889791 | .053517 | Fail |
| Full-block mesh, same minimal renderer | 31.065138 | .889785 | .053547 | Fail |
| Fine full-block mesh, matched calibrated graph cut | 29.078770 | .890198 | .047623 | Fail |

[Minimal matched F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/original_minimal_three_views/review_F/hand.png),
[fine full-block F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_fine_three_views/review_F/hand.png).
All five controls have finite face metrics, native visual verdicts and hashes;
none is promoted to temporal reconstruction.

### A blurrier primary camera must not be exempt from source-quality checks

The train-only bandwidth audit at anchor J identifies the angular primary
K004_D005_121016 as softer than several neighboring sources. Against I004_D005_1210Q7,
the signed relative blur variance has fit/held medians -1 / -1.5625 pixel-squared
over 102/23 patches. This is relative projected bandwidth, not a unique optical
defocus estimate. The old prior penalizes only *positive* variance in alternatives:
it cannot demote a primary that is itself blurrier. Native train-image patches
corroborate the hand-detail difference.

`--seam-cut-bandwidth-allow-primary` admits reliable signed estimates, then adds a
common offset to keep source costs nonnegative. Unqualified alternatives retain
primary-equivalent cost; neither source detail nor visibility is modified. No
physical-camera exception, anatomical mask or held-out RGB enters this decision.
The common three-view canary uses bandwidth penalty .003 and rank penalty .001.

The paired three-view requests differ **only** in this new boolean (and the
derived request hash). F predictions are byte-identical. At J, the primary's
selected fraction falls from 87.86% to 2.70%, and I004_D005 rises from 10.46% to
89.99%. At L, source rank one rises from 14.47% to 83.75%. Native J hand/lipstick
and L face details improve. Nevertheless F's neck seam is unchanged, and a native
J lipstick crop still shows an inaccurate stepped tip/contour. Overall fail, not
a passed general recipe. Both F outputs score **29.047241 / .889427 / .047545**
face PSNR / SSIM / LPIPS under the unchanged study protocol.

[Matched J comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth_three_views/review_J/hand_neck.png),
[native J lipstick](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth_three_views/review_J_native/lipstick.png),
[native train-camera hand patches](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/J_primary_bandwidth_trace/point_00_train_patches.png).

Post-hoc region summaries suggest a limitation of a single quality score per
camera: F source F004_A005 is sharper in two accepted hand patches but blurrier
over 39 face patches. Two local samples are insufficient to fit a new local/depth
model. Denser train-only verification is needed before introducing that model;
these diagnostic boxes are never passed to prediction.
[Regional bandwidth evidence and caveat](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth_three_views/regional_bandwidth_audit.json).

The focused suite passes **116 tests in an isolated index snapshot**, excluding
unrelated working-tree edits. The common-sample audit replay is numerically
identical, including sample and visibility hashes. Existing rendering and model
defaults remain unchanged.

The final [artifact audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/color_refit_bandwidth_findings.json)
verifies 21 full-resolution renders from seven failed control configurations,
their face-only metrics and review records, 62 raw depth maps, calibration and
runtime-script provenance (289 retained hashes). This is an audit pass, not a
visual repair pass; temporal promotion remains disabled.

### Dense local bandwidth audit and remaining right-neck seam

The next audit uses disjoint spatial blocks: a full patch/search footprint must
fit inside one 128-pixel block, and held blocks do not overlap fit footprints.
Counts of overlapping patches are reported separately from independent blocks.
Depth-bin boundaries use fit observations only; a patch crossing a depth jump
is excluded. These are train-camera diagnostics, not reconstruction metrics.

At J, the primary-versus-I004_D005 blur estimate remains negative on 30 fit /
5 held blocks (block medians -1 / -1.28125 pixel-squared). At F, G004_B005 remains
softer on 22 fit / 5 held blocks (.36 / .64). This corroborates the earlier
global-quality result with stricter support accounting. It does **not** validate
a local skin/hand-specific model.

The 48-pixel patch plus search margin misses narrow disocclusions even when all
28 camera pairs are compared: all 7,392 retained observations have a visible
angular primary. A 24-pixel diagnostic patch with the unchanged 8-pixel search
margin, stride 8 and the same single-depth-layer check yields 66,638 observations,
including 292 where the primary is occluded. Those 292 occupy only two spatial
blocks, neither in the predeclared held split. More patches are therefore not
independent validation; no local prior is fitted or promoted from these data.
The smaller patch is an audit opt-in; the renderer's 48-pixel default is unchanged.
[All-pairs narrow-patch evidence](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_F_dense_bandwidth_audit/local_bandwidth_all_pairs24.json).

A direct trace of the remaining **right** neck patch at F (580,660) finds that
E004_C005 projects .801943 but all 25 native stereo taps lie near .754480: it sees
foreground hand. E004_B005 likewise sees hand near .642079 versus projected
.692188. G004_B005 sees neck: projected .716629, raycast .716629 and native median
.716865, with all 25 taps within .001 normalized units. Two neighboring points
reproduce this pattern. Thus this localized patch has supported neck geometry;
it is not the already repaired false foreground leaf at (530,472), and its RGB
is not room filling or source averaging.
[Native depth footprints](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/F_remaining_neck_trace/native_raw_footprints.json),
[fixed-camera source crops](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/F_remaining_neck_trace/point_00_train_patches.png).

Three further three-anchor controls use the full-block mesh and signed bandwidth
prior. Raising its penalty from .003 to the helper's existing .01 scale changes
the sampled neck pixels from G004_B005 (rank 2) to E004_A005 (rank 3), but leaves a
visible seam. Adding the already tested accurate harmonic gain leveling does not
remove it. Removing the rank preference with the bandwidth prior still enabled
softens F facial detail and introduces a conspicuous J chest/clothing boundary.
Unlike the old zero-rank test, this one includes color calibration, exact
visibility, repaired full-block geometry and the signed bandwidth prior.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Signed bandwidth .01, rank .001 | 29.041328 | .889583 | .047575 | Fail |
| Same + harmonic gain leveling | 29.046722 | .889692 | .047160 | Fail |
| Signed bandwidth .01, rank 0 | 28.297064 | .908267 | .059281 | Fail |

[Matched F gain control](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth01_gain_three_views/review_F/hand.png),
[remaining J lipstick contour](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth01_gain_three_views/review_J/lipstick.png),
[zero-rank J regression](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/all_source_bandwidth01_zero_rank_three_views/review_J/hand_neck.png).
All three controls have native F/J/L reviews, finite face-only metrics and hashes.
No temporal or fly-through confirmation is claimed. The next useful question is
whether locally overlapping alternative cameras provide a consistent relative
bandwidth graph under withheld-camera-pair checks, not another unconditional
increase of the global source penalty.

An isolated index snapshot passes **125 tests**. The old runtime helper and the
current default produce exactly identical observations for 139 real F/G patches.
The [final audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/local_bandwidth_findings.json)
verifies 241 retained hashes, nine new renders, face metric inputs, audit runtime
snapshots and all 62 raw depth hashes. Audit pass is not a visual pass.

### Local camera-pair graphs and spatial-plus-angular color controls

Two opt-in controls test the remaining source-appearance discontinuity without
changing mesh, camera poses, visibility or RGB source averaging. The local bandwidth
control fits relative camera quality in depth-separated 128-pixel cells, using
24-pixel train patches. Each camera-pair edge is predicted while withholding that
whole edge. This is **internal cycle consistency, not independent spatial/scene
validation**. Reliable cells contribute scalar source costs; only those costs are
smoothed, in float64 with a true-residual gate. Unknown cameras keep the global
prior. F qualifies 50 of 85 cells, assigning 438,456 pixels; the solve takes 576
iterations with relative residual 8.90e-8.

The matched rank .001 versus .0001 requests differ only in rank penalty and
request hash. Neither removes the neck seam. Weaker rank softens F facial detail
and introduces a J chest/clothing texture boundary. Native J tube contours remain
inaccurate. The new helper is an explicitly rejected diagnostic, not a default.

The second control composes the existing mesh-attached angular gain with the
spatial camera calibration. Fitting and rendering now explicitly agree on the
camera-response mode; old RGB-mode manifests remain compatible. The calibration
is the train-only full-block-mesh refit. With smoothness 20, held train-pair median
display L1 changes .0171341 -> .0169209. Reducing only smoothness to .2 improves
that diagnostic to .0164175 (4.18% below no angular correction), with converged
true residual 6.96e-5. Held-out F/J/L RGB is never used for fitting or rendering.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Spatial refit, no angular field (matched control) | 28.965292 | .889463 | .047677 | Fail |
| Local bandwidth .01, rank .001 | 29.036213 | .889520 | .047626 | Fail |
| Local bandwidth .01, rank .0001 | 28.851383 | .886384 | .050197 | Fail |
| Spatial + angular field, smoothness 20 | 28.449343 | .888701 | .047824 | Fail |
| Spatial + angular field, smoothness .2 | 28.675110 | .889403 | .045877 | Fail |

The two angular requests differ only in fitted-field hash and derived request
hash. The .2 field improves face LPIPS but leaves the F neck discontinuity and
J soft/incorrect lipstick contour. Native ear, face, hand/neck and lipstick crops
were reviewed on all three anchors. These are three views of one temporal frame,
not temporal validation. Neither control is promoted to every-40th-frame runs.

[Local bandwidth F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/local_bandwidth_graph_three_views/review_F/hand.png),
[weak-rank J regression](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/local_bandwidth_graph_weak_rank_three_views/review_J/hand_neck.png),
[spatial/angular F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/angular_spatial_s02_full_block_three_views/review_F/hand.png),
[native J lipstick](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/angular_spatial_s02_full_block_three_views/review_J/lipstick.png).

A current-code F rerender with the new options disabled is **byte-identical** to
the previous spatial-refit control (PNG SHA-256
`f5e605bfe9153bdfbfb0d2d4a4321c42aa883a94c2dc62257f102bc92582f387`).
Original campaign metrics and outputs are unchanged. The study ROI protocol is
not numerically interchangeable with the old campaign's face polygon.

The isolated index snapshot passes **144 distinct tests**. The
[artifact audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/local_field_controls_findings.json)
checks 12 full-resolution renders, four explicit failed visual verdicts, matching
GT/ROI hashes, runtime source snapshots, both angular fits and all 62 raw depth
hashes (309 retained hashes). The audit passing does not make these visual controls
successful. Further response-model work must separate color mismatch from projected
texture bandwidth; a lower face LPIPS alone is not a reason to promote a recipe.

### Independent spatial RGB fields and native-depth bandwidth regression

The old `spatial` response is achromatic: it combines scalar exposure with one
16x9 scalar field, not the separately fitted diagonal RGB gains. An opt-in
`--spatial-rgb` calibration now fits three independent gain fields, with the same
fit-only sample selection and regularization. Rendering selects them explicitly
with `--camera-color-model spatial-rgb`; no source/channel mixing, RGB filtering,
pose change or semantic mask is introduced. All existing 62-camera scalar/RGB
parameters exactly reproduce the previous native/exact fit.

On the same 1,698,892 held train-pair samples, median display L1 improves only
.01741528 -> .01714472 (1.55%); p90 changes .06131827 -> .06113207. The native
F/J/L review retains the neck seam and soft/incorrect J lipstick contour. A
current-code scalar control is PNG-byte-identical to the old native/exact scalar
render (`298620da074396441212d9aa57acd6d2390ffd840684cf2abad56603f6eee3a1`).
This rejects the missing chromatic fields as a sufficient explanation or repair.
[RGB-field F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/spatial_rgb_response_three_views/review_F/hand.png).

A second audit tests whether camera bandwidth varies with **native source-camera
inverse depth**, accounting for the local source-to-target projection Jacobian.
This avoids a target-depth lookup tied to F. The fitted values are relative
projected blur-variance regressions, not identifiable physical lens PSFs. Patch
observations are collapsed into camera-pair/spatial-block medians; full fit and
held footprints remain disjoint. Normalization, fitting and evaluated-depth bounds
use fit observations only. The audit changes no prediction.

| Train-only bandwidth audit | Fit / held spatial blocks | Constant median / p90 error | Linear inverse-depth median / p90 | Quadratic median / p90 |
|---|---:|---:|---:|---:|
| F camera neighborhood | 44 / 6 | .271024 / .723475 | .210602 / .583191 | .217004 / .639762 |
| J camera neighborhood | 49 / 9 | .312146 / .947971 | .288571 / .701379 | .288329 / .715513 |

F uses 946 fit / 134 held camera-pair blocks, J 1,092 / 171. Linear depth improves
F median held error by 22.3% and J by 7.55%; the quadratic is not clearly better.
These are separately fitted camera neighborhoods in one temporal frame, not a
transferred rig model or a passed temporal gate.
[F bounded audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_conditioned_bandwidth_F_bounded.json),
[J bounded audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/depth_conditioned_bandwidth_J_bounded.json).

The paired F selection control uses the same immutable **RGB8 source warps** and
visibility for both models (not a fresh native-reprojection claim), bandwidth
weight .01 and rank .001. Inverse-depth evaluation is clamped to observed fit
support. Linear depth changes the traced right-neck pixels to F004_C005 (rank 5),
but introduces a more conspicuous hand-adjacent texture patch. Adding the existing
one-sided harmonic gain solver keeps source labels byte-identical and softens the
patch but does not remove it. All three controls fail native face/ear/hand review.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Spatial RGB camera response, native F/J/L canary | 28.589523 | .888543 | .048607 | Fail |
| Constant native-bandwidth F control | 29.045212 | .889527 | .047618 | Fail |
| Linear native-depth bandwidth F control | 29.008965 | .889442 | .047485 | Fail |
| Same labels + harmonic gain | 29.042618 | .889678 | .047234 | Fail |

[Matched depth-model comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/native_depth_bandwidth_linear_F/review_F/hand.png),
[matched gain comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/native_depth_bandwidth_linear_gain_F/review_F/hand.png).
The first diagnostic writer attempt saved the prediction but failed to save its
source map. Incomplete directories/logs were retained as `*_failed_writer`; one
clean retry after fixing the path produced byte-identical RGB and complete
manifests. No incomplete attempt is treated as complete.

The lesson is narrower than a focus diagnosis: native depth predicts part of the
inter-camera detail mismatch, but selecting a sharper source still leaves a
visible appearance boundary. Neither RGB-field correction nor this source-cost
control is promoted to temporal/fly-through reconstruction.

An isolated index snapshot passes **154 tests**, including rigid-rig coordinate
invariance, length-unit invariance and fit/held isolation. The
[final artifact audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/rgb_depth_bandwidth_findings.json)
checks six candidate renders, four failed visual verdicts, exact source identity
for unlevelled RGB8 controls, identical labels for the gain control, 62 raw depth
hashes and 299 retained hashes. Artifact-audit pass is not a visual-repair pass.

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
