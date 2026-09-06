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

### Mesh-attached per-camera RGB gain fields: rejected three-view control

The next control fits smooth **per-camera, per-mesh-vertex** log RGB gains from
all 62 train cameras, after the native/exact scalar spatial calibration. Unlike
the earlier primary-relative screen-space correction or first-order angular
field, it solves all camera differences together on the mesh graph. Shared
albedo cancels from the objective. The renderer still projects one train camera
per pixel and applies its interpolated gain; it does not render a consensus RGB,
average source images, change geometry/visibility, use a semantic mask, or read
F/J/L RGB for prediction. Corrected colors can change graph-cut labels.

The full-block mesh, scalar camera calibration, hard seam-cut recipe and F/J/L
cameras are unchanged. An off-control with the new renderer exactly reproduces
the native/exact scalar baseline PNG SHA `298620da074396441212d9aa57acd6d2390ffd840684cf2abad56603f6eee3a1`.

The initial solver correctly refused publication: its true relative residual
was 1.80e-6, above the required 5e-7. Projecting the preconditioner into the
zero-camera-mean invariant subspace removes numerical gauge drift without
changing the fit objective. The same data, smoothness 64 and ridge .01 then
converge in 1,904 iterations, with true residual 9.08e-8 and mean-gauge error
4.76e-16. The failed log is retained. Gain clipping affects 0.376% of coefficients.

This is **not** a successful color fit: over 2,279,308 held train-pair samples,
median display L1 worsens .01719648 -> .01757403 (2.20%). Validation withholds
spatially grouped mesh vertices; it is not claimed to enforce disjoint native
RGB interpolation footprints.

| 000973, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Matched scalar camera-field baseline | 28.724285 | .889098 | .047989 | Fail |
| Additional mesh-attached camera fields | 28.537418 | .888003 | .048547 | Fail |

All seven native-resolution F/J/L face/ear/hand/neck/lipstick comparisons were
viewed. The F ochre strip beside the lipstick remains; the J tube remains soft
with an incorrect contour; the L neck/chest transition is still visible. Missing
room is ignored, but incomplete hair silhouette is not excused. No temporal or
fly-through repair is accepted.
[F hand/neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mesh_camera_color_s64_three_views/review_F/hand.png),
[J lipstick comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mesh_camera_color_s64_three_views/review_J/lipstick.png),
[L neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mesh_camera_color_s64_three_views/review_L/neck.png).

The result rejects a smooth per-camera surface gain as a **sufficient repair**,
not every possible camera-response model. Earlier native-depth traces establish
that selected points in the right-neck patch have supported neck geometry and
single-camera skin RGB, not averaged room pixels. That local result does not
prove the hand/tube contour or entire mesh correct. Further trials must separate
detail mismatch from contour errors rather than assume another gain field will
fix both.

The isolated index suite passes **177 tests**, including held/hidden RGB
isolation, shared high-frequency albedo cancellation, rigid-motion invariance,
true-residual/gauge checks, barycentric binding and fail-closed provenance.
[Artifact audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/mesh_camera_color_findings.json)
checks three candidate views, the byte-identical off-control, 62 source RGB and
raw depth hashes, finite face-only metrics and the inspected-crop inventory.
The exact fit/render runtime is archived; subsequent guard-only changes make
checksum failure an explicit exception and reject RGB sampling overrides that
would bypass the fitted fields. Existing defaults and the original campaign
remain unchanged.

### Fixed-source detail restoration and the noise/blur ambiguity

A separate diagnostic freezes the categorical source map and the same immutable
RGB8 train warps. It applies a bounded, within-one-source inverse-heat detail step
in exposed-linear RGB; it is explicitly **not** unchanged pointwise reprojection
or an identified optical deconvolution. The four-neighbor coefficient is capped
at .125 (2x maximum constant-coefficient spectral gain). A full 3x3 valid,
unclipped, single-depth-layer footprint is required. No camera RGB is averaged,
no geometry/visibility changes, and no eval RGB or anatomy-specific exception
determines the filter. The amplitude uses the prior linear native-depth
bandwidth regression minus twice its **fit-only** median error, never held RGB.

Both before/after images use exactly the same source labels. The available F
warps have `nearest_fill8` labels; J has `seam_cut8`. These are paired filter
controls within each view, not a comparison of those selection algorithms.
The fixed-label RGB8 baseline differs from its native float renderer by at most
one RGB8 level. The initial F invocation assumed a nonexistent `seam_cut8` path
and failed before creating output; the log is retained, and one retry explicitly
selected the actual existing variant. J's original runtime script is archived.

| Held train-patch control | Pair blocks / spatial blocks | Median NCC before | Median NCC after | Median paired block change | Blocks improved |
|---|---:|---:|---:|---:|---:|
| F neighborhood | 134 / 6 | .904201 | .894196 | -.003342 | 4.48% |
| J neighborhood | 171 / 8 | .887257 | .875847 | -.007402 | 4.68% |

NCC is an internal correspondence diagnostic, not a reported reconstruction
quality metric. Original correspondences remain fixed; dense patches are
aggregated into pair/block summaries rather than treated as independent votes.
The filter changes 34,701 F pixels and 497,648 J pixels after RGB8 rounding.

| 000973 F, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Exact RGB8 fixed-label control | 28.978622 | .888390 | .051534 | Fail |
| Same labels + bounded source detail | 28.977867 | .888360 | .051521 | Fail |

Native F face/ear/hand and J hand/neck/lipstick comparisons were viewed. F retains
the conspicuous source patch and thin wrong contour beside the lipstick; J
retains its soft, oversized tube contour. The minute F LPIPS improvement does
not override either failed visual gate. No L/path/temporal promotion is attempted
after the paired train and F/J gates fail.
[F paired detail control](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/bounded_source_detail_F/review_F/hand.png),
[J paired detail control](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/bounded_source_detail_J/review_J/lipstick.png).

This prompted a concrete check of the measurement, not further sharpening
strength tuning. With equal true blur and independent noise standard deviations
.015/.003, the old two-image Gaussian/NCC profile reports nonzero blur in **all
30 synthetic checks** (median sigma .8 pixels). Filtering one noisier image can
improve correlation even though neither camera has a different true PSF.

`audit_triplet_source_bandwidth.py` tests a third-camera instrumental moment
estimate. Its small-blur model is `B ~= gain * (A + variance/2 * Laplacian(A))`;
moments against a third camera remove the independent-noise cross term under
that model. The same noise-only canary estimates variance .00604 instead of
interpreting sigma .8 as optical blur. Known additional variances .36 and 1.0
are estimated .32023 and .68014: the larger-blur bias explicitly limits the
first-order approximation. Shared scene correspondence and independent sensor
noise remain assumptions, not established capture facts.

On real data, the third camera is ordered by rig geometry and must have an
existing registration cycle closing within .5 pixel. A/B use symmetric half
shifts. Moments are pooled per camera-pair/spatial block, with both directions
reported; ill-conditioned moments and implausible gains are rejected.

| Same admitted held pair blocks | Blocks | Median absolute old variance | Median absolute instrumental variance | Median estimator difference | Forward/reverse disagreement |
|---|---:|---:|---:|---:|---:|
| F neighborhood | 129 | .500000 | .262025 | .292250 | .078279 |
| J neighborhood | 110 | .640000 | .529883 | .235190 | .299508 |

The audit uses 63,407 F and 79,273 J triplet observations. These differences do
not prove that all real-camera bandwidth differences are noise, or that the
entire patch has correct geometry. They demonstrate why the old relative
Gaussian fit is unsafe as a physical deblurring parameter. J's substantially
larger directional disagreement also cautions against promoting this new
small-blur diagnostic directly to a rendering model.
[F triplet audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/triplet_bandwidth_F.json),
[J triplet audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/triplet_bandwidth_J.json).

An isolated index snapshot passes **192 tests**. The
[artifact audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/source_detail_triplet_findings.json)
rechecks source/selection identity, the single-source filtered lookup, finite
face-only metrics, inspected crops, both triplet audits, runtime hashes and
62 original raw depth hashes. No renderer or campaign default changes in this
step; both detail controls remain rejected.

### Direct seam-gradient correction and adjacent-time correspondence audit

A fixed-label additive **gradient-domain color correction** tests whether the
remaining patch is just a color discontinuity at source boundaries. Each seam
edge uses a difference from one train camera visible at both endpoints. Within
a source region its original gradient remains the objective. True depth edges
and seams without a common visible camera are disconnected. A float64 screened
Poisson solve (ridge 1e-6, true residual <5e-9) determines the additive RGB offset.
Unlike the earlier one-sided gain field, the primary can also change. This is
explicit color correction, not unchanged pointwise reprojection; there is no
source RGB average, new source selection, geometry edit or eval RGB input.

The controlled F `nearest_fill8` and J `seam_cut8` labels are their existing
immutable RGB8 labels, as in the preceding detail test. F converges in 7,328
iterations, J in 7,456, with true residuals 8.85e-10 / 9.01e-10. Both conserve
their pre-clipping mean color to numerical precision. F clips 1,114 channels,
J 186; the large maximum F offset (.485) is retained in the audit, not hidden.

On 956 same-depth guided edges in the F diagnostic neck box
`[500,550,680,740]`, mean disagreement with the chosen train-camera gradient
drops **.033786 -> .005653**. On 2,192 hand/neck edges it drops
.041914 -> .006732. These are internal color-continuity checks, not GT-based
reconstruction metrics. Despite this large reduction, native F/J review still
shows the neck patch, the thin wrong F contour and the soft oversized J tube.
Simply increasing a seam correction is not justified by this result.

| 000973 F, unchanged study face polygon | PSNR | SSIM | LPIPS | Visual gate |
|---|---:|---:|---:|---|
| Exact fixed-label RGB8 control | 28.978622 | .888390 | .051534 | Fail |
| Additive seam-gradient correction | 28.462755 | .888527 | .050800 | Fail |

[F gradient comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/gradient_leveling_F/review_F/hand.png),
[J gradient comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/gradient_leveling_J/review_J/lipstick.png).
All five F face/ear/hand and J hand/neck/lipstick panels were viewed. No temporal
rendering recipe is promoted.

A separate read-only audit then tests **adjacent source-frame identity**, rather
than assuming that static rig stability proves synchronization of moving hands.
It uses no mesh or held-out RGB. E004_C005 reference SIFT features are tracked
forward/backward within that camera; moving features must pass an LK round trip
<.5 pixel and move at least .5 pixel per available frame. Mutual SIFT matches in
E004_B005 and G004_B005 are checked against the frozen epipolar geometry over
offsets -2..+2 available frames. Each available step is two numeric source-frame
IDs; no FPS or millisecond timing is inferred.

| Source time | Secondary train camera | Same features present at all five offsets | Median absolute epipolar error at offsets -2 / -1 / 0 / +1 / +2, pixels |
|---|---|---:|---|
| 000973 | E004_B005 | 48 | 3.210 / 2.561 / **.762** / 3.681 / 5.299 |
| 000973 | G004_B005 | 30 | .775 / .463 / **.363** / .573 / .654 |
| 001059 | E004_B005 | 79 | 1.868 / 1.213 / **.482** / 2.213 / 5.738 |
| 001059 | G004_B005 | 80 | 1.176 / .653 / **.591** / 2.470 / 4.645 |
| 001139 | E004_B005 | 78 | 5.763 / 3.402 / **.636** / 3.402 / 6.957 |
| 001139 | G004_B005 | 58 | 5.555 / 2.813 / **.516** / 4.379 / 8.428 |

Zero offset is best in all six paired sets. This does **not** establish perfect
synchronization: per-feature linear zero crossings are broad, and G004_B005
medians vary -.478 / -.336 / -.193 available frames across these times. Fixed
calibration/localization bias divided by changing motion can mimic such an
offset. Almost no near-static cross-camera tracks survive this audit, so it
cannot independently separate calibration, subframe timing, shutter effects or
view-dependent feature localization. No source time, pose or intrinsics is changed.

Six sheets with 36 spatially spread correspondences were visually inspected;
recognizable hand, clothing, hair, eye and neck features are present. Repetitive
hair/clothing and view-dependent highlights remain correspondence caveats.
[Temporal audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/audit.json),
[000973 E004_B005 identities](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/review/000973_E004_B005_1210I7.png),
[001139 G004_B005 identities](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/review/001139_G004_B005_1210FG.png).
The 39 EXR reads use the same display curve, per-image exposure and JPEG98 4:4:4.
The audit initially omits JPEG entropy optimization; enabling it reproduces the
three original 000973 JPEG hashes exactly, confirming unchanged image content.
The two inspected EXR headers contain no capture timecode/timestamp attributes.

The isolated index suite passes **200 tests**. The
[closing audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/gradient_temporal_findings.json)
rechecks both color controls, finite face-only metrics, source hashes, all six
temporal review sheets, the 39 EXRs and the 62 original raw depth hashes. This
is diagnostic progress, not a successful skin/contour repair or the requested
every-40-frame fly-through validation.

### Held-time epipolar controls and an independent 001219 window

#### What was tested

Read-only train-camera correspondence models separate a spatial residual field,
a motion-dependent residual term, and valid pairwise epipolar matrices. Nothing
exports/applies a new rig calibration, changes source times, uses held-out RGB,
or produces a new prediction. Three earlier anchor times (000973, 001059, 001139)
provide leave-one-time-out fits. A fourth, **001219**, and its entire five-frame
source window are disjoint from the fitting windows.

**Diagnostic correction:** the first unconstrained spatial design used both
reference and zero-offset secondary xy. The latter also defines the response;
this can explain localization error algebraically and is high-severity response
leakage. Its apparent improvement is withdrawn, not evidence for calibration or
timing. The original JSON/runtime is preserved and explicitly superseded by a
reference-xy-only model, with a regression test rejecting four-coordinate input.
This correction changes no renderer, source image or campaign metric.
[Supersession record](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/spatial_model_supersession.json).

The corrected field uses fit-only normalization, one total weight per
time/128-pixel block, Huber .5-pixel weighting, fixed slope ridge .01, and motion
predictor `(e(+1)-e(-1))/2`, excluding the zero-time response. It is deliberately
not a physical calibration model. Pairwise essential fits preserve the supplied
intrinsics; fundamental fits are less constrained. Both use MAGSAC .75-pixel
thresholds and at most eight spatially spread fitting points per time/block.
**No held correspondence is rejected using a fitted model's inlier mask.**
The closing audit also caught repeated fitting indices when SIFT emitted
identical reference xy with different orientations. The new diagnostic sampler
now deduplicates xy within each block before spatial selection; both model
audits were rerun. The first model tables are superseded, with runtime and JSON
preserved. This does not modify the legacy shared sampler or production code.
[Sampling correction](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/epipolar_sampling_supersession.json).

#### Results

Median of spatial-block median absolute epipolar errors, pixels; these are
correspondence diagnostics, **not face/image reconstruction metrics**:

| Secondary train camera | Held time | Frozen rig | Essential, fixed intrinsics | Fundamental |
|---|---|---:|---:|---:|
| E004_B005 | 000973 | .486967 | .523761 | .245505 |
| E004_B005 | 001059 | .562672 | .384027 | .207821 |
| E004_B005 | 001139 | .616447 | .469205 | .214534 |
| G004_B005 | 000973 | .509046 | .523200 | .334034 |
| G004_B005 | 001059 | .528510 | .501672 | .212938 |
| G004_B005 | 001139 | .477533 | .494729 | .288848 |
| E004_B005 | **001219, new window** | **.627138** | **.409409** | **.174180** |
| G004_B005 | **001219, new window** | **.405746** | **.417887** | **.322371** |

The fourth-time comparison retains all 240 E004_B005 / 214 G004_B005 matches
from the preexisting feature-selection protocol. This is a different cohort
from the all-five-offset moving-track table above and must not be numerically
merged with it. There are 62/64 such all-five-offset tracks at 001219; zero whole
available-frame shift is again best for each pair. The inspected calibration
declares zero lens-distortion coefficients for these three train cameras.

Reference-only spatial/timing fits remain inconclusive: joint motion
coefficients across held folds are .0116 / .1682 / -.0251 for E004_B005 and
-.0225 / .0167 / .1570 for G004_B005, in available-frame units. Joint fits worsen
some held errors. Neither a stable timing correction nor a unique causal
separation follows. The model coefficient explains the residual; a hypothetical
zeroing source shift has the opposite sign. No such shift was applied.

[Held-time matrices](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/multitime_epipolar_models_unique_xy.json),
[reference-only residual models](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/spatial_vs_timing_reference_only.json),
[independent 001219 results](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence_001219/independent_holdout_unique_xy.json).
Both native 001219 correspondence sheets were viewed: twelve spatially spread
fabric, hair, neck/finger and ear/earring patches. Repetition and highlights
remain subpixel-localization caveats, not independently verified correspondences.
[E004_B005 sheet](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence_001219/review/001219_E004_B005_1210I7.png),
[G004_B005 sheet](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence_001219/review/001219_G004_B005_1210FG.png).

The isolated staged-index suite passes **212 tests**. The
[closing evidence audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/temporal_source_correspondence/held_time_findings.json)
checks unique observation keys, finite coordinates, all 52 source EXRs across
four windows, fit/held separation, matrix ranks and held-distance calculations,
the corrected spatial fits, the original 62 raw depth hashes and unchanged mesh.
It resolves the old helper hash to its archived runtime and verifies that the
only helper used by the matrix audit (`held_block_summary`) is AST-identical.
[Inspectable check notebook](assets/dec5_held_time_epipolar_checks.ipynb).

#### Insights

Free fundamental models improve all eight held pair/time comparisons, whereas
fixed-intrinsics essential fits are less consistent. This justifies a separate
**common-rig calibration control**, but does not prove that intrinsics are the
cause: stable feature-localization bias, restricted scene geometry and other
model mismatch remain alternatives. Only two camera pairs were tested, not a
joint 62-camera rig, and an epipolar improvement is not a repaired hand/neck mesh.
There is no new PSNR/SSIM/LPIPS result, accepted skin-seam repair, or every-40th
frame/fly-through reconstruction validation in this diagnostic stage.

Applying a replacement calibration would depart from the pinned-template
constraint. Explicit authority has been requested for an isolated train-only,
multi-time common-rig experiment; it has not been received. Original source
times, template, published campaign, and rendering/model defaults remain intact.

### Authorized common-rig control on clever-shadow (2026-09-06)

#### What was tested

The user subsequently authorized autonomous experiments, including common-rig
calibration refinement, and preferred the more powerful clever-shadow GPU.
This supersedes the authorization wait above; it does not change source EXRs,
the original template, the published campaign, or the held-out RGB exclusion.

`build_multitime_rig_tracks.py` extracts duplicate-xy-free SIFT features from
62 train cameras at 000973 / 001059 / 001139. Same-time mutual descriptor matches
are checked by a free fundamental model; track unions reject conflicting
observations from the same camera. There is no cross-time matching or semantic
mask. The 25,312 triangulated tracks supply 150,382 observations. Cameras are
shared exactly across times; 3D points from different times remain independent.
Virtual observation images concatenate those point sets into exactly 62 camera
parameter blocks, so scene motion cannot create per-time camera poses.

`refine_multitime_camera_rig.py` compares poses-only, focal and full-pinhole
adjustment. All three original eval-camera calibration rows remain unchanged.
Sparse BA uses PyCOLMAP 4.1.1; this is not the dense PatchMatch binary. A separate
regularized control uses a common absolute-pose prior (rotation std .2 degrees,
translation std .03 original world units) and bounds focal changes to +/-5%
and principal-point changes to +/-32 pixels. The
[COLMAP pose-prior implementation](https://github.com/colmap/colmap/blob/4.1.1/src/colmap/estimators/cost_functions/pose_prior.h)
defines rotation followed by translation in the sensor frame. PyCeres 2.6 is
installed privately, without changing the Nerfstudio environment or defaults.

#### Results

Unregularized intrinsics are rejected: some camera centers shift by 3.47–4.06
world units, and full pinhole changes a principal point by more than 1,500 pixels.
Lower fitting error is insufficient evidence of a plausible physical rig.

Two independent held times, **000979 and 001219**, provide another 124 train
EXRs and 15,822 tracks. A fixed inventory of 153,643 pair correspondences across
716 camera/time pairs is evaluated without candidate-dependent inlier removal.
Errors below are symmetric epipolar distances: medians within 128-pixel blocks,
then within camera/time pairs. They are not face reconstruction metrics.

| Shared-rig variant | Held pair/block median, px | Held p90, px | Pairs improved | Optimization |
|---|---:|---:|---:|---|
| Original fixed template | .545406 | 1.009393 | — | Fixed |
| Poses only | .476349 | .807722 | 68.58% | Iteration limit 200 |
| Poses + focal, regularized | .462737 | .806897 | 72.49% | Converged, 90 iterations |
| Full pinhole, regularized | .458867 | .793589 | 73.60% | Iteration limit 1000 |

The converged poses+focal candidate is selected **before rendering** for the
000973 reconstruction control. Full pinhole adds little held improvement and
has not converged. Some focal parameters reach the common bounds; the selected
candidate's maximum aligned center shift is .203 world units and maximum
rotation change .509 degrees. It is not yet an accepted calibration or recipe.
[Selection receipt](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/selection_before_render.json),
[fixed held comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/held_rig_scores.json).

The exact dev3 CUDA COLMAP binary was copied into a private clever-shadow bundle
with its required libraries; both executables have SHA-256
`27bfbe22c358062444495c2b105991d5847e1eab92dfaa8c9cae4b46f5c9c66e`.
It reports 3.13.0.dev0 / commit 5509fffe / CUDA. A 13-train-camera canary runs
both three-iteration passes at 1920, with source count 12 and the frozen 6/2
geometric gates. It produces **13/13 finite 1080x1920 geometric maps**, mean
coverage **.373476**, minimum **.329585**. This confirms execution on the RTX PRO
6000, not full-rig reconstruction quality. The system COLMAP and libraries were
not replaced. [Canary receipt](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/local_colmap_canary/result.json).

The full 62-camera 000973 pipeline completed locally with **62/62 finite
1080x1920 geometric maps**, coverage mean **.386363**, minimum **.251123**.
Matched full-block TSDF has **86,340 vertices / 167,030 triangles / one component**.
It is an extracted mesh, not a serialized raw TSDF volume. The importer hit a
metadata-only `shutil.copy2` timestamp error after copying all provenance bytes;
all imported arrays were checked for equality with the raw geometric maps and
all hashes rechecked before resuming the unchanged pipeline. No depth/source
bytes were changed. [Recovery receipt](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/reconstruct_000973/import_recovery_receipt.json).

Fresh train-only native/exact spatial color calibration and hard F/J/L renders
use the baseline settings. Because optional renderer helpers had changed since
the historical baseline, all three original-rig images were also rerendered
with the current runtime: **all three PNG hashes exactly match the historical
control**. This permits reuse of its metrics and comparison crops; merely
matching command-line settings would not have established equivalence.

| 000973 reconstruction | Face PSNR | Face SSIM | Face LPIPS | Native F/J/L gate |
|---|---:|---:|---:|---|
| Original rig, matched full-block/native-exact renderer | 28.724285 | .889098 | .047989 | Unresolved defects |
| Shared poses+focal, regularized | 16.915396 | .655352 | .417596 | **0 pass / 3 fail** |

**Later coordinate audit:** the historical BA control above did not propagate its
post-BA similarity transform to the untouched query cameras. The
[common-query-gauge control below](#completing-the-train-rig-gauge-on-query-cameras-2026-09-06)
isolates that inconsistency and recovers much of the large regression without
changing the mesh or fitting held RGB. The historical number must not be used as
unconfounded evidence that train-only rig refinement intrinsically fails.

Both rows use exactly the same held F GT hash and GT-only face polygon/protocol.
No full-frame or room metrics are added. The strong regression is not a changed
face ROI. Eight actual native crops were inspected: F face/hand/ear/overview,
J hand-neck/tube, and L neck/face-ear. The candidate retains the neck source seam,
distorted tube/hand junctions, and jagged or missing hair silhouettes. It also
displaces facial features substantially relative to the unchanged held cameras.
The L crop does not show the tube; its tube verdict is explicitly not applicable,
not a pass. Missing room alone is ignored.
[F hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/reconstruct_000973/review_F/hand.png),
[J tube comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/reconstruct_000973/review_J/lipstick.png),
[L face/ear comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/reconstruct_000973/review_L/face_ear.png),
[visual verdict](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/reconstruct_000973/visual_review.json).

The closing audit verifies 310 train EXRs across disjoint fit/held times,
retained depth/mesh/render hashes, explicit train/eval separation, unchanged
held camera rows, exact original-runtime replay, and matched face metric
definitions. The isolated staged-index suite passes **223 tests**, including
11 new track/BA/held-score checks.
[Evidence audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/calibration_control/shared_rig_findings.json),
[inspectable comparison notebook](assets/dec5_shared_rig_control_checks.ipynb).

#### Insights

The selected candidate is **rejected**, not promoted to the temporal recipe.
Even bounded, converged shared-rig optimization can reduce held train-pair
epipolar errors without preserving reconstruction in the coordinate system of
unchanged held cameras. This is an observed projection failure, not proof of
which physical calibration parameter is wrong. Before another intrinsics-based
reconstruction, an absolute train-camera-held-out projection/gauge test is needed;
adjusting F/J/L using their evaluation RGB would conceal the failure and is not
allowed. No every-40th-time or continuous fly-through repair has passed.

### Depth-separated source graphs and narrower full-block TSDF (2026-09-06)

#### What was tested

The image-grid source graph coupled neighboring pixels even across a real depth
discontinuity, and evaluated cross-source colors at both endpoints. This can
discourage putting a source boundary at a natural object boundary. The opt-in
`--seam-cut-depth-log-jump .0075` removes such graph edges without changing RGB,
visibility, geometry or source averaging. Zero remains the existing default.
The threshold reuses the earlier same-surface depth gate; it is not fitted to GT.
Paired F/J/L runs test rank penalty .001 and zero, both with and without the new
graph. Only the graph threshold differs within each pair.

A separate geometry control fixes the original 62 raw depths, calibration and
native/exact spatial camera response. Full-block TSDF first narrows truncation
from .004 to .0015 at voxel .00025, then halves voxel to .000125 while retaining
the narrow band. This isolates the combination missing from earlier controls:
the original narrow/fine test lacked full-block integration, and the later
full-block fine test retained truncation .004. No new PatchMatch run, semantic
mask, source-camera fit or evaluation-RGB prediction input is introduced.

#### Results

| 000973 control | Face PSNR | Face SSIM | Face LPIPS | Native gate |
|---|---:|---:|---:|---|
| Image-grid graph, rank .001 | 28.724285 | .889098 | .047989 | Fail |
| Depth-separated graph, rank .001 | 28.721514 | .889168 | .047968 | Fail |
| Image-grid graph, rank 0 | 28.821976 | .913522 | .062508 | Fail |
| Depth-separated graph, rank 0 | 28.806019 | .913621 | .062285 | Fail |
| Full-block voxel .00025 / trunc .004, frozen native color response | 28.748835 | .889802 | .047824 | Fail |
| Full-block voxel .00025 / trunc .0015, same response | 28.755987 | .891535 | .048329 | Fail |
| Full-block voxel .000125 / trunc .0015, same response | 28.766863 | .891860 | .048132 | Fail |

All rows use the same F GT hash and GT-only face polygon. These seven rows are
diagnostic variants of one temporal frame, not seven frames. The off graph
control reproduces all three historical baseline PNG hashes. On the paired graph
changes, every pixel retaining its source label also retains exactly the same
RGB. F/J/L lose 2,856 / 2,546 / 2,255 cross-depth graph edges. This changes
12,289 / 5,931 / 5,223 labels at rank .001, and 142,635 / 212,363 / 88,096 at
zero rank; it still does not yield a correct skin/tube boundary.

The narrow .00025 mesh has 311,429 vertices, 603,889 triangles and two connected
components. The .000125 mesh has **1,353,172 vertices, 2,649,029 triangles and six
components** after the unchanged relative component filter. These are extracted
meshes, not saved raw TSDF volumes. Original raw-depth coverage is unchanged.

Twenty-eight native comparison crops were actually viewed across the two
controls: face/ear/hand for F, hand-neck/tube for J, and neck/face-ear for L.
All four new candidate variants fail their three-anchor gate (0 pass / 12 fail).
At zero rank the neck patch weakens, but a wrong skin-colored rim around the
tube is conspicuous and F facial detail softens. Narrow/finer full-block fusion
alters contours and notches without repairing the neck patch or broad/sheared
J tube; hair/shoulder silhouette defects remain. Missing room alone is ignored.
[Graph F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/depth_separated_source_control/review_rank0_F/hand.png),
[graph J tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/depth_separated_source_control/review_rank001_J/lipstick.png),
[narrow/finer F comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/full_block_narrow_control/review_finer_narrow_F/hand.png),
[narrow/finer J tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/full_block_narrow_control/review_finer_narrow_J/lipstick.png).

The paired audits check request differences, RGB identity, all 62 original
1080x1920 depth arrays and hashes, mesh statistics, identical camera normalization,
source-camera response, GT/ROI protocol, finite face-only metrics and native
verdict inventories. [Graph audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/depth_separated_source_control/findings.json),
[TSDF audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/full_block_narrow_control/findings.json),
[executed comparison notebook](assets/dec5_surface_graph_tsdf_controls.ipynb).
The isolated staged-index suite passes **242 tests**. Nineteen added cases cover
exhaustive binary graph minima with depth cuts, scale invariance, unchanged
planar/default graphs, exact single-source RGB gathering and fail-closed CLI
depth/aggregation validation. These are implementation checks, not visual passes.

#### Insights

Neither a depth-separated label graph nor more triangles is a sufficient repair.
The code remains an opt-in diagnostic; no existing model, single-frame runner or
campaign default changes. Next, the specific incorrect tube/skin boundary must
be compared with original train RGB and stereo depths to distinguish upstream
depth error from source appearance before adding another fusion setting.
No continuous fly-through or every-40th-available-time recipe is accepted.

### Native tube-boundary trace and support-guarded shell control

**What was tested.** The original full-block mesh, fixed rig, native/exact
spatial camera correction and hard seam-cut8 rank .001 were replayed for held J.
The replay PNG is byte-identical to the matched previous control. Six diagnostic
pixels were traced; these coordinates never enter geometry or source selection.
Three have no target surface. The trace helper previously unprojected zero depth
to the camera centre; it now reports `no_target_surface` with no invented world
point or source correspondence. Tests cover zero, negative, non-finite depth and
image bounds. Historical trace artifacts remain intact.

`audit_patchmatch_trace_depth_support.py` independently checks all 62 original
train-depth maps using native 5x5 taps, not bilinear depth across layers. At least
20 of 25 taps must agree within .001 normalized units for strict near support.
Free space requires the existing compact farther-layer criterion: at least 80%
of taps beyond both .005 normalized units and 1% of projected depth. Zero maps
and missing taps never vote. No RGB or semantic masks enter this audit.

**Results.** These are three surface points at one time, not a temporal sample:

| J pixel | Original triangle | Strict near views | Farther-layer views | Interpretation |
|---|---:|---:|---:|---|
| (875,310) | 13702 | 60 | 0 | Supported tube surface |
| (860,310) | 13700 | 61 | 0 | Supported tube surface |
| (835,335) | 11602 | 0 | 19 | Unsupported tube-adjacent shell |

The last point has 19 nominally "near" cameras at the older .005 tolerance, but
their observed surfaces are roughly .0032–.0037 closer, not at the extracted
surface. The chosen K004_D005 source sees room at this projected point. This is
distinct from the earlier supported right-neck patch: one-camera RGB lookup can
still paint room onto a wrong triangle. Exact visibility against the same mesh
does not independently validate that triangle.

[Numbered GT/control diagnostic crop](/mnt/data/lookcloser_dec5_5a3_surface_repair/J_tube_boundary_trace/target_points.png),
[native train patches at the shell point](/mnt/data/lookcloser_dec5_5a3_surface_repair/J_tube_boundary_trace/trace_v2/point_04_train_patches.png),
[all-62 native evidence](/mnt/data/lookcloser_dec5_5a3_surface_repair/J_tube_boundary_trace/native_all62_v2.json).

The new opt-in `--near-gap .001 --maximum-near-views-to-carve 1` adds a support
guard to the existing free-space carver. Three contradicting cameras are still
required, but two agreeing native footprints preserve a triangle. Default
behavior is unchanged. The same whole-mesh rule is applied without pixel,
anatomical or held-image conditions. Fixed camera color correction is not refit.

It removes 4,730 triangles and 810 subsequent small-component triangles, retaining
77,946 vertices / 149,512 triangles / one component. Another 1,188 triangles have
at least three contradictions but are protected by two or more near observations.
The two tested tube depths remain exactly .770851 and .770777. The shell triangle
is removed: the ray at (835,335) now reaches a farther surface at .824254 instead
of .773073. Screen-space hole filling does not change any of these three values.

| Same 000973 study face polygon | PSNR | SSIM | LPIPS | Native F/J/L gate |
|---|---:|---:|---:|---|
| Matched original full-block control | 28.724285 | .889098 | .047989 | Fail |
| Support-guarded shell removal | 28.753834 | .889354 | .047553 | 0 pass / 3 fail |

Seven native F/J/L face, ear, hand/neck and lipstick comparisons were inspected.
The J brown tube-adjacent rim is reduced, but the tube remains broad/soft and
small slits remain. F retains its conspicuous neck source patch and has a small
black hand/tube slit. L retains irregular hair/shoulder boundaries. Missing room
alone is not counted as failure. The small face-metric improvement does not
override these failures.

[F hand/neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/review_F/hand.png),
[J lipstick comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/review_J/lipstick.png),
[L face/ear comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/review_L/face_ear.png),
[integrity audit and verdict](/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/findings.json),
[reproducible checks](assets/dec5_supported_shell_checks.ipynb).

**Insights.** A supported neck/source-color patch and an unsupported tube-adjacent
shell coexist; neither "all blur is source averaging" nor "all nearby brown is
skin" is justified. Native strict support is materially different from extraction
weight or a loose near-depth count. The support guard fixes the sampled false
surface while retaining sampled true tube points, but is not a sufficient skin
or fly-through repair. Do not strengthen carving blindly or promote this recipe
to every-40th-time validation. All 62 original maps remain unchanged and valid
1080x1920 (mean/min coverage .3827300892 / .2478824267). The audit also verifies
unchanged texture settings, GT/ROI protocol, finite renders and retained hashes.
Source data, original calibration, published campaign and model defaults remain
unchanged. No raw TSDF volume is serialized.
An isolated staged-index snapshot passes **261 tests** across 38 files, including
native missing/invalid-depth cases, independent support/free-space tolerances,
support-guard behavior and existing fusion/visibility/color/source-graph controls.
The companion notebook executes successfully and rechecks the retained hashes.

## Harmonic source-base transport controls (2026-09-06)

### What was tested

On the same frozen 000973 full-block mesh, replace secondary sources' smooth
color bases by a depth-connected harmonic continuation from a lower-rank source;
retain their own detail residuals and the exact original hard labels. This is
color transport, not pointwise averaging of cameras. Unanchored components remain
unchanged. Low-pass and transport systems use float64 with true residual gates.
The primary source remains unchanged. No target RGB or semantic masks enter it.

Seven matched variants test base scales 16/64, an optional source-color edge
threshold .04, a same-point exposed-chromaticity seed threshold .04, scalar
exposed-linear luminance transport, and RGB transport bounded to gains .5–2 and
chromaticity RMS change .025. The chromaticity-only variant failed the F visual
gate and was not run on J/L. All others use the identical rule on F/J/L.

### Results

| Same study face polygon | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Matched hard RGB8 control | 28.724285 | .889098 | .047989 |
| RGB base16 | 28.734015 | .889217 | .047707 |
| RGB base64 | 28.754614 | .889353 | .047549 |
| Color-edge base16 | 28.762484 | .889495 | .047406 |
| Color-edge base64 | 28.765400 | .889468 | .047402 |
| Chromaticity-seeded base64 | 28.763680 | .889453 | .047398 |
| Scalar luminance base64 | 28.781874 | .889424 | .047485 |
| Bounded RGB base64 | 28.765402 | .889437 | .047424 |

All **19 variant/anchor outputs at one time** fail the complete native gate;
45 face, ear, hand/neck and tube crops were inspected. Unrestricted transport
weakens a broad neck patch but creates a conspicuous gray tube-adjacent patch.
Color-edge/seed gates do not prevent it. Scalar/bounded transport controls that
strong recoloring, but leaves the neck patch, hand/tube seam, broad soft J tube,
and irregular hair/shoulder outlines. No new all-black output pixels were found.

[F bounded hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/base_transport_control/F/review_bounded_rgb/hand.png),
[J scalar tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/base_transport_control/J/review_luminance/lipstick.png),
[audit and all verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/base_transport_control/findings.json).

### Insights

Correctly bounded color changes do not establish correct material appearance or
surface correspondence. The face ROI excludes most neck/hand pixels: tiny face
metric gains cannot certify repair there. Historical code snapshots reproduce
the earlier controls; disabled-option replays are byte-identical. All actual RGB,
camera, depth and label inputs match by path and SHA-256 across each anchor's
variants. The 62 raw maps remain valid and unchanged. This family is not promoted.

## Explicit multi-camera RGB and low-band blending (2026-09-06)

### What was tested

Following explicit user authorization to take the skin patch from multiple
cameras, the standalone `run_visible_source_blend_control.py` consumes verified
frozen eight-source warps. It does not alter the original hard-render recipe.
The 62 train cameras, mesh, calibration, camera-response correction, exact
visibility and native pixel centers remain fixed. F/J/L held RGB is read only
after prediction, for evaluation and native comparison.

Both controls use the same normalized camera-distance prior and a 32-pixel
feather toward geometric visibility boundaries. Only train sources visible at
the target surface receive weight; there is no semantic skin/face segmentation.
One control averages complete RGB in sRGB-linearized display space (Reinhard
remains applied). The other averages source low-pass bases at smoothness 64 and
adds detail from the original hard-selected source. Geometry-only weights and
the same parameters apply to all three held camera anchors.

### Results

| Same study face polygon, F | PSNR | SSIM | LPIPS | Native F/J/L gate |
|---|---:|---:|---:|---|
| Matched hard RGB8 control | 28.724285 | .889098 | .047989 | Fail |
| Full RGB mixture | 29.767834 | .932165 | .115152 | 0 pass / 3 fail |
| Low-band mixture + hard detail | 29.149364 | .890657 | .046164 | 0 pass / 3 fail |

All 14 native face/ear/hand/neck/tube comparisons were inspected. Full RGB
mixing weakens the neck patch but visibly blurs skin detail, hair, the ear and
the hand; LPIPS worsens by about 2.40x despite better PSNR/SSIM. Low-band mixing
largely retains detail and slightly improves face LPIPS, but the conspicuous
neck patch and hand/tube seam persist. A dark neckline fringe is also visible.
The wrong broad tube/rim in J and irregular hair/shoulder contours in L are not
repaired by either appearance-only control. Missing room alone is not a fail.

[F full-RGB face](/mnt/data/lookcloser_dec5_5a3_surface_repair/visible_source_blend_control/F/review_full_rgb/face.png),
[F low-band hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/visible_source_blend_control/F/review_low_band/hand.png),
[J low-band tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/visible_source_blend_control/J/review_low_band/lipstick.png),
[L low-band face/ear](/mnt/data/lookcloser_dec5_5a3_surface_repair/visible_source_blend_control/L/review_low_band/face_ear.png),
[audit and six verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/visible_source_blend_control/findings.json),
[inspectable companion checks](assets/dec5_source_base_blend_checks.ipynb).

### Insights

Multi-camera mixing is not intrinsically forbidden now, but averaging all detail
does introduce the blur that the hard recipe avoided. This is direct paired
evidence, not a claim that the original patch was caused by averaging: original
RGB came from exactly one selected source per pixel. Mixing only smooth tone is
the more promising of these two controls, but it has not passed the skin/tube
gate. Do not promote either result or confuse three held anchors at one time
with temporal/fly-through validation. The study ROI is identical across these
controls, and differs from the old campaign ROI; old campaign LPIPS is not a
paired baseline. Output hashes, normalized visible-only weights, finite EXRs and
their PNGs are verified; full-RGB compositing is independently reconstructed from
the saved weights. Model and single-frame defaults remain unchanged.
An isolated staged-index snapshot passes **290 tests** across 40 files, including
visible-only weight normalization, invalid-source exclusion, feathered handoffs,
retained high-frequency detail, bounded response changes and default-off field
behavior. The companion notebook executes successfully and rechecks retained
hashes and paired metric definitions.

## Local seam and primary-occlusion mixtures (2026-09-06)

### What was tested

Two further matched controls address the global mixture's sharpness regression.
`run_seam_local_source_control.py` blends only source labels found within a
32-pixel four-connected path on the same target depth layer. It does not cross
log-depth jumps of .0075. An eight-pixel geometric visibility feather fades each
source before its own occlusion. Pixels outside actual mixtures keep their exact
hard-source RGB, rather than undergoing a global color transform.

The recorded F neck point (580,660) exposed a limitation of that rule: five train
sources are visible, but only the old rank-2 source was selected nearby. The local
rule therefore leaves its interior unchanged. A separate
`run_primary_fallback_source_control.py` mixes **all** visible fallback sources
where the primary is occluded, and smooths that transition over a 32-pixel
same-depth band. Outside the band, the original hard RGB remains exact.
Each control tests full linear-display RGB versus low-pass bases plus the
original hard detail. There are no semantic masks or eval-RGB predictor inputs.
The full-block mesh, camera calibration/response, visibility and frozen eight
warps remain identical to the paired baseline. Each rule is shared across F/J/L.

### Results

| Same 000973 study face polygon, F | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Matched hard RGB8 control | 28.724285 | .889098 | .047989 |
| Local seam, full RGB | 28.784077 | .890537 | .048324 |
| Local seam, low-band | 28.749506 | .889178 | .047669 |
| Primary fallback, full RGB | 28.817654 | .891388 | .050104 |
| Primary fallback, low-band | 28.786356 | .889264 | .047435 |

All **12 variant/anchor outputs at one temporal frame** fail the complete native
gate; 28 native crops were inspected. The global face blur is avoided, but the
neck patch, hand/tube seam, broad soft J tube and irregular hair/shoulder outlines
remain. The all-fallback mixture makes angular tube-adjacent patches in F more
conspicuous. Low-band variants retain fine detail but are not complete repairs.
Missing room alone is not counted as failure.

At F x=580, the source switch across y=648/649 has a smooth mesh-depth ratio
(absolute log jump .000322). Local mixing reduces its RGB8 boundary from
`[152,116,78] / [156,119,86]` to `[156,119,86] / [156,119,85]`. Yet the patch
interior at (580,660) remains `[168,125,87]`: local seam smoothing is not the same
as changing that source's texture across the patch.

The all-fallback control gives the five visible ranks 2/3/4/5/7 weights
.298/.249/.176/.168/.108 at that interior point. Their cached RGB8 channel ranges
are `[165..171,119..127,80..91]`. The mixed result is `[169,124,87]`, barely
different from the old source. A strictly post-hoc GT check is `[160,117,83]`;
its R/G values lie below all five sources, so no convex mixture of those cached
colors can reproduce them. This is evidence for this sampled pool/point, **not**
a claim about all 62 cameras or proof of a unique exposure/BRDF/geometry cause.

[Local F hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/seam_local_source_control/F/review_full_rgb/hand.png),
[all-fallback F hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/primary_fallback_source_control/F/review_full_rgb/hand.png),
[local scanline evidence](/mnt/data/lookcloser_dec5_5a3_surface_repair/seam_local_source_control/F/seam_profiles.json),
[post-hoc five-source point audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/primary_fallback_source_control/F/neck_point_audit.json),
[local integrity/verdict audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/seam_local_source_control/findings.json),
[fallback integrity/verdict audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/primary_fallback_source_control/findings.json),
[inspectable checks](assets/dec5_local_fallback_blend_checks.ipynb).

### Insights

The sampled patch is not explained by one radically different fallback color.
Changing mixture weights can reduce a boundary's color step while preserving a
visible interior texture/appearance mismatch. Do not continue broad weighting
sweeps or promote a tiny face-metric gain. The next investigation must separate
native texture/focus/noise differences from incorrect surface projection, using
train evidence and post-hoc held comparisons with explicit scope limits.

Both audits verify identical input hashes, normalized visible-only weights,
finite 1920x1080 PNG/EXR outputs and exact preserved RGB outside the respective
mixture domain. Exact float identity is checked against the runner's CUDA RGB8
normalization; NumPy CPU division can differ by one float32 ULP. PNG identity is
also checked directly. The study face ROI is unchanged, excludes most neck/hand,
and is not the old campaign ROI. No candidate has passed temporal or continuous
fly-through validation. Production model/renderer defaults remain unchanged.
The isolated staged-index snapshot passes **303 tests** across 42 files,
including depth-separated seam bands, exact inactive RGB, all-visible fallback
mixing, missing support and invalid input guards. The companion notebook executes
successfully and rechecks paired hashes, metrics, weights and native verdicts.

## Native train-image noise-floor control (2026-09-06)

### What was tested

The earlier pairwise bandwidth fit was confounded by high-frequency noise/texture.
The opt-in `run_native_noise_source_control.py` tests whether conservative native
train-image filtering reduces that appearance mismatch. It holds the mesh,
calibration, spatial camera response, exact visibility, UVs and eight-source hard
labels fixed. A fresh unfiltered reprojection reproduces all three frozen F/J/L
RGB8 controls byte-for-byte; all 24 source-warp replays have zero valid-pixel error.

`native_source_noise_filter.py` estimates an HH-MAD high-frequency floor in the
lowest-base-variation quartile of unclipped 32x32 native tiles. This statistic is
**not identified sensor noise or a lens PSF**: fine texture, JPEG coding and prior
processing can contribute. Native RGB8 channelwise non-local means uses
`h = clip(floor, .5, 3) * strength`, with strengths 1 and 2, template/search windows
7/21. It runs before the frozen camera-response correction, without additional
color-space conversion, random detail, semantic masks or another camera's RGB.
The same cached native image is used across target views. There are 23 unique
train images across the three eight-source pools; held F/J/L RGB is excluded.

### Results

| Same 000973 study face polygon, F | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Exact unfiltered native replay | 28.724285 | .889098 | .047989 |
| Native NLM, strength 1 | 28.726891 | .889567 | .049411 |
| Native NLM, strength 2 | 28.786858 | .898417 | .130522 |

All **6 variant/anchor outputs at one time fail**; all 14 native crops were
inspected. Mild filtering leaves the neck patch and hand/tube seam with slight
detail softening. Stronger filtering produces plasticky skin and loses face/ear
detail; LPIPS is 2.72x worse despite higher PSNR/SSIM. J's broad soft tube/rim and
L's ragged hair/shoulder contours remain. No new all-black pixels were introduced.

The native high-frequency floor is 1.114 RGB8-luma units for F's primary E004_C005
and .741 for fallback G004_B005. This documents differing fine-scale appearance,
but does not prove a sensor-noise cause or explain the entire patch.

[F mild hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/native_noise_control/F/review_nlm1/hand.png),
[F stronger face](/mnt/data/lookcloser_dec5_5a3_surface_repair/native_noise_control/F/review_nlm2/face.png),
[J mild tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/native_noise_control/J/review_nlm1/lipstick.png),
[audit and six verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/native_noise_control/findings.json).

### Insights

Native denoising is not an accepted patch repair. In particular, noise-floor
differences must not be reinterpreted as a validated deblurring kernel. The audit
checks all source/cache hashes, exact disabled replays and hard-source gathers,
unchanged selection labels, finite 1920x1080 PNG/EXR pairs and identical study
GT/ROI/protocol. The actual parser implementation is hashed as a runtime input;
unrelated worktree changes are not included in this experiment's code commit.
Most neck/hand pixels lie outside the face ROI, so that metric is not their gate.

## Subpixel-spacing full-block TSDF control (2026-09-06)

### What was tested

One further geometry control tests `voxel=.0000625`, `truncation=.0005` against
the previous finest `.000125/.0015` full-block control. At F's focal length and
the sampled neck depth .714, the new voxel projects to about .83 pixels rather
than 1.66. The new truncation projects to about 6.64 pixels; **truncation span is
not itself an estimate of actual geometric error**.

Both controls use the same 62 raw full-resolution geometric depths, normalized
scale .10075768902732685, full-block integration, extraction weight 2, depth
truncation 4, crop +/- .15 and component threshold `max(100,.002*largest)`.
The three-camera render path, calibration, spatial color response, eight-source
hard seam selection and native/exact reprojection remain fixed. This control
does not use the native denoising above or multi-source RGB averaging.

### Results

| Same 000973 study face polygon, F | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Prior finest `.000125/.0015` | 28.766863 | .891860 | .048132 |
| Subpixel `.0000625/.0005` | 28.761492 | .891466 | .047728 |

The extracted mesh has **5,450,524 vertices, 10,657,121 triangles and 7 connected
components**, from 40,314 integrated blocks. Component triangle counts are
10,410,840 / 87,186 / 39,872 / 39,026 / 31,499 / 24,555 / 24,143. Mesh SHA-256:
`ea237fd05ad9c32dc01ff341805a6ed14cc420b4be7c947c3be22cc1ce72cb40`.
Only the mesh and manifests are serialized, not the raw TSDF volume.

All **3 held camera anchors fail** after inspecting all 7 native crops. F's
neck patch and tube seam remain, with extra holes/notches through the hand and
skin silhouette. J retains a broad tube, wrong rim, small chin holes and irregular
hair boundaries. L retains ragged hair/shoulder contours. More triangles and
slightly lower face LPIPS do not constitute a successful surface gate.

[F hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/subpixel_tsdf_control/review_subpixel_F/hand.png),
[J tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/subpixel_tsdf_control/review_subpixel_J/lipstick.png),
[L face/ear](/mnt/data/lookcloser_dec5_5a3_surface_repair/subpixel_tsdf_control/review_subpixel_L/face_ear.png),
[audit and verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/subpixel_tsdf_control/findings.json),
[companion checks for both controls](assets/dec5_native_noise_subpixel_checks.ipynb).

### Insights

Further voxel-size sweeps lack supporting evidence. The audit independently
loads the mesh and recomputes connected components; all 62 original depth arrays
are finite 1080x1920, with unchanged mean/min coverage .382730089/.247882427.
Input hashes, normalization, paired GT/ROI, three finite PNG/EXR pairs, source
selection images and reprojection audits are checked. No held RGB or semantic
masks enter reconstruction/prediction. All workers are terminal and no OOM/CUDA
error was found. Failed workspaces are retained; production defaults and the
original 50-frame campaign remain unchanged. Neither experiment is promoted to
the every-40th-available-frame or continuous fly-through validation.
The isolated staged-index snapshot passes **310 tests** across 43 files, including
native noise-floor scaling, exact disabled/constant-image behavior, edge-preserving
filtering on a synthetic noisy step and invalid-strength rejection. The companion
notebook executes top-to-bottom, verifies 309 native-control and 111 subpixel
retained hashes, paired metric definitions and the native dimensions/input hashes
of all 21 inspected crops. These are comparison-integrity checks, not visual passes.

## Integer-depth lookup convention (2026-09-06)

### What was tested

The pinned COLMAP MVS code copies the camera K unchanged and unprojects depth
at integer `(column,row)` coordinates. Open3D 0.19 tensor integration projects
voxels with K, then truncates positive UV coordinates to select a depth sample.
This creates a half-pixel-centered lookup relative to those integer rays.
See the pinned [COLMAP MVS model](https://github.com/colmap/colmap/blob/5509fffe/src/colmap/mvs/model.cc),
[depth computation](https://github.com/colmap/colmap/blob/5509fffe/src/colmap/mvs/patch_match_cuda.cu)
and [Open3D integration kernel](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h).
The importer preserves the raw depth values and K; this is camera-z depth, not
Euclidean ray range. This audit does **not** establish that the pinned MVS
integer convention itself is physically correct for the original RGB centers.

`colmap_integer_depth_fusion.py` adapts only the integration lookup: K's principal
point gets +.5 so the kernel chooses the nearest integer sample. Block discovery
retains the original K. The isolated `run_colmap_integer_depth_fusion.py` entry
point restores the Open3D factory even on failure and writes an explicit adapter
request/manifest. Existing fusion, renderer and model defaults are unchanged.

### Results

An analytical tilted plane, integer-centered 64x64 depth, voxel .002 and truncation
.02 tests the actually installed Open3D CPU and CUDA implementations. Both give
identical interior residual statistics (synthetic world units, not measured mm):

| Depth lookup | Mean signed plane residual | Plane residual RMSE |
|---|---:|---:|
| Existing floor lookup | .001439298 | .001562367 |
| Nearest integer lookup | -.000033946 | .000600948 |

The real 000973 canary uses unchanged 62 raw depths, calibration and spatial RGB
response, the original full-block voxel/truncation .0005/.004, weight 2, crop and
component rule. The bounded block inventory is identical: 2,222 blocks with the
same coordinate hash. The new mesh has **80,193 vertices / 154,966 triangles /
one component**, SHA-256
`3cb841303aadd9a3dfef2fca00a1bd2fe4e7bc6ef3bde01f9127e27f588b47de`.
It is a serialized mesh, not a saved raw volume.

| Same study face ROI, F | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Existing floor control | 28.724285 | .889098 | .047989 |
| Nearest integer depth | 28.694189 | .887771 | .048337 |

All seven native crops were inspected: **0 pass / 3 fail** at one time. The F
neck patch and tube seam, J broad tube/wrong rim and adjacent false surface, and
L ragged hair/shoulder remain. The synthetic bias is real but its correction is
not a skin-seam repair. All 62 finite 1080x1920 raw arrays retain coverage
.382730089 mean / .247882427 minimum. Paired GT/ROI, finite render pairs,
source-selection/reprojection audits, actual mesh components and hashes pass.

[F hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/integer_depth_convention/real_000973/review_F/hand.png),
[J tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/integer_depth_convention/real_000973/review_J/lipstick.png),
[synthetic results](/mnt/data/lookcloser_dec5_5a3_surface_repair/integer_depth_convention/synthetic/findings.json),
[real-frame audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/integer_depth_convention/real_000973/findings.json).

### Insights

Keep depth-array indexing explicit and test it against the producer, not only
the downstream renderer. Do not infer a large physical geometry improvement from
a correct small synthetic test. This opt-in control is not promoted temporally.

## Completing the train-rig gauge on query cameras (2026-09-06)

### What was tested

The earlier regularized shared-rig BA already had absolute pose priors, then
applied a common post-BA similarity to the reconstructed scene and all train
cameras. Its three untouched held camera rows did not receive that similarity.
`complete_rig_similarity_gauge.py` applies the **recorded** scale/rotation/translation
to those query c2w poses, without changing their intrinsics, image identity, train
camera rows, source RGB/response or mesh. The scale is .998096972; the rotation is
about .249 degrees. No transform is fitted to held RGB. This is coordinate-gauge
completion, not a new physically verified held-camera calibration.

An exact current-runtime replay of the old BA renders precedes the new variant:
**all three old PNG hashes match**. A new calibration file is kept inside the
experiment; the original fixed template and published campaign are untouched.

### Results

| Same 000973 study face ROI, F | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original fixed rig | 28.724285 | .889098 | .047989 |
| BA with old, untransformed query poses | 16.915396 | .655352 | .417596 |
| Same BA mesh, common query gauge | 25.986305 | .816415 | .071790 |

This recovers **9.07 dB** and reduces LPIPS by .345806 relative to the old BA
evaluation. The large facial displacement mostly disappears. However it is
still worse than the original rig, and all seven four-panel native comparisons
fail the complete gate: **0 pass / 3 fail**. The F skin patch/tube seam, J wrong
broad tube/rim and L irregular hair/shoulder boundaries remain. There is no
accepted temporal or continuous fly-through result.

[F hand/neck](/mnt/data/lookcloser_dec5_5a3_surface_repair/complete_gauge_control/review_F/hand.png),
[F face](/mnt/data/lookcloser_dec5_5a3_surface_repair/complete_gauge_control/review_F/face.png),
[J tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/complete_gauge_control/review_J/lipstick.png),
[comparison audit](/mnt/data/lookcloser_dec5_5a3_surface_repair/complete_gauge_control/findings.json),
[inspectable checks for both coordinate controls](assets/dec5_coordinate_control_checks.ipynb).

### Insights

The old BA render failure was substantially confounded by an inconsistent query
coordinate gauge; it did not isolate the value of rig refinement. The same
recorded transform applies to arbitrary query cameras, not only one eval view.
Synthetic reprojection invariance and exact old-render replays test that claim.
They do not validate the remaining focal/pose changes or eliminate the surface
artifact. Future rig experiments must preserve a coherent query coordinate
system before interpreting held renders. The corrected bounded focal model is
still not accepted. Further geometry/calibration controls require a common
train-only rule and the full native gate, not selection by one face metric.
The isolated staged-index snapshot passes **322 tests** across 45 files. Added
tests cover actual CPU VBG plane bias, nearest lookup, unchanged input K,
camera-z depth, invalid inputs, factory restoration after a failed fusion,
similarity reprojection invariance, unchanged train rows and rejection of an
already-transformed query. The companion notebook executes successfully and
rechecks 195 depth-control / 144 gauge-control retained hashes, paired metric
definitions, exact old BA replays and all 14 native crop/input inventories.
All workers are terminal; no OOM/CUDA errors were found and no scratch was deleted.

## Shared pose-only rig with fixed intrinsics (2026-09-06)

### What was tested

The opt-in `run_pose_only_rig_control.py` isolates shared pose corrections from
the previous focal-plus-pose candidate. One regularized 62-camera rig fits
25,312 points / 150,382 observations across 000973, 001059 and 001139. All original
intrinsics remain exactly unchanged. The recorded post-BA similarity is also
applied to F/J/L query poses; held-camera RGB never enters fitting or prediction.

The request records thresholds before this fit/render: median held-pair
improvement at least .02 px, at least 60% improved pairs, improved overall p90
and median/p90 on each held time, solver convergence, maximum rotation .6 degrees
and camera-center shift .25 original world units. The same 716 pairs / 153,643
correspondences on 000979 and 001219 are retained across candidates. These times
are disjoint from fitting but have appeared in earlier diagnostics; they are
not an untouched final generalization test.

After eligibility, pinned CUDA COLMAP `5509fffe` recomputes all 62 full-resolution
depths for 000973. Matched full-block TSDF `.0005/.004`, weight 2, crop ±.15,
train-only spatial 16x9 response fitting, native/exact visibility and hard
seam-cut8 rendering remain unchanged. No RGB averaging or semantic masks are
introduced by this control. A fresh original-rig replay uses the same renderer.

### Results

| Fixed held-train pair inventory | Median error (px) | Pair-block p90 (px) | Improved pairs |
|---|---:|---:|---:|
| Original rig | .545406 | 1.009393 | — |
| Pose only, regularized | .475516 | .811580 | 68.72% |

BA converges in 78 iterations. Maximum rotation is .306276 degrees and maximum
camera-center change .040595 original world units. Each held time improves:
000979 median `.555579 → .485016`, 001219 `.537349 → .462576`.
The sparse eligibility gate passes; the actual surface gate does not.

| Same study face polygon, F | PSNR | SSIM | LPIPS | Native F/J/L gate |
|---|---:|---:|---:|---|
| Original rig, byte-identical replay | 28.724285 | .889098 | .047989 | Fail |
| Earlier focal+pose, common query gauge | 25.986305 | .816415 | .071790 | 0 pass / 3 fail |
| Pose only, common query gauge | 27.008524 | .822349 | .059505 | **0 pass / 3 fail** |

All seven native crops and the context overview were actually inspected. The F
neck patch and jagged hand/tube seam remain; J retains an incorrect broad tube
side/rim and adjacent background-colored slab. L retains irregular hair and
neck/shoulder silhouettes with attached background-colored fringes. Missing
room alone is ignored. The face ROI excludes most neck/hand pixels and cannot
override those visible failures.

All 62 geometric maps are finite 1080x1920 arrays, exactly equal to their
imported depths. Coverage mean/min is `.385430932 / .249664834`; the extracted
mesh has 81,452 vertices, 157,427 triangles and one component. It is not a saved
raw TSDF volume. Import initially failed only when the filesystem rejected
`shutil.copy2`'s provenance timestamp operation. All 62 arrays, intrinsics and
provenance bytes were verified before resuming the unchanged pipeline request;
the failure logs and recovery receipt are retained. No PatchMatch rerun or
source-byte change was needed.

[F hand/neck comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/pose_only_control/reconstruct_000973/review_F/hand.png),
[J lipstick comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/pose_only_control/reconstruct_000973/review_J/lipstick.png),
[L face/ear comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/pose_only_control/reconstruct_000973/review_L/face_ear.png),
[audit and retained hashes](/mnt/data/lookcloser_dec5_5a3_surface_repair/pose_only_control/reconstruct_000973/findings.json),
[companion checks](assets/dec5_pose_only_rig_checks.ipynb).

### Insights

The completed query gauge removes a confound, but neither regularized BA variant
repairs this surface. Better sparse epipolar agreement and slightly higher depth
coverage are insufficient for acceptance. Retain the original calibration;
do not expand either candidate to the every-40th-frame validation. This result
does not prove that calibration is exact or identify a unique cause of the
remaining appearance/geometry defects. Multi-camera mixtures above also remain
unpromoted: smoothing source transitions did not correct the patch interior.

An isolated staged-index snapshot passes **326 tests across 46 files**. The final
artifact audit verifies 365 retained hashes, identical original intrinsics,
three byte-identical original-rig replays, paired face-metric definitions and
the native verdict inventory. The companion notebook executes top-to-bottom and
independently recomputes the held-pair summaries. All reconstruction workers are terminal, with no
active CUDA/OOM error. Scratch and source data were not deleted.

## Exposure correction before stereo: paired input proxy (2026-09-06)

### What was tested

Previous response corrections changed texture inputs after depth reconstruction.
`audit_stereo_input_exposure.py` instead tests whether correcting the images
fed to stereo has a strong photometric signal. The original 000973 mesh and
calibration remain fixed. All 62 train EXRs are encoded as JPEG98 4:4:4 using
original per-image gains, one geometric-mean train gain (cancel ingest), or
original gain times the existing train-only scalar correction. Every original
JPEG replay is byte-identical. F/J/L RGB and semantic masks are excluded.

The proxy uses the identical 165,404 fully visible 11x11 mesh-warped patch pairs
over 248 **directed** camera pairs (four nearest neighbors per train camera).
Eligibility is fixed by geometry, not candidate colors; low-variance patches
remain in the inventory with NCC -1. This is ordinary grayscale NCC, **not** an
exact reproduction of the [pinned COLMAP bilateral photometric cost](https://github.com/colmap/colmap/blob/5509fffe/src/colmap/mvs/patch_match_cuda.cu).
Before measurement, the request records a minimum .005 median paired NCC gain
and 60% improved camera-pair medians for dense-canary eligibility.

### Results

| Response before stereo | Pair-block median NCC | Median paired NCC change | Improved directed pairs |
|---|---:|---:|---:|
| Original | .683486 | — | — |
| Cancel known ingest gain | .685846 | +.000369 | 54.03% |
| Train-fitted scalar correction | .686020 | +.000637 | 57.26% |

The median paired change is not the difference between the two aggregate
medians. Neither variant reaches the recorded gate. Across individual patches,
the share with proxy NCC >=.1 changes only `93.7801% → 93.7970% / 93.8097%`;
these are photometric proxy counts, **not** newly reconstructed valid depths.

An original-reference-contrast stratification avoids concealing a large
low-texture effect under clothing/hair observations. In the lowest contrast
bin (gray standard deviation <.01; 7,452 patch pairs), median per-patch changes
are +.001376 / +.002784. In the .01–.03 bin (89,923 pairs), they are
+.000194 / +.000247. These descriptive strata use fixed original-image contrast,
not semantic skin masks or candidate-dependent exclusions.

The independent audit verifies 260 input/output hashes and recomputes 24 evenly
spaced observations directly from the source images and mesh rays: maximum NCC
discrepancy is zero. Three native train-image comparisons were actually viewed;
the corrections change brightness but retain the captured shading and contours.
They are input reviews, not reconstructed-view visual passes.
[E004_C005 input comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/stereo_input_exposure_control/input_review_00018/hand_neck.png),
[E004_B005 comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/stereo_input_exposure_control/input_review_00043/hand_neck.png),
[G004_B005 comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/stereo_input_exposure_control/input_review_00027/hand_neck.png),
[integrity audit and contrast strata](/mnt/data/lookcloser_dec5_5a3_surface_repair/stereo_input_exposure_control/integrity_audit.json),
[inspectable companion](assets/dec5_stereo_input_exposure_checks.ipynb).

### Insights

Known exposure mismatch changes source brightness but is not a strong NCC driver
on this measured interior pool. No new dense job is justified by this proxy.
This does **not** rule out effects at silhouettes or occlusion boundaries: those
are excluded by the common full-patch visibility rule. The current mesh can also
bias the sampled correspondence geometry. The scalar fit uses train data, and
this one-time paired diagnostic is not an untouched final generalization test.
No new render, face-quality measurement, temporal or fly-through pass is claimed.
The skin seam remains unresolved; the original campaign and defaults are unchanged.

The isolated staged-index snapshot passes **333 tests across 47 files**. The
companion notebook executes top-to-bottom, rechecks retained hashes and
recomputes the camera-pair summaries. No scratch was deleted and all workers
are terminal. A next appearance control must be distinguished from the already
rejected per-camera/first-order angular gain fields and eight-source mixtures;
simply repeating them is not a justified next experiment.

## Common mesh RGB base from 62 cameras, hard source detail (2026-09-06)

### What was tested

The user permits multi-camera color aggregation. This opt-in control fits a
mesh-attached low-frequency display-RGB base to the visible mean of all 62
train cameras, and a separate base to each source's observations. The output is
`source RGB - source base + common base`. Thus source detail remains hard-selected,
while the common base is independent of the target camera. This differs from the
previous log-gain fields and eight-source target-space mixtures, but shares their
fundamental assumption that a smooth correction can reconcile camera appearance.

The 000973 full-block mesh, original rig, spatial camera response, exact visibility,
native half-pixel projection, hole-fill rule and hard labels are unchanged. Both
bases use the same mesh Laplacian (smoothness 64, ridge .0001); all visible train
vertices participate. F/J/L RGB is used only afterward for review, never fitting
or prediction. No semantic masks or anatomy-specific rules are used. The
production runners and model defaults are unchanged.

### Results

The fit converges in 1,760 iterations (true normalized residual `8.65e-8`;
required `<5e-7`, absolute RHS normalization floor `1e-6`). Of 80,221 vertices,
77,921 have at least one observed source and 77,066 have at least two.
All three off controls replay their original PNGs **byte-identically**, and all
three categorical source maps are unchanged.

| F held-out face, same study ROI | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original hard source | 28.724285 | .889098 | .047989 |
| Common RGB base + hard detail | 29.360821 | .888843 | .048490 |

The PSNR increase is not surface acceptance. The face polygon excludes neck,
hand, tube and ear; it is also not numerically comparable to the original
50-frame campaign ROI. All seven native region crops and one contextual overview
were actually inspected: **0 visual pass / 3 fail** across F/J/L.
F retains sharp face detail but gains a darker, more conspicuous polygonal neck
strip. J retains the wrong broad jagged tube rim. L retains jagged shoulder/hair
fringe. Missing room alone is ignored.

Independent barycentric checks localize the new darkening to the added base
correction, not a changed mesh or stale render. At F pixel `(580,648)`, the
source/common bases differ by `[-14.78,-10.25,-7.56]` RGB8 levels: prediction
`[152,116,78] -> [137,106,71]`, while post-hoc GT is `[151,115,83]`. At the
previous patch-interior probe `(580,660)`, prediction barely changes
`[168,125,87] -> [169,125,88]`, versus GT `[160,117,83]`. These are illustrative
previously identified points, not new image-region quality metrics or fit targets.

[F neck/hand comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/canonical_surface_base_control/review_F/hand.png),
[J tube comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/canonical_surface_base_control/review_J/lipstick.png),
[L face/ear comparison](/mnt/data/lookcloser_dec5_5a3_surface_repair/canonical_surface_base_control/review_L/face_ear.png),
[findings and point checks](/mnt/data/lookcloser_dec5_5a3_surface_repair/canonical_surface_base_control/findings.json),
[native verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/canonical_surface_base_control/visual_review.json),
[reproducible validation companion](assets/dec5_canonical_surface_base_checks.ipynb).

### Insights

This particular multi-camera base is rejected. A common smooth base plus
independently smoothed source residuals does not guarantee continuous color at
source transitions, and can introduce a new broad bias. The measured darkening
is caused by the computed color offset; why the underlying camera observations
disagree remains a separate question, not proof of exposure alone. Keeping mesh
and labels fixed also means this control cannot repair an incorrect silhouette.
No temporal or continuous fly-through pass is claimed, and no new dense job is
justified by this failed appearance gate. Source data and original campaign
outputs are preserved.

The isolated staged-index snapshot passes **363 tests across 49 files**. The
audit verifies 157 input/output hashes, paired ROI definitions, independent
float64 face PSNR, exact PNG replay and barycentric color-offset spot checks.
The companion notebook executes top-to-bottom without errors. All fit/render
workers are terminal; no scratch was removed.

## Direction-conditioned RGB base and source-view identity (2026-09-06)

### What was tested

The previous common base ignored viewing direction. `view_conditioned_surface_base.py`
fits the **source low-frequency display RGB itself**, using constant, linear or
quadratic unit-direction features with a shared mesh Laplacian. This is not the
earlier first-order log-gain-difference fit. Its inputs are the independently
smoothed source bases from the preceding control, unchanged fixed geometry and
62 train cameras. The constant/shared base is not a fitting target.

Before rendering, every fifth physical-camera name in sorted order is withheld:
49 cameras fit the directional model and 13 evaluate it on the identical visible
vertex inventory with at least two fitting-camera observations. The pre-recorded
gate requires >=10% reduction in median camera mean absolute low-band RGB error
and >=60% improved cameras. Quadratic is chosen over linear only for another
>=5% error reduction. Earlier camera-response calibration remains fixed and used
all train cameras; this is a holdout of the **directional fit**, not of the entire
photometric pipeline. F/J/L RGB never enters fitting or parameter selection.

The selected model is refitted to all 62 train cameras. Two render controls keep
the mesh, visibility, labels and original float RGB fixed:
`RGB_source - independently_smoothed_source_base + fitted_query_base`, then
`RGB_source + fitted_query_base - fitted_source_base`. The latter has exact
source-view identity: when query and source camera coincide, its correction is
zero. Both are opt-in diagnostics; no production defaults change.

### Results

| Direction model | Median held-camera mean absolute base RGB error | Improved cameras |
|---|---:|---:|
| Constant | .0206383 | — |
| Linear | .0119866 | 13/13 |
| Quadratic | .0102058 | 13/13 |

Quadratic reduces this low-band proxy by **50.55%** and passes the recorded
eligibility gate. All solves converge; the final all-train model uses 416
iterations with true normalized residual `6.40e-8` (<`5e-7`). Independent NumPy
recomputation from retained source bases and held-model coefficients differs by
at most `2.38e-9` in per-camera mean error. This is not a reconstructed-image
quality improvement.

| Same F study face ROI | PSNR | SSIM | LPIPS | Native F/J/L verdict |
|---|---:|---:|---:|---|
| Original | 28.724285 | .889098 | .047989 | Fail |
| Direct directional base | 28.110579 | .887674 | .048963 | 0 pass / 3 fail |
| Source-anchored directional transfer | 28.229683 | .888448 | .048384 | 0 pass / 3 fail |

All seven native detail comparisons plus overview were actually viewed, with
both candidates alongside GT and the original. Direct replacement retains sharp
face detail but leaves a more conspicuous dark neck strip. Source anchoring
removes much of that additional darkening, yet leaves the original neck/hand
patch and tube mismatch. J retains a wrong broad jagged tube rim; L retains the
attached hair/shoulder fringe. Missing room alone is ignored. Face ROI excludes
neck, hand, tube and ear, and is not comparable to the old campaign ROI.

[F paired neck/hand](/mnt/data/lookcloser_dec5_5a3_surface_repair/view_conditioned_surface_base_control/paired_review_F/hand.png),
[J paired tube](/mnt/data/lookcloser_dec5_5a3_surface_repair/view_conditioned_surface_base_control/paired_review_J/lipstick.png),
[L paired face/ear](/mnt/data/lookcloser_dec5_5a3_surface_repair/view_conditioned_surface_base_control/paired_review_L/face_ear.png),
[six visual verdicts](/mnt/data/lookcloser_dec5_5a3_surface_repair/view_conditioned_surface_base_control/visual_review.json),
[findings](/mnt/data/lookcloser_dec5_5a3_surface_repair/view_conditioned_surface_base_control/findings.json),
[validation companion](assets/dec5_directional_surface_base_checks.ipynb).

### Insights

Better camera-held low-frequency prediction does not suffice to remove the
visible seam. Source-view identity avoids baking model residual error into a
captured source view, but does not correct correspondence or silhouette errors.
Both variants are rejected. These results do not justify further increasing
angular polynomial degree merely to lower this proxy; the next repair evidence
must address the remaining surface/correspondence defect itself. Temporal frames
`000899, 000979, 001059, 001139, 001219` and continuous fly-through remain required
for any future accepted repair; this one-time failed control does not satisfy
either requirement. Original source data and campaign outputs remain unchanged.

The isolated staged-index snapshot passes **388 tests across 52 files**. The
artifact audit verifies 230 hashes, all six byte-identical off replays, unchanged
source maps, independently recomputed held-camera scores and exact face PSNR.
The validation notebook executes top-to-bottom. All workers are terminal, no
CUDA/OOM failure is active and no scratch was removed.

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
