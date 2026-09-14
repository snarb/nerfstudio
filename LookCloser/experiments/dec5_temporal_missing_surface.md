# DEC5 temporal missing-surface stage attribution

## What was tested

The preceding goal turn made progress: a complete, audited 150-time dynamic
camera workaround was published. It did **not** achieve artifact-free video or
general mesh recovery. This experiment addresses remaining actual geometry loss,
not another camera-only workaround.

`diagnose_temporal_missing_surface.py` casts identical calibrated rays into the
original pre-carving TSDF and the final boundary-only movie mesh at 001033 and
001041. It compares geometry misses with saved hard-source RGB eligibility.
Red overlays identify original hits lost after carving; blue identifies final
geometry with no eligible RGB. Original misses remain unlabelled because they
include true background, not automatically missing anatomy.

`probe_forearm_train_depth.py` saves the three closest real train views and their
original-mesh depth. No eval RGB, new camera calibration, or original data edits.
`analyze_forearm_depth_stages.py` defines fixed manual train-RGB-only palm/forearm
polygons and will compare photometric depth, filtered geometric depth, original
mesh and independently reprojected geometric-depth support.

A matched 001033 reconstruction is supervised on clever-shadow, using the
validated CUDA COLMAP 3.13.0.dev0 commit5509fffe bundle. It reproduces the original
12-source-per-reference, 62-train recipe and retains all depth maps. The existing
`run_temporal_full_block_control.py` then compares per-view TSDF block activation
and full bounded block-union integration on **the same observed depths**. Geometry
ingest reproduces historical JPEG exposure; video RGB profiles remain fixed.

## Results

Status: stage attribution, controlled depth reconstruction, two matched RGB/clay
renders, and real-train inspection complete. No new mesh is promoted.

| Frame | Original TSDF triangles | Later lost hits, lower context | Final hits without RGB, lower context |
|---|---:|---:|---:|
| 001033 | 152042 | 3960 | 304 |
| 001041 | 141616 | 6345 | 297 |

The context rectangle is portrait `[0,1150,1080,1920]`, **not** an anatomical mask
or quality ROI. Counts include unrelated shirt edges. Native clay/RGB comparison
shows the principal forearm hole and later palm fragmentation already in the
original TSDF. The main holes are therefore not caused by the later train-mask
carving or hard-source RGB rejection. This does not yet isolate the earlier
PatchMatch/filtering/TSDF extraction stage.

- [001033 matched stage comparison](/mnt/data/dec5_temporal_missing_surface_diagnosis/001033/comparison.png)
- [001041 matched stage comparison](/mnt/data/dec5_temporal_missing_surface_diagnosis/001041/comparison.png)
- [001033 real train views](/mnt/data/dec5_forearm_train_probe/001033/train_references.png)
- [001041 real train views](/mnt/data/dec5_forearm_train_probe/001041/train_references.png)

Real same-time references show the descending hand near/beyond image boundaries
and visible hand motion blur. Both can reduce useful matching evidence; they are
candidate explanations, not yet a measured causal attribution. The original mesh
bounds lie strictly inside the configured ±0.15 normalized crop, so that outer
crop plane does not explain these holes.

### Matched TSDF control: negative

The pinned local run completed photometric/geometric passes in315.3/455.1seconds;
all62 scalar1080x1920 geometric maps are finite and nonempty. Per-view and full-block
fusion took4.3/5.5seconds. Both same-camera renders were actually inspected with
matched clay. The principal forearm hole persists in both; no repair promoted.

| New same-depth fusion | Vertices | Triangles | Forearm visual decision |
|---|---:|---:|---|
| Per-view blocks | 78937 | 152035 | Large hole persists |
| Full bounded block union | 79120 | 152387 | Large hole persists |

The original historical mesh had152042triangles: the new GPU run is a recipe
reproduction, not a claim of byte-identical historical geometry. All original
source EXR hashes match the historical inventory. The150-time campaign did not
retain historical JPEG hashes for this frame, so exact JPEG equivalence is not
claimed. The two new fusion arms share identical newly computed depths, making
their block-update comparison controlled. Their raw meshes are not silhouette-
carved; both RGB controls use identical existing train foreground eligibility.

- [Matched RGB controls](/mnt/data/dec5_forearm_depth_control_001033/rgb_comparison.png)
- [Matched clay controls](/mnt/data/dec5_forearm_depth_control_001033/clay_comparison.png)

### Where the depth disappears

The fixed G004_B005 train forearm/palm polygons do **not** capture the actual
missing lower patch. They have zero original-mesh misses, despite being near it.
Geometric valid fractions are96.76% and95.48%, with median26 and21 other measured
depth agreements. They serve as preservation controls, not proof the whole arm
is reconstructed. This prevented a falsely reassuring ROI-only conclusion.

Three actual target-ray probes near portrait(365,1810/1850/1890) extrapolate a
plane from adjacent visible skin, then test101 depth hypotheses over±0.02
normalized depth. This is a diagnostic bracket, **not ground truth or added mesh**.
The two missing rays fall inside21–34 and15–26 real train image frusta across
their brackets, respectively, but have at most one agreeing observed depth map.
Thus the stronger hypothesis that no camera can see the area is rejected;
frustum membership alone does not establish usable depth.

Real RGB crops at the middle probe show skin, not background or cuff, in six
nearby cameras whose images contain the projected query. Geometric valid fractions
in11x11windows are0%,0%,8.3%,0%,1.7%,0%; photometric values exist everywhere but
are inconsistent. The observed local loss is already present at geometric depth
filtering, before TSDF. Weak texture and blur are plausible upstream causes;
this experiment does not separate them or prove the plane's depth is exact.

- [Train polygon/depth comparison](/mnt/data/dec5_forearm_depth_control_001033/depth_stage_analysis/comparison.png)
- [Actual missing-ray locations](/mnt/data/dec5_forearm_depth_control_001033/missing_ray_coverage/queries.png)
- [Six real views of the missing region](/mnt/data/dec5_forearm_depth_control_001033/missing_ray_coverage/inside_camera_references.png)

The learned-prior agent now has a separately bounded one-time forearm follow-up:
conservative plane vs aligned learned prior, trusted measured boundary anchors,
real-train region restrictions, old geometry preservation, and other-view checks.
It must not call inferred geometry a measured observation or bridge into the cuff.
This follow-up is not yet an accepted reconstruction change.

Focused tests:7passed (stage attribution and existing camera-path regression).
No source, published movie, existing mesh, or production default was changed.

The independent learned-prior study completed at two times with no candidate
promoted: [confidence-gated depth report](dec5_confidence_depth_prior.md). The
12/24/36-source comparison remains separately supervised on dev3:
[source-count report](dec5_patchmatch_source_count_ablation.md).

## Insights

Large camera count is not the same as local usable observations. Before filling
a black region, distinguish original missing geometry, later deletion, lack of
RGB eligibility and true background. Use matched camera rays and real images;
coverage counts alone cannot establish anatomical correctness. No full-frame
PSNR/SSIM/LPIPS is computed for these unmatched novel views.

Artifacts: `/mnt/data/dec5_temporal_missing_surface_diagnosis`,
`/mnt/data/dec5_forearm_train_probe`, `/mnt/data/dec5_forearm_depth_control_001033`.
