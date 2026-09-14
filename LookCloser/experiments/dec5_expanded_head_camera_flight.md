# Expanded DEC5 dynamic camera flight and head-defect diagnosis

## What was tested

The user requested one more horizontal column on each side and one more vertical
row above/below the previous D..K / A..D loop, without a slow movie, and asked to
address dark areas under the subject's right cheek and on the crown.

The actual rig has only vertical rows A..E. The previous shot already reached A;
there is no real row above it. C/A also does not exist. The new C..L / A..E path
uses smooth mean-value coordinates over five real train anchors C/B, D/A, L/A,
L/E, C/E. It never fabricates a corner camera or extrapolates outside their
convex hull. The prior loop phase and non-centered screen composition are kept,
with the same fixed 0.70 virtual focal scale and no crop/stabilization/zoom.
150 real source times 000899–001197 remain chronological; camera and actor move.
Only 24 fps / 6.25 s is encoded. The actor clip is not a seamless temporal loop.

Diagnosis used original and filtered mesh clay raycasts, saved depth/source IDs,
native matched-camera RGB controls, and the closest **real train** images.
Held-out RGB and generated RGB were not used to construct predictions.

Local opt-in changes, with existing defaults unchanged:

- Zero-displacement splitting of non-manifold vertex joins exposes otherwise
  rejected boundary cycles without moving original triangles.
- Small head loops receive isolated ear triangulation; curved loops may receive
  a bounded membrane prior. Existing surface detail remains fixed. This is not
  additional measured PatchMatch evidence, nor a globally watertight guarantee.
- Existing semantic carving is retained, including on the head. Restoring all
  removed head triangles was tested and rejected because expanded views exposed
  additional brown background-bearing fringe geometry.
- Small head notches are proposed by depth-mask closing, inverse-depth boundary
  fitting, and at least two train silhouette supports. Proposals are lifted to
  actual 3D triangles, appended to the mesh, and textured by the existing hard
  single-source renderer. No RGB inpainting or source averaging is performed.
  These patches depend on the requested camera; they are not a universal
  temporal reconstruction method or independent depth observations.

## Results

Status: expanded, first camera-avoidance, and final elevated controls all rendered
150 frames. Final elevated video has completed encoded visual validation. No
variant is an artifact-free reconstruction.

The initial interpretation that every black ray miss was a geometry defect was
too broad. The real train images show a dark cast shadow under the subject's
right cheek, and some neighboring viewpoints see room background between chin
and shoulder. A ray miss there can be the real silhouette, not a hole through
skin. This is distinct from the spurious crown opening and ragged mesh borders.

- [001083 real-train evidence](/mnt/data/dec5_expanded_head_repair_diagnosis/001083/train_evidence.png)
- [001123 real-train evidence](/mnt/data/dec5_expanded_head_repair_diagnosis/001123/train_evidence.png)
- [Original vs filtered geometry](/mnt/data/dec5_expanded_head_repair_diagnosis/001123/comparison.png)
- [Matched crown before/after](/mnt/data/dec5_head_repair_matched_v3/comparison/001123_crown.png)
- [Matched jaw before/after](/mnt/data/dec5_head_repair_matched_v3/comparison/001083_jaw.png)

Controls retained separately:

| Hypothesis/control | Observation / decision |
|---|---|
| MeshLab selected-face hole filling | Also touches neighboring holes; rejected strict audit |
| Isolated planar-loop filling | Preserves other boundaries but skips important curved/open contours |
| Curved-loop membranes | Adds bounded geometry, but cannot close a notch connected to the outer boundary |
| Local depth notches, RMSE <= 0.004 | Closes the main crown opening; keeps the real under-chin gap |
| More permissive RMSE <= 0.009 | Covers part of the gap with an incorrect flat skin strip; rejected |
| Restore all original head triangles | Additional fringe/spikes at the expanded side view; rejected |
| Preserve filtered surface and complete only bounded holes/notches | Initially selected at 000899/001123; subsequent camera control exposes a stretched notch membrane, so final video excludes notch additions |

The rejected permissive control lives in
`/mnt/data/dec5_head_repair_matched_v4`; its unfinished expanded preparation
`/mnt/data/dec5_expanded_head_dynamic_150` was stopped before movie rendering.
The intermediate expanded root ending `_v2` contains only four canary renders,
not a finished movie. The completed wider control root is
`/mnt/data/dec5_expanded_head_dynamic_150_v3`; expanded-view controls are
in `/mnt/data/dec5_expanded_head_preserve_canary`.
The RMSE numbers are normalized scene-coordinate depth-fit tolerances, not
physical millimetres or photometric quality metrics.

The matched native crop review confirms a substantial reduction of the black
crown opening, but some jagged/brownish hair-border facets remain. The cheek
shadow is not erased. Ragged neckline/torso and occasional texture seams remain;
do not call this artifact-free or claim all black image background is missing skin.

Focused camera/temporal/audit regression tests: **50 passed**, including the
elevated spline's physical-speed regression test.

### User-authorized camera-path workaround

The expanded control completed all 150 unique times in 721.9 seconds with eight
workers. Remaining upper-side crown gaps and jaw-border seams prompted a
separate shot-level workaround, as requested by the user. It is not evidence
that the underlying mesh was recovered correctly.

`artifact_aware_camera_flight.py` smoothly maps the same saved loop phase into
the rig-coordinate envelope x=[-0.8,3.82], y=[-1.92,0.95], relative to H/C.
Its actual sampled span is 4.619995 horizontal and 2.869454 vertical camera
intervals, with 32.109964 degrees maximum viewing-direction difference. Adjacent
camera step-length ratios stay below 1.028707; no snapping or per-frame camera
exceptions are used. Fixed lens and non-centered framing are unchanged.

Three controls are retained:

| Camera control | Result |
|---|---|
| Less restricted envelope x>=-2.5, with refitted local depth notches | Smaller black gap, but remaining incorrect chin border; not selected |
| More frontal envelope x>=-0.8, identical expanded-control meshes | Reveals a long spurious chin-to-shoulder membrane; rejected |
| Same more frontal path, fixed boundary-only meshes, no depth-notch additions | Selected: three native canaries remove the long membrane and avoid the large chin opening; narrow hair gaps and texture seams remain |

The second control exposes a concrete **new repair failure**: the apparent skin
strip is a view-conditioned completion triangle. It is not original COLMAP
geometry. Primitive-ID provenance isolates it in red in
[001083 patch provenance](/mnt/data/dec5_camera_avoidance_pilot_v2/patch_provenance/001083.png).
Consequently, the proposed final workaround discards all depth-notch additions,
uniformly across times, and uses the already-computed fixed boundary-only stage.
No source, exposure, or PatchMatch changes are mixed into this shot experiment.
The actual native canary verdict is saved at
`/mnt/data/dec5_camera_avoidance_pilot_v3/canary_review.json`. The first workaround
output `/mnt/data/dec5_camera_avoidance_dynamic_150` completed 150 frames in
691.6 seconds. Full inspection still found small black under-chin breaks at
000975 and 001191, where the low arc exposes the underside. It is therefore a
retained intermediate control, not a completely successful workaround.
The wider completed control remains at
`/mnt/data/dec5_expanded_head_dynamic_150_v3/video.mp4` for comparison, not as an
artifact-free alternative.

### Elevated final shot

The final selected controller is `elevated_camera_workaround.py`, output
`/mnt/data/dec5_elevated_camera_dynamic_150`. It changes only camera poses relative
to the boundary-only stage. Raising the camera envelope to y=[0.2,0.95] keeps the
underside of the chin from opening toward the camera. Horizontal range remains
x=[-0.8,3.82]. This intentionally sacrifices vertical range for the user's
visibility workaround; do not describe it as the original tall 4-row orbit.

A naive vertically compressed ellipse has a 6.31:1 speed range. A periodic cubic
spline, densely evaluated and resampled by physical 3D camera distance, reduces
that to 1.0124:1; maximum adjacent speed ratio is 1.00786. The resulting maximum
view-direction difference is 31.5128 degrees. This is real pose motion, not
animated cropping, translation, zoom, or stabilization.

Native canaries 000975/001083/001191 no longer show the conspicuous under-chin
opening or the rejected stretched membrane. Thin dark boundaries, hair-crown
notches/brown fringe, strong skin/neck color seams, and incomplete lower-body
geometry remain. Saved verdict:
`/mnt/data/dec5_elevated_camera_arc_canary/canary_review.json`.
The final 150-frame render completed in **691.9 seconds** with eight workers.
All 150 overview images and seven native frames (000899, 000973, 000975, 001049,
001083, 001123, 001191) were inspected. All 15 decoded MP4 contact sheets covering
all 150 frames were also inspected; no additional gross encoding corruption was
seen at overview resolution. This is not exhaustive native-resolution review.

- [Final normal-speed MP4](/mnt/data/dec5_elevated_camera_dynamic_150/video.mp4)
- [Four distributed camera/actor phases](/mnt/data/dec5_elevated_camera_dynamic_150/dynamic_extremes.png)
- [Final review and audit root](/mnt/data/dec5_elevated_camera_dynamic_150)
- [Representative decoded middle sequence](/mnt/data/dec5_elevated_camera_dynamic_150/decoded/sheet_070.png)

| Final audit | Measured result |
|---|---:|
| Unique source times / meshes / RGB frames | 150 / 150 / 150 |
| Sampled horizontal / vertical rig span | 4.61827 / 0.74994 intervals |
| Maximum viewing-direction difference | 31.51280 degrees |
| Maximum / minimum physical camera step | 1.012379 |
| Fixed-landmark screen travel, x / y | 384.00 / 153.57 pixels |
| Actual rendered foreground centroid travel, x / y | 285.49 / 154.67 pixels |
| Independent fresh depth raycasts | 5 / 5 match saved camera depth |
| Geometry preservation/locality audit | 150 / 150 pass |

All RGB output is verified as a rotation of the native renderer image, with no
post-render crop, translation or stabilization. Foreground-centroid travel mixes
actor and camera movement; the fixed-landmark and fresh-raycast checks establish
the camera contribution independently. Novel views have no matched GT, so no
full-frame PSNR/SSIM/LPIPS is reported.

Native final-frame checks also retain an incorrect blue/brown patch behind the
lipstick at 000975. The overview reveals serious pre-existing lower-forearm
openings at 001029–001037, followed by hand fragmentation at 001039–001043, near
the open lower reconstruction boundary. These are
not fixed by the head-view workaround. Consequently this is a limited head-shot
visibility improvement, not an artifact-free full-body reconstruction.

### What is and is not established about the upstream cause

Visibility in two cameras is necessary but not sufficient for usable stereo.
Weak texture, view-dependent highlights, mixed-depth windows near boundaries,
and the actual 12-source selection can prevent reliable correspondence. The
geometric consistency filter, TSDF extraction weight threshold 2, component
filter and later silhouette filtering are separate possible loss stages.
TSDF weight is not a direct count of cameras in which skin is visible.
[COLMAP documents weak-texture limitations and patch/resolution tradeoffs](https://colmap.github.io/faq.html#improving-dense-reconstruction-results-for-weakly-textured-surfaces).

For 001083 the retained TSDF metadata confirms 62 train views, tensor extraction
weight 2, voxel 0.0005, truncation 0.004, and crop [-0.15,0.15]^3. Its mesh bounds
are strictly inside that crop, so the outer crop plane does not explain these
head holes. The non-manifold-edge cleanup left the triangle count unchanged
(147722 before and after), so that specific cleanup is not a demonstrated cause
for this frame. Small-component removal remains a separate stage.

We have not traced these exact missing patches through retained unfiltered and
filtered PatchMatch depth maps and voxel weights. Consequently, blaming their
original absence specifically on insufficient PatchMatch matches would be an
unproven hypothesis. The current repair treats the output geometry; it is not a
demonstration that the upstream reconstruction failure has been eliminated.

The user also asked about increasing the 12-source limit. Inspection of
`build_colmap_patch_match_config.py::explicit_source_views` confirms that these
are the nearest 12 camera centers for **each** of 62 reference images, not only
12 cameras in the whole reconstruction. Selection is not per-patch occlusion-
aware. A 12/24/36-source, otherwise identical two-frame control is justified,
but has **not** completed in this video task. At the user's request, a separate
clean-context agent launched the six serial dev3 controls; another is testing
confidence-gated learned depth priors separately for skin and hair. Those studies
have independent reports and do not gate this shot workaround. More sources may provide a missing
useful baseline, but do not guarantee better geometry and do not fix later
filtering losses. Do not describe 12 as experimentally optimal for these holes.

## Insights

### All-time native jaw follow-up

The selected elevated movie was additionally inspected at native crop resolution
for **all 150 times**, using 25 six-frame sheets produced by
`review_temporal_camera_workaround.py`. The 500x300 crops track only the saved
fixed-landmark projection; no source image, video, camera or geometry is changed.
The separate [review bundle](/mnt/data/dec5_elevated_camera_jaw_review_150)
binds original image checksums, crop coordinates, sheet hashes and explicit LLM
notes. `finalize_temporal_camera_review.py` verifies complete ordered coverage.

No broad under-chin opening or stretched membrane was seen in these crops.
However, this is **not** a zero-hole verdict: tiny contact notches occur while
the hand overlaps the chin, and small black holes recur at **001193/001195**.
Fresh original/final raycasts and source-ID attribution show **44/45** and
**73/74** black spot pixels respectively already miss the original mesh.
One boundary pixel was later removed at 001193; one lacks RGB at 001195.
Major skin-color seams remain, especially late in the clip.
The native jaw crop does not include the crown or lower forearm. This follow-up
does not upgrade the existing imperfect-video publication to artifact-free,
and no full movie was rerendered for these small defects. The review/inventory
and existing camera-path regression tests pass: **12 passed**.

- A black pixel with zero mesh depth is not by itself proof of missing anatomy.
  Compare real train silhouettes before filling an apparent opening.
- Keep geometry-vs-texture diagnosis separate from cosmetic improvements. A
  skin-colored patch can hide a black area while inventing the wrong surface.
- Boundary-loop repair alone misses silhouette notches connected to the large
  open reconstruction boundary. Local 3D completion can help the selected shot,
  but requires explicit prior/provenance and native inspection.
- Use the actual non-rectangular rig hull; requested row names are not proof
  those cameras exist.
