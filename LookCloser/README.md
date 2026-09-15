# LookCloser

This directory extends Nerfstudio with the LookCloser model and its measured preprocessing and
rendering experiments. The paper implementation remains available as `lookcloser`; optional
geometry/rendering tools do not change that model unless explicitly selected.

## Selected geometry backend for the DEC5 rig

The selected no-training actor geometry path is **fixed-pose COLMAP PatchMatch -> continuous
TSDF**. Feed-forward DA3, MapAnything, MonoMVSNet and MVSMamba did not meet the visual or
actor-only metric bar. Splatfacto depth remains a diagnostic reference, not a per-frame
requirement.

Use `scripts/run_colmap_patchmatch_tsdf.py` on one display-referred JPG frame dataset. The input
must contain calibrated `frames`, explicit filename train/eval splits and at least two train
cameras. The runner:

1. exports the supplied calibration without feature matching or bundle adjustment;
2. runs full-resolution photometric and then geometric PatchMatch on train cameras only;
3. fuses the geometric depths into one strict CUDA TSDF;
4. builds a geometry-only 16-camera angular texture subset;
5. raycasts the mesh and renders by hard nearest-source selection, using later cameras only for
   visibility holes.

No image/person mask is accepted. RGB sources are never averaged, and eval RGB is unavailable to
geometry and prediction construction. By default the command uses the measured full-resolution
recipe (`1920`, three iterations per pass, geometric gates `6/2`, TSDF voxel/truncation
`0.0005/0.004`, tensor extraction weight `2`).
After cropping, it also removes disconnected islands smaller than the larger of 100 triangles or
`0.2%` of the dominant component. This scale-aware rule removed the residual ear/hair fragment in
the validation frame without using image coordinates and remains proportional when later frames
produce a denser or sparser mesh.

```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python LookCloser/scripts/run_colmap_patchmatch_tsdf.py \
  --data /path/to/one_frame_jpeg_dataset \
  --output-dir /path/to/output \
  --colmap-bin /usr/local/bin/colmap
```

The validated binary is CUDA COLMAP `3.13.0.dev0`, commit `5509fffe`. The runner fails closed on
another build because COLMAP 4.1.1 and a packaged 3.13 binary produced sparse, zero, or duplicated
depth maps with identical inputs. `--allow-unverified-colmap-build` is only for a build that has
passed a depth-map canary.

The final image is
`OUTPUT/render/nearest_fill16/eval_pred_0000.png`; the mesh is
`OUTPUT/colmap_patchmatch_tsdf.ply`. `pipeline_request.json` prevents `--resume` from silently
mixing artifacts made with different inputs, binaries or parameters, while
`pipeline_manifest.json` records the completed recipe and hashes.

Metrics are optional and must use an independent actor surface plus a region JSON; room/full-frame
metrics are intentionally not part of the DEC5 decision:

```bash
python LookCloser/scripts/run_colmap_patchmatch_tsdf.py \
  --data /path/to/one_frame_jpeg_dataset \
  --output-dir /path/to/output \
  --colmap-bin /usr/local/bin/colmap \
  --score-metrics \
  --metric-surface-depth-manifest /path/to/fixed_surface/mesh_depth_manifest.json \
  --roi-boxes-json /path/to/actor_roi.json
```

See `experiments/dec5_000899_offtheshelf_geometry.md` for the measured comparison and ear-artifact
ablation.

## Fifty-frame PatchMatch-TSDF campaign

`scripts/run_colmap_patchmatch_tsdf_campaign.py` is the opt-in, resumable controller for the
first 50 numeric DEC5 5A-3 frames. It stages one temporary JPEG dataset at a time, transfers the
fixed calibration by unique `physical_camera`, runs the pinned single-frame recipe on `dev3`,
verifies 62 full-resolution geometric maps and retained hashes, and publishes a frame only after
manual-GT-only face metrics and a visual verdict exist. It never creates a permanent 50-by-65
JPEG copy and never sends a face ROI to the geometry or rendering host.

Initialize and verify the pinned remote environment before reconstruction:

```bash
python LookCloser/scripts/run_colmap_patchmatch_tsdf_campaign.py init --preflight
python LookCloser/scripts/run_colmap_patchmatch_tsdf_campaign.py reconstruct \
  --frames 000899 000901 000903
```

Face polygons live in the output root under `config/face_polygons/FRAME.json`. Each file is
bound to the held-out display GT hash and must declare that it was drawn manually on GT without
using the prediction. `score`, `review`, and `finalize` are explicit states; thus reconstructed
or scored scratch cannot appear as a completed CSV row. The independent final checker is
`scripts/audit_colmap_patchmatch_tsdf_campaign.py`. The durable 3D output is the extracted TSDF
mesh plus its manifests—not a serialized raw Open3D TSDF volume.

## Opt-in shared temporal color/texture calibration

The DEC5 helper `scripts/joint_temporal_texture.py calibrate` combines patch preparation
and joint fitting in one command. Several head poses fit one fixed RGB profile per
physical train camera and one fixed display exposure. A separate time tests transfer;
its local texture registration may adapt, but not the shared camera profiles.

From the repository root, preview the operation without creating files or loading images:

```bash
python LookCloser/scripts/joint_temporal_texture.py calibrate \
  --output /mnt/data/lookcloser_dec5_5a3_joint_texture_v2 \
  --fit-frames 000899 000973 001139 001197 \
  --held-frames 001059 --dry-run
```

Remove `--dry-run` to run. Use a new output root for different code/configuration;
the helper refuses to overwrite a mismatched hash-pinned calibration. Calibration
alone does not bake or visually approve a mesh. Use the existing `bake_joint_temporal_mesh.py`
entry point afterward. New times can use `prepare --frames ...` followed by `adapt`
without refitting camera color or exposure. This helper currently targets the fixed
62-train-camera DEC5 dataset and existing meshes, not arbitrary rigs.

Unlike the original hard-source renderer above, this experimental GLB path bakes
robust mixtures of registered train RGB into a static UV texture. Both paths render
the actual mesh; neither uses target RGB for prediction. The GLB no longer needs the
train images at viewing time. The calibration has modest perceptual benefits but
does not repair geometry and is not promoted to production defaults.

Measured results and native comparisons: [joint temporal texture report](experiments/dec5_joint_temporal_texture.md).

### Sharp, single-source mesh texture (opt-in)

`bake_joint_temporal_mesh.py bake --hard-source --calibration-root FITTED_ROOT
--output NEW_ROOT` reuses the frozen calibration and geometry but selects one
camera on connected regions of the mesh adjacency graph. Native detail is never
averaged across cameras. Train-only low-frequency color agreement guides source
labels; those low-pass pixels are never baked. A selected source is replaced only
where it fails per-texel visibility. The output GLB embeds the texture and needs
no train images at viewing time. The graph helper additionally requires
`PyMaxflow==1.3.2` (the experiment installed it without changing other packages).

Use a separate output root: configuration/source hashes prevent incompatible
resume, and `hard_texture_complete.json` is written only after GLB round-trip
validation. Run native review and the independent audit before visual acceptance.
This is **texture only**, not temporal mesh completion; geometry repair is deferred.
Commands, known limitations and comparisons:
[hard-source texture report](experiments/dec5_hard_surface_texture.md).

### Local geometry repair and smooth mesh flythrough (opt-in pilot)

`diffusion_mesh_repair.py` records three masked synthetic-view proposals, a matched
real-only stereo control, independent depth-support carving, selected chin-hole
filling and an explicitly disclosed cylindrical object prior for frame 000973.
`bake_local_mesh_repair.py` textures only added faces and preserves the old atlas;
`finalize_local_mesh_repair.py` publishes an embedded GLB and checks hashes.
`smooth_mesh_flythrough.py` renders a slow, arc-length-parameterized central rig
loop with fixed intrinsics and nonempty-frame checks, without image morphing.
These scripts change no model or existing runner defaults. Generated views are
not real observations; this pilot improves two defects but remains visually
imperfect. Masks/loop IDs are frame-specific, not a general moving-scene recipe.
See [diffusion-assisted repair report](experiments/dec5_diffusion_mesh_repair.md)
for replay commands, negative controls, final assets and known limitations.

### Slow 150-time-frame mesh video (opt-in)

`render_smooth_temporal_mesh_video.py` streams the existing 150 temporal meshes
with frozen exposure/camera profiles, one-source RGB and a small smooth central
camera path transferred through each mesh's normalization. No new training is
required. `run_smooth_temporal_workers.py` supervises disjoint render workers;
`temporal_texture_view_prior.py` optionally favors close-angle texture sources.
`review_encode_smooth_temporal_video.py` builds native review panels while workers
run and encodes only a complete chronological inventory. Explicit visual reviews
and `audit_smooth_temporal_video.py` distinguish verified files from artifact-free
quality. See [150-time video report](experiments/dec5_smooth_temporal_mesh_video.md).

For an isolated defective time, `diagnose_temporal_mesh_shelf.py` records explicit
local deletion proposals, and `run_temporal_full_block_control.py` supervises a
matched real-depth TSDF block-activation control. `review_temporal_full_block_control.py`
keeps real-view evidence and native render comparisons separate from acceptance.
`compose_verified_temporal_mesh_video.py` can reuse the other 149 unchanged frames
with checked input equivalence and explicit original-receipt ancestry; it does
not claim a rerender or skip the replacement's visual gate. See the
[000971 shelf investigation](experiments/dec5_temporal_shelf_repair.md).

For a slower, wider **open** arc inside the central train-camera space, use
`central_space_temporal_flythrough.py init`, then the existing supervised temporal
workers. It reuses corrected meshes but rerenders all 150 actual time instants;
the audit checks actual convex containment and excludes a nonexistent loop join.
`publish_smooth_temporal_video.py --report PATH` can attach the matching experiment
report. See [central-space flight and fringe controls](experiments/dec5_central_space_flight.md).

That first open arc was visually too small: smoothness alone did not validate
multi-row travel. `diagnose_camera_grid_flight.py` provides frozen-actor 3x3/4x4
cubic-spline controls with explicit two-axis extent and loop-speed gates;
`audit_camera_grid_flight.py` checks PNG/MP4 inventories and decoded samples.
See [camera-grid diagnosis](experiments/dec5_camera_grid_diagnosis.md).

Those static controls are not the dynamic deliverable. The opt-in
`dynamic_grid_flythrough.py` uses 150 distinct source meshes/RGB instants and an
open cubic 3x3/4x4 grid traversal with a minimum achieved two-axis extent gate.
`train_foreground_guard.py` tests train-only silhouette carving and background
RGB rejection without changing the original mask-free runner. The dynamic
worker supervisor supports up to eight render workers on the 96GB host;
`finalize_dynamic_grid_video.py` audits time diversity and actual camera travel
and encodes at the request FPS. See
[dynamic grid and foreground-guard experiment](experiments/dec5_dynamic_grid_background_guard.md)
for measured limitations, including unresolved internal lipstick occlusion.

For the wider **horizontal -4..+4, full-height vertical, closed-return** dynamic
route, use `wide_dynamic_camera_flight.py` and `run_wide_dynamic_workers.py`.
The rig has only five vertical rows; the helper explicitly records that limit.
An identical-mesh camera probe plus independent saved-depth raycasts validate
actual rendering, not just pose metadata. `prepare_wide_dynamic_geometry.py`
rechecks view-dependent contour deletions for the new poses;
`finalize_wide_dynamic_flight.py` checks all 150 changing times and encodes them.
See [wide dynamic camera flight](experiments/dec5_wide_dynamic_camera_flight.md).

To reuse the **exact earlier static 4×4 loop with a moving actor**, use
`replay_dynamic_camera_flight.py`, then the same geometry/worker/finalizer helpers.
It reads the saved pilot poses and resamples the full loop over 150 source times,
preserving both translation and orientation. The full-route duration difference
is explicit; see [dynamic replay](experiments/dec5_replayed_4x4_dynamic.md).

For visible **object movement inside the image**, use the opt-in
`screen_travel_camera_flight.py`: D..K camera translation with non-centered
composition and a fixed wider virtual lens. The finalizer checks actual screen
displacement and rotation-only image output; `diagnose_screen_travel.py` isolates
camera motion on identical geometry. See [screen-travel diagnosis](experiments/dec5_screen_travel_camera_flight.md).

`expanded_head_camera_flight.py --output NEW_ROOT` expands that route to C..L /
A..E using five real anchors (C/A and rows above A do not exist). It combines
bounded head-boundary and local 3D-notch completion with the unchanged hard-source
renderer. The finalizer emits only the normal-speed movie for this opt-in mode.
This expanded depth-notch variant is retained as an experimental control, not
the selected general-view reconstruction: its patches can fail from other angles.
See [head-defect controls and limitations](experiments/dec5_expanded_head_camera_flight.md).

For the subsequent user-authorized shot workaround, use
`artifact_aware_camera_flight.py --output NEW_ROOT`. It uses a smoothly restricted
camera envelope and the fixed boundary-only mesh stage; view-conditioned depth
patches are excluded because they can become stretched membranes from other views.
`elevated_camera_workaround.py --output NEW_ROOT` further raises the lower arc
to hide exposed under-chin boundaries. It trades vertical range for visibility
and resamples a periodic cubic spline by 3D distance to keep camera speed even.

## Experimental prior-guided local completion

For the fixed DEC5 `001193` diagnostic, `run_mhr_dense_sampling_control.py`
tests train-only silhouette fitting at triangle-interior samples, not only
vertices. `--order 4` / `--order 8` select explicit quadrature controls;
`--dry-run` validates their adapters without launching a fit. These are prior-only
outputs: existing geometry guards, local patch extraction, measured-depth
admission and native RGB review are still required. This is not yet a general
frame/campaign completion command. See [density tests and limits](experiments/dec5_mhr_dense_surface_sampling.md).

`recover_original_surface_texture.py` separately tests a narrow texture fallback
after inferred completion: it can reuse a verified baseline train reprojection
only at the same original triangle and surface intersection. It does not fill
new geometry, average RGB, use GT or change production defaults. See
[visibility-backoff protocol](experiments/dec5_inferred_visibility_backoff.md).

`run_mhr_radius_seed_control.py --candidate-root AUDITED_CANDIDATE --output NEW_ROOT`
tests all radius/normal-eligible verified depth seeds instead of the nearest24
cap, with unchanged hull/fit/free-space gates. It changes evidence selection,
not the prior or existing runner defaults. Use `audit_mhr_radius_seed_control.py`
and `render_mhr_radius_seed_control.py` afterward; this remains a fixed-frame
experiment, not an approved temporal repair. See
[measured-support neighborhood control](experiments/dec5_mhr_radius_seed_support.md).
