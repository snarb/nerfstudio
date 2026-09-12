# DEC5 000973: diffusion-assisted local surface repair

## What was tested

User-authorized hypothesis: edit a few calibrated mesh renders to remove the
lipstick's false rear slab and close only the chin slit, then use the generated
views as explicit synthetic priors for local mesh repair. Source EXRs, the fixed
rig and the existing published meshes remain immutable. This is a new experiment,
not a continuation of the deferred multi-time body/neck reconstruction.

Output: `/mnt/data/lookcloser_dec5_5a3_diffusion_mesh_repair_000973`.
Input: the hard-source 000973 atlas with the original full-block TSDF geometry.
Three synthetic poses interpolate train cameras I004_D005 and K004_D005 at
0.25/0.5/0.75. Their camera parameters and the exact native-crop rotation/scale
are recorded. Only original **train** RGB supplies shape/material references;
F/J/L held-out RGB is unavailable to repair construction.

The built-in imagegen editor receives one edit target, a binary edit-area guide
and a four-camera train reference sheet. Its API does not expose a hard mask
parameter, so the guide is not a pixel-preservation guarantee. Returned views
must be independently aligned and clipped to the recorded repair masks. Raw
generated outputs are retained separately and never mislabeled as captured views.
Prompts require the same cylinder position/diameter/length, unchanged hand/face,
natural shadow under the chin, and no relighting or global beautification.

Exposure remains 10.570320292648677 for every input, with the previously frozen
train-camera RGB profiles and bounded registration fields. This experiment does
not refit camera profiles, optimize poses, or access held-out RGB during repair.
Manual edit masks are explicitly authorized for this experiment; they do not
change the mask-free defaults of the original PatchMatch campaign.

## Results

Completed **single-time-frame pilot**, 000973, on clever-shadow, 2026-09-12.
The rear slab is substantially reduced, and the selected chin opening is closed.
This is **not an artifact-free mesh**: small tube/finger contact defects, a chin
texture seam, old face/neck source boundaries and hair-rim artifacts remain.
No 150-time-frame reconstruction was launched or claimed by this pilot.

### Hypothesis → test → decision

| Hypothesis | Measured/visual result | Decision |
|---|---|---|
| Three locally edited views can directly repair stereo | Matched 8-real + 3-synthetic control improves local coverage, but leaves a large chin gap and rough surface | Reject as replacement mesh |
| The lipstick slab has insufficient independent depth support | 647 triangles in the reviewed multi-view tube ROI have fewer than 2 near real-depth observations and at least 3 free-space contradictions | Remove these triangles locally |
| Only the chin slit should be closed | Reviewed loops have 228 + 3 edges; selected MeshLab fill adds 304 vertices / 808 triangles, closes 231 edges, leaves other hole boundaries unchanged | Accept local geometry patch; retain visible texture seam as defect |
| A round tube prior can constrain an unobserved rear side | Fit to 141 independently supported visible shaft points; median / p95 radial residual 0.0000280 / 0.0001228 normalized units | Use disclosed cylinder prior, not a measured backside |
| First cylinder ends too early at the finger | Train I/D, K/D, H/C show an exposed lower cap/contact gap | Extend occluded lower endpoint by 0.0015, upper by 0.0003; replace 31 old cap faces; retain first candidate |
| Residual brown stripe is partly texture, not just mesh | Train K/D ray audit hits new cylinder faces in the brown stripe: mesh self-visibility admitted skin/background samples | Apply a disclosed neutral-metal source prior only to new tube texels |
| A wider object-volume cut might remove the last slab remnants | The first 114-face cut damages the upper finger; 89 proposed faces must be protected by the real-depth guard | Reject broad cut; final additional removal is only 25 independently contradicted, unsupported faces |
| Repair carving can detach tiny islands | Final connected-component cleanup removes 5 tiny-island triangles | Final surface has 2 components: 154,907 actor triangles and 384 tube triangles |
| A small analytic loop avoids anchor jumps | Fixed intrinsics, convex rig weights, arc-length speed, continuous look-at rotation, no duplicate endpoint | Deliver 360-frame / 12-second static-mesh flythrough |

The tube ROI is an intersection of projected edit regions, not an exact semantic
object segmentation: its 1,391 original triangles include occluded surfaces.
768 have fewer than two strict near observations; only the 647 also contradicted
by real free-space evidence are carved in the conservative candidate.
The reused native 62-view audit is pinned to original mesh SHA-256
`ab4436f42a2c047c1208a1912844743f8316f5b0b2acab897b624f89b32d23a9`.
Its near tolerance is 0.001 normalized units, native radius 2, required tap
fraction 0.8, minimum free-space gap max(0.005, 0.01 × depth).

### Diffusion ablation, including the failed direct route

All three built-in imagegen results were generated, visually inspected, aligned
and clipped to the masks. Median SIFT similarity-alignment error was
0.640 / 0.634 / 0.580 pixels in the 2× edit crop. The first raw generated image
changed pixels outside the mask by mean 8.50/255: raw whole-image output is unsafe
as an invariant-preserving input. **Final native masked images differ by exactly
zero outside their masks**. Alignment does not certify subpixel 3D consistency.

Selected project assets and exact prompts:
[view 00](assets/dec5_diffusion_mesh_repair/synthetic_prior_00.png),
[view 01](assets/dec5_diffusion_mesh_repair/synthetic_prior_01.png),
[view 02](assets/dec5_diffusion_mesh_repair/synthetic_prior_02.png);
[prompt 00](assets/dec5_diffusion_mesh_repair/prompt_00.txt),
[prompt 01](assets/dec5_diffusion_mesh_repair/prompt_01.txt),
[prompt 02](assets/dec5_diffusion_mesh_repair/prompt_02.txt).
Raw outputs, masks, transforms, native composites and receipts remain in
`OUTPUT/views/00..02/`; generation used the **built-in imagegen tool**, not CLI.

Localized fixed-pose COLMAP uses the verified 3.13.0.dev0 / 5509fffe CUDA bundle,
8 cropped real train images, optionally 3 cropped synthetic views, 512-pixel
native crops, three photometric and three geometric iterations, NCC 0.1,
geometric gates 6/2, two consistent views and a 1° triangulation angle.
Depth range is the original 4.5..20 converted by dataparser scale 0.1007576890.
The experimental local TSDF uses voxel 0.00025, truncation 0.0015, extraction
weight 2 and full discovered block-union integration. These are isolated local
experiment settings, not replacements for the frozen full-frame recipe.

| Same local camera subset | Vertices | Triangles | Middle query chin edit-region ray-hit fraction |
|---|---:|---:|---:|
| Original 62-view full mesh, reference only | 80,221 | 155,052 | 0.9370 |
| 8 real train views | 35,999 | 65,948 | 0.0984 |
| Same 8 real + 3 synthetic views | 43,403 | 80,324 | 0.2161 |

Ray-hit fraction is a **geometry diagnostic, not an image metric or correctness
score**. Synthetic views help the matched local control but do not close the
gap. Both local reconstructions are rougher than the original 62-view mesh;
the real-only control prevents incorrectly blaming that entire degradation on
diffusion. See `matched_mvs_control.json` and
[matched normals](assets/dec5_diffusion_mesh_repair/matched_mvs_control.png).
Positive stereo depth within synthetic chin masks is only 34.2–46.7%.

An initial narrow normalized depth range 0.55..1.1 was rejected because the J/B
camera is farther away. It also caused block-discovery failure when no depths
were below the fusion cap; this was **not a CUDA defect**. The rejected workspace
is retained as `mvs_narrow_range_rejected`; all published comparisons use the
corrected common depth range. No source or previous experiment was deleted.

The cylinder has radius 0.00108581 and a fixed fitted axis. Generated outlines
are a weak shape prior, not real support; their bright-metal masks understate
the full width by roughly 6–7 pixels in the enlarged edit crop. Endpoint edits
are explicit train-reviewed sculpting priors. The unseen rear side, idealized
flat end cap and occluded lower extent are **not physically certified**.

### Texture preservation and held-out face metrics

The entire original 4935×4937 atlas remains byte-identical in the top of an
extended atlas. Existing vertices and retained triangles are unchanged; only
new chin/tube faces receive new UVs. One real train camera supplies each accepted
texel, with geometry-based fallback; no cross-camera RGB average and no diffusion
RGB is baked. Unsupported new texels use explicitly labeled nearest observed
texel copying **within the same new material part**, not invented camera support.
The bake receipt distinguishes appearance completion from real observations.

The last train-reviewed iteration exposed the same visibility circularity in the
new patch: K/D crop samples at x=85..125, y=140 hit new cylinder faces, but some
sampled RGB was brown skin/background. Its opt-in `--neutral-cylinder` gate keeps
only display samples with max channel >0.25 and channel range <0.23 × max channel.
This is a **material-color prior**, not a semantic segmentation, physical visibility
test, new gain, or recoloring. It can discard legitimate colored metal reflections.
All RGB still comes from one train camera per sampled texel; inferred texels copy
a nearby accepted metal texel. Final bake: 805,474 new patch texels, 122,238
appearance-completed (15.2%). A real camera sample is not claimed as independently
verified geometric support. An attempted 114-face object-volume cut was visibly
unsafe at the upper finger. Final code hard-protects any old triangle with two
near real observations and requires three contrary free-space observations before
additional removal. This preserves 89 proposed faces and removes 25. Its dedicated
regression test prevents an object mask from overriding supported finger geometry.
The broad-cut candidate is retained and **rejected**, under `object_completion/`,
`final_neutral/`, `flythrough_neutral/`. Earlier conservative/cylinder publications
remain under `final/` and `flythrough_final/`. The selected deliverables are
**`final_supported/` and `flythrough_supported/`**; do not confuse these versions.

Only held-out F004_B005_1210O9 is scored, with the same fixed manual face ROI and
display-domain protocol as the hard-source pilot. No full-frame/actor/room metric
or loss is reported. The face ROI does not establish lipstick correctness.

| Variant | Face PSNR ↑ | Face SSIM ↑ | Face LPIPS ↓ |
|---|---:|---:|---:|
| Original hard-source mesh | 26.3360 | 0.876896 | 0.113736 |
| Conservative carve + chin fill | 26.3694 | 0.877281 | 0.113338 |
| Final round-tube + neutral-material prior + chin fill | 26.3598 | 0.877142 | 0.113294 |

Native-derived GT/before/conservative/final comparisons cover four train views
I/D, K/D, J/C, H/C and all three held-out views F/B, J/D, L/B.
Chin crops additionally track the actual repaired 3D boundary, since fixed
tube-relative crops can miss the chin in oblique views.
[Tube comparison](assets/dec5_diffusion_mesh_repair/tube_comparison.png),
[chin comparison](assets/dec5_diffusion_mesh_repair/chin_comparison.png).
Strict artifact-free verdict remains **fail**, with diagnosed residuals; local
geometric improvement and artifact integrity must not be called global quality pass.

### Smooth flythrough and downloadable artifacts

`OUTPUT` below means `/mnt/data/lookcloser_dec5_5a3_diffusion_mesh_repair_000973`.

- Textured GLB: `OUTPUT/final_supported/frames/000973/dec5_000973_repaired.glb` (~25 MiB).
- OBJ + texture archive: same directory, `dec5_000973_repaired_obj.zip`.
- Repaired normalized geometry PLY + operations/UV/provenance manifests: same directory.
- Video: `OUTPUT/flythrough_supported/smooth_000973.mp4` (~8.2 MiB), 1080×1920,
  30 fps, 360 frames, 12 seconds, **one static source time 000973**.
- 360 lossless rendered PNGs: `OUTPUT/flythrough_supported/frames/`.
- [Overview contact sheet](assets/dec5_diffusion_mesh_repair/flythrough_contact_sheet.png)
  and native detail sheets under `OUTPUT/flythrough_supported/detail_review/`.
- `OUTPUT/visual_review.json`, `OUTPUT/audit.json`,
  `OUTPUT/heldout_review_supported_final/face_metrics.json`, `OUTPUT/checks.jsonl`.

The loop uses central G/B, I/B, I/D, G/D anchors, with strictly positive convex
weights (minimum 0.03787), fixed H/C intrinsics and a fixed optical-axis target.
Translation speed is 0.0334963–0.0334972 normalized units/s; angular speed
2.395–2.719°/s, including a normal seam step at the loop closure. Maximum
orientation difference from the H/C reference is 10.19° (that reference is not
the loop's mean position). No piecewise anchor transition or image morphing.
The previous 150-frame path traversed 24 linear/Slerp segments in just 5 seconds.

A framing bug was caught visually in the first path prototype: using the mean
rig-center position in place of the reference camera center moved the optical
target off the subject. `flythrough_rejected_framing` preserves that failed run.
The corrected implementation fixes the reference optical target and rejects empty
quarter-turn preflights and empty frames. An asymmetric-rig regression test covers it.
Final mesh rendering took **36.1 s**, including preflights, using four workers;
rendering plus H.264 encoding took **41.6 s** on clever-shadow. This is texture/mesh
rendering throughput, **not a claim that full PatchMatch reconstruction became faster**.

### Replay workflow and tests

All commands run from the repository root with the existing nerfstudio environment.
This is a hash-tied 000973 experiment; the reviewed loop IDs/masks cannot be reused
blindly on another moving frame. `prepare` and final publication reject incompatible
requests; intermediate experiment directories are not a generic campaign resumer.

```bash
python LookCloser/scripts/diffusion_mesh_repair.py prepare
python LookCloser/scripts/diffusion_mesh_repair.py edit-masks
# Built-in imagegen: inspect targets, submit the saved prompts with mask/train references,
# save each result as OUTPUT/views/XX/generated_raw.png. No hidden API/CLI fallback.
python LookCloser/scripts/diffusion_mesh_repair.py import-generated
python LookCloser/scripts/diffusion_mesh_repair.py prepare-mvs
python LookCloser/scripts/diffusion_mesh_repair.py run-mvs
python LookCloser/scripts/diffusion_mesh_repair.py fuse-mvs
python LookCloser/scripts/diffusion_mesh_repair.py real-control
python LookCloser/scripts/diffusion_mesh_repair.py compare-mvs
python LookCloser/scripts/diffusion_mesh_repair.py inspect-geometry
# Visually inspect boundary overlays and independent depth evidence before these edits.
python LookCloser/scripts/diffusion_mesh_repair.py local-repair
python LookCloser/scripts/diffusion_mesh_repair.py cylinder-repair
python LookCloser/scripts/diffusion_mesh_repair.py refine-cylinder
python LookCloser/scripts/diffusion_mesh_repair.py object-completion
# bake_local_mesh_repair.py --candidate CANDIDATE --output SEPARATE_ASSET_ROOT
# Bake local_repair -> local_asset, cylinder_refined -> cylinder_refined_asset.
# Selected final: object_completion_supported -> neutral_supported_asset with --neutral-cylinder.
python LookCloser/scripts/finalize_local_mesh_repair.py publish
python LookCloser/scripts/review_local_mesh_repair.py --cylinder-asset final_supported --review-suffix _supported_final
python LookCloser/scripts/review_local_mesh_repair.py --heldout --cylinder-asset final_supported --review-suffix _supported_final
# smooth_mesh_flythrough.py --asset OUTPUT/final_supported/frames/000973 --output OUTPUT/flythrough_supported
python LookCloser/scripts/finalize_local_mesh_repair.py video-crops
# Inspect the saved native crops and write an honest visual_review.json before audit.
python LookCloser/scripts/finalize_local_mesh_repair.py audit
```

67 tests pass across local repair, hard-source texture, joint temporal texture
and campaign tests. Coverage includes exact selected-hole boundaries, preservation
of old geometry/UV texel coordinates, uniform periodic path speed, convex-hull
constraints, asymmetric framing, and checksum/path-escape failures.
Measured environment: Open3D 0.19.0, PyMeshLab 2025.7.post1, trimesh 4.12.2,
xatlas 0.0.11, PyMaxflow 1.3.2, torch 2.7.1+cu128; model defaults are unchanged.

Terminal audit passes artifact integrity: original geometry/atlas invariants,
synthetic outside-mask equality, real/synthetic/held-out split, finite face-only
metrics, every retained final asset checksum, all 360 PNG checksums, and ffprobe
360 frames / 12.000 s / 1080×1920. Visual quality remains explicitly fail.
Selected GLB SHA-256:
`9fa79ebfc166a233ca3c327f7e1105142d2faf7e02343f9992bb147383d35f67`.
Selected video SHA-256:
`7e52317706170a1279d7a1b91bb09d2a3e3f4fd78d74cea9187ce57d6b49351e`.
No workers remain active after the terminal supervision check; rejected workspaces
and previous publications remain recoverable and are not download targets.

## Insights

1. **A visible triangle is not necessarily an observed surface.** The false slab
   is real mesh geometry, not two-camera color averaging. TSDF weight and mesh
   self-visibility do not independently establish native cross-view surface support.
   A component filter also misses a slab attached to the main mesh. The local
   independent depth veto establishes a concrete missing-confidence-check failure;
   it does not isolate every upstream PatchMatch/TSDF numerical cause.
2. **Specularity is plausible, not proven to be the sole cause.** The train views
   show view-dependent metal highlights and substantial defocus in K/D. Fitting
   a robust low-dimensional shape to supported shaft points is useful here, but
   its backside must stay labeled as an object prior. A static unlit texture also
   cannot reproduce physically moving specular highlights under a changing view.
3. **Diffusion edits are useful proposals, not automatically consistent cameras.**
   The direct local MVS experiment failed its quality gate despite better coverage
   than the matched real-only control. Iterative edited-dataset optimization, as in
   [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/), is a related research
   direction, not evidence that three independent edits will produce valid TSDF geometry.
4. **Keep natural shadows distinct from baked source seams.** The real underside
   of the chin is dark; globally erasing that shadow would be incorrect. Residual
   polygonal color boundaries on face/neck are retained and flagged. Twelve
   native-scale flythrough samples were inspected, but later source times were
   not processed here, so temporal shadow-band stability is not yet validated.
5. **Do not promote this to all moving frames unchanged.** The reusable workflow
   is inspect → mask/provenance → matched hypothesis test → local geometry edit →
   train review → frozen export → held-out review → smooth render → integrity audit.
   Region tracking and rigid-object motion evidence remain necessary before a
   temporal geometry campaign; the previously deferred multi-time body reconstruction
   was not silently started.
