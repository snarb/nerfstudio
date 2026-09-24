# Task for a new session: mesh-teacher → LookCloser, DEC5 `000973`

## Objective and authority

Train **LookCloser**, not a new mesh model, to reproduce the current mesh teacher's
novel-view detail, then improve real-image fidelity through restrained fine-tuning.
This is one static actor instant viewed from many cameras, **not dynamic NeRF**.
The preparation session did not start training. In this new session implement the
opt-in two-stage training workflow, run the controlled comparisons, inspect outputs,
and document measured results. Preserve unrelated dirty changes and model defaults.

Read `LookCloser/AGENTS.md`, `README.md`, `Paper LookCloser.md` locally; do not download
the paper. Read `experiments/dec5_mesh_distillation_preparation.md` and the dataset
audits before launching. Historical mesh/texture experiments used this real eval
camera for model selection: it is held out from fitting, not a pristine unseen benchmark.

## Prepared inputs

Host: `clever-shadow`. Dataset root:

```text
/mnt/data/dec5_000973_mesh_distillation_v1
  request.json                       frozen inputs, seed, scripts, all camera poses
  config/                            calibration, fixed exposure, camera profiles, scripts
  mesh/teacher.ply                    exact selected geometry, not a raw TSDF volume
  mesh/original_tsdf.ply              original full-block TSDF before later repairs
  mesh/normalization.json             original coordinate normalization record
  mesh/face_matches_original_tsdf.npy exact triangle-position provenance, not truth labels
  synthetic/transforms.json           300 train + 24 synthetic validation views
  synthetic/images/train_XXXX.png     UNIQUE stems for frequency-map compatibility
  synthetic/images/val_XXXX.png
  synthetic/depth/*.npy.gz            gated camera-z depth; zero = unknown
  synthetic/full_depth_z/*.npy.gz     ungated mesh z-buffer; zero = no hit
  synthetic/masks/*.png               valid RGB supervision; outside = unknown
  synthetic/confidence/*.png          heuristic weight /255; NOT automatically loaded
  synthetic/views/VIEW/               additional coverage, source IDs, provenance, receipt
  real/transforms.json               62 real train RGB + one real held-out RGB
  real/images/                       fixed-display PNG, no per-image exposure
  real/face_roi.json                  GT-only face polygon for the NEW display protocol
  heldout_teacher/frames/000973/      teacher reference; evaluation only
  review/                            contact sheets and native review patches
  dataset_hashes.json                 retained data checksums
  packaging.json                     unique-stem packaging receipt
  independent_audit.json             actual Nerfstudio parser/depth validation
  visual_review.json                 reviewed scope and known teacher defects
```

All images are native **1920×1080**, not rotated portrait images and not 6K.
RGB is display-domain 8-bit PNG in [0,255] (normal loader scales to [0,1]); do not
apply EXR/PQ/HDR exposure normalization or another Reinhard/sRGB transform to it.
Synthetic train: all 62 actual train poses rendered by the teacher, plus 238 local
barycentric camera interpolations. Validation: 24 distinct interpolated poses.
Virtual poses use only train calibration, central columns D..K / rows B..D;
position weights are nonnegative and sum to one, rotations use weighted SO(3) means.
No camera positions are extrapolated. This is not proof that every visible surface
is correct; per-pixel support remains necessary. Every one of the 62 real train
poses is included, including outer cameras outside the central virtual envelope.

Teacher: selected repaired mesh + hard single-source train RGB reprojection,
incidence² quality, 4-degree angular prior, native footprint visibility, frozen
camera RGB profiles and exposure, **zero UV registration and no RGB averaging**.
Source raycasts/images are cached; one virtual-view comparison to the unmodified
renderer verifies byte-identical RGB and source IDs (`parity/train_0062.json`).
No diffuse generative fill or target-GT substitution is used in these teacher views.

Native real/teacher comparisons expose an important limitation: the lipstick tube
is largely missing in central `train_0033`, despite a good face, and some side views
have a chipped tube/hand edge. Hair silhouettes are blocky. Do NOT interpret these
as correct geometry or claim an artifact-free teacher. The missing tube is a useful
real-finetuning recovery test; never use teacher coverage to exclude it from real
evaluation. See `review/train_0033_real_pair.png`.

`real/` uses the same fixed exposure and train camera profiles as the teacher.
Held-out camera `F004_B005_1210O9` uses the fixed exposure with identity response;
its RGB has never been used to fit a profile. `J004_D005_1210TA` and
`L004_B005_12106A` are excluded. EXRs and their source transforms are unchanged.

## Critical coordinate, depth, mask contracts

Use these dataparser settings for **both phases**:

```text
orientation_method = none
center_method = none
auto_scale_poses = False
scale_factor = 1.0
downscale_factor = 1
depth_unit_scale_factor = 1.0
eval_mode = filename
load_3D_points = False  # unless implementing a separately verified point seed
```

For the initial controlled comparison also keep `camera_optimizer.mode=off` and
`appearance_embedding_dim=0`: calibration is fixed, and 300 synthetic image IDs
must not silently become 62 differently indexed appearance embeddings. If testing
appearance embeddings later, implement an explicit identity mapping/initialization
at the transition rather than loading a mismatched table.

Explicit filename lists are authoritative. Do not reorient/recenter each dataset:
that silently moves the pretrained field when switching datasets. `transforms.json`
contains a descriptive depth-unit field, but this parser still needs the explicit
`depth_unit_scale_factor=1.0` config. The audit actually checks loader results.

Depth is **camera z**, in the same normalized units as camera translations/mesh,
not metres and not unit-ray distance. Keep `is_euclidean_depth=False`. If converting,
multiply z by `sqrt(1+((u+.5-cx)/fx)^2+((v+.5-cy)/fy)^2)` exactly once.
Independent barycentric-hit checks validate this convention.

Choose and freeze one appropriate scene AABB before all comparison arms; mesh was
cropped to `[-.15,.15]^3`, and audit uses `scene_scale=.15`. Verify the actual actor
bounds before choosing a tighter AABB. Room/background is absent in the teacher;
this pilot is actor reconstruction, not proof of full-room fidelity. A background
extension must be explicit and identical across comparison arms, not silently
change field coordinates/capacity after pretraining.
Real images are deliberately unmasked. Before fine-tuning a bounded actor field,
decide and record a train-only actor/frustum sampling policy shared with the real-only
baseline; do not waste the objective fitting room pixels that cannot be represented
inside the chosen AABB. Do not use the teacher's missing-surface mask to exclude
real lipstick or hair supervision. Real held-out face evaluation stays GT-defined.

- `mask.png`: a mesh hit with at least one admissible RGB source. Black pixels
  outside this mask mean **unknown**, NOT a black target or empty-space label.
  Valid genuinely dark pixels inside remain valid; validity is not based on RGB.
- `visibility_count.png`: number of cameras passing the SAME MESH visibility and
  sampling gates. **Not independent PatchMatch votes**, not a calibrated probability.
- `confidence.png`: `min(count/3,1)`, zero near unsupported pixels (one-pixel erosion),
  ×.25 for triangles absent from original TSDF and ×.25 for local depth jumps.
  It is a heuristic reliability weight, not evidence that original triangles are correct.
- `depth_supervision.npy.gz`: zero except at interior pixels with at least two
  visible sources, no local >.003 relative depth jump, and an exact original-TSDF
  triangle match. This avoids treating later prior repairs as measured depth.
- `inferred_geometry.png`: triangle does not exactly match original TSDF; covers
  repairs/moved/retriangulated surfaces, not just hand-labelled anatomical regions.

The current stock loader does **not** consume `confidence_file_path` automatically.
Implement/test an opt-in weighted RGB path if using it. Never silently advertise
weighted training while only reading the RGB mask. For invalid depth, verify that
zero is excluded from *all* depth objectives and depth-guided sampling. Do not
force rays outside coverage to empty space or teach the silhouette of mesh holes.

## Required experimental workflow

1. Re-run `audit_mesh_distillation_dataset.py audit`; inspect `visual_review.json`
   and representative synthetic/real images. Do not regenerate the teacher to
   improve each student's metric or hide known defects. Audit remaining source
   leakage and normalization invariants before training.
2. Build LookCloser frequency maps from prepared RGB in a **derived run/preprocessing
   directory**, not by mutating this hash-pinned bundle. They are not precomputed
   here: progressive patch-frequency fitting belongs to the next training session.
   Preserve unique image stems. Teacher holes/background must not dominate frequency
   estimates. Use real-train frequency evidence at trusted train-pose mesh depths
   where needed to avoid inheriting blurred teacher hair as a low-frequency prior;
   no real held-out patches may enter grid initialization. Implement/measure this
   opt-in combination rather than assume synthetic and real frequencies are equal.
   In particular, do not equate an unobserved teacher voxel with permanently zero
   representational capacity: verify a conservative frequency fallback leaves room
   to recover the missing lipstick/edges after the Frequency Grid is frozen.
3. Train a **real-only LookCloser baseline** with the same coordinate system,
   display preprocessing, representation, evaluation, and recorded compute budget.
   Reuse an old run only if these conditions and input hashes genuinely match.
4. Pretrain a fresh LookCloser on 300 synthetic train views with masked/weighted RGB
   and gated teacher-depth assistance. Evaluate on the 24 synthetic val poses and
   separately on real held-out RGB. First establish whether the student can reproduce
   teacher details at unseen virtual poses; don't obscure a representation/sampling
   failure by moving directly to real fine-tuning. Depth support is a warm start,
   not a perpetual restriction to the exact teacher surface.
5. Save a pre-finetune checkpoint and renders. Then switch to 62 real train images
   with an explicitly lower field LR (start ~5–10× lower than pretrain, validate,
   not a claimed optimal setting). Keep the **Frequency Grid** values fixed after
   the transition. Trainable hash-grid features and appearance/density MLP weights
   should remain trainable. **Do not confuse Frequency Grid with occupancy grid.**
   Occupancy must be able to update so real evidence can recover missing surfaces.
6. Preserve field coordinates, frequency-grid buffers and initialization/warmup
   state at the transition. `model_parameters_only` deliberately restores only
   fields, not all required grid state; it is NOT sufficient by itself. Full resume
   has separate optimizer/scheduler/LR controls; inspect the current trainer and
   implement a tested opt-in phase transition. `grid_update_interval=0` disables
   periodic updates but does not automatically protect against initialization/reset.
   Test hashes of frequency buffers before loading, after loading, and after several
   fine-tune steps; verify actual field gradients/updates and actual LR.
7. If needed use decaying teacher replay/geometry support only on trusted regions.
   Relax prior/depth constraints where real observations disagree; small LR alone
   cannot correct a systematically wrong lipstick surface. Log real vs synthetic
   ray fractions and teacher weights. Do not feed eval RGB into training/calibration.
8. Compare **real-only**, **synthetic-pretrained before fine-tune**, and
   **synthetic→real fine-tuned**. Report PSNR/SSIM/LPIPS, not loss. Real face ROI must
   be fixed from GT and must include missing predicted parts, never intersect it
   with candidate coverage. Report masked synthetic-teacher fidelity separately;
   it is distillation fidelity, not evidence of real reconstruction accuracy.
9. Render a smooth identical novel-camera path from teacher and each student, for
   this same static time. Inspect hair, ears, lips/lipstick, jaw, silhouettes and
   view-dependent flicker. This does not test flicker across actor time. Select
   checkpoints by `eval_all_psnr`, LPIPS tie-break within .07 dB, per AGENTS; keep
   synthetic-validation selection and real-validation selection explicitly labelled.
10. Write `experiments/dec5_lookcloser_mesh_distillation.md` with What was tested /
    Results / Insights, checkpoint/input hashes, native crops, compute time and
    failure cases. Stop and diagnose catastrophic geometry collapse, empty rays,
    wrong depth scale, non-finite metrics or normalization drift. Minor teacher
    artifacts are known limitations, not grounds for scene-specific regeneration.

## Practical commands and supervision

```bash
cd /home/brans/repos/nerfstudio/LookCloser
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1 \
  ../.venv/bin/python scripts/audit_mesh_distillation_dataset.py audit
```

Use the existing quiet LookCloser runner for training, after implementing and
testing the phase/mask contracts. Do not invent a ready-to-run training command
with unimplemented flags. Inspect `--help` / use `--dry-run` first. Keep mandatory
hourly supervision active until training ends, then immediately evaluate/review.
Do not end the task merely because a worker was detached. No new SfM/PatchMatch
is needed for this experiment, and no per-image exposure fitting is permitted.

The initial bundle's main report and audit record whether preparation completed;
this task file alone is not a completion receipt.
