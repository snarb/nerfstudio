# DEC5 `000973`: mesh-teacher data preparation

## What was tested

Prepare a frozen single-time teacher dataset for the proposed LookCloser
synthetic-pretrain → real-finetune experiment. No NeRF training in this session.

Root: `/mnt/data/dec5_000973_mesh_distillation_v1`.
The selected mesh is the same `000973` repaired geometry used by the later
wide-spiral production renderer (SHA-256
`68487f5cb375723ba8121c38954c21d8d9f7dfad26aac13336a3e678da3bfa69`).
RGB uses hard-source incidence²/4° angular selection, native visibility,
frozen train-camera profiles/exposure and zero UV registration. No averaging.

300 synthetic training poses = 62 exact train poses + 238 local barycentric
interpolations; 24 additional validation interpolations. All inputs are native
1920×1080. Virtual camera parents are train-only, central D..K/B..D. Positions
are convex combinations and rotations use SO(3) means, not independent Euler
interpolation. Seed is `20260924`. Synthetic targets are never real GT images.

Each view includes RGB, full camera-z mesh depth, gated depth supervision,
RGB validity, visibility count, heuristic confidence, source IDs and an
inferred-geometry flag. Full original TSDF and selected repaired meshes are
retained with metadata. Visibility count is **same-mesh visibility**, not a new
measurement of independent PatchMatch support. Confidence is not calibrated.
The real 62-train/1-eval split is separately exported using fixed display
calibration; held-out camera response remains identity, never fitted.

## Results

**Completed and audited on 2026-09-24. Training was not launched.**
`independent_audit.json` reports pass for all retained data hashes, all original
source RGB/transforms hashes, mask/depth consistency and the actual parser splits:
synthetic **300/24**, real **62/1**. Camera poses/intrinsics at the 62 common poses
match exactly. Independent camera-z/ray-distance checks have maximum absolute
z error **3.37e-7 normalized units**. Five focused tests pass.

Cached-source export matches the unchanged historical renderer **byte-for-byte**
in RGB and source IDs at both `train_0062` (virtual) and `train_0033` (real pose).
All 324 views were inspected on 14 contact sheets; eight native/detail/GT-review
panels were also inspected. This is not a claim of 324 native-detail approvals.
`visual_review.json` admits an imperfect teacher for controlled distillation,
explicitly **not artifact-free**. The bundle occupies approximately 4 GiB locally.

| Data property | Min | Median | Max |
| --- | ---: | ---: | ---: |
| RGB-valid fraction of image | .26365 | .41874 | .46682 |
| Trusted-depth fraction of image | .25451 | .40992 | .45915 |

These fractions describe actor framing/coverage, not face-quality metrics.
The mesh has 149,223 triangles; 147,855 exactly match the original TSDF and 1,368
do not. The latter are excluded from gated geometry supervision. Summed render
worker time is 8,493 seconds (parallel, not wall time).

The supervisor first used four workers, then gracefully repartitioned remaining
work across eight after measuring RAM/VRAM headroom. It did not change the frozen
render recipe. Interrupted unfinished views are recomputed, completed receipts
are reused after checksums. Checks are recorded every 30 seconds, not just hourly.

Useful review artifacts (absolute paths under the root):

- `review/camera_sampling.png`: actual rig-space pose distribution.
- `review/train_0033_real_pair.png`: native same-pose real versus teacher.
- `review/train_0000_real_pair.png`, `review/train_0058_real_pair.png`: outer controls.
- `review/heldout_face_pair.png`: independent real RGB versus teacher projection.
- `review/contact_*.jpg`: all synthetic camera views, chronological by view ID.

## Insights

This is a **usable imperfect teacher**, not ideal synthetic ground truth. In the
central `train_0033` view, the tube of lipstick is largely absent; side views also
show chipped tube/hand edges. Hair silhouettes are polygonal; outer shoulder/torso
coverage is incomplete. Native skin detail is largely retained. No per-view mesh
repair, GT paste or generated image was added to make this dataset look cleaner.

Unknown pixels must not become negative RGB/empty-space supervision. Repaired
triangles receive reduced RGB confidence and are excluded from gated depth.
Even original TSDF triangles can be wrong: provenance is not a correctness label.
The actual target for the next session is to retain teacher detail and recover
missing real surfaces without freezing occupancy or hash features incorrectly.

The old `000973` face polygon visibly included background/hair when overlaid on
the new GT. A new inset frontal-face polygon was manually drawn on GT only and
visually checked; old face metrics are therefore not numerically comparable.
No new PSNR/SSIM/LPIPS claim is made before training/evaluation. The real held-out
camera has historical model-selection exposure, which must be disclosed.

Frequency maps are intentionally left to the next training session (progressive
patch fitting, using the new image domain); all image stems are made unique for
the frequency-map cache. Frozen grid transition, masks/weights and actual depth
semantics need integration tests before starting expensive training.

Detailed handoff: [training task](dec5_mesh_distillation_training_task.md).

Replay from the LookCloser directory, using the existing `.venv` on clever-shadow:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1
../.venv/bin/python scripts/prepare_mesh_distillation_dataset.py init
../.venv/bin/python scripts/prepare_mesh_distillation_dataset.py real
../.venv/bin/python scripts/run_mesh_distillation_preparation.py --workers 8
../.venv/bin/python scripts/prepare_mesh_distillation_dataset.py finalize
../.venv/bin/python scripts/audit_mesh_distillation_dataset.py package
../.venv/bin/python scripts/audit_mesh_distillation_dataset.py audit
```

Native panels/held-out-reference export and manual visual review are separate
steps, not implied by replaying the automated audit. Do not rerun `finalize` over
an already packaged/sealed dataset: it intentionally emits the pre-packaging
transforms. For unchanged completed inputs, run only the independent audit.
