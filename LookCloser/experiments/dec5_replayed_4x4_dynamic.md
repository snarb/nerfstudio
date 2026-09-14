# DEC5: earlier static 4×4 camera path, now with moving actor

## What was tested

The user selected the camera motion from
`/mnt/data/dec5_camera_grid_diagnosis_v2/4x4/video.mp4`, but wanted the actor to
move as in the subsequent dynamic clip. This explicitly replaces the interim
wide sweep and two-row-inset designs. Neither interim candidate is the delivery.

`replay_dynamic_camera_flight.py` reads the **saved 720 camera poses** of that
specific static pilot. It resamples the full periodic curve to 150 phases using
periodic cubic translation/weights and rotation interpolation, preserving both
position and orientation. It does not substitute a new look-at target, use only
the first 150 poses, or animate a single source mesh. Exact coincident samples
are tested against the saved matrices. Source times remain 000899–001197, with
one verified mesh and its own train RGB per chronological time.

The pilot SHA-256 is
`8998c29383f323f4c14671752104be1ba2a4ef8b111addcd7ab5d3d1bf46f345`;
its request SHA-256 is
`cade4747cf21f6951c44becda9ff7b8022030bd6cfe23cc1c0747a113207f04d`.
The route spans F..I horizontally and A..D vertically, as in that pilot. It is
not compatible with the superseded two-row vertical edge margin: the rig has
only five vertical levels. No new path is disguised as an exact replay.

Timing is explicitly different: the original 720-frame, 24-fps static flight was
30 seconds; 150 actual temporal samples at 24 fps are 6.25 seconds. The full
camera loop therefore runs 4.8 times faster in the normal dynamic version. The
separate 12-fps copy is 12.5 seconds, slowing actor and camera equally at lower
cadence; no interpolated actor frames or temporal repetitions are fabricated.
Camera periodicity does not imply periodic actor motion or a seamless movie join.

Rendering retains the existing fixed camera color profiles, fixed exposure,
hard-source texture selection and train-only source masks. New-pose monotonic
restoration rechecks old silhouette deletions; it cannot fill geometry missing
from the original mesh. Eight disjoint workers are supervised every 30 seconds
on clever-shadow. No training, SfM, diffusion or changed model defaults.

## Results

**150/150 rendered and encoded. Camera/actor motion verified; reconstruction is
not artifact-free.** Eight workers finished in **722 seconds** without CUDA/OOM
errors. All 150 overview frames, all 150 actual MP4-decoded frames and four native
distributed-phase images were visually inspected. The old-static/new-dynamic
quarter-cycle comparison was also inspected.

| Check | Result |
|---|---:|
| Distinct chronological source times / meshes / RGB outputs | 150 / 150 / 150 |
| Achieved horizontal / vertical grid span | 2.879997 / 2.879452 intervals |
| Maximum pairwise viewing-angle change | 26.9883° |
| Closed-loop translation step max/min | 1.000281 |
| Pose error against selected pilot after normalization round trip | 1.78e-15 |
| Independent saved-depth / fresh-raycast checks | 5 / 5 match |
| Focused camera, replay, normalization and audit tests | 43 passed |
| Normal / slowed movie | 150 frames at 24 / 12 fps; 6.25 / 12.5 s |

Both real actor motion and the chosen camera loop are visible in the decoded
sequence. Known failures remain: neck openings near 000963, lipstick-side
contamination near 000975 onward, fragmented lowering hand at 001039–001043,
crown openings roughly 001099–001149, clothing-edge cutouts and skin source seams.
Visual findings are recorded in groups of ten with their scope explicit:
100 frames inherit a conservative batch `fail`, and 50 are
`reviewed_known_artifacts`; these are **not** 50 artifact-free passes or a precise
100-frame defect prevalence estimate. No pending reviews remain.

Outputs: `/mnt/data/dec5_replayed_4x4_dynamic_150_v2`.
The final audit checks all source times, input/output hashes, exact selected
trajectory poses and independent fresh raycasts against saved target depths.
Visual review remains separate from integrity: inherited neck holes, lipstick
background contamination and texture seams must not be relabeled artifact-free.
No novel-view ground truth exists, and no full-frame PSNR/SSIM/LPIPS is substituted
for the original held-out face-only campaign.

Retained evidence:

- [Chosen path diagram](/mnt/data/dec5_replayed_4x4_dynamic_150_v2/camera_path.png).
- [Old static vs new dynamic, matched camera phases](/mnt/data/dec5_replayed_4x4_dynamic_150_v2/reference_comparison.png).
- [All-frame render sheets](/mnt/data/dec5_replayed_4x4_dynamic_150_v2/contact_sheets/).
- [Actual decoded movie sheets](/mnt/data/dec5_replayed_4x4_dynamic_150_v2/decoded/).
- [Independent integrity audit](/mnt/data/dec5_replayed_4x4_dynamic_150_v2/integrity_audit.json).

Movie SHA-256:

- `video.mp4`: `67b8dab316bd20c31985ffae70009e250f3177de943a1fc396f78fa4e8f48392`.
- `video_slow_12fps.mp4`: `9ceefb55198f51b47dbc7c96b9f997bf5fa503b3194219ebb0a62f0ba3ad162d`.

## Insights

- Bind a chosen reference video and its path by checksum; a newly designed
  "similar" spline does not satisfy a request to reuse an earlier camera motion.
- Camera phase and actor time are independent. Keep both changing, and verify
  actual rendered rays and decoded movie images, not only metadata.
- State duration explicitly. A full 30-second route and only 6.25 seconds of
  distinct scene samples cannot preserve both original speeds without additional
  temporal reconstruction or synthetic interpolation.

Replay:

```bash
python LookCloser/scripts/replay_dynamic_camera_flight.py
python LookCloser/scripts/prepare_wide_dynamic_geometry.py \
  --parent /mnt/data/dec5_replayed_4x4_dynamic_150 \
  --output /mnt/data/dec5_replayed_4x4_dynamic_150_v2
python LookCloser/scripts/run_wide_dynamic_workers.py supervise \
  --output /mnt/data/dec5_replayed_4x4_dynamic_150_v2 --workers 8
```
