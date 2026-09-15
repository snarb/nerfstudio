# DEC5: neighborhood completion at earlier face/hair times

## What was tested

Transfer the unchanged [observed-neighborhood surface prior](dec5_poisson_jaw_completion.md)
to actual frames **001083 and 001123**. No time-specific threshold tuning,
cross-time mesh substitution, mask dilation or production-surface replacement.
The [corrected native texture footprint](dec5_native_texture_footprint.md)
is applied to **both** production and repaired geometry, isolating geometry.

`run_neighborhood_completion_transfer.py` produced the two candidate meshes
and four moving/F004_E005_1210FP render pairs. The new opt-in
`study_head_neighborhood_transfer.py` adds matched native train views:
E004_B005_1210I7 at 001083 and G004_A005_121071 at 001123. Twelve fresh RGB
images in total. All source profiles, exposure, camera parameters and texture
selection are unchanged within each pair. Production defaults remain untouched.

Review uses previously fixed, train-GT-defined face/skin and coarse hair regions.
GT is regenerated from immutable EXR with the renderer's exact mean-centered
log-gain and fixed exposure response. It is for post-hoc review only.
No held-out RGB, PSNR/SSIM/LPIPS or full-frame quality metric is used in this
coverage experiment. These counts must not be compared to face fidelity scores.

## Results

| Actual frame | Added triangles | Verified seeds | Certified vertices | Moving RGB pixels changed | F/E RGB pixels changed | Region-view RGB pixels changed |
|---|---:|---:|---:|---:|---:|---:|
| 001083 | 382 | 1017 | 375 | 0 | 0 | 144 |
| 001123 | 332 | 1031 | 162 | 0 | 0 | 0 |

| Train-region coverage diagnostic | 001083 before → after | 001123 before → after |
|---|---:|---:|
| Face-interior zero-depth pixels | 0 → 0 | 0 → 0 |
| Face-interior black RGB pixels | 0 → 0 | 0 → 0 |
| Coarse hair-region zero-depth pixels | 11561 → 11519 | 6349 → 6349 |
| Coarse hair-region black RGB pixels | 11595 → 11553 | 6353 → 6353 |

The coarse hair polygon includes real spaces between strands and some exterior
background; its zero-depth count is **not an anatomical missing-surface count**.
The face polygon excludes portions of the silhouette: zero interior misses
does not establish that the cheek/neck boundary is correct.

Independent replay verifies original geometry preservation, seed depth support,
all local interpolation certificates and **248 fresh native ray checks**
(62 cameras × two pixel offsets × two times), with zero qualified free-space
violations. Final meshes have 161/122 connected components and zero nonmanifold
edges; these counts do not imply watertightness or anatomical correctness.
Nine focused certificate, admission-guard and native-footprint tests pass.

The main agent actually inspected all ten saved panels: face and hair native
crops plus moving/F/E/region head comparisons at each time. Both times retain
visible crown breaks; 001123 retains a particularly clear opening with detached
fringe and a dark cheek/neck boundary. Moving views also retain source seams.
No new conspicuous defect was observed, but **broad artifact-removal verdict
is fail for both times**. Byte-identical moving RGB is not a video improvement.

![001083 hair comparison](/mnt/data/dec5_neighborhood_completion_transfer/001083/head_review/hair_native.png)
![001123 hair comparison](/mnt/data/dec5_neighborhood_completion_transfer/001123/head_review/hair_native.png)
![001123 face boundary](/mnt/data/dec5_neighborhood_completion_transfer/001123/head_review/face_skin_native.png)

Artifacts: `/mnt/data/dec5_neighborhood_completion_transfer/{001083,001123}`.
`head_review/visual_review.json` binds each actual inspection to image hashes.
`freeze_head_neighborhood_transfer.py --check` verifies the separate immutable
`head_transfer_artifact_manifest.json`; the earlier 001195 manifest is unchanged.
All workers terminated normally. No video was rerendered or replaced.

## Insights

The late-jaw gain at 001193/001195 does **not** generalize to a useful crown
repair at these earlier times. The current method only admits small, locally
supported additions; it cannot be promoted as a general head-completion fix.
Do not loosen guards per time or launch a 150-frame rerender for these results.

The user's camera-path workaround remains valid but separate: the existing
[phase-shifted dynamic flight](dec5_temporal_camera_phase.md) reduces late jaw
exposure while retaining camera excursion and actor motion. It still exposes
hair and hand defects. Any further artifact-aware path should be evaluated
across actual actor times and all visible regions, not merely the cheek or one
static mesh. Passing observed-depth safety checks and hiding a defect are
neither sufficient proof of correct geometry nor artifact-free video.
