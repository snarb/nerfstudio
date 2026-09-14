# DEC5 residual jaw holes: small camera-height workaround

## What was tested

After the elevated 150-time movie avoided the broad under-chin defect, native
review still found small interior black spots at 001193 and 001195. Their
previous attribution is overwhelmingly original mesh misses, not RGB averaging.
The user authorized a smooth camera-path workaround while geometry research
continues. No image generator, mesh filling, exposure change or GT input is used.

`probe_elevated_jaw_visibility.py` checks height offsets 0, 0.1, 0.25 and 0.4 rig
rows at 000899, 001191, 001193, 001195 and 001197. Zero offset must reproduce the
published pose. All positions remain inside the calibrated five-camera anchor
polygon. Fixed intrinsics, horizontal travel and scene-landmark composition are
unchanged; only the camera's rig height increases. The offset is uniform in time,
not a per-frame jump. Clay is CPU raycast from the same hash-verified meshes.

`study_elevated_jaw_rgb.py` then renders those five actual times at +0.4 rows,
using the exact published angular-prior/foreground-mask wrappers. The 150-pose
request contains 150 distinct positions with unchanged chronological actor data;
only five RGB canaries are rendered so far. It does **not** replace the movie.

## Results

CPU controls: `/mnt/data/dec5_elevated_jaw_height_probe` (20 clay views).
RGB controls: `/mnt/data/dec5_elevated_jaw_rgb_plus04` (five matched RGB views).

| Original selected interior jaw component | +0 rows | +0.1 rows | +0.25 rows | +0.4 rows |
|---|---:|---:|---:|---:|
| 001193 | 45 pixels | 34 | 19 | no corresponding isolated spot visible |
| 001195 | 73 pixels | 55 | 18 | 5 |

These are selected ray-miss component diagnostics, not anatomical masks or
whole-image quality scores. Components can split or join; visual checks are
required. All five native jaw RGB comparisons and head overviews were inspected.
The conspicuous late jaw spots disappear or become much smaller at +0.4. The
original mesh, camera profiles and RGB algorithm are unchanged.

![001193 native jaw comparison](/mnt/data/dec5_elevated_jaw_rgb_plus04/review/001193_jaw.png)
![001195 native jaw comparison](/mnt/data/dec5_elevated_jaw_rgb_plus04/review/001195_jaw.png)

The route still spans rig x **-0.7983..3.8200**, y **0.6000..1.3500**. Maximum
rig step is 0.06743 and maximum second difference 0.03351; adding a constant
height preserves the original rig-coordinate motion derivatives. All 150 poses
are distinct. This is a trajectory adjustment, not freezing the camera, image
cropping or replacing the dynamic subject with a single-time mesh.

Limitations: native neck/face color seams, opaque/ragged crown geometry and the
small early hand/chin contact notch remain. At 000899 the latter looks somewhat
more exposed from the raised camera. The correction is therefore a **partial
shot improvement**, not an artifact-free movie or a repaired mesh. No new full
150-frame video is accepted on these five canaries alone.

## Insights

Small camera-height changes can hide an original surface hole through visibility
without repairing it. The useful late-frame improvement must be weighed against
early hand/chin exposure over the whole path. Do not use the reduced visible hole
count as evidence that TSDF geometry improved. Confidence-prior and source-count
experiments remain separate; rejected gradient color correction is not included.
