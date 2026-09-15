# Larger horizontal and vertical dynamic camera choices

## What was tested

The user watched the four previous choices and found camera movement too small.
This separate opt-in campaign retains the production 150 distinct actor times
`000899..001197` (step 2), meshes, masks, fixed exposure/profiles and incidence-2
hard-source unwarped RGB. It increases actual camera translation and viewpoint
change, with four open spline paths and independently constrained endpoints.
No experimental geometry from the parallel crown/forearm work is included.
Normal speed remains 24 fps / 6.25 seconds. No slowed alternatives are planned.

The physical anchors are the real train cameras C/B, K/B, K/D and C/D. Convex
interpolation and endpoint blends do not extrapolate outside the train-camera
hull. C..K avoids horizontal outermost A/N and penultimate B/M columns. There
are only five vertical rows A..E: avoiding both outermost and penultimate rows
entirely would leave only C, eliminating vertical motion. These paths stay
strictly inside B..D with at least 0.127 row-coordinate margin and never reach
A/E or the B/D row-center coordinates. The initial camera center is exactly the
published production center; the final center is the actual G/C camera. There
is no periodic wrap into the initial chin/hand pose.

## Results

**All four selected RGB gates pass for full-sequence evaluation with known
residuals. Three paths are rendering; the refined fourth is queued next under
the same six-worker limit. No new video is published yet.**

| Path | Physical center-ray separation | Row-coordinate range | Frozen-actor rendered travel |
|---|---:|---:|---:|
| Diagonal sweep | 53.54° | −0.782..+0.861 | 558 × 511 px |
| Wide oval | 50.66° | −0.759..+0.873 | 576 × 483 px |
| Left high arc | 51.62° | −0.826..+0.809 | 564 × 531 px |
| Refined right high arc | 50.24° | −0.841..+0.824 | 560 × 449 px |

Relative to the previous S curve's physical 18.32° azimuth / 6.01° elevation,
the new paths achieve 50.06–53.50° azimuth and 15.89–17.37° elevation. Both axes
increase by about 2.7×. These angular measures use camera-center rays to a fixed
3D target, so they cannot be inflated by changing camera orientation alone.

These are measured raycasts of the **same actor mesh** at six actual movie
cameras, not an inference from camera metadata or changing actor pose. A pair
of fixed depth-separated 3D landmarks also shows over 1.5× the relative image
movement of the previous S curve. Pure framing translation cannot cause that
relative parallax. The full dynamic RGB movement audit is still pending.

- [Previous S-curve frozen-actor camera probe](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/motion_probe/previous_s_curve_contact.png)
- [New diagonal frozen-actor camera probe](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/motion_probe/new_contact.png)
- [Explicit motion gate](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/motion_gate.json)

The first geometry-only gate (`dec5_large_motion_choices_v1`) achieved the large
motion but clipped some crown views. The second gate (`v2`) used one fixed
0.85× production focal length and 520 × 500 pixel composition to preserve the
head; it still exposed the broken descending lower forearm. Both failed gates
are retained, with requests and clay images, and were not full-video renders.

The active `v3` keeps exactly the same physical paths and fixed 0.85× lens. A
smooth time-scheduled camera tilt during hand descent adds
`400*exp(-((index-70)/19)^4)` vertical composition pixels. This puts the lower
forearm outside the actual field of view while the hand lowers. It is not an
image crop, tracked ROI, stabilization, animated zoom or actor retiming. Screen
target travel is approximately 517–520 px horizontally and 528–845 px vertically.
The head is not locked at image center. Natural distance/scale variation along
the convex path is retained rather than extrapolating to an unsupported sphere.

All four v3 clay sheets were inspected. The diagonal path's first 12 RGB canaries
were inspected as a chronological contact sheet, plus native lipstick `000995`,
hand `001029/001037/001045`, crown `001123`, and jaw `001193/001197` crops.
The broad face stays coherent; the descending lower-forearm gap is excluded.
Crown shell/fringe openings, rough hair, lipstick/hand membranes and source/neck
boundaries persist. This is not an artifact-free reconstruction.

The other three paths' twelve-frame RGB contact sheets and native `000995`,
`001029/001037`, and `001123` crops were inspected. Oval and left-high arc pass
the same limited gate with residual crown/fin artifacts. Original right-high
arc fails: `000943` exposes an under-chin cutout and `001037/001045` still show
a conspicuous descending-arm remnant. That failed twelve-frame gate is retained
without a full video. The separate `right_high_arc_refined` candidate blends
the first 60 camera-center samples from the safer diagonal corridor into the
right arc, using quintic smoothstep, and adds 150 pixels of descent-time camera
tilt. Its real center-ray angle remains 50.24°. Native refined `000943` confirms
the large initial chin cutout is hidden. All twelve refined RGB canaries and
native `000995`, `001029/001037/001045`, and `001123` crops are now inspected:
the large lower-forearm gap is outside the field of view, but lipstick/hand
membranes, crown openings and source boundaries remain. This fourth path is
approved only for full-sequence evaluation with those explicit residuals.

- [Diagonal RGB canaries](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/canary_review/contact.png)
- [Native crown residual](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/canary_review/001123_head_native.png)
- [Native hand/fin residual](/mnt/data/dec5_large_motion_choices_v3/diagonal_sweep/canary_review/001037_head_native.png)

The first six-process canary wrapper completed 12 valid frames, then failed when
a process switched variants: the frozen renderer's installer is not re-entrant.
No input or valid receipt was modified. The separate fresh-process shard
supervisor resumes missing frames, caps concurrency at six and records process,
GPU, inventory and free-space checks every 30 seconds. Original failure logs
remain under `v3/canary_workers`; resumed logs use `v3/canary_fresh_workers`.

## Insights

The requested larger motion needs both a genuinely wider 3D arc and sufficient
room for the head to move across the image. Broad paths can expose geometry that
was hidden by the previous 18–19° shots. Analytic camera timing can exclude the
lowering forearm, but cannot repair unsupported crown geometry or texture seams.
Native RGB gates, full-sequence review and honest residual disclosure remain
required before publication. Pixelwise PSNR/SSIM/LPIPS between different novel
views would not be a valid quality comparison, so none is reported here.

Active replay commands (existing hash-bound requests are immutable):

```bash
../.venv/bin/python scripts/visible_large_motion_choices.py init
../.venv/bin/python scripts/visible_large_motion_choices.py screen
../.venv/bin/python scripts/finalize_large_motion_choices.py motion
../.venv/bin/python scripts/supervise_large_motion_choices.py --canary
../.venv/bin/python scripts/finalize_large_motion_choices.py panels
# Initialize the bounded refinement; mixed mode starts its canaries while
# the three previously approved paths render, with six processes total:
../.venv/bin/python scripts/right_arc_visibility_refinement.py
../.venv/bin/python scripts/supervise_large_motion_choices.py --mixed
# Only after the actual native canary gate:
../.venv/bin/python scripts/supervise_large_motion_choices.py
../.venv/bin/python scripts/finalize_large_motion_choices.py sheets
../.venv/bin/python scripts/finalize_large_motion_choices.py audit
../.venv/bin/python scripts/finalize_large_motion_choices.py encode
```
