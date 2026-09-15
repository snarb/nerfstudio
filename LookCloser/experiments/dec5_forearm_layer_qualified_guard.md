# Layer-qualified measured-depth protection

## What was tested

An opt-in continuation of the [lower-row forearm experiment](dec5_foundation_lower_forearm.md)
at actor time `001037`. Keep its exact raw stereo proposal, calibration, fixed
camera color profiles, four train observations, masks and edge/consistency gates.
Change only whether a corroborated farther PatchMatch point can veto the proposed
nearer arm. No held-out image, temporal geometry transfer or camera-path change.

The same rule applies to all 62 query cameras: the query is warm throughout a
3×3 neighborhood (`R-B>8`); the nearer point is warm and at least three rectified
pixels inside the coarse arm region in **all four** witnesses; the farther point
is blue (`B-R>8`) in at least three; both projections are within native image
margins in all four. This disqualifies some wrong-object far-layer witnesses. It
does **not** measure the correct depth or establish a generic skin classifier.
Skin-to-skin disagreements retain the original measured-depth protection.

The effective recipe is bound by the outer `qualification_request.json` and
`qualification_result.json`. The reused inner geometry request still hashes the
base guard as provenance; it must **not** be read as an unchanged algorithm.
Each new RGB request binds both outer files explicitly. Existing runner and model
defaults and the published 150-time video are unchanged.

## Results

**Partial coverage gain; all three RGB canaries still fail. Not a video repair.**

| Geometry check | Strict lower-row control | Layer-qualified candidate |
|---|---:|---:|
| Eligible proposal pixels | 21,196 | 21,196 |
| Proposed triangles | 40,925 | 40,925 |
| Retained additional triangles | 34,926 | 37,385 |
| Old mesh vertex/triangle prefix preserved | Yes | Yes |
| Strict-control added faces lost | — | 0 |

The candidate retains **2,459** extra faces. Its final pass still has **3,253
ray observations contradicting the original guard**; all are disqualified by
the new color/region rule. The qualified guard has zero remaining vetoes over
62 cameras × two pixel-center offsets. These are ray observations, not unique
surface points. Do not describe this as zero original geometric contradictions.

| View, same actor time and camera | Changed RGB pixels vs strict | Newly visible depth pixels | New black pixels | Visual verdict |
|---|---:|---:|---:|---|
| H/A train pose | 713 | 425 | 0 | Fail: wrist gap and peppering remain |
| E/D train pose | 1,582 | 35 | 8 | Fail: scattered blue/dark arm defects |
| Moving-video pose | 1,034 | 555 | 1 | Fail: large wrist/forearm break remains |

Counts above are diagnostics, **not anatomical coverage or image-quality
metrics**. The upper 1,200 portrait rows are byte-identical to the strict control
for all three renders. No full-frame PSNR/SSIM/LPIPS or loss was introduced.

The main agent inspected all three native detail panels, the moving overview,
and the five-sample NCC plot. Other saved overviews are not claimed as inspected.

- [Moving forearm comparison](/mnt/data/dec5_forearm_layer_qualified_guard/review/moving_detail.png)
- [H/A versus real train RGB](/mnt/data/dec5_forearm_layer_qualified_guard/review/H004_A005_1210M6_detail.png)
- [E/D versus real train RGB](/mnt/data/dec5_forearm_layer_qualified_guard/review/E004_D005_1210L4_detail.png)
- [Explicit negative verdict](/mnt/data/dec5_forearm_layer_qualified_guard/visual_review.json)

### Remaining skin-to-skin ambiguity

A read-only diagnostic examines the five already frozen F/E veto samples using
rectified E/E–F/E RGB. It compares mean-centered RGB NCC in fixed 7×7, 15×15,
and 31×31 windows along a half-pixel disparity sweep. The reference patch is
fixed in F/E; only the E/E correspondence changes. This is **not COLMAP's
internal slanted-plane/bilateral matching cost** and does not alter geometry.

| Frozen sample | Nearer-prior NCC, 15×15 | Far-PM NCC, 15×15 | Query channel-std, 7×7 |
|---|---:|---:|---:|
| 0, palm | .9069 | .9363 | 3.52 |
| 1, forearm | .8143 | -.0246 | 1.99 |
| 2, forearm | .5296 | .0860 | 1.49 |
| 3, forearm | .6972 | -.2666 | 1.99 |
| 4, forearm | .5762 | .0625 | 1.87 |

Forearm hypotheses differ by roughly 39–43 disparity pixels. Larger-context
matching generally favors the nearer hypothesis over the far PM one, but the
small skin windows have weak variation and unstable alternative peaks. The
palm result depends on window size; the last forearm sample's best large-window
peak is not at the proposed depth. These observations cannot justify blindly
accepting the prior or disabling all measured-depth vetoes.

An independent audit checked every enclosing remap contributor throughout the
largest window, complete search and exact hypotheses. All belong to photographed
1080×1920 source content with a native-pixel margin: black rectification padding
does not explain these particular curves. All reported NCC estimates are finite.

- [NCC curves](/mnt/data/dec5_forearm_layer_qualified_guard/ambiguity/ncc_curves.png)
- [Numerical patch diagnostics](/mnt/data/dec5_forearm_layer_qualified_guard/ambiguity/result.json)
- [Independent audit](/mnt/data/dec5_forearm_layer_qualified_guard/audit.json)
- [129 retained/input hashes](/mnt/data/dec5_forearm_layer_qualified_guard/artifact_manifest.json)

Nine helper/regression tests passed. Three disjoint RGB workers, geometry and
diagnostic jobs all terminated; GPU/free-space evidence is in `terminal_check.json`.
Recheck without modifying artifacts:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python \
  scripts/freeze_forearm_layer_qualified_guard.py --check
```

## Insights

Cross-view depth agreement can preserve a wrong layer; the agreement count alone
is insufficient confidence. Explicit object/color evidence safely narrows some
contradictions, but here only yields a small visual improvement. Its warm/blue
thresholds are dataset-specific and need temporal/other-region negative controls
before generalization. This branch has not repaired the hand or cheek.

A useful next test is a train-only, availability-aware photometric consistency
check on competing layers with multiple independent baselines and uncertainty
for low-texture windows. Do not choose depth from the five diagnostic samples or
use the proposed geometry's own rendered texture as validating evidence.

The user-authorized camera workaround remains separate. The previous exact
train-waypoint loop hid a late jaw fleck but exposed an early under-chin cutout
([report](dec5_local_train_waypoint.md)). Any replacement path must preserve
dynamic actor times and substantial smooth travel, and check the whole affected
sequence—including the return to the beginning—not just a favorable camera.
