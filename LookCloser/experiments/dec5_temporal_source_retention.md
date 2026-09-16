# DEC5: relative hard-source retention on two moving-camera clips

## What was tested

The [earlier sparse-view control](dec5_surface_ray_source_prior.md) weakened a
transported jaw shadow but left temporal source switching untested. This
experiment freezes that exact `axis_relative` rule: retain a valid graph source
only if its pixel quality is at least 50% of the best visible source; otherwise
use that best source. No RGB averaging, ray-angle modification or parameter sweep.

Two contiguous 24-frame, 24-fps clips use actual `wide_spiral_free` camera poses
and distinct moving-actor meshes: lipstick 000973–001019, late face 001087–001133.
All indices precede the cinematic train-view dissolve. Meshes, source masks,
graph labels, fixed profiles/exposure, calibration, source images and visibility
tests are unchanged. The controller uses three GPU render workers, not PatchMatch
jobs; checks every ten seconds retain live PIDs/stages, GPU memory, errors and
free space. All 48 workers finished with exit0 and no recorded OOM/CUDA errors.

Root: `/mnt/data/dec5_temporal_source_retention`.
These are **1080×1920 diagnostic renders**, paired into 2160×1920 comparison
videos. They do not replace, upscale or modify the delivered 3456×6144 video.

## Results

Actual target-depth arrays and face-source graph labels are exactly equal to the
matched original in all 48 frames. Cameras and per-time meshes are unchanged;
each clip contains 24 distinct actor meshes and 24 distinct camera centers.

| Clip | Changed RGB pixels/frame min / median / max | Newly black RGB pixels, total |
|---|---:|---:|
| Lipstick | 4,213 / 7,811.5 / 17,575 | 6, across four times |
| Late face | 9,195 / 11,129.5 / 16,206 | 0 |

All six new black pixels have a positive target depth and a valid source ID.
They are not missing-ray/no-source holes: the new source supplies black RGB.
Actual inspected crops localize them beside hair/ear/neck boundaries. Five had
very dark original colors; 001015 at portrait(42,1095) changes from[101,84,47]
to black. Do not dismiss that last point as mere one-level quantization. No
per-frame exception or rerender hides these side effects.

### Temporal source diagnostic

Baseline-only quarter-resolution Farneback flow, with forward/backward cycle
error below0.5px, maps source IDs between each adjacent pair. Both arms use the
same accepted samples in a fixed head/hand screen window; all missing sources
and out-of-bounds tracks are excluded. This is **not ground-truth flow, a
flicker metric, or a face-fidelity score**.

| Clip | Common tracked samples across23 transitions | Original source switches | Relative-source switches |
|---|---:|---:|---:|
| Lipstick | 1,034,998 | 210,277 | 207,260 |
| Late face | 1,653,476 | 30,504 | 25,333 |

This sampled diagnostic does not flag an overall increase in camera-source
switching. It cannot establish perceptual temporal consistency or correct source
ownership. No PSNR/SSIM/LPIPS or full-frame quality metrics were computed here.

Main agent inspected all four chronological overview sheets (all48 times), all
twelve strongest-change native sheets, six native nose strips, six individually
marked black-pixel crops, and the localized nose attribution panel:29 images.
These are actual contact/crop inspections, **not continuous normal-speed video
playback**. Both comparison MP4s decode to24 frames at2160×1920/24fps.

The predominant changes are at hair boundaries and, briefly, the hand/lipstick
membrane. Existing rough contours remain; the membrane still has visibly
incorrect geometry. At000993 a thin dark line next to the lipstick persists.
At001007 the conspicuous polygonal neck/tube overlap remains. The later
nose-side seam persists throughout the inspected sequence. The control is
therefore **not promoted as an artifact fix**; no full150-frame/6K rerender.

- [Lipstick comparison video](/mnt/data/dec5_temporal_source_retention/review/lipstick/matched_diagnostic.mp4)
- [Late-face comparison video](/mnt/data/dec5_temporal_source_retention/review/late_face/matched_diagnostic.mp4)
- [Lipstick/neck native changes](/mnt/data/dec5_temporal_source_retention/review/lipstick/largest_change_native_04.png)
- [Late nose native sequence](/mnt/data/dec5_temporal_source_retention/review/late_face/nose_native_04.png)

### New localized nose evidence

At001123, the posthoc rectangle[885,610,945,750] and a luma deficit of20/255
against its5×5 median select156 dark-ridge samples. This is a diagnostic
selection, not an anatomical segmentation: the inspected magenta panel also
includes a few eye pixels. All156 have geometry and an admitted source; four
are exactly black. Source ownership is H/C17, I/C123, J/C16. **All156 RGB values
and their source IDs remain unchanged** under relative retention.

[Localized comparison](/mnt/data/dec5_temporal_source_retention/nose_attribution/nose.png)
and saved native/presentation coordinates, RGB, source IDs and depths provide a
reproducible next diagnostic. Positive depth does not establish correct geometry.
The experiment rules out missing target rays or absent source IDs at these
samples, but does not yet distinguish wrong surface placement, inaccurate source
occlusion tests and transported view-dependent shadows. It does not identify the
line as a natural shadow or authorize painting over it.

Eight focused tests cover contiguous actual times, immutable camera/source
selection, changed-geometry rejection, static-source tracking, and the existing
hard-source policy invariants. The final seal rechecks matched geometry, receipts,
frozen scripts and every retained comparison. Production remains unchanged.

## Insights

The prior jaw-shadow gain does not generalize to the conspicuous cinematic
nose seam. Relaxing graph-label loyalty alone is not enough when the retained
source already meets the relative-quality rule. Further generic weight sweeps
are not justified by this result. The saved nose intersections should instead
be traced into real I/C, H/C and J/C pixels and their visibility/depth evidence.

The lip/neck membrane separately needs geometry correction across time. The
reviewed000995 hand/tube masks and gap-carving policy are a concrete seed for a
tracked, train-only temporal mask experiment; copying those masks or context
polygons unchanged onto moving frames would not be valid. This remains a next
step, not work claimed completed by the present texture control.

Reproduction uses the repository venv with two OpenMP/OpenBLAS threads:

```text
scripts/study_temporal_source_retention.py prepare
scripts/study_temporal_source_retention.py supervise
scripts/review_temporal_source_retention.py --clip lipstick
scripts/review_temporal_source_retention.py --clip late_face
scripts/inspect_temporal_retention_black_pixels.py
scripts/diagnose_temporal_retention_nose.py
```

The experiment root is intentionally no-overwrite. After actual image review,
`seal_temporal_source_retention.py --notes REVIEW_JSON` checks the explicit
viewed-image inventory. A seal verifies this negative/partial control, not the
full requested artifact-free video or mesh objective.
