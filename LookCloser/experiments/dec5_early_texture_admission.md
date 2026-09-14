# DEC5 target-angle prior before source admission

## What was tested

The renderer rejected camera candidates using an incidence-only quality threshold
before the target-angle prior was applied. A camera whose centroid visibility
passed could therefore receive zero weight even at its own target pose. The
later angular preference could not recover it. This is particularly harmful
with noisy or inferred mesh normals and the existing eighth-power cosine score.

`diagnose_forearm_texture_admission.py` reproduces centroid visibility, normal
quality, the 12%-of-best filter and the later angle preference. Its diagnostic
does not change images or count centroid visibility as final pixel visibility.
At three real H004_A005_1210M6 target views, it finds:

| Time | Skin pixels with geometry | Own-camera centroid visible | Visible own-camera pixels discarded before angle prior |
|---|---:|---:|---:|
| 001029 | 24863 | 24829 | 16067 |
| 001033 | 16586 | 16278 | 13838 |
| 001037 | 9562 | 8981 | 8513 |

`study_early_texture_prior.py` changes only the order: apply the existing
calibration-only target-angle weights, then the same 12% filter, then the same
graph labeler. It keeps normals, graph costs, sigma=4 degrees, visibility,
pixel fallback, static registration, RGB sampling, fixed exposure/profiles and
single-source RGB unchanged. No averaging, sharpening or new color fit is added.

The legacy renderer has no admission hook. This opt-in study compiles a verified
single-statement substitution in an isolated process and hashes that exact
implementation. It refuses unexpected source text. The original renderer file
and all model defaults remain unchanged; the old late-angle wrapper is not also
installed. Existing source-mask checks are installed normally.

Forearm roots: `/mnt/data/dec5_forearm_texture_admission` (diagnosis) and
`/mnt/data/dec5_forearm_early_texture_prior` (RGB control). This control reuses the
same completed color-qualified quadratic meshes from the previous experiment;
it is not a new mesh repair. Baseline symlinks explicitly point to the unchanged
production geometry renders. The matched visual pairs compare early versus
late texture on the **same repaired mesh**.

## Results

### Fixed real-train forearm ROI

| Time | Late angle prior PSNR / SSIM / LPIPS | Early angle prior |
|---|---|---|
| 001029 | 28.272 / .8514 / .2054 | **31.526 / .9405 / .0795** |
| 001033 | 20.618 / .7236 / .3646 | **21.087 / .8181 / .2884** |
| 001037 | 19.053 / .5608 / .5583 | **19.200 / .6327 / .5405** |

These are fixed manual **train forearm-skin** metrics, not face or held-out
scores. The actual final own-camera source counts rise from 6628 to 18189,
1734 to 11048, and 382 to 6101 skin pixels. The skin hole counts stay exactly
18 / 1433 / 2203. All six moving/real-view target depth arrays are byte-equal
as arrays to their late-prior controls, and mesh/camera/source inventories match.
Global black-pixel counts change by at most three pixels (source/coverage
diagnostic, not a full-frame quality score).

![Same geometry; source admission order only](/mnt/data/dec5_forearm_early_texture_prior/001029/moving_matched_native.png)

Native review finds much less skin/hand tessellation and neck/garment source
seams, but the geometric wrist/forearm cuts remain. Some source RGB is genuinely
motion-blurred. The result does not synthesize sharpness or repair finger depth.

### Independent held-out face gate

The prior gradient experiment already froze GT-only face polygons at 000899,
000973 and 001193. `study_early_prior_heldout.py` reuses those GT images and ROIs,
the same original meshes and calibrated F004_B005_1210O9 pose. No held-out RGB
enters prediction, calibration or parameter selection. This is a separate
fixed-exposure ROI protocol, not the old per-image-exposure 50-frame table.

| Time | Late angle prior face PSNR / SSIM / LPIPS | Early angle prior |
|---|---|---|
| 000899 | 26.237 / .9016 / .1067 | **28.760 / .9333 / .0740** |
| 000973 | 29.079 / .9029 / .0837 | **31.605 / .9327 / .0647** |
| 001193 | 28.206 / .9242 / .0862 | **30.277 / .9397 / .0761** |

All three metrics improve at all three held-out times; target depths remain
exactly equal. Native GT/late/early panels were directly inspected: broad
cheek/neck seams are reduced, without removing real shadows. Hair/fringe mesh
limitations, existing neck holes and lipstick defects remain. This is a passed
**non-regression/improvement gate**, not artifact-free approval.

![Held-out face and hair](/mnt/data/dec5_early_texture_prior_heldout/evaluation/000973/native_face_hair_comparison.png)

An initial body-control launch correctly refused a changed producer-controller
hash after the preparation-only plane flag was added. The old producer was
verified against its archived hash, its render function AST was proven identical,
and that ancestry is recorded in the new request. Old requests were not changed.
One held-out worker returned SIGTERM status after writing a valid completed
render; all output hashes verified and a receipt-only resume exited normally.
No image was regenerated for that event and no CUDA/OOM traceback was present.

### Full dynamic video

`run_early_texture_video.py` replays the exact 150 distinct source times, meshes
and phase+30 camera poses from `/mnt/data/dec5_phase30_dynamic_150`, changing only
source admission order. It reuses the existing lifecycle lock, disjoint worker
assignment and 30-second PID/GPU/free-space checks, but installs the early prior
without the late wrapper. Six workers run on clever-shadow.

Candidate root: `/mnt/data/dec5_phase30_early_texture_dynamic_150`.
The immutable gate explicitly permits full-sequence **evaluation with known
geometry failures**. It does not promote the three isolated forearm meshes,
freeze the actor, reduce camera travel or call remaining holes repaired.
All 150 frames finished, with 150 distinct meshes, RGB images and calibrated
camera positions. Rendering took 13m17s wall time including one SIGTERM recovery;
the resumed supervisor validated existing receipts and completed the remaining
seven frames, exiting normally. The first process group disappeared without a
CUDA/OOM traceback; the origin of SIGTERM is unknown. Twenty-seven compact
PID/GPU/disk checks are retained. No completed image was regenerated for quality.

The independent audit confirms 31.51 degrees of view-angle span, a 1.0124
maximum/minimum camera-step ratio, 384 x 154 px landmark travel and 301 x 150 px
foreground-centroid travel. Native RGB is only rotated to portrait, never
recentered or cropped. The MP4 is 1080 x 1920, 24 fps, 150 frames, 6.25 seconds;
no slow version was created. SHA-256:
`ce1ff5f7fd612a46e39b6ae89d10aee7534f895ea7e7887d722dc10903bccd40`.

Main-agent inspection covered all 150 overview frames, all 150 native jaw/lipstick
crops, four distributed native full images, and all 150 decoded MP4 frames in
15 sheets. The phase workaround avoids the broad late sub-cheek opening, but
tiny black jaw flecks remain around 001191--001195. Serious forearm/hand holes
remain around 001029--001045 (2.7--3.05 s), as do the lipstick rear fin, crown and
right-hair notches, some skin stitching and the incomplete lower torso.
Overview groups 060--079 are explicitly marked fail; the other 130 frames are
reviewed-with-known-artifacts, **not 130 artifact-free passes**. Jaw review does
not excuse defects outside the jaw crop. This is a texture improvement and a
camera workaround, not elimination of the underlying geometry problem.

Artifacts: [video](/mnt/data/dec5_phase30_early_texture_dynamic_150/video.mp4),
[overview sheets](/mnt/data/dec5_phase30_early_texture_dynamic_150/contact_sheets),
[native jaw reviews](/mnt/data/dec5_phase30_early_texture_dynamic_150/jaw_review/visual_review.json),
[integrity audit](/mnt/data/dec5_phase30_early_texture_dynamic_150/integrity_audit.json).
Face numbers above are three held-out canaries, not a 150-frame aggregate.
The focused source-admission and forearm regression suite passes 20 tests;
all new Python entry points compile. Existing model and renderer defaults are
unchanged. Scripts, logs, frame receipts and publication hashes retain the
exact experimental implementation, including the isolated source transform.

## Insights

Candidate admission must use the intended ranking objective. Applying a strong
view prior after irreversible incidence culling can make that prior ineffective,
even when the desired camera sees the surface. This experiment changes that
ordering, not the sharp one-source texture policy. The positive held-out result
supports full-sequence evaluation; unresolved depth holes still need geometry
work and cannot be excused by better face metrics.
