# Multiview admission of agreed forearm geometry

## What was tested

The previous no-final-veto control still had wrist holes. A read-only layer
diagnostic tested successively earlier domains: eligible foreground pixels,
all two-pair agreement, masked reference prediction, and reference prediction
without semantic masks. The last two are diagnostic bounds, not accepted meshes.

Seven interior wrist-hole components (four H/A, three moving) are completely
covered by **already agreed** stereo geometry within .01 camera-z of their
boundary depths. Reference-only eligibility, not stereo disagreement, had
excluded these surfaces. An additional cuff-side component remains outside the
agreed domain; removing semantic masks is not adopted as its repair.

The opt-in change preserves existing eligibility and adds agreed grid points
where the old mesh leaves a missing foreground layer in **at least two of the
62 actual train cameras**. Camera rays pass through the point itself; native
photographed bounds and camera-z separation .003 are enforced. No target camera
or held-out RGB determines admission. Four-view stereo/mask agreement, disparity
alignment and max edge .002 remain unchanged.

The comparison uses the same explicitly **no-final-PM-veto diagnostic** on both
sides. This isolates the admission change; it does not claim the final measured
depth guard has passed. Existing original vertices/triangles remain unchanged.

## Results

Eligible points increase **21,196 → 28,349** (+7,153); proposed added triangles
**40,925 → 54,211**. All newly admitted points have at least two train-view
missing-layer votes. Independent replay verifies the complete vote arrays.

| Matched RGB view | New depth pixels | Lost depth pixels | New black pixels |
|---|---:|---:|---:|
| H/A train pose | 468 | 0 | 0 |
| E/D train pose | 7 | 0 | 0 |
| Moving-video pose | 505 | 0 | 1 |

These are diagnostics, not anatomical quality metrics. Upper 1,200 portrait
rows remain byte-identical in all three views. No full-frame PSNR/SSIM/LPIPS.

In the seven already selected interior wrist components, missing pixels fall
from **386 to 8 in H/A** and **360 to 6 in the moving view** (746→14 combined,
not unique 3D points). The separate 67-pixel cuff-side component stays at 67.

**Substantial local wrist improvement, not full-frame/video acceptance.** Native
review shows most internal wrist holes disappearing and fewer E/D skin speckles.
Hand distortion/holes, cuff-side truncation, a visible old/new layer boundary and
two-tone forearm texture remain. The scene is still not artifact-free.

- [H/A against real train GT](/mnt/data/dec5_multiview_forearm_admission/review/H004_A005_1210M6_detail.png)
- [Moving wrist comparison](/mnt/data/dec5_multiview_forearm_admission/review/moving_detail.png)
- [Moving overview](/mnt/data/dec5_multiview_forearm_admission/review/moving_overview.png)
- [Earlier-domain diagnosis](/mnt/data/dec5_forearm_proposal_gap_diagnosis/moving.png)
- [Per-component residuals and vote replay](/mnt/data/dec5_multiview_forearm_admission/audit.json)
- [Explicit partial visual verdict](/mnt/data/dec5_multiview_forearm_admission/visual_review.json)

Both diagnostic-domain montages, all three new native detail panels and the
moving overview were inspected. Other saved overviews are not claimed as
inspected. A synthetic ray test verifies camera-z separation and frustum
availability. Three render workers completed normally. Recheck retained/input
hashes with `scripts/freeze_multiview_forearm_admission.py --check`.
The freeze binds 72 retained/input hashes; the preceding patch-guard study
separately binds 335 hashes.

## Insights

An apparent hole from a novel/train view can be hidden behind an old surface in
the chosen reference view. Restricting additions to that reference's holes
silently discards useful, mutually agreed geometry. The multiview rule corrects
this without broadening the stereo masks or changing learned depths.

This is an append-only local inference pilot, not a watertight merged surface,
a temporal validation or a cheek repair. Next: test the remaining layer/texture
seam and transfer the same admission rule to another actor time before rollout.
The production 150-time video, camera path, source data and model defaults are
unchanged. The broader moving-video and improved-cheek objective remains open.
