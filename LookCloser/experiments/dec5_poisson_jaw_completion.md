# DEC5: continuous surface prior with observed-neighborhood certification

## What was tested

One-time 001193 canary following the [curved-cap controls](dec5_curved_jaw_caps.md).
`study_poisson_jaw_completion.py` samples 150000 oriented points from the existing
head mesh and runs installed Open3D 0.19 Screened Poisson, depth 10, scale 1.05,
linear fit, four threads, sample seed 17. This is a **geometric prior**, not new
stereo observations or a human anatomical model. It does not replace the face.

Only local proposals near original boundaries are added: original-surface and
boundary distance <= .003 normalized units, edge <= .0015, normal dot >= .25,
and closest-point boundary restrictions. Every original vertex/triangle remains
exact. The unfiltered local proposal visibly overgrows hair and is not accepted.
Semantic masks and the previously audited measured D/D mask override are reused;
source RGB masks, camera profiles, exposure and renderer are unchanged.

Three admission controls:

- Direct depth: the previous two-view sample rule and measured free-space veto.
- Nearest anchors: also permit a prior triangle if its three nearest original
  surface points each have at least three agreeing train depth views.
- Interpolation: find nearby verified original-mesh vertices, require agreement
  with the Poisson surface within .0005, compatible normals, and at least eight
  seeds within .003. The query must lie inside their tangent-plane convex hull.
  A local quadratic must pass leave-one-seed-out p90 error <= .0005 and predicted
  offset <= .0005. This checks geometric interpolation, not held-out RGB accuracy.

All added surfaces retain the existing sample free-space veto and strict final
two-offset measured-ray guard across all 62 train cameras. Missing depth remains
unknown: successful interpolation is explicitly labelled inferred, never counted
as a direct depth observation. `local_surface_certificate.py` is an opt-in helper.

## Results

| Fixed F/E train skin ROI | Interior depth misses | Edge-inclusive depth misses | Edge-inclusive black RGB |
|---|---:|---:|---:|
| Production | 30 | 45 | 58 |
| Previous small curved cap | 14 | 16 | 26 |
| Poisson + direct depth | 19 | 25 | 39 |
| Poisson + nearest anchors | 19 | 25 | 39 |
| Poisson + certified interpolation | **3** | **3** | **19** |

These are the two previously frozen GT-drawn **train** skin polygons, not
full-face/full-frame quality metrics. The interpolation improves this local
mesh coverage substantially, but does not establish whole-head anatomical
accuracy, temporal transfer or an artifact-free movie.

Of 41320 local raw triangles, 4091 pass semantic screening. Final direct,
nearest-anchor and interpolated variants retain 1401/1442/1601 triangles after
native pruning. Interpolation uses 1838 verified seeds and certifies 993 of 3015
candidate vertices. All final original geometry prefixes are exact. Surfaces
are appended, not topologically welded to the original mesh.

![Native jaw comparison](/mnt/data/dec5_poisson_jaw_completion/review_interpolated/F004_E005_1210FP_detail.png)
![Moving camera comparison](/mnt/data/dec5_poisson_jaw_completion/review_interpolated/moving_detail.png)

Thirteen fresh RGB renders cover the three variants in moving/physical F/E
baseline-repaired pairs, plus one interpolated held-out F/B render. All three
moving predictions are byte-identical and differ from production at only 13
pixels: the existing camera workaround hides most of this particular defect.
The visible local improvement is in the lower F/E view, not a large movie change.

The independently frozen held-out face protocol gives exactly unchanged PSNR
**30.276545**, SSIM **.939692**, LPIPS **.076123**. Baseline and repaired PNGs
are byte-identical. This is one-view non-regression, not evidence of improved
geometry in that view. No held-out RGB enters geometry or source selection.

Main-agent visual review inspected three raw-added clay views, all eight native
head/detail comparison panels, and the held-out panel. The broad raw hair shell
is rejected; the filtered interpolation substantially reduces the isolated jaw
spot, but a few missing pixels and a ragged dark edge remain. Hair/crown,
lipstick, hand/forearm and the full temporal movie are not repaired here.

### Where confidence and RGB coverage were lost

Before admission, both full Poisson and the bounded local proposals cover all
16 pixels missed by the earlier curved cap. Those local triangles pass masks
and sample free-space checks, but many have few direct depth votes or weak
nearest anchors. Neighborhood interpolation addresses this specific limitation.

The final edge ROI has 16 black RGB pixels **with valid mesh depth**; every one
has source ID 255 (no selected source). A read-only visibility replay finds
1–4 initially visible source cameras per pixel, then zero after the renderer's
unconditional four-tap depth checks. Hypothetically ignoring taps whose bilinear
weight is <= .001 restores one source at all 16 pixels. This is diagnostic only:
no texture check, RGB sampling or rendering rule has been changed. The next
test must preserve foreground isolation and avoid background contamination.

Independent audits replay local geometry assembly, recompute 3701 seed depth
queries and all 3015 interpolation certificates, and perform 372 fresh native
ray checks over the three final meshes, with zero qualified free-space violations.
The Poisson solve itself is not rerun; its point cloud/raw mesh are hash-bound.
Four focused tests pass, including hull containment, noisy/insufficient support,
all-anchor requirements and free-space/semantic veto preservation.

Root: `/mnt/data/dec5_poisson_jaw_completion`. The best local pilot mesh is
`interpolated/001193/mesh.ply`; it is not a serialized TSDF volume or a promoted
replacement for the 150-frame movie. Source EXRs and published artifacts remain
untouched. Initial review code was archived before adding the third comparison.

## Insights

Checking depth only at an actual hole can prevent any completion. A bounded,
cross-validated neighborhood prior can improve coverage while preserving a
measured free-space veto. This is different from declaring unsupported depths
measured or simply lowering the required number of views.

Geometry and texture coverage now have distinct remaining mechanisms. Fix the
near-zero-weight visibility rejection and test temporal transfer before any
video publication. The requested artifact-free dynamic video and general mesh
improvement remain incomplete; this is a verified positive local step.
