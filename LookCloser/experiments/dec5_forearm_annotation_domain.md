# DEC5 forearm: annotation-domain and measured-boundary controls

## What was tested

Hypothesis: some residual forearm cuts come from proposal assembly, not absent
camera coverage. `diagnose_forearm_residual_topology.py` traces reference-mask
admission, initial triangulation, curvature transfer and color-qualified carving.
Its plane-envelope coverage is a diagnostic, **not** an RGB/face metric.

Two separate controls retain the same three train-only masks, measured depths,
fixed camera calibration, production-mesh prefix and early hard-source texture:

1. Rebuild the accepted grid with a quadratic depth prior, solving a regularized
   Laplacian residual pinned only to trusted measured boundary depths (16/71/20
   pins for 001029/001033/001037). Maximum displacement remains 0.012. This is
   inferred geometry, not measured anatomy. Test alone on 001037 and with the
   annotation correction on all three times.
2. Keep the previous boundary-conditioned curved vertices exactly. Correct only
   the semantic admission domain: the frozen polygons leave an unannotated image
   border. A valid continuous renderer coordinate at x=2.05..2.48 rounds to
   pixel 2, outside the known annotation. Treat this as unknown, not a negative
   skin vote. Still require two known positive cameras and veto every known
   disagreement. No polygon dilation or per-frame parameter exception.

The second control is opt-in `--known-annotation-domain` in
`study_forearm_production_delta.py`; existing defaults are unchanged. Its source
proposal, quadratic fit, boundary feather, triangle extent and color-qualified
free-space guard are unchanged. Three-frame preparation, rendering and audit
were parallelized on clever-shadow with separate outputs; no new PatchMatch run.

## Results

All numbers below use the same fixed manual **train forearm skin ROI** in
H004_A005_1210M6. These are reprojection controls, not held-out face metrics and
not comparable to the face campaign CSV. No room/full-frame metrics are computed.

| Frame | Variant | PSNR | SSIM | LPIPS | Black skin pixels |
|---|---|---:|---:|---:|---:|
| 001029 | Previous curve | 31.5263 | 0.940505 | 0.079503 | 18 |
| 001029 | Measured pins + domain | 30.9765 | 0.936891 | 0.106913 | 38 |
| 001029 | Domain only | 31.5263 | 0.940505 | 0.079503 | 18 |
| 001033 | Previous curve | 21.0872 | 0.818142 | 0.288355 | 1433 |
| 001033 | Measured pins + domain | 22.1499 | 0.833980 | 0.293309 | 1100 |
| 001033 | Domain only | 21.0872 | 0.818142 | 0.288355 | 1433 |
| 001037 | Previous curve | 19.1997 | 0.632703 | 0.540476 | 2203 |
| 001037 | Measured pins only | 18.5444 | 0.620030 | 0.534244 | 2775 |
| 001037 | Measured pins + domain | 18.6536 | 0.640555 | 0.513944 | 2712 |
| 001037 | Domain only | 19.3682 | 0.653651 | 0.521395 | 2121 |

The measured-pin shape is rejected: it increases lateral missing skin on 001037
and worsens all three metrics on 001029. Its better LPIPS on 001037 does not
override the visible hole regression.

The isolated annotation correction removes the thin transverse cut on 001037
without moving any old retained proposal vertex or removing an old face. It
admits 158 more pre-carve triangles, of which 143 survive the unchanged guard.
Skin depth-hole pixels decrease 2141 -> 2074. On 001029/001033 geometry is
unchanged and the scored PNG hashes are byte-identical to the previous control.
This is a local assembly fix, **not** an artifact-free forearm reconstruction.

The main agent inspected all six native moving/train panels for each three-frame
control (12 panels total). Small lower-arm pinholes/source boundaries remain on
001029; 001033/001037 still have substantial side holes and coarse hand seams.
No candidate is promoted into the production video. The original production
meshes, all source data and the published 150-frame movie remain untouched.

Independent audits verify exact original production prefixes, exact preservation
of earlier retained curved vertices, and face-set differences. Each final
candidate also passes fresh 62-camera, two-offset ray checks under the
**color-qualified** guard. This does not mean the old depth-only guard passes:
it still reports 726/5172/12930 veto pixels on the domain-only controls.
All three measured-pin/domain candidates pass their separate assembly audits;
passing structural checks does not certify visual quality. 22 focused tests pass.

Artifacts:

- [Topology diagnosis](/mnt/data/dec5_forearm_residual_topology/001037)
- [Negative measured-pin comparisons](/mnt/data/dec5_forearm_measured_boundary_review)
- [Domain-only metrics and native comparisons](/mnt/data/dec5_forearm_annotation_domain_review)
- [Domain-only meshes and fresh audits](/mnt/data/dec5_forearm_annotation_domain_only)

![Isolated domain correction, moving camera](/mnt/data/dec5_forearm_annotation_domain_review/001037/moving_comparison.png)

## Insights

Continuous projection availability and discrete annotation availability are
different contracts. A subpixel sliver can falsely veto a whole triangle even
when two other cameras support it. Explicit unknown annotation borders fix that
specific inconsistency; they do not establish reliable depth for all skin.

Trusted boundary pins alone are insufficient to infer a good missing surface:
sparse anchors and a quadratic prior can improve one image statistic while
worsening the silhouette. Keep the annotation fix opt-in and reject this shape
replacement. Remaining side coverage needs stronger surface evidence, not a
larger blanket hole fill.

The separately accepted shot-level workaround remains the unchanged smooth
phase+30 path: 150 real actor times and a 31.51-degree camera excursion. It hides
the broad late cheek opening without narrowing travel or freezing the actress;
tiny late jaw flecks and other documented defects remain. See
[camera phase study](dec5_temporal_camera_phase.md) and
[current 150-frame texture/video result](dec5_early_texture_admission.md).
