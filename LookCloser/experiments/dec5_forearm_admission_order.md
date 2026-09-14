# DEC5 forearm: evaluate shape before admission

## What was tested

The preceding [depth-observability diagnosis](dec5_forearm_depth_observability.md)
found that planar projections rejected candidates before the inferred curved
surface was tested. This opt-in control changes the geometry queried by initial
admission, not the camera calibration, masks, texture sources or exposure.
It retains the existing two-skin-view rule, available-view disagreement veto,
trusted free-space veto, 100-pixel anchor-distance limit, raw grid triangulation,
10-pixel boundary-conditioned curvature, final semantic/extent checks and
62-camera/two-offset contrastive depth/color guard. Original production geometry
is preserved exactly. The quadric is an inferred local prior, not measured anatomy.

`study_forearm_production_delta.py --admission-shape plane` must exactly replay
the former accepted bitmap, raw mesh arrays and final mesh bytes. `quadric`
queries the same policy at the inferred depth before making that raw grid.
The normalized 0.01 displacement limit applies to each candidate. An initial
implementation incorrectly stopped 001037 because an unused candidate exceeded
the bound; its failed workspace is retained separately. No bound was increased.

One additional matched control, `--observed-ring-feather`, blends towards the
old plane only near actual old-depth boundary samples instead of every mask
boundary. This keeps old ring vertices exact but is **rejected** below.
No defaults or production video have changed.

## Results

All three plane controls reproduce the former transferred and guarded PLYs
byte-for-byte. The same shape-first rule was run on all three times:

| Frame | Old/new accepted points | Newly admitted / removed | Final added triangles, old/new |
|---|---:|---:|---:|
| 001029 | 1136 / 1136 | 0 / 0 | 2452 / 2452 |
| 001033 | 7585 / 7627 | 51 / 9 | 14732 / 14743 |
| 001037 | 8861 / 9776 | 915 / 0 | 16606 / 17656 |

On 001037, 400 otherwise eligible candidates exceed the unchanged 0.01 bound.
The largest selected displacement is 0.0099824. The previous point-only probe's
1319 candidates used different annotation-domain lookup; it is not the number
of recovered mesh vertices or image pixels in this matched control.

Scores use the same fixed manual **train forearm-skin ROI** in
H004_A005_1210M6, including black holes. These are neither held-out face metrics
nor full-frame metrics. The main face campaign CSV is unchanged.

| Frame | Initial admission | PSNR | SSIM | LPIPS | Black skin pixels |
|---|---|---:|---:|---:|---:|
| 001029 | Plane | 31.52622 | 0.940515 | 0.079498 | 18 |
| 001029 | Quadric | 31.52622 | 0.940515 | 0.079498 | 18 |
| 001033 | Plane | 21.73409 | 0.835507 | 0.256374 | 1260 |
| 001033 | Quadric | 21.72905 | 0.836815 | 0.257388 | 1259 |
| 001037 | Plane | 19.64523 | 0.675119 | 0.440426 | 1992 |
| 001037 | Quadric | 20.16230 | 0.692228 | 0.426160 | 1775 |

Native moving/train comparisons were actually inspected at all three times.
001029 is identical; 001033 is essentially unchanged with severe side cavities;
001037 gains modest side coverage but retains holes, a rough lateral sliver,
texture seams and coarse wrist/hand geometry. **Partial improvement only; no
production promotion or isolated frame replacement.**

![001037 moving comparison](/mnt/data/dec5_forearm_admission_order_transfer_review/001037/moving_comparison.png)

### Boundary-feather control: rejected on 001037

| Admission / feather | PSNR | SSIM | LPIPS | Black skin pixels |
|---|---:|---:|---:|---:|
| Plane / all edges | 19.64523 | 0.675119 | 0.440426 | 1992 |
| Plane / observed ring | 18.81549 | 0.664602 | 0.431790 | 2633 |
| Quadric / all edges | 20.16230 | 0.692228 | 0.426160 | 1775 |
| Quadric / observed ring | 19.50362 | 0.688312 | 0.395653 | 2162 |

Both observed-ring variants enlarge lateral cavities in directly inspected RGB,
despite improving LPIPS. They pass geometric safety checks but fail the visual
comparison; neither was propagated to neighboring times. More retained triangles
and lower LPIPS do not override worse visible coverage.

Independent audits replay admission arrays, boundary-conditioned deformation and
transferred mesh arrays; verify unchanged production geometry and final semantic
limits; and bind fresh 124-ray guard checks with zero qualified veto pixels.
38 focused tests pass. All workers finished normally. Provenance snapshots and
the final retained-output manifest preserve both successful and rejected controls.

Artifacts:

- [Matched plane meshes](/mnt/data/dec5_forearm_admission_plane_bounded)
- [Shape-first meshes and audits](/mnt/data/dec5_forearm_admission_quadric_bounded)
- [Three-time RGB, metrics and review](/mnt/data/dec5_forearm_admission_order_transfer_review)
- [Rejected plane/ring comparison](/mnt/data/dec5_forearm_plane_ring_review)
- [Rejected quadric/ring comparison](/mnt/data/dec5_forearm_quadric_ring_review)
- [Retained-output manifest](/mnt/data/dec5_forearm_admission_order_manifest.json)

Replay: use `prepare --frame FRAME --root NEW_ROOT --curved-anchor-root
/mnt/data/dec5_forearm_multiview_anchors --boundary-conditioned
--photometric-free-space --known-annotation-domain --witness-rgb-limit .12
--witness-comparison-margin .01 --admission-shape quadric`, then the existing
`study_confidence_boundary_completion.py render --output NEW_ROOT --frame FRAME`.
Use fresh roots; the requests reject changed source/configuration hashes.

## Insights

Premature planar admission is one real contributor, not the fundamental solution
to every hole. The remaining patch is a first-reference 2.5D surface bounded by
manual evidence; it does not cover the entire wrist/hand or reconcile every old
surface boundary. Weakening those constraints or changing the feather domain
can increase apparent surface area while worsening novel-view holes.

The authorized [camera-phase workaround](dec5_temporal_camera_phase.md) remains
separate: the published early-texture movie has 150 real actor times, 31.5-degree
view span and about 384 by 154 pixels of fixed-landmark screen travel. It hides
the conspicuous cheek cavity without freezing the camera, but forearm, crown and
lipstick defects remain. A hidden hole is not repaired geometry. The current
local results do not justify another full-sequence render or an artifact-free claim.
