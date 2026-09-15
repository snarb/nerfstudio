# Surface-ray and relative hard-source retention controls

## What was tested

The cinematic actual-train ending bypasses mesh defects; it does not satisfy the
underlying reconstruction objective. This follow-up tests the transported jaw
shadow identified in [the mid-sequence diagnosis](dec5_midsequence_jaw_completion.md).
The preceding goal turn made concrete progress by publishing four verified
cinematic presentations, but did not repair the mesh or achieve the broad goal.

Two opt-in source-selection hypotheses, without changing geometry, masks,
incidence exponent (2), source RGB, fixed color profiles/exposure, registration
(zero), visibility tests, source sampling, or graph smoothness:

- Replace source/target optical-axis angle with the angle of rays from each
  reconstructed point to the respective camera centers, using the same 4°
  Gaussian width and 12% post-prior source admission threshold. This is
  invariant to camera panning at an unchanged center for the same 3D points.
- Retain a valid graph-preferred source only if its pixel quality is at least
  50% of the best currently visible source. Otherwise select that best source.
  RGB is still taken from exactly one camera, never averaged.

Three arms isolate relative retention alone (`axis_relative`), ray angle alone
(`ray`), and both (`ray_relative`). Initial matched native K/B views at `001193`
and `001195` reuse the original production mesh, not the attempted completion.
The exact compiled renderer text and dependency hashes are pinned in each
request. Existing production renderer/model defaults are unchanged. The new
`ray_source_control.execution_sha256` or `retention_transfer.execution_sha256`
records the executed experimental implementation; inherited implementation
fields describe the parent, not the new implementation.

## Results

All six initial render workers exited normally. Each new target depth array is
exactly equal to its baseline. None adds a black RGB pixel. The fixed GT-defined
jaw polygon contains 8,611 pixels. The previously recorded dark-ridge diagnostic
requires prediction luma more than 20/255 below GT and more than 15/255 below
its own 5×5 median. This is a local artifact diagnostic, **not a face metric**.

| Time | Production ridge pixels | Relative retention | Ray angle only | Both |
|---|---:|---:|---:|---:|
| 001193 | 110 | 29 | 104 | 29 |
| 001195 | 67 | 16 | 67 | 15 |

Native jaw panels were visually inspected at 1:1. Relative retention makes the
continuous dark line substantially less prominent but leaves a thin dotted
residual. Ray angle alone does not materially change it. No hole repair or
artifact-free approval follows from this result. Full-head/actor overview shows
the existing rough hair/shoulder boundaries unchanged in character.

- [193 native jaw comparison](/mnt/data/dec5_surface_ray_source_prior/001193/review/jaw.png)
- [195 native jaw comparison](/mnt/data/dec5_surface_ray_source_prior/001195/review/jaw.png)
- [Initial numerical comparison](/mnt/data/dec5_surface_ray_source_prior/comparison.json)
- [Initial supervisor checks](/mnt/data/dec5_surface_ray_source_prior/checks.jsonl)

Transfer checks use three actual moving views (`000995`, `001193`, `001195`),
the earlier native `000995` K/B view, and an existing frozen held-out face
benchmark at `001193`. This repeatedly consulted benchmark is a regression
check, not a fresh blind evaluation. All transfer inputs/thresholds are frozen
before scoring; held-out RGB is unavailable to prediction construction.

All ten transfer renders also exited normally, preserving target depth exactly
and adding zero black pixels. Native-resolution moving/head panels were reviewed
for all five cases; held-out GT comparisons were reviewed for both candidates.

| Frozen held-out face, 001193 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Production reference | 30.85838 | 0.943257 | 0.072251 |
| Relative retention only | 30.84649 | 0.943220 | 0.072204 |
| Ray angle + relative retention | 30.36706 | 0.943975 | 0.061146 |

Relative retention alone is nearly neutral on this face benchmark and changes
2,035/3,334/3,286 RGB pixels in the three moving views. Combining the ray prior
changes 201,002/13,753/12,554 pixels, respectively. In the early moving frame
the lips look softer under the ray prior; that remains a visual sharpness concern
despite the better LPIPS at a different, late held-out time. The early native
view still has a clear crown hole and the moving early frame still has a small
lipstick/hand membrane. Neither control repairs those geometry defects.

**Decision:** keep relative retention as a promising local rendering control;
do not promote either arm to the full video yet. Ray-angle plus retention has
a measured face-fidelity tradeoff, not an unqualified quality win. The existing
production movies and meshes remain unchanged. No temporal flicker claim is
made from these sparsely sampled time instants.

- [Moving lipstick time, 000995](/mnt/data/dec5_surface_ray_source_prior/transfer/000995_moving/head.png)
- [Moving late head, 001195](/mnt/data/dec5_surface_ray_source_prior/transfer/001195_moving/head.png)
- [Held-out GT / production / ray-retention](/mnt/data/dec5_surface_ray_source_prior/transfer/heldout/ray_relative_heldout.png)
- [Face-only metrics](/mnt/data/dec5_surface_ray_source_prior/transfer/heldout/metrics.json)

## Insights

The dominant tested mechanism is unconditional retention of a valid graph
source, even when another source has much better pixel quality—not averaging
and not principally optical-axis versus surface-ray angles. At traced late
K/B jaw points the offending J/C prior is already similar under both angle
definitions (approximately 0.0453 versus 0.0422; own K/B weight is 1).

This establishes a local texture improvement and a remaining residual, not a
general temporal improvement or a repaired surface. Relative retention can
increase source switching, so moving-view and face-fidelity checks must precede
any production promotion. True missing geometry requires a separate repair.

Four focused tests verify rigid-motion invariance, own-center preference,
invalid geometry rejection, hard RGB/visibility preservation, baseline behavior
at zero retention threshold, and prior-before-admission order.
