# DEC5 dynamic hard-texture color seams

## What was tested

A separate control on the **current** elevated dynamic movie, initially frame
001193. The old 000973 gradient-domain experiment failed to repair its geometric
lipstick/neck fragment; that does not establish whether the newer movie's large
face/neck color mosaics have the same cause. No frozen renderer, model defaults,
source exposure, camera profiles, mesh or published video is modified here.

`study_temporal_gradient_seams.py` reconstructs CPU train-RGB warps using the exact
current mesh, fixed camera profile, static subpixel warp, depth-footprint tests,
and the existing train-only foreground masks. Its selected RGB8 pixels must
reproduce the published image within one quantization level before correction.
Every selected source must pass the reconstructed visibility check. Held-out RGB
is never read. Geometry, source labels and missing-pixel support remain fixed.

The existing `hard_source_gradient_leveling.py` supplies seam gradients from one
train camera that sees both neighboring mesh points. Inside a source region,
the original source gradient remains the objective. A float64 screened Poisson
solve changes display RGB offsets; it is **explicit local color correction**,
not unchanged RGB reprojection and not source-detail averaging. It is currently
view-dependent, not a temporally validated static camera calibration.

The first 500x630 native portrait crop is `[200,740,700,1370]` on 001193. The crop
is only a diagnostic; it has not been composited into the movie. Full-frame and
adjacent-time validation are required before a video candidate is accepted.

## Results

Initial crop artifacts:
`/mnt/data/dec5_temporal_gradient_color_001193`.
The CPU reconstruction exactly matches selected published RGB8 pixels (maximum
error **0**, invalid selected pixels **0**). Runtime **32.81 seconds**. The solver
converged in 4,448 iterations, true relative residual `9.90e-10`.

![Native matched comparison](/mnt/data/dec5_temporal_gradient_color_001193/comparison.png)

The large forehead, cheek and neck mosaics substantially decrease. Existing jaw
holes remain. Hair becomes warmer/darker, and the solver creates clipping:
maximum absolute display offset **0.73562**, **2,908 clipped channels**, including
**383 newly black supported pixels**. Therefore the raw additive control is not
accepted as a final correction, despite its clear skin-seam improvement.

An independent audit verifies original RGB, source labels, depth, missing-pixel
preservation, hashes, and offset-to-output reproduction. Internal seam RGB-jump
median decreases **0.027451 -> 0.014379**; p90 **0.109804 -> 0.056209**. These are
continuity diagnostics, **not GT fidelity/PSNR/SSIM/LPIPS**. Within-source gradient
change has median approximately zero, p90 0.005229 and p99 0.027451; high-frequency
detail is not averaged but its gradients are not claimed to be exactly unchanged.

### Bounded display-gain control

`bound_temporal_gradient_offset.py` limits the proposed value of each channel to
0.5..2 times its original display value and preserves originally nonzero channels
after RGB8 quantization. This is a fixed explicit additional constraint, not
calibrated radiance or a changed global exposure. It uses no spatial averaging,
additional source selection, GT, geometric filling or learned image generator.

![Bounded native comparison](/mnt/data/dec5_temporal_gradient_color_001193_bounded/comparison.png)

The bounded crop retains the visible face/neck improvement and creates **zero**
new fully black supported pixels. The skin still has some color variation and
hair changes tone; no claim of exact GT color or temporal stability is made.
The gradient, bounded-gain and actual CUDA-equivalence tests pass: **8 passed**.
The full 001193 image and 001191/001195 actor extents were subsequently inspected
at native scale. All three retain the strong skin-mosaic reduction, though warmer
hair, the original jaw holes, ragged torso boundaries and some color differences
remain. Three nearby instants are not a full temporal-flicker test.

### Execution-only CUDA acceleration

The new opt-in `--solver-device cuda` keeps CPU warp reconstruction and the exact
native float64 solver equations, guidance, ridge and true-residual gate. It does
not downsample the image or replace the algorithm. Default remains CPU. The two
matched actor-extent controls include the entire rendered foreground; the omitted
outer rectangle contains only empty background.

| Time | CPU total | CUDA total | CUDA solver | End-to-end speedup | Maximum RGB8 difference |
|---|---:|---:|---:|---:|---:|
| 001191 | 425.60 s | 19.39 s | 5.30 s | 21.95x | 1, in 14 channels |
| 001195 | 450.61 s | 18.54 s | 4.48 s | 24.31x | 1, in 4 channels |
| 001193 full image | 1118.73 s | 30.43 s | 14.57 s | 36.76x | 1, in 3 channels |

Iteration counts are identical (6,112 and 6,736); maximum saved offset differences
are 1.16e-6 and 1.45e-6. This is verified numerical equivalence within RGB8
quantization, not byte identity. CPU/CUDA runs had concurrent machine load, so
timings are operational measurements rather than isolated hardware benchmarks.
Full-image 001193 also passes independent equivalence: both executions take
6,112 iterations, true residual 9.17e-10, maximum saved offset difference 1.01e-6.
All three CPU/CUDA pairs are terminal and independently audited.

Artifacts: `/mnt/data/dec5_temporal_gradient_full_001193_cuda_bounded` and
`/mnt/data/dec5_temporal_gradient_actor_{001191,001195}_cuda_bounded`.
The bound preserves zero newly black supported pixels in all reviewed candidates.
Published movie, geometry and source files remain unchanged.

### Additional poses and explicit native verdicts

Full-image 000899 and 000973 bounded controls were also inspected at native scale.
Both reduce skin mosaics. On 000899 the natural hand cast shadow remains; on
000973 the brown/orange rear lipstick patch remains. Crown fringe/notches,
warmer hair and existing unsupported geometry remain in these controls.
The unbounded proposals generate respectively 31 and 340 newly black supported
pixels; the bounded variants generate zero. Raw numerical audits and independent
bounded-output reproduction checks pass.

All five bounded canaries have hash-bound `visual_review.json` files with
**whole-image `fail`** despite improved color seams: this does not hide the
remaining geometric artifacts or declare novel-view GT fidelity. The five are
000899, 000973, 001191, 001193 and 001195; none is promoted to the published movie.
`audit_bounded_temporal_gradient_control.py` independently reproduces the output,
checks unsupported pixels and binds the explicit review to the inspected PNGs.

## Insights

The current broad color mosaic is at least partly a source-seam problem: it can
be substantially reduced without moving or filling a single triangle. This is
different from the original unsupported lipstick geometry and the remaining
black jaw holes. Improved color continuity cannot certify mesh recovery.

Next checks are full-frame composition, hair/lip detail, independent skin/color
fidelity where a real camera exists, and adjacent-frame stability. Do not expand
to a 150-time corrected movie on the strength of one cropped screenshot.
