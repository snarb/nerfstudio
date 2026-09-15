# DEC5: conservative weak-fringe removal and replacement

## What was tested

2026-09-15. Follow the [original crown support diagnosis](dec5_original_crown_support.md)
with a real, opt-in geometry edit. Test the same rule on actual frames 001083/001123,
over **all original head triangles**, not the hand-picked diagnostic triangle IDs or
train hair polygons. No held-out image or target pose selects the edit.

An original triangle is eligible for removal only when:

- all vertices lie in the fixed head band `x > -0.03`;
- at least six physical cameras classify **all four** sample locations (vertices and
  centroid) as clear background: each 9×9 native neighborhood contains no foreground
  pixel in the previously measured-refined mask;
- **every** sample has fewer than two corroborated train depth votes. Any sample with
  at least two protects the entire triangle. The previously diagnosed five-vote example
  is protected by this rule rather than discarded by a low median.

Compare four arms with the same original geometry and current fixed texture recipe:
production, the previous [.001 inset shell](dec5_inset_head_completion.md), removal-only,
and removal plus that shell. No original vertex moves. Remaining original triangles are
exact subsets. Shell additions remain an inferred prior, not new measurements or a raw
TSDF volume. All output meshes are separate; production artifacts/defaults are untouched.

Deleting old faces may reveal previously hidden shell faces. Therefore replacement reruns
the 62-camera, two-offset corroborated free-space guard; the previous shell audit is not
silently reused. A separate audit recomputes background neighborhoods using integral images
instead of the producer's maximum filter, recomputes depth votes, reconstructs both meshes
and performs 124 fresh native safety casts per replacement.

Eight fresh RGB controls cover removal/replacement at moving and exact native train poses
for both times; four previous production/inset pairs are reused with validated receipts.
Fixed camera profiles/exposure, hard-source selection, zero RGB warp and original texture
masks remain unchanged. The native diagnostic target alias avoids source-mask clipping of
the target raycast itself. Native GT is the same hash-bound fixed-profile image used before.

## Results

| Frame | Original head triangles | Clear-background candidates | Removed original triangles | Extra shell triangles pruned after removal |
|---|---:|---:|---:|---:|
| 001083 | 64,206 | 440 | 290 | 0 |
| 001123 | 62,939 | 546 | 309 | 1 |

Remaining original triangle coordinates are unchanged. Both replacement guards converge,
and independent replay passes all 248 fresh native checks. The newly revealed conflicting
shell triangle at 001123 illustrates why removal requires a fresh safety pass.

### Matched depth/RGB change diagnostics

| Frame / view | Removal vs production: lost depth / changed RGB | Replacement vs inset: lost depth / changed RGB | Combined vs production: new depth / lost depth |
|---|---:|---:|---:|
| 001083 moving | 353 / 790 | 353 / 791 | 16 / 353 |
| 001083 native train | 668 / 1,876 | 662 / 1,909 | 772 / 662 |
| 001123 moving | 462 / 1,142 | 462 / 1,143 | 43 / 462 |
| 001123 native train | 514 / 1,447 | 363 / 1,448 | 1,380 / 363 |

Lost depth is expected when removing a spurious foreground fragment, but is **not** itself
proof of a correct edit. Conversely, new depth from an inferred shell does not establish
anatomical accuracy. These are change diagnostics, not full-frame quality scores. No
PSNR/SSIM/LPIPS improvement or loss value is reported for these novel-view comparisons.

All three comparisons at both native poses have **zero new/lost depth and zero new/removed
black pixels in the frozen train face-skin regions**. This does not certify all face color,
the whole silhouette, or unseen/held-out views; it is a limited gap non-regression check.

The main agent inspected all 12 saved panels: removal and replacement crown comparisons
plus face/jaw comparisons for both views and times. Detached upper fringe is visibly less
conspicuous in the moving views, particularly the arch on the left of the crown. The side
train view still has a large opening and ragged/tan hair rim. The inset shell retains its
partial native-gap fill; it does not restore the full crown. No conspicuous new face/jaw
defect was observed in the inspected crops. Existing source/neck boundaries remain.

Verdict: **partial fringe improvement, not promoted to the 150-frame video**. It is a real
mesh edit rather than camera hiding, but it does not complete the requested broad repair.
No per-frame exception or hand-picked triangle deletion was used.

- [083 moving removal](/mnt/data/dec5_weak_fringe_replacement/review/001083/moving_removal_crown.png)
- [123 moving replacement](/mnt/data/dec5_weak_fringe_replacement/review/001123/moving_replacement_crown.png)
- [123 native residual opening](/mnt/data/dec5_weak_fringe_replacement/review/001123/native_unmasked_replacement_crown.png)
- [Native face/jaw comparison](/mnt/data/dec5_weak_fringe_replacement/review/001123/native_unmasked_jaw.png)
- [Independent geometry audit](/mnt/data/dec5_weak_fringe_replacement/geometry_audit.json)
- [Visual verdict](/mnt/data/dec5_weak_fringe_replacement/visual_review.json)
- [Experimental replacement mesh](/mnt/data/dec5_weak_fringe_replacement/001123/replace/mesh.ply)

Two focused tests verify protection by any supported sample and agreement between the
maximum-filter and integral-image background tests. All study workers completed normally.
The final artifact audit sealed and independently rechecked **438 SHA-256 hashes**;
the focused tests passed (2/2). The repository-wide pytest defaults request unavailable
xdist (`-n=4`), so the focused invocation explicitly clears `addopts` as shown below.
The saved `.ply` is an untextured geometry diagnostic; source EXRs and original meshes
were not deleted. Removing triangles affected only these new experimental copies.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/freeze_weak_fringe_replacement.py --check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python -m pytest -q -o addopts='' tests/test_weak_fringe_replacement.py
```

## Insights

1. Preserving every old face was retaining some visibly detached fringe. A cautious
   multiview background/depth test can trim part of it without losing native face-skin
   coverage. Removal and completion need separate controls; the shell alone did not trim it.
2. Whole-triangle protection is deliberately conservative. A triangle containing one
   supported sample can still include an erroneous extension. Subtriangle-level treatment
   is a possible next test, not justification to lower protection globally or per frame.
3. The large remaining crown gap is a surface-completion problem beyond this sparse trim.
   Future work must verify wider-angle/temporal behavior and held-out fidelity before any
   production rollout. The separately running larger-motion videos still use production
   geometry, not these two-frame candidates.

### Wider-path transfer check

The main agent subsequently raycast all three meshes on the **actual four new wide
camera paths** at both times (24 full-resolution CPU-only depth/clay renders).
Camera requests and baseline mesh hashes match the video campaign exactly. No GPU
worker slot, production mesh or video request was changed. These are eight isolated
time/pose checks, not complete temporal validation or new RGB renders.

| Frame / path | Removal: lost depth pixels | Replacement: newly covered pixels | Replacement: lost depth pixels |
|---|---:|---:|---:|
| 001083 diagonal | 382 | 337 | 352 |
| 001083 oval | 417 | 127 | 404 |
| 001083 left arc | 420 | 256 | 396 |
| 001083 right arc | 238 | 31 | 238 |
| 001123 diagonal | 583 | 45 | 583 |
| 001123 oval | 344 | 22 | 344 |
| 001123 left arc | 489 | 35 | 489 |
| 001123 right arc | 326 | 19 | 326 |

All eight three-arm head panels were actually inspected. The detached top arch is
reduced, especially at 001123, but the ragged outline remains. The refined right arc
still shows a side hair/temple opening. There is no conspicuous new broad face/neck
deformation in these clay comparisons. This does **not** establish texture quality,
absence of RGB seams, anatomical accuracy or improvement over held-out images.

Removal produces no new or nearer surface in all eight checks, as required by exact
triangle deletion. Replacement also brings an inferred shell in front of existing
depth at 1,841–2,309 pixels (>0.0001 normalized depth); it is not solely hole filling.
Consequently it still requires matched RGB/held-out checking before promotion.
The depth-change counts are diagnostics, not full-frame quality metrics.

The initial independent count replay exposed two float32 threshold-boundary pixels
in 001083 oval: subtraction-first and addition-first comparisons differ by one ULP.
The audit now reproduces the documented producer operation exactly; no mesh or
render was changed to satisfy it.

- [Wide-path audit](/mnt/data/dec5_wide_fringe_geometry/audit.json)
- [Eight-panel visual verdict](/mnt/data/dec5_wide_fringe_geometry/visual_review.json)
- [001123 left-arc geometry comparison](/mnt/data/dec5_wide_fringe_geometry/001123/left_high_arc/head_comparison.png)
- [001083 diagonal geometry comparison](/mnt/data/dec5_wide_fringe_geometry/001083/diagonal_sweep/head_comparison.png)

Replay with `scripts/screen_wide_fringe_geometry.py`, then
`scripts/audit_wide_fringe_geometry.py`. The latter verifies hashes, inventory,
24 finite depth arrays, recorded change counts and deletion monotonicity;
the final check passed and rechecked 64 SHA-256 bindings.
