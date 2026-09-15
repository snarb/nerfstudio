# DEC5: local silhouette-envelope head completion

## What was tested

2026-09-15. The [crown stage attribution](dec5_crown_completion_gap.md) showed
that admitting more of the same Poisson shell did not repair the visible crown.
Construct a **different surface in 3D** from all 62 real train-camera silhouettes,
then use measured depth to reject false additions. This is an inferred envelope,
not measured TSDF surface or a new trained model. No existing defaults change.

The same recipe applies to actual times `001083` and `001123`:

- Reuse the frozen train-only foreground masks refined by measured witnesses.
  Do not dilate them further; use a one-pixel inward signed-distance margin.
- CUDA evaluates the minimum signed silhouette distance on a .00025 normalized
  grid in the existing head bounds, padded by .003. At least three available
  cameras are required; all available negative silhouettes constrain occupancy.
- Explicit 64-bit camera availability distinguishes unknown FOV from background.
  Reject marching-cubes faces at box caps and cells with changing camera sets.
- Keep additions only within .006 of both the original mesh and its boundary,
  with all vertices x > -.03 and maximum edge .0015. Require all triangle vertices
  plus centroid to have >= 2 mask supports and zero available outside votes.
  No nearest-original-facet normal constraint is used for this new envelope.
- Append the surviving faces without changing original vertices/triangles.
  No target camera, target RGB, manual screen polygon, or current movie framing
  constructs this geometry. Texture source masks stay unchanged.
- Prune only additions that violate independently corroborated measured free
  space: depth separation > .003, three other agreeing physical cameras, native
  ray offsets 0/.5, all 62 train cameras. Repeat until no removals remain.

The first eight-round budget expired with 2/1 removals in the final round.
Those attempts remain failed and immutable. A separate continuation increases
the **common iteration budget**, not a depth threshold, to 16 total rounds.
It admits no new triangles. Convergence occurs at 13/9 total rounds, followed
by another 124 fresh native-ray checks per mesh. The explicit overlay uses
hash-verified symlinks for original proposal data, not copied/stale workspaces.

Roots:

- `/mnt/data/dec5_head_silhouette_completion`: proposals, CPU replays, rejected
  eight-round guards, logs and tests.
- `/mnt/data/dec5_head_silhouette_depth16`: converged guards, native/old-moving
  RGB comparisons, visual review and final integrity manifest.
- `/mnt/data/dec5_head_silhouette_cinematic`: four matched RGB controls at the
  **actual saved poses/lenses of both delivered wide-spiral movies**.

RGB uses the frozen incidence-2, zero-registration, hard-source renderer and
fixed camera profiles/exposure. No source averaging or generated RGB. Camera,
lens, frame identity and source settings match each baseline exactly. Changing
geometry can still change texture-source graph labels on existing surfaces;
unchanged renderer settings do not mean all old RGB pixels remain identical.

## Results

| Frame | Raw hull triangles | Local proposals | Mask-admitted | Depth-guarded additions | Total guard rounds |
|---|---:|---:|---:|---:|---:|
| 001083 | 536,938 | 230,137 | 211,749 | 177,666 | 13 |
| 001123 | 554,040 | 266,781 | 246,478 | 204,570 | 9 |

Unknown-domain filtering rejects 10,752/16,958 raw faces; box-edge filtering
rejects 7,598/8,135 (categories can overlap). Most retained faces overlap or
sit behind existing geometry; triangle counts are not a visual-quality score.

Post-guard matched RGB against the original production mesh:

| Frame / view | New depth pixels | Lost depth pixels | Changed RGB | Black removed / introduced |
|---|---:|---:|---:|---:|
| 001083 old moving | 142 | 0 | 8,603 | 247 / 11 |
| 001083 native train | 2,544 | 0 | 4,879 | 2,610 / 10 |
| 001123 old moving | 518 | 0 | 10,897 | 567 / 4 |
| 001123 native train | 3,178 | 0 | 5,949 | 3,214 / 0 |

Within the previously frozen coarse native crown query, missing depth falls
from 6,043 to 4,191 at 001083 and from 4,038 to 1,358 at 001123. The full hair
query changes from 8,055 to 6,126 and 5,146 to 2,242. These polygons contain true
background/strand gaps: these are coverage diagnostics, **not anatomical error
rates**, and no face/full-frame PSNR, SSIM, LPIPS or loss is computed here.

Visual inspection of four crown and four jaw comparisons finds a **substantial
partial native-view crown improvement**, especially at 001123: the central
black opening acquires hair texture. It improves more than the preceding inset
prior. Smaller holes, detached/rough upper fringe and tan/brown rim remain.
The old moving views improve less. No conspicuous new cheek/jaw shape defect
was seen in these inspected crops; existing fine source-color lines persist.

Transfer to exact current cinematic poses, before the real-train ending:

| Shot / frame | New depth pixels | Changed RGB | Black removed / introduced |
|---|---:|---:|---:|
| Look-at / 001083 | 433 | 39,861 | 737 / 9 |
| Look-at / 001123 | 522 | 10,194 | 740 / 4 |
| Free / 001083 | 412 | 33,082 | 690 / 5 |
| Free / 001123 | 520 | 10,188 | 739 / 5 |

All four cinematic comparisons were inspected at native resolution. The
001083 top arch/opening and strong brown fringe remain in both shots. At
001123 the crown is partly outside the **unchanged** framing; the residual
face/neck lines and rough outline persist. No strong new macrogeometry defect
was seen, but these are **not material repairs of the cinematic crown opening**.
No full-150 video or production mesh is replaced. Native improvement is not
presented as successful temporal or held-out validation.

- [001123 native crown: GT / original / inset / silhouette](/mnt/data/dec5_head_silhouette_depth16/review/001123/native_unmasked_crown.png)
- [001083 native crown comparison](/mnt/data/dec5_head_silhouette_depth16/review/001083/native_unmasked_crown.png)
- [001123 native face/jaw preservation control](/mnt/data/dec5_head_silhouette_depth16/review/001123/native_unmasked_jaw.png)
- [Actual free-spiral 001083 comparison](/mnt/data/dec5_head_silhouette_cinematic/review/wide_spiral_free/001083_head.png)
- [Actual look-at 001123 comparison](/mnt/data/dec5_head_silhouette_cinematic/review/wide_spiral_lookat/001123_head.png)
- [Visual verdict](/mnt/data/dec5_head_silhouette_depth16/visual_review.json)
- [Experimental geometry, untextured](/mnt/data/dec5_head_silhouette_depth16/001123/guarded/mesh.ply)

Three tests cover CPU/CUDA field equivalence, 62-camera availability bits and
unknown/known transition rejection. Independent CPU bilinear replay samples
4,096 grid locations per frame, including near-surface locations; maximum
field discrepancy is 0.000323/0.000334 pixels. Audits reconstruct marching cubes,
locality, masks and exact triangle assembly, replay eight proposal raycasts,
and repeat all 248 final native depth checks. All eight RGB workers' frame jobs
completed; four two-render processes exited normally. The initial geometry
producer's `inf-inf` diagnostic warning refers to excluded ray misses; it is
not an OOM or non-finite foreground render. No failed convergence attempt is
silently relabeled complete.

The grid-field GPU stage alone took 1.56/1.43 seconds for 14.36/15.94 million
points; this excludes masks, I/O, extraction, guards and RGB. It is not an
end-to-end reconstruction timing. Existing retained depth/mask data are reused.
Process/GPU/disk observations live in `dec5_head_silhouette_depth16/checks.jsonl`.
All source, code, geometry, render and review bindings are sealed and rechecked.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/seal_head_silhouette_completion.py check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/pytest -q -o addopts='' tests/test_head_silhouette_completion.py
```

## Insights

1. A spatial silhouette-constrained surface fills more of the native crown
   than admission-only changes to the same Poisson prior. This is real but
   view-dependent partial progress, not a sufficient production repair.
2. The envelope must be checked against measured depth: the raw candidate
   changed tens of thousands of existing visible depths, and the guard removed
   34,083/41,908 additions. Silhouette consistency alone is not true geometry.
3. Preserving all original triangles also preserves wrong fringe. Additional
   surface behind it cannot remove projecting brown islands or source seams.
   Next distinguish the surviving fringe's geometric and source-color evidence
   in the actual cinematic view before another completion or deletion rule.
   Do not enlarge the view crop or hide the residual opening to claim a repair.
