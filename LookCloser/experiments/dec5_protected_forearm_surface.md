# DEC5 protected-production forearm completion

## What was tested

The preceding constrained solve improved 001037 but damaged good geometry on
001029/001033 by replacing production triangles and subsequently carving their
replacement. This control changes the **composition rule**, not the fitted
shape, discrete solver, masks, camera calibration, exposure or depth thresholds.

`protected_production_grid.py` requires the published production mesh to be an
exact vertex/triangle prefix of the current local baseline. It preserves every
production triangle, permits bounded removal of earlier inferred additions
only, and creates new grid faces only around holes in the production raycast.
The adjacent sampled ring takes its depth from that protected mesh. The grid
does not overwrite already observed surface pixels. Ring sampling is not a
claim of topologically welded vertices or an anatomical ground truth.

The controller's `--preserve-production-surface` is opt-in and requires the
existing `--feasible-depth-constraints`. All three times use the same 16-pass
guard work limit, same 49 depth candidates, exact native pins, semantic rules,
extent limit and 62-view contrastive evidence checks. Model defaults and the
single-frame PatchMatch recipe are unchanged. No original/publication file is
modified. Root: `/mnt/data/dec5_protected_constrained_forearm`.

## Results

| Frame | Protected production triangles | Replaced earlier-prior triangles | Active hole pixels | Retained new triangles | Guard passes |
|---|---:|---:|---:|---:|---:|
| 001029 | 143776 | 2452 | 1103 | 2389 | 1 |
| 001033 | 144265 | 14728 | 7671 | 15133 | 4 |
| 001037 | 139053 | 17640 | 10023 | 18745 | 5 |

Independent audits reproduce the constrained solve, production-hole grid,
bounded prior removal and final assembly. All production prefixes are exact;
372 native ray checks pass. No nonmanifold edges are introduced. The raw grid
and later carving still have multiple index-connected components; this is not
an artifact-free/manifold-mesh certificate.

Same fixed **train H/A forearm ROI**, same hard-source RGB renderer:

| Frame / variant | PSNR | SSIM | LPIPS | Black RGB pixels | Missing rendered depth |
|---|---:|---:|---:|---:|---:|
| 001029 previous | 31.52622 | 0.940515 | 0.079498 | 18 | 18 |
| 001029 protected | 32.01306 | 0.945495 | 0.047098 | 3 | 3 |
| 001033 previous | 21.72905 | 0.836815 | 0.257388 | 1259 | 1241 |
| 001033 protected | 26.32301 | 0.892099 | 0.207159 | 289 | 275 |
| 001037 previous | 20.16230 | 0.692228 | 0.426160 | 1775 | 1737 |
| 001037 protected | 21.90890 | 0.758793 | 0.355989 | 966 | 910 |

All three metrics and both hole counts improve on all three times versus the
previous best local quadric baseline. These are **not held-out face metrics**
and do not modify the face campaign CSV. No full-frame quality metric is used.

001029 was rendered and visually checked first: the good forearm was preserved,
with fewer tiny black marks and a remaining mild color seam. Only then were
001033 and 001037 processed with identical parameters. All six candidate RGB
images are fresh renders. Native moving-view and GT/train comparisons were
inspected and retained in `/mnt/data/dec5_protected_forearm_transfer_review`.

![001029 protected control](/mnt/data/dec5_protected_forearm_transfer_review/001029/H004_A005_1210M6_comparison.png)
![001033 moving view](/mnt/data/dec5_protected_forearm_transfer_review/001033/moving_comparison.png)
![001037 moving view](/mnt/data/dec5_protected_forearm_transfer_review/001037/moving_comparison.png)

The 001029 local forearm gate passes with a known color seam. 001033/001037
show real coverage gains but still fail artifact-free acceptance because wrist,
hand and side-skin holes remain. The common method passes the **regression
gate for wider testing**, not production-wide rollout. No per-frame winner was
substituted into the video. Upper 1,400 image rows are RGB-identical in all six
matched views, so this local control neither improves nor damages the head.

### Regression cause rechecked

New geometry misses over previously visible **production** faces are now zero
at every time (previously 1000/553/5 in the rejected replacement control).
There remain 1/6/193 newly missing pixels over earlier inferred additions,
while the overall missing area decreases. On 001037, 190 of those losses arise
in final carving. Protecting production removes the observed regression
mechanism; it does not yet make every replacement prior coverage-monotonic.

Twenty-four focused tests pass, including no-hole/no-replacement, exact sampled
boundary depth and production-prefix protection cases. Requests, evidence,
raw/guarded meshes, metrics, native reviews and stage logs are hash-bound.

## Insights

Preserving reliable existing geometry is essential when adding a shape prior.
The measured-pin solve and depth guard alone did not provide that guarantee.
The new composition rule transfers the coverage benefit to all three test
times without the previous catastrophic regression. It should remain the
starting point for further completion tests, not a return to wholesale surface
replacement or per-frame parameter tuning.

Remaining wrist/hand holes and losses of earlier inferred surface still need
work. Additional temporal frames need their own valid observations/annotations;
the three-time result does not establish quality across 150 times. The published
dynamic-camera/dynamic-actor video is unchanged and still contains known
artifacts. The full user objective remains unfinished.
