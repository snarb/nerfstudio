# DEC5 feasible-depth constrained surface: gain fails temporal transfer

## What was tested

The previous diagnosis found that many annotation conflicts could be resolved
by changing depth without changing masks. `feasible_forearm_surface.py` tests
49 residual depths at 0.0005 spacing within +/-0.012, using both original
integer-admission and final-vertex mask conventions. The original G/A plane
bound 0.01, two positive annotations, no available disagreement, trusted-free
veto and 100-pixel support distance remain. Bounds apply to the solved points,
not solely the initial shape guess.

`discrete_surface_constraints.py` minimizes a residual smoothness/prior objective
by checkerboard coordinate descent. Each update stays in its discrete feasible
set; gaps between feasible depths are not silently filled. Valid native pins
remain exact. Trusted pixels conflicting with these constraints are excluded,
not moved to a convenient depth. Convergence is coordinatewise, not a claim of
global optimality. This is a shape prior, not new measured anatomy.

The opt-in `--feasible-depth-constraints` branch leaves existing defaults alone.
Initial root: `/mnt/data/dec5_constrained_forearm_surface`. The eight-pass
native guard exhausted its budget on 001029; the result remains failed there.
We reran **all three frames** with a common 16-pass work limit in
`/mnt/data/dec5_constrained_forearm_surface_guard16`, without changing geometry
thresholds. `--guard-max-rounds` defaults to eight. The old producer is archived
in the initial root's `config/`; failed/earlier results are retained.

## Results

| Frame | Domain pixels | Exact native pins | Solve sweeps | Guard passes through clean pass | Final new triangles |
|---|---:|---:|---:|---:|---:|
| 001029 | 22678 | 15981 | 9 | 12 | 41936 |
| 001033 | 17088 | 6470 | 6 | 8 | 31749 |
| 001037 | 11435 | 187 | 5 | 5 | 21300 |

All three final independent audits replay the feasible sets, solve, bounded
face removal and assembly, then pass 124 native ray checks each. Nonmanifold
edge counts remain zero. Index-connected components increase after carving;
this still does not establish a coherent or anatomically correct surface.

Fixed **train H/A forearm ROI**, unchanged hard-source renderer and GT:

| Frame / variant | PSNR | SSIM | LPIPS | Black RGB pixels | Missing rendered depth |
|---|---:|---:|---:|---:|---:|
| 001029 previous | 31.52622 | 0.940515 | 0.079498 | 18 | 18 |
| 001029 constrained | 21.20350 | 0.764674 | 0.440045 | 1007 | 1001 |
| 001033 previous | 21.72905 | 0.836815 | 0.257388 | 1259 | 1241 |
| 001033 constrained | 21.21483 | 0.798014 | 0.323681 | 836 | 811 |
| 001037 previous | 20.16230 | 0.692228 | 0.426160 | 1775 | 1737 |
| 001037 constrained | 21.95092 | 0.761920 | 0.339919 | 957 | 900 |

These are not held-out face scores or full-frame metrics. No main campaign CSV
was changed. Six candidate images cover moving and fixed train views at all
three times: four are fresh transfer renders; two 001037 renders are reused
only after exact mesh bytes and unchanged rendering inputs were verified.
The initial canary comparison also retains the unsuccessful unconstrained
coherent variant. All six final native comparison panels were actually viewed.

001037 has a real local coverage gain, but still has wrist/hand holes and color
seams. 001029 acquires extensive black speckling over previously good forearm
skin. 001033 fills part of the old void but adds speckling and worsens all three
RGB scores. All three verdicts are **fail for artifact-free acceptance**; the
general method is rejected for production because of the transfer regressions.
No per-frame choice of winner and no full-video rerender were applied.

![001029 regression](/mnt/data/dec5_constrained_forearm_guard16_review/001029/H004_A005_1210M6_comparison.png)
![001033 transfer](/mnt/data/dec5_constrained_forearm_guard16_review/001033/moving_comparison.png)
![001037 local gain](/mnt/data/dec5_constrained_forearm_guard16_review/001037/moving_comparison.png)

### What caused the transfer regression

`diagnose_constrained_surface_regression.py` verifies the original production
mesh prefix, identifies which removed triangles came from it versus earlier
inferred additions, and raycasts old/raw/guarded geometry in the fixed train ROI.
Counts below describe newly missing geometry, not RGB or target-mask clipping.

| Frame | Removed production triangles | Removed earlier-prior triangles | New holes over old production / prior | Created during assembly / carving |
|---|---:|---:|---:|---:|
| 001029 | 3157 | 2452 | 1000 / 0 | 79 / 921 |
| 001033 | 1618 | 14728 | 553 / 4 | 20 / 537 |
| 001037 | 437 | 17640 | 5 / 191 | 5 / 191 |

Thus all 1,000 newly missing pixels in 001029 were previously covered by
production geometry. The primary failure is not merely missing data: the
replacement discarded an existing surface, then carving perforated its
replacement. Exact sparse/native depth pins did not protect that surface as
rendered from other rays. On 001037, where the replacement mostly concerns prior
geometry, the tradeoff is different. This explains why one good canary did not
generalize; it is not a justification for a hand-picked per-time exception.

### Additional held-out preflight

Two extra, never-trained eval camera GTs (J/D and L/B) were inspected at 001037
solely to check whether they could validate the full forearm. J/D clips most
of it at the image edge; L/B does not provide a useful full forearm view.
No predictions or metrics were generated for them. They did not enter geometry,
camera profiles, source selection or texture. The original F/B campaign protocol
is unchanged. Files and receipts:
`/mnt/data/dec5_constrained_forearm_additional_heldout/001037`.

## Insights

Feasible-depth constraints help the proposed missing surface, but wholesale
replacement is the wrong composition rule. The next justified experiment must
protect existing production geometry and apply the constrained prior only to
missing regions / earlier inferred additions, with explicit boundary joining
and the same native evidence checks. Do not weaken the checks or keep only the
successful frame. The requested artifact-free video and improved mesh remain
unfinished; the published moving-camera/dynamic-actor movie is untouched.

Twenty focused tests pass, including discrete-gap preservation, exact pins,
non-increasing objective, disconnected grid handling and invalid feasible sets.
Default model/single-frame recipes are unchanged. Final reviews, provenance,
six comparison panels, meshes and compact logs are frozen with hashes.
