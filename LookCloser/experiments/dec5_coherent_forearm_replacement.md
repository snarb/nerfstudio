# DEC5 coherent bounded forearm replacement

## What was tested

The secondary-reference append-only canary improved a few pixels but left a
fragmented surface. This control rebuilds the bounded H/A forearm domain as
one height field, rather than appending more hole grids. The existing shared
world quadric supplies the shape prior. Native depth pins must satisfy the
previous trusted three-other-view test; a residual Laplacian solve retains
them exactly. Source RGB, masks, poses and renderer settings are unchanged.

`study_coherent_forearm_replacement.py` runs only on 001037. A source face can
be replaced only when **all three vertices** project inside the accepted skin
domain and lie within 0.012 normalized depth of the replacement. All other
faces and all original vertices are preserved. New faces undergo the same
strict skin/extent checks and 62-camera contrastive depth guard, at integer
and half-pixel rays. This is an inferred surface, not measured anatomy.

The source is the previous local quadric control, **not the published video**.
Output: `/mnt/data/dec5_coherent_forearm_replacement`.
Review: `/mnt/data/dec5_coherent_forearm_replacement_review`.

## Results

The domain has 10,130 pixels and 174 trusted native pins. Solved depth residuals
range from -0.001171 to +0.001649. The bounded operation removes 17,682 old faces
and proposes 19,698 new faces. The guard removes 768, retaining 18,930; its next
full pass is clean. Independent replay verifies the pin solve, bounded removal,
raw assembly, preserved outside faces, and 124 final native ray checks.

| Fixed H/A train forearm ROI | PSNR | SSIM | LPIPS | Black RGB pixels | Rendered missing depth |
|---|---:|---:|---:|---:|---:|
| Previous local quadric | 20.16230 | 0.692228 | 0.426160 | 1775 | 1737 |
| Coherent replacement | 20.06015 | 0.716512 | 0.351966 | 1855 | 1765 |

These are train-reprojection diagnostics, **not held-out face metrics**. The
main face CSV is unchanged; no full-frame quality metric was computed.
Two fresh native RGB renders were inspected against the previous result,
including H/A GT and the moving-camera view. The forearm interior becomes
smoother, but side voids and wrist/hand defects remain. Black pixels increase,
despite better LPIPS. Verdict: **fail / not production accepted**. No transfer
to other times and no video rerender were started after this gate.

![Moving control](/mnt/data/dec5_coherent_forearm_replacement_review/001037/moving_comparison.png)
![Train GT and control](/mnt/data/dec5_coherent_forearm_replacement_review/001037/H004_A005_1210M6_comparison.png)

Index-connected component counts are previous 101, raw replacement 100, guarded
295; nonmanifold edges remain zero. Duplicated grid ring vertices mean that this
is not a count of physically detached objects. Still, the carving stage again
fragments the initially coherent grid. Passing its native-depth checks does
not establish a visually complete surface.

### Residual attribution

`diagnose_coherent_forearm_residual.py` re-raycasts raw and guarded meshes using
the renderer convention. The source-mask wrapper also clips a target when it
is a physical train camera; the diagnostic explicitly replays that rule before
checking the saved depth. It does **not** cause any of the missing pixels in
this fixed skin ROI. The first unwrapped full-image equality check failed and
is retained in the diagnostic log; production rendering was not changed.

Of 1,765 final depth misses, 1,385 already lack a raw surface; carving adds 380.
Ordered, mutually exclusive attribution at the actual half-pixel target rays:

| Cause | Pixels |
|---|---:|
| Native guard removes previously present surface | 380 |
| Grid or final semantic boundary | 172 |
| Original model/plane displacement bound | 217 |
| Fewer than two positive skin annotations | 770 |
| Available skin-annotation disagreement | 226 |
| No intersection / initial trusted-free veto / distance / target mask only | 0 |

![Native GT and residual classes](/mnt/data/dec5_coherent_forearm_replacement/001037/residual_attribution.png)

Annotation failure alone does not prove incorrect masks: an incorrect depth can
project real skin outside the correct annotation. A second **diagnostic-only**
probe searches 121 depths at 0.0005 spacing, +/-0.03 around the shared quadric,
with masks unchanged. It retains the full discrete feasible set, not a convex
interval that could conceal invalid depths. This search neither accepts a new
displacement bound nor applies the final 62-view guard or constructs a mesh.

| Residual class | Tested | Feasible within +/-0.01 | Feasible within +/-0.03 | No feasible tested depth |
|---|---:|---:|---:|---:|
| Model/plane bound | 217 | 217 | 217 | 0 |
| Fewer than two skin positives | 770 | 576 | 600 | 170 |
| Annotation disagreement | 226 | 224 | 224 | 2 |

The depth search changes H/A ray depth, whereas the earlier model/plane bound
is in G/A coordinates. Its first row must not be read as satisfying that old
bound; some points already satisfy mask checks at zero offset. Among the
annotation-failure groups, **800 of 996** have a feasible sample within 0.01;
824 within 0.03. The closest valid offsets for the two groups have medians
0.0010 and 0.0015. All three train-GT projection panels were inspected: feasible
samples mainly trace the skin side/cuff margins. They do not prove correct
depth, a coherent shape, successful texturing or recovered pixels.

## Insights

Rebuilding one smooth grid is insufficient while proposal eligibility is
decided on an imperfect fixed shape. The next justified test is a bounded
surface optimization using **feasible depth sets as constraints**, plus measured
depth pins and the unchanged final evidence checks. Blind mask dilation is not
supported: most annotation failures can be resolved by changing geometry with
the original masks. Conversely, hard pruning alone creates fragments again.

Sixteen focused tests pass, including new rejection tests for faces crossing
the replacement boundary, depth-displaced faces, missing depth and nonfinite
projections. Requests, raw/guarded meshes, RGB, diagnostics and native reviews
are retained with hashes. The published dynamic movie and previous studies
remain unchanged. This is progress in diagnosis, not completion of the requested
artifact-free video or general mesh repair.
