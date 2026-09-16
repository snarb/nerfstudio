# DEC5 001123: fresh-pose local anatomical completion

## What was tested

Frozen local MHR completion on a fourth source time, with a fresh pose fitted
from that time's train RGB and measured depths. No previous fitted pose is
copied. The hypothesis is that a constrained anatomical patch can improve the
ragged underchin boundary without replacing the measured face.

Roots: [fresh fit](/mnt/data/dec5_mhr_transfer_001123),
[fit inputs](/mnt/data/dec5_mhr_transfer_001123_inputs),
[production-base completion](/mnt/data/dec5_mhr_completion_001123).
Specifications: [prior](/mnt/data/dec5_mhr_transfer_001123_spec.json),
[completion](/mnt/data/dec5_mhr_completion_001123_spec.json).

The existing `transfer_local_mhr_prior.py`, `run_local_mhr_completion.py` and
`review_local_mhr_transfer.py` are used unchanged. The recipe retains the
54 fitting / 8 reserved train split, measured conformance, 2px silhouette
tolerance, capped 100-step fit, neutral anatomical band 135..153, unsafe-parent
exclusion, .002/.003 surface/boundary locality, .00002 centroid clearance,
strict multiview support or certified local interpolation, and the final
62-camera × two pixel-offset measured-free-space veto. True held-out views
are excluded. Current cinematic cameras, texture profiles and masks remain
unchanged. The delivered 6K video is not modified by this experiment.

## Results

Status: **completed negative transfer for noticeable repair; not promoted**.

Fresh landmarks detect 57/62 train views. Five extreme views have no landmark
detection (A/B, A/C, A/D, B/B, D/A); all 62 still contribute measured-depth
evidence and pass input validation. There are 12,400 measured anchors,
including 1,600 reserved samples. The coarse neck group includes chin, neck
and clavicle samples, not exclusively the underside.

Final silhouette fitting reaches the 100-step cap, **not convergence**.
Reserved fixed-sample mean excess over the 2px tolerance falls from 18.300586
to .002095px; outside samples fall 2,212→36 on 11,593 samples. This is not an
image-quality metric or proof of useful hole repair. All 100 iterates replay
exactly. Final prior topology has 144 strict crossing pairs (16 new) and 87
normal changes over 90 degrees. Such parent facets are excluded locally.
The whole prior is rejected: actually inspected C/E, E/D, G/B and M/B clay
panels retain eye/nose distortions despite plausible lower-neck alignment.

| Candidate/admission stage | Count |
|---|---:|
| Original vertices / triangles retained exactly | 62,525 / 120,898 |
| Unsafe parents excluded in the band | 74 |
| Subdivided triangles | 740,864 |
| Local proposals before centroid gate | 155,137 |
| Centroid rejections | 4,668 |
| Raw proposals | 150,469 |
| Semantic-admitted proposals | 76,156 |
| Strict initial / final | 39,363 / 39,297 |
| Certified interpolation initial / final | 40,944 / 40,866 |
| Independently validated observed seeds | 2,280 |
| Certified vertices / queries | 22,201 / 41,263 |

Twelve focused tests pass (`test_local_mhr_transfer.py`,
`test_local_mhr_completion.py`, `test_local_mhr_frontier.py`). An initial orchestration call requested
`check` before `build`; it failed closed on the absent candidate request.
The failed log is retained. Correct order `build → check → admit → audit`
was then used; no thresholds or inputs changed. The active audit stdout was
moved outside the audited tree before inventory creation so a growing log
cannot invalidate its own inventory.

All 15 native HD diagnostic RGB renders completed: five independent CPU
workers (one view each, two computation threads), with three sequential mesh
arms per worker. This parallelizes only independent views, not the algorithm.
All five clay triplets, all five RGB triplets, and three localized interpolation
side-effect panels were actually viewed. No broad new facial fold or untextured
patch appeared, but the existing underchin fringe remained. These diagnostic
stills are not the requested 6K deliverable and do not replace it.

| View | Enclosed misses, baseline → either branch | New colored hits | Newly black | Nearer >.003, strict / interpolation |
|---|---:|---:|---:|---:|
| Current moving | 9→9 | 0 | 0 | 0 / 0 |
| F/E | 20→20 | 0 | 0 | 1 / 1 |
| M/B | 12→10 | 2 | 0 | 1 / 1 |
| C/E | 35→35 | 5 | 0 | 0 / 0 |
| G/B | 2→2 | 0 | 0 | 0 / 1 |

No old depth hits are lost. There are no untextured new hits or common-hit
farther changes >.003. Largest nearer change is .013736 at a tiny M/B
underchin/silhouette component; F/E's .003150 change is a shoulder-edge pixel,
and interpolation's G/B .003649 change is a chin-edge pixel. These are not
evidence of a successful broad underside repair. Clay counts differ for M/B
(15→13) and C/E (39→39), because current-policy named-train rendering applies
the existing target foreground mask; the final audit replays that distinction.

Examples: [current moving](/mnt/data/dec5_mhr_completion_001123/native_rgb/current_moving.png),
[C/E underside](/mnt/data/dec5_mhr_completion_001123/native_rgb/C004_E.png),
[F/E](/mnt/data/dec5_mhr_completion_001123/native_rgb/F004_E.png).

### Why further hole filling is not justified here

New opt-in [missing-ray attribution helper](../scripts/diagnose_local_mhr_frontier.py)
traces the same diagnostic rays through the whole prior, anatomical band,
safe band, locality gate, raw proposals, semantic gate, depth admission and
final native veto. It changes no meshes or thresholds. Evidence:
[/mnt/data/dec5_mhr_frontier_001123/result.json](/mnt/data/dec5_mhr_frontier_001123/result.json).

For the nine enclosed missing rays in the moving-view review crop, **even
the whole prior has zero intersections**. This rules out the later depth
admission as the reason these particular rays remain missing; it does not
prove that they represent missing anatomy. The four named train views have
zero unmasked-production misses within the review crop and independently
predicted skin confidence ≥230/255. The skin masks are nonempty: full staged
crop support is 107,990 / 181,373 / 212,868 / 196,908 pixels for C/E, F/E, G/B,
M/B, respectively; maxima are 247/247/248/248. The four native diagnostic
overlays were actually viewed. Their empty selected cohorts must not be
reported as positive repair results.

This test deliberately distinguishes confidently classified interior skin
from uncertain contour/hair pixels. It cannot certify the thin silhouette
as correct, and skin semantics are not ground truth. The observed remaining
artifact is chiefly ragged existing geometry at the edge in these views;
the append-only rule cannot remove or reposition that original fringe.
Full-face PSNR/SSIM/LPIPS are not computed: this is a geometry-transfer
diagnostic, not an independent held-out fidelity evaluation.

## Insights

The positive 001193/001195 puncture closure does not establish a general cure
for jagged silhouettes. Both 001083 and this fresh 001123 pose give negligible
visible benefit despite tens of thousands of admitted triangles. Stop expanding
this append-only fit to other poses as a generic contour repair. Next geometry
work must diagnose and correct the existing boundary surface, with multiview
foreground/free-space evidence and protected measured interior—not loosen
depth gates to force the prior into pixels with no established skin hole.

This is evidence changing the next experiment, not a successful new mesh for
the user. No production asset, previous sealed experiment, input image or
model default was changed. The full artifact-free dynamic-video goal remains
incomplete. Final geometry/render replay and inventory are bound in
[final_seal.json](/mnt/data/dec5_mhr_completion_001123/final_seal.json).
