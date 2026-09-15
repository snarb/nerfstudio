# Mid-sequence jaw completion and the remaining dark seam

## What was tested

Transfer the existing [close-boundary observed-neighborhood completion](dec5_close_boundary_completion.md)
to `000995`, and validate it together with the previously repaired `001193` and
`001195` on the actual wider `left_high_arc` video pose and native train H/C and
K/B poses. Geometry thresholds, original mesh prefix, production source masks,
fixed exposure/profiles and hard one-source unwarped RGB remain unchanged.
There are 13 new full-resolution renders and nine paired comparisons; five
production images are reused only after hash validation.

The new `000995` proposal adds 2,233 triangles, with 2,339 verified seeds and
1,550 certified vertices before final filtering. Native free-space checks remove
12 proposed faces; all 124 final camera/check combinations pass. Independent
audit recomputes 4,841 seed queries and 4,333 local certificates. Original mesh
vertices/triangles remain an exact prefix. This is inferred local completion,
not measured surface everywhere, and not welded/watertight repair.

The late meshes retain 2,007/1,989 additions. Their older evidence preparation
included one automatically measured camera-mask override; `000995` uses the
original masks. Consequently this is **not** proof of a uniform 150-frame mask
pipeline. No new PatchMatch, pose optimization or held-out data is used.

## Results

| Frame | View | Added depth pixels | Changed RGB pixels | Removed black RGB pixels |
|---|---|---:|---:|---:|
| 000995 | Moving | 1 | 3 | 1 |
| 000995 | H/C | 0 | 0 | 0 |
| 000995 | K/B | 6 | 23 | 6 |
| 001193 | Moving | 0 | 4 | 0 |
| 001193 | H/C | 0 | 5 | 0 |
| 001193 | K/B | 68 | 78 | 68 |
| 001195 | Moving | 0 | 1 | 0 |
| 001195 | H/C | 0 | 8 | 0 |
| 001195 | K/B | 43 | 46 | 44 |

These are diagnostic counts, not quality metrics. Every pair has zero lost
depth pixels and zero new black pixels. Nevertheless the moving-view gain is
negligible. The main agent inspected all nine native jaw comparison panels:
isolated gaps improve at late K/B, but the conspicuous thin dark under-jaw line
remains. None establishes the intended material broad repair. No full-head,
full-video, artifact-free or held-out metric pass is claimed; production is not
changed.

- [Mid-sequence K/B comparison](/mnt/data/dec5_midsequence_jaw_completion/000995/review/K004_B005_1210DS/head.png)
- [Late K/B native jaw](/mnt/data/dec5_midsequence_jaw_completion/001193/review/K004_B005_1210DS/jaw_native.png)
- [Visual review](/mnt/data/dec5_midsequence_jaw_completion/visual_review.json)
- [303-binding artifact seal](/mnt/data/dec5_midsequence_jaw_completion/artifact_manifest.json)

### The thin dark line is not a geometry hole

A fixed 8,611-pixel polygon was selected from real K/B train GT, then reused at
both late times. In both production and completed renders it contains **zero**
missing-depth pixels, zero absent-source pixels and zero exactly black RGB
pixels. This is a local diagnostic region, not a face metric mask or a complete
anatomical hole inventory.

Tracing darker-than-GT local ridge pixels through the actual target intersection
and source-ID buffer gives:

| Frame | Selected ridge pixels | From J004_C005_1210I4 | J/C near measured depth | J/C at a source boundary |
|---|---:|---:|---:|---:|
| 001193 | 110 | 108 | 93 | 96 |
| 001195 | 67 | 67 | 60 | 66 |

Near means depth residual within .0015 normalized units. None of these J/C
samples has the >.005 farther-depth discrepancy seen in the separate blue
lipstick-fin diagnosis. A NumPy replay of the exact single-source linear RGB,
fixed camera profile and display transform matches the output within **0/1
uint8 levels**, respectively. All four saved source witnesses were visually
inspected: J/C contains an actual strong under-jaw shadow; the target K/B view
has a softer/differently located transition. There is **no two-camera RGB
averaging** in this artifact.

- [193 actual J/C witness](/mnt/data/dec5_midsequence_jaw_completion/001193/seam_diagnosis/source_trace/case_01.png)
- [195 actual J/C witness](/mnt/data/dec5_midsequence_jaw_completion/001195/seam_diagnosis/source_trace/case_00.png)
- [Classification evidence](/mnt/data/dec5_midsequence_jaw_completion/001193/seam_diagnosis/classification.json)
- [Exact source replay](/mnt/data/dec5_midsequence_jaw_completion/001193/seam_diagnosis/source_trace/result.json)

## Insights

Local supported completion can fill tiny isolated gaps without a measured depth
regression, but does not materially repair this wider moving view. The prominent
late K/B line is transported source shadow on valid geometry. Filling more
triangles cannot by itself remove it. This attribution does not yet uniquely
separate source-label boundaries, residual surface/calibration error and shadow
registration; those require a matched rendering experiment.

Two untested renderer hypotheses follow from code inspection: graph-preferred
sources currently remain selected whenever valid, even if another source has
much better weight; and the target-angle prior uses optical-axis differences,
which can change under panning at an unchanged camera center. A point-to-camera
ray prior would be panning-invariant. Neither hypothesis is implemented or
promoted here. Cinematic trajectory experiments continue using unchanged
production geometry and RGB selection.

The focused transfer suite passes two tests: request/inventory preservation and
rejection of modified original geometry. Supervisor records cover all nine view
workers; all 13 renders are terminal and hash-verified. Retained meshes are not
serialized raw TSDF volumes.
