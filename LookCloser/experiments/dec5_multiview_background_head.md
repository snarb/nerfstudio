# DEC5: multiview background pruning of weak head triangles

## What was tested

2026-09-15. The previous goal turn delivered two audited wide-spiral movies
(`5ee91919`); that was progress toward camera motion, not proof of repaired
geometry. This experiment resumes the [original crown support diagnosis](dec5_original_crown_support.md).

Apply one uniform rule to all original triangles whose three vertices have
normalized head coordinate x > −.03, at both001083 and001123. Delete only when
at least12 physical train cameras classify **all seven samples** (vertices,
edge midpoints, centroid) as clear background in9×9 neighborhoods, AND every
sample has fewer than two consistent PatchMatch observations. Any sample with
two observations protects the whole triangle. Depth agreement uses .001 units,
the existing1.5px roundtrip gate and1-degree parallax. Missing depth alone
never justifies deletion. Foreground masks are fallible train-RGB evidence.

Original retained vertex coordinates are exact; no smoothing, component cleanup,
heldout view, manually selected crown ROI, generated pixels or color change.
Test pruning alone and pruning plus the existing guarded .001 inset shell.
The latter shell has **not** rerun its free-space guard after pruning exposes
new parts; it is a diagnostic, not a certified combined mesh.

Opt-in code: `prune_multiview_background_head.py`,
`study_multiview_background_head.py`, `seal_multiview_background_head.py`.
Output root: `/mnt/data/dec5_multiview_background_head`.

## Results

| Frame | Head triangles | Clear-background candidates | Protected by a two-view sample | Removed |
| --- | ---: | ---: | ---: | ---: |
|001083|64,206|200|43|157|
|001123|62,939|255|78|177|

Independent replay recomputed every mask-window vote with summed-area tables
instead of dilation, recomputed all eligible depth votes/references, and checked
the exact retained triangle assembly and coordinates. Both audits passed.
Four tests cover all seven samples, any-sample protection, configurable gates,
nonfinite evidence and valid projection requirements. Resume revalidates all
current request inputs rather than trusting an existing result marker.

Eight fresh matched RGB renders: two times × native/moving × two edited meshes.
Original and inset-only controls were reused only with verified artifacts and
matched cameras, source masks, exposure and calibration. All renders use the
unchanged production incidence-2, unwarped, hard-source renderer.

| Frame / view | Arm / matching baseline | Lost depth pixels | Introduced black pixels | Changed RGB pixels |
| --- | --- | ---: | ---: | ---: |
|001083 moving|pruned / original|303|305|625|
|001083 moving|pruned+inset / inset|303|305|629|
|001083 native|pruned / original|487|481|1,285|
|001083 native|pruned+inset / inset|481|476|1,308|
|001123 moving|pruned / original|278|278|588|
|001123 moving|pruned+inset / inset|278|278|590|
|001123 native|pruned / original|333|319|1,145|
|001123 native|pruned+inset / inset|285|272|1,185|

These counts are coverage/change diagnostics, **not anatomical error, face or
full-frame quality metrics**. Some removed surface correctly reveals background;
the counts alone do not establish improvement or regression. No PSNR/SSIM/LPIPS
or loss is reported for novel views without matching GT.

Main-LLM inspection covered all eight crown/jaw panels at native resolution:
some projecting thin fragments disappear or thin, but the main crown opening
and brown/ragged rim remain. No material moving-view repair. The combined inset
surface does not replace the missing part. No conspicuous new cheek/jaw defect
was observed in these crops; this is not heldout or full-temporal acceptance.

**Verdict: no material crown repair; not promoted.** The production meshes and
recently delivered cinematic videos remain unchanged. Further expensive guard
certification or150-frame rollout of this visually insufficient arm is unwarranted.

- [001083 native crown, five-way comparison](/mnt/data/dec5_multiview_background_head/review/001083/native_unmasked_crown.png)
- [001123 native crown, five-way comparison](/mnt/data/dec5_multiview_background_head/review/001123/native_unmasked_crown.png)
- [001123 moving crown](/mnt/data/dec5_multiview_background_head/review/001123/moving_crown.png)
- [001123 native jaw](/mnt/data/dec5_multiview_background_head/review/001123/native_unmasked_jaw.png)
- [Actual visual verdict](/mnt/data/dec5_multiview_background_head/visual_review.json)

The study seal rechecks retained render hashes, unchanged source EXRs/depths,
matching renderer settings and terminal workers. Meshes are untextured geometry
diagnostics, not raw TSDF volumes or promoted laptop-ready models.

## Insights

1. Positive evidence protection matters: a median-only policy would erase some
   triangles that have coherent support at a corner or midpoint. This experiment
   preserves those explicitly rather than tuning exceptions for each frame.
2. Removing conflicting old geometry is not sufficient to repair the missing
   surface behind it. The additive inset shell and deleted fragment occupy
   different visible regions. A replacement must cover the actual opening while
   respecting reliable neighboring depth, not merely shrink an existing shell.
3. The next geometry test should fit/replace the local missing surface from
   reliable surrounding observations and a constrained shape proposal, then
   validate its newly visible parts. Increasing deletion strength alone is not
   supported by this experiment. The overall artifact-free video/mesh goal
   remains unmet, despite completed camera-path delivery.
