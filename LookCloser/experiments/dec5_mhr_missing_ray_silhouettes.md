# Missing-ray silhouette/depth attribution

## What was tested

Post-hoc001193 diagnosis only; no mesh, mask, fit, target prediction or production
changes. The44 previously stored missing-ray hits on the smooth025 prior are
sampled at401 offsets from−0.01 to+0.01 along their target-camera rays. These are
normalized scene distances, not meters. Each sample is projected into the62
train masks, with the existing independently measured mask override. A second
diagnostic uses a uniform two-pixel dilation, not a mask change in admission.

Root: `/mnt/data/dec5_mhr_local_patch_admission/ray_silhouette_probe`.
This supplements, but does not alter, the sealed original admission control.

## Results

The original smooth400 local candidate's14 matching proposed facets are all
rejected by some train masks. Main LLM inspected four full-resolution witness
crops: C/E,D/E,E/D,E/E. The red samples lie on actual background beyond the neck,
not merely on dark skin misclassified by segmentation. Do not weaken those masks.

[C/E witness](/mnt/data/dec5_mhr_local_patch_admission/mask_veto_witnesses/C004_E005_1210X7.png),
[E/D witness](/mnt/data/dec5_mhr_local_patch_admission/mask_veto_witnesses/E004_D005_1210L4.png).

| Diagnostic mask | Rays with a feasible sampled point | Nearest feasible absolute offset min / median / max | Outside cameras at prior point min / median / max |
|---|---:|---|---|
| unchanged |44/44|0.00860 /0.00925 /0.00970|4 /12 /16|
| two-pixel diagnostic dilation |44/44|0 /0.008925 /0.00940|0 /9 /13|

For unchanged masks, the first feasible points lie farther from the target
camera than the prior's front hits. Their nearest-original-surface distances
are0.000351 /0.000483 /0.000686 (min/median/max); all44 satisfy the0.002 locality
limit.25/44 have at least two agreeing train-depth views, with depth-vote
min/median/max0 /2 /6.

The audit replays all2,187,856 camera/sample mask classifications and the
independent measured-depth check. Numeric evidence and hashes are retained.
Finite sampling and silhouette agreement are not ground-truth depth or proof
that every one of these44 pixels must contain skin.

## Insights

The prior's first front intersection is wrong for these rays. Closing a shallow
gap at that depth would create a neck protrusion visible against real background
in other cameras. A deeper surface near measured geometry is plausible and has
partial direct depth support; nearest-rim proximity alone cannot choose it.

The wider local-domain control nevertheless leaves44/44 missing pixels in all
three strict native reviews. Main inspected G/B,M/B,E/B and the requested-hole
comparison: mostly lower-neck facet shading changes, no requested repair.
Thus removing the open-edge gate is not sufficient. The next meaningful change
should constrain anatomical conformance with multiview silhouettes/depth ordering
or construct the supported deeper surface; not relax semantic rejection or merely
add more overlapping prior facets. Both remain hypotheses, not accepted fixes.
