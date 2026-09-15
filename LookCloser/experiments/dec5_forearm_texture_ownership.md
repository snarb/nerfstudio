# Forearm source-seam attribution and coherent ownership

## What was tested

Follow-up to [multiview geometry admission](dec5_multiview_forearm_admission.md)
on `001037`. Fix its inferred mesh, camera poses, actual actor time, source RGB,
profiles/exposure and zero registration. Identify the remaining forearm seam by
actual pixel source IDs, face labels, depth and old/added triangle identity.

Three opt-in label policies keep hard single-source RGB, never averaging:

1. One owner per connected added component, selected by area coverage; among
   sources within one percentage point of best coverage, choose highest mean
   normalized existing source quality. Require coverage≥80% and ≥100 faces.
2. Weight coverage by **target-visible pixel counts**, not hidden triangle area.
3. The same visible weighting, joining original faces within .0015 normalized
   distance of the added surface and absolute normal agreement≥.8.

An owner replaces labels only where its existing visibility/quality is positive.
Normal per-pixel visibility fallback remains. The baseline graph energy is
explicitly recorded as **before** the ownership override, not final optimality.
No model/renderer default, geometry, exposure or color profile is changed.

## Results

**Reject all three ownership variants.** The lower source band is reduced, but
the moving view gains a conspicuous light wrist patch. Joining nearby original
faces moves the boundary and adds mottling. No candidate is promoted to video.

Attribution distinguishes two problems: old/new surfaces have zero shared
topological graph edges, but the broad lower band also crosses source labels
**inside** the added surface. Among strong warm-color adjacent-pixel jumps
(mean RGB difference≥10), 515/636 in H/A and 610/646 in the moving crop coincide
with source switches. These are adjacency-event diagnostics, not anatomical or
image-quality metrics and not proof that every jump is artificial.

- [RGB / source / geometry / fallback maps](/mnt/data/dec5_forearm_seam_attribution/moving_maps.png)
- [H/A attribution](/mnt/data/dec5_forearm_seam_attribution/H004_A005_1210M6_maps.png)

The largest added component has 49,602 faces. Weighting hidden surface area
initially chooses G/A even for the H/A target. Target-visible weighting corrects
that to H/A. In the moving view, both visible variants select `J004_B005_1210GR`.
For the joined region its coverage is 87.88%; H/A covers 74.98%, H/C 68.84%, and
the original upper-wrist source I/B only 22.98%. No eligible single source covers
the whole visible region under the current visibility and source-quality gates.

| Variant | H/A changed RGB | E/D changed RGB | Moving changed RGB |
|---|---:|---:|---:|
| Geometric-area owner | 7,037 | 40 | 6,837 |
| Visible added-region owner | 440 | 40 | 6,970 |
| Visible joined-region owner | 1,913 | 542 | 9,478 |

All nine depth arrays are **byte-identical** to the control; there are zero new
black pixels and zero changed pixels in the upper 1,200 portrait rows. These
integrity checks do not establish visual quality. The joined policy changes
341/104/883 original-face labels in H/A, E/D and moving views; labels outside
the declared nearby regions are verified unchanged. The other two policies
preserve all original-face labels.

- [Three visible-policy wrist comparisons](/mnt/data/dec5_visible_patch_owner/review/moving_detail.png)
- [Hand/arm overview and new patch](/mnt/data/dec5_visible_patch_owner/review/moving_overview.png)
- [H/A comparison](/mnt/data/dec5_visible_patch_owner/review/H004_A005_1210M6_detail.png)
- [Negative visual gate](/mnt/data/dec5_visible_patch_owner/visual_review.json)

The main agent inspected both source/geometry maps and legends, all three
area-policy detail panels, all three visible-policy detail panels and the moving
overview. Other saved overviews are not claimed as inspected. Four helper tests
cover connectivity, invisible-source rejection, original-face preservation,
near-surface region joining and target-visible weights. All nine workers ended
normally. Recheck artifacts with `scripts/freeze_forearm_texture_ownership.py --check`.

Diagnostic erratum: the first fresh H/A depth comparison omitted the existing
renderer's native-target mask and correctly stopped before writing panels.
Replaying the same mask restores agreement. Virtual moving targets are not
masked this way. The failed log is retained; this was not a new mesh defect.

## Insights

Source continuity alone is insufficient when no source covers the whole region
and overlapping sources have appreciably different colors. Hard ownership moves
the seam rather than eliminating the disagreement. Better coverage weighting is
necessary but does not solve the visible patch. Do not accept the controls merely
because depth is unchanged or no black pixels were introduced.

The next useful diagnostic is matched-point source radiometry in overlapping
train views: distinguish camera/spatial-response differences from incorrect
correspondence and real view-dependent skin appearance before fitting a shared
correction. Any correction must preserve native high-frequency detail and the
fixed temporal exposure protocol. Geometry joining/hand repair and full temporal
validation remain separately unfinished; the production video is unchanged.
