# DEC5 000995: query quorum, wrong texture layer, and corrected blue-wedge diagnosis

## What was tested

Two isolated controls keep the production video unchanged:

1. `study_query_support_quorum.py`: a triangle is removable only when **each**
   vertex and centroid has fewer than three native close-depth observations,
   at least six stable farther-depth footprints, and six farther observations
   each corroborated by at least three other cameras. Near tolerance is .0015
   in normalized coordinates. This deliberately weakens the earlier protection
   by any one near view; it is a counterfactual, not a proven-safe policy.
   All mesh triangles are tested; no RGB, ROI or masks select deletion.
2. `combine_query_quorum_texture_guard.py`: on exactly that mesh, reject source
   RGB where its measured 5x5 depth footprint lies stably farther than the
   surface. This reuses the frozen measured-texture guard, without averaging,
   new color fitting, camera changes or geometry changes in this second stage.

Each arm has three 1080x1920 diagnostic stills: one moving-path pose, H/C and
K/B. These are **not** replacements for the delivered native-6K video and are
not temporal validation. H/C and K/B are train views, not held-out evaluation.

Roots:

- `/mnt/data/dec5_query_support_quorum/000995`
- `/mnt/data/dec5_query_quorum_measured_texture/000995`

## Results

**Neither arm passes the requested visual result.** The quorum trims an
obvious flap above the fingernail. The main blue wedge directly behind the
barrel survives. The combined guard largely turns it skin-colored, but the
false protrusion remains. Existing crown/fringe and chin/neck defects remain.
No candidate was promoted; no delivered video, source dataset, or model default
was changed.

The quorum removes 300 triangles, 104 more than the previous any-near rule,
retaining all previous removals and leaving all vertices unchanged. An
independent native-footprint audit replays all 1200 removed samples across
62 cameras and verifies exact triangle-subset geometry. Far corroboration
still reuses its original helper; the audit does not certify physical truth.

| View | Quorum RGB changes vs any-near | New black pixels | Lost depth hits |
|---|---:|---:|---:|
| Moving | 217 | 5 | 0 |
| H/C | 643 | 187 | 185 |
| K/B | 423 | 280 | 286 |

| View | Combined guard: contradictory chosen sources before / after | RGB changes vs quorum | New black pixels |
|---|---:|---:|---:|
| Moving | 613 / 0 | 1422 | 8 |
| H/C | 954 / 0 | 1845 | 11 |
| K/B | 2205 / 0 | 2744 | 51 |

These are diagnostic counts, **not** image-quality metrics or missing-anatomy
counts. Much removed geometry was already false background-colored surface.
The source replay independently checks native footprints at the actual
float32 ray-hit points for every selected colored source. Zero contradictions
means that particular guard passed, not that every remaining color is correct.

### Correction to the diagnostic localization

The previous 86-face polygon largely described the flap above/behind the nail,
not the main blue wedge beside the barrel. The representative removed face
48941 belongs to that earlier cohort. Its removal must not be reported as
removal of the main blue wedge. Actual remaining first-hit tracing also rejects
the hypothesis of newly exposed, previously uncounted faces inside that old ROI.

A native coordinate grid established a small post-hoc blue-wedge core:
`[(156,1230),(164,1230),(164,1248),(156,1248)]`, portrait pixel coordinates.
It contains 171 pixels and has **zero overlap** with the old polygon. Neither
this core nor its RGB selects geometry, texture sources or thresholds.
[Old orange region and corrected cyan core](/mnt/data/dec5_query_support_quorum/000995/blue_wedge_support/regions.png).

All 171 pixels originally select `J004_A005_121014`. Its measured center depths
are .04260 to .04426 normalized units farther than the rendered points
(median .04331). The stricter stable-footprint guard rejects 169/171; the other
two do not satisfy its full neighborhood test. In the combined arm all 171 core
pixels remain colored and zero violate that stable-footprint guard.

However, the actual core points have 0..49 nearby measured depths across all
62 cameras, median **19**, so simply demanding more close-depth votes would not
reliably remove the defect. `diagnose_blue_wedge_semantics.py` projects these
same points into the five previously reviewed native hand/lipstick crops:

| Train camera | Core points with close depth | Close observations clearly outside both hand and tube | Close observations clearly inside either |
|---|---:|---:|---:|
| H/C | 90 | 88 | 0 |
| K/B | 95 | 94 | 0 |
| I/C | 20 | 20 | 0 |
| J/C | 55 | 53 | 0 |
| J/A | 0 | 0 | 0 |

The remaining five close observations are in the two-pixel uncertainty band.
Counts sum observations, not unique world points. All five witness panels
were actually viewed: the first four project into the visible room gap beside
the barrel/finger; J/A projects onto clothing behind that gap. Thus the near
measurements can be mutually consistent **and still describe a false foreground
extension into the gap**. This is stronger evidence than assuming that every
near vote measures the finger. Prompted masks are fallible semantic priors,
not independent depth truth, and were used only for this diagnostic.

[Native semantic witnesses](/mnt/data/dec5_query_support_quorum/000995/blue_wedge_support/semantic_witnesses).
[Combined K/B comparison](/mnt/data/dec5_query_quorum_measured_texture/000995/K004_B005_1210DS/review/lipstick_native.png).

The root agent inspected three quorum lipstick panels, three native sheets
covering all 22 new-black components, all twelve combined head/lipstick and
new-black panels, the corrected region image, and five semantic witness panels.
The separate [final review receipt](/mnt/data/dec5_query_support_quorum/000995/final_review.json)
records hashes and explicit fail verdicts; immutable computation records keep
their original pre-review status. This still-image review makes no all-frame,
continuous-playback or artifact-free claim. No PSNR/SSIM/LPIPS was computed here.

Finalization verified 164 artifact/code bindings. The four focused test modules
for query quorum, measured-free pruning, measured texture visibility and
instance-mask utilities pass all 21 tests. Test success does not override the
explicit visual failures.

## Insights

There are two distinct problems: wrong-layer texture selection explains the
cloth color, while a false surface explains the remaining skin-colored fin.
The guard addresses the former, not the latter. Close-depth vote counts alone
cannot resolve correlated foreground leakage at a narrow occlusion boundary.

The next geometry experiment should use explicit visible-background evidence
from several train views to constrain that gap, with an object-depth/occlusion
bound protecting real neck and clothing behind it. It must distinguish
"outside the hand" from "empty foreground space"; blindly carving the union's
complement is unsafe. This is a different evidence model from increasing the
quorum or overriding just five of 62 near votes. It should be checked first on
the corrected barrel-side gap, then transferred to other times before video
production. The separate cheek-hole objective remains incomplete.
