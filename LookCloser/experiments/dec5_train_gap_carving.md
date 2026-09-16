# DEC5 000995: train-visible gap constraints with foreground protection

## What was tested

The corrected [blue-wedge diagnosis](dec5_query_quorum_and_blue_wedge.md)
showed that several close PatchMatch depths lie in the visible room gap beside
the lipstick. Counting those correlated depths cannot establish a real surface.
This experiment lets explicitly reviewed semantic evidence constrain the mesh.
It does not replace COLMAP, invent synthetic RGB, or change delivered video.

`study_train_gap_carving.py` uses the previously reviewed hand/nail and lipstick
SAM masks in H/C, K/B, I/C and J/C, with manually drawn **train-RGB context
polygons**. The polygons were drawn on actual native source crops, not the
rendered diagnostic core. Four source/constraint overlays were reviewed before
carving. The masks are fallible priors, not measured empty-space ground truth.
J/A is excluded because its gap projects onto clothing rather than the room.

The semantic negative mask is the context polygon outside the hole-filled
hand/tube union, with a two-native-pixel safety band. A complete 2x2 sampling
footprint must be negative. Each camera also restricts the operation to a
foreground depth slab: 5th..95th percentiles of measured depths inside the
three-pixel-eroded object union and context, expanded by .003 normalized units.
This is an empirical layer bound, not proof of object identity.

Three **same** cameras must exclude all three vertices and centroid of a
triangle. All mesh faces are queried; no projected defect core selects deletion.
Vertices never move and only complete triangles are removed. The parent is the
previous quorum mesh, not an unmodified original TSDF. Existing head repairs
and previous pruning are inherited. No component cleanup is added.

After this rule damaged the visible pink tip, the paired
`study_train_gap_positive_veto.py` added a foreground veto: any vertex/centroid
inside or within two pixels of the object union in **any** one depth-slab-valid
view protects the triangle. Any of four neighboring pixels can establish this
protection. Context polygons, masks, slabs, parent mesh and rendering stay fixed.

Roots:

- `/mnt/data/dec5_train_gap_carving/000995`
- `/mnt/data/dec5_train_gap_positive_veto/000995`

## Results

**Material local improvement, not a finished artifact-free reconstruction.**
The large blue flap beside the barrel is substantially reduced. The first arm
notches the pink lipstick tip in K/B; reject it. The foreground veto restores
that tip while retaining most gap removal. A small jagged blue remnant near
the finger/tube junction remains. Moving-view improvement is modest because
the protrusion is less exposed there. Existing crown/fringe and chin/neck
defects remain unchanged in the inspected comparisons; the cheek-hole goal
is not solved by this local constraint.

| Control | Removed faces | Faces protected relative to first arm |
|---|---:|---:|
| Three negative views | 94 | — |
| Same + foreground veto | 79 | 15 |

| Control | View | RGB changes vs quorum | Newly black pixels | Lost depth hits |
|---|---|---:|---:|---:|
| Negative only | Moving | 309 | 1 | 0 |
| Negative only | H/C | 550 | 413 | 415 |
| Negative only | K/B | 908 | 835 | 836 |
| With veto | Moving | 264 | 1 | 0 |
| With veto | H/C | 565 | 355 | 357 |
| With veto | K/B | 790 | 725 | 727 |

These are diagnostic counts, not PSNR/SSIM/LPIPS or counts of missing anatomy.
The intended deletion makes the real room gap black because the actor-only mesh
does not model the room. Missing skin or lipstick must not be excused this way.

Using the actual train masks as a **non-independent** boundary check, the
negative-only arm creates 14 newly black pixels inside the K/B lipstick mask,
including one more than two pixels inside it. The veto arm creates **zero**
newly black pixels inside either hand or lipstick mask in both H/C and K/B.
These same masks helped define the constraint, so this is a preservation test,
not an unbiased accuracy metric.

The corrected 171-pixel post-hoc blue core contains 171 rendered hits before,
48 after negative-only carving, and 61 with foreground protection. That core
does not determine the operation or its settings. More removed pixels alone
are not better: the first control also removes part of the visible tip.

[Veto K/B comparison](/mnt/data/dec5_train_gap_positive_veto/000995/review/K004_B005_1210DS/lipstick_native.png)
shows the GT gap, original false wedge, and corrected mesh render side by side.
[Veto H/C comparison](/mnt/data/dec5_train_gap_positive_veto/000995/review/H004_C005_1210SZ/lipstick_native.png).

The main agent actually viewed all four source/mask panels, all six head and
six lipstick panels, and six native sheets covering every newly black component.
The [review receipt](/mnt/data/dec5_train_gap_positive_veto/000995/visual_review.json)
records these 22 viewed files and explicit limitations. Independent arithmetic
replays negative/positive projection footprints and verifies unchanged vertices
and exact triangle subsets. The 62 native measured-depth receipt is rechecked
at finalization. This does not independently certify mask truth or all triangle
interiors from four samples.

Finalization verified 144 artifact bindings. All 25 focused tests pass across
the new negative/veto sampling rules and the existing quorum, free-space,
texture-visibility and instance-mask helpers. Compact stage logs, including
the failed parse attempt, are retained under the veto root's `logs/`; the
test output is `validation_tests.log`.

Six RGB renders completed on clever-shadow, three concurrently per arm, about
22..25 seconds per worker. These are 1080x1920 diagnostic stills, **not** a new
6K video. No source RGB averaging, retraining, exposure fit or new source mask
was introduced into the renderer. The first review command failed at parsing
an extra f-string brace, before executing or writing review artifacts; its log
is retained. The corrected reviewer and both geometry/RGB pipelines completed.

## Insights

The useful distinction is not simply low versus high depth confidence. A
visible-background constraint can contradict a mutually consistent but false
foreground extension. It must also honor positive foreground evidence: a
negative-view majority alone can cut a real-looking tip in another view.

The remaining narrow fringe should be localized against the coarse triangles
and the protected uncertainty band before changing thresholds. Geometry-preserving
local subdivision is a plausible next control now that the evidence model has
changed; earlier subdivision with near-depth voting did not solve this defect.
Temporal transfer requires new tracked/reviewed masks and matching layer bounds
at other times, not copying these four frame-specific context polygons. Neither
arm is promoted to the production sequence, and the full video/mesh goal remains
open.
