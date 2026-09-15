# DEC5: actual RGB witnesses and measured-depth alignment of new body patches

## What was tested

Follow-up to [body-prior admission](dec5_body_neighborhood_completion.md) at
actual frame **001037**. Two questions: are the depth vetoes confusing skin with
background/cloth, and can genuinely displaced proposals be aligned rather than
discarded? No source EXR, calibration, published mesh/movie or model default is
changed. No held-out RGB or temporal frame is used for construction.

`diagnose_body_rgb_veto.py` inspects the actual four-native-tap veto events on
semantic triangles intersecting the previously fixed train H/A forearm misses.
`calibrated_depth_witness.py` uses the renderer's exact mean-centered log-gain
response, also comparing the legacy diagnostic's uncentered response.
The existing contrastive rule requires three depth/color-compatible witnesses;
observed-depth color must beat the nearer proposed-depth color by .01, with the
existing unavailable-comparison fallback. Every one of four taps must qualify.

After viewing those witnesses, `align_poisson_to_measured_depth.py` tests bounded
deformation of **new** below-head Poisson vertices only. At each of three rounds,
up to eight nearby native depth references per vertex are considered. Each
observation needs three other depth/color-compatible views, and each vertex needs
two qualifying reference equations. A regularized graph solve moves neighboring
proposal vertices smoothly: prior weight .05, graph weight .2, step bound .002,
total bound .006 in the existing normalized scene gauge (not meters).

Final candidates must pass semantic/proximity/edge/orientation checks and the
unchanged **direct** multiview depth admission and depth-only free-space guards.
The earlier inferred-seed certificate is not used in this alignment control.
Observed vertices/faces remain exact; this is not a wholesale mesh replacement.

## Results

### RGB diagnosis

The suspected gain-formula difference is **not a cause here**: mean log-gain is
approximately [-1.44e-9, 1.20e-10, 6.01e-10]. All 62 generated source RGB arrays
are byte-identical between centered and legacy formulas, and zero decisions
change across 5,952 inspected four-tap veto events. The new helper makes the
response convention explicit for future use without altering production profiles.

| Diagnostic on fixed forearm rays | Depth-only | Color-qualified |
|---|---:|---:|
| Veto events retained | 5952 | 2562 |
| Affected semantic triangles vetoed | 273 | 234 |
| Nearest semantic hits passing free-space check | 320 | 362 |
| Three-depth shape certificate **and** free-space pass | 2 | 2 |
| One-depth shape certificate **and** free-space pass | 3 | 7 |

There are 687 semantic hits and 527 unique hit triangles in this diagnostic.
Correlated events are not independent missing pixels. Counts at the nearest
semantic triangle are not a counterfactual raycast after removing occluders.
Despite many event abstentions, photometric qualification does not unlock a
large set of already shape-certified forearm proposals. No such override is
deployed to the mesh/video.

The main agent viewed all six saved native witness panels: two retained vetoes,
two RGB mismatches and two ambiguous comparisons, each with up to four witness
cameras. The retained examples show **real forearm skin**, not background; the
nearer prior shifts projections toward a different skin/cloth boundary or skin
brightness gradient. Farther measured positions match better across cameras.
Other cases indeed straddle cloth/skin or remain photometrically ambiguous and
are rejected as decisive free-space evidence. This is local evidence, not a
claim that every depth veto in the video is correct.

![Retained skin-depth contradiction](/mnt/data/dec5_body_rgb_veto_diagnosis/001037/case_00_retained.png)
![Cloth/skin witness mismatch](/mnt/data/dec5_body_rgb_veto_diagnosis/001037/case_02_rgb_mismatch.png)

### Bounded alignment canary

The three rounds constrain 10022/11029/11382 proposal vertices. Maximum total
displacement is .0059995565. After semantic and shape limits, 19,041 candidate
triangles remain; direct depth admission accepts 7,405. Native pruning removes
66/5/0, retaining **7,334** additions. Independent replay recomputes the RGB/depth
equations, all three sparse solves, geometric limits, direct admission and
**124 fresh native ray checks**, with zero qualified contradictions.

Four fresh RGB images compare protected baseline/aligned geometry at moving
and real H004_A005_1210M6 views. Same native-footprint renderer, hard source RGB,
profile/exposure and cameras. Baseline image hashes match the previous controls.

Fixed **train forearm** ROI — not held-out face and not full-frame quality:

| Variant | PSNR | SSIM | LPIPS | Depth misses | Black RGB |
|---|---:|---:|---:|---:|---:|
| Protected baseline | 22.169533 | .760377 | .356042 | 910 | 911 |
| Previous three-depth prior | 22.232553 | .763148 | .353475 | 893 | 894 |
| Aligned direct-depth proposal | 22.296850 | .766704 | .355607 | 879 | 880 |

Alignment has slightly better coverage/PSNR/SSIM but worse LPIPS than the previous
three-depth proposal. It is not an unqualified improvement. There are zero RGB
changes in the upper 1,400 rows in both matched views: this does not repair the
cheek/crown. Geometry-valid black RGB pixels increase from 740 to 1087 in the
moving image and 25 to 48 in the train image; added depth is not guaranteed RGB.

Both added-surface clay images and all four native RGB hand/torso panels were
actually inspected. Broad wrist/hand breakup persists. Clothing gains coexist
with ragged, fragmented patches; final index-connected components increase to
1742 (zero nonmanifold edges). The patches are not welded to the original mesh.
**Visual hand/video gate: fail. No 150-frame rollout or new movie.**

![Aligned proposal, moving hand](/mnt/data/dec5_observed_poisson_alignment/001037/review/moving_hand.png)
![Aligned proposal, real train comparison](/mnt/data/dec5_observed_poisson_alignment/001037/review/H004_A005_1210M6_hand.png)

Seventeen focused tests pass, including gain-gauge invariance, calibrated camera
depth sign, refusing unqualified observations, zero-evidence identity, smooth
displacement and norm bounds. Native-data replay tests the real pipeline beyond
these synthetic solver tests. A diagnostic-only list/array error occurred during
panel construction; the failed workspace/log was retained separately, and the
corrected diagnostic completed. All reconstruction/render workers ended normally.

Artifacts: `/mnt/data/dec5_body_rgb_veto_diagnosis/001037` and
`/mnt/data/dec5_observed_poisson_alignment/001037`.
`freeze_body_rgb_alignment.py --check` verifies inputs, outputs and actual reviews.
The earlier published movie and face campaign metrics remain unchanged.

## Insights

The hypothesized gain mismatch is ruled out for this frozen calibration. Some
vetoes are ambiguous, but genuine skin-depth contradictions remain; removing
them indiscriminately would preserve a displaced surface. Bounded alignment is
locally measurable but still does not produce a continuous, video-ready hand.

Preserving every existing face also preserves existing bad hand geometry and
source seams. Further geometry work needs evidence-based replacement of unreliable
surface, not another small relaxation of append-only admission. Separately, the
next practical camera-workaround test should render the difficult actor time from
exact central train views before building a smooth, train-aware path. Neither a
slightly better ROI nor hiding the object by cropping constitutes the full goal.
