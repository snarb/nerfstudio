# DEC5: continuous body-prior completion and depth-support ablation

## What was tested

The earlier head transfer did not repair crown defects. This experiment targets
the larger remaining wrist/hand failure at actual time **001037**, starting from
the [protected-production forearm mesh](dec5_protected_forearm_surface.md).
No source EXR, existing mesh, movie or model default is modified.

`study_body_neighborhood_completion.py` samples 300,000 oriented points from
the baseline and computes a depth-10 Screened Poisson proposal. Only append-only
proposals below normalized x=-.03 survive; maximum original/boundary distance
is .006 and maximum triangle edge .0015. These distances are in the normalized
scene gauge, not meters. Original triangles and vertices remain exact.

`body_surface_neighborhood.py` chooses at most three neighboring seeds per
angular sector, eight sectors within radius .006. This avoids collecting only
the densely sampled nearest edge of a hole. It does **not** certify a surface.
The existing normal compatibility, enclosing tangent-plane hull, quadratic
leave-one-out error and predicted offset limits still apply. Seeds require
three measured train-depth views and distance to the Poisson proposal <=.0005.
Unchanged semantic and measured-depth vetoes reject conflicting additions.

After a first RGB/geometry review, `study_body_single_depth_seed.py` tests exactly
one weaker condition: at least **one** measured seed depth instead of three.
Same Poisson proposal, distance/shape checks, masks, sample-free-space veto and
final 62-camera/two-offset native guard. This is explicitly weaker inferred
geometry, not multiview-confirmed anatomy. No temporal observation or learned
body model participates in either control.

Important guard scope: these new additions use the jaw study's **depth-only**
corroborated free-space rule. The protected input previously used a contrastive
color-qualified body guard. This experiment preserves that input but does not
establish that the two guard definitions are equivalent.

## Results

| Variant | Qualified seeds | Certified proposal vertices | Initially admitted triangles | Final additions |
|---|---:|---:|---:|---:|
| Three-depth seeds | 12557 | 5659 | 8985 | 8856 |
| One-depth seeds | 18144 | 6765 | 9489 | 9315 |

Final pruning removes 124/5/0 and 164/10/0 triangles respectively. Separate
audits replay seed support, certificates and assembly; the primary audit also
recomputes semantic and sampled depth evidence. Each final mesh passes 124 fresh
native checks with zero qualified violations. Original prefixes are exact.
There are 961/979 index-connected components and zero nonmanifold edges.
Append-only patches are **not welded** to the original surface; this is not a
watertightness or correct-anatomy certificate.

Eight fresh RGB images compare both variants against freshly rendered protected
baselines at the moving camera and real train H004_A005_1210M6. Both sides use
the corrected native texture footprint, same camera/profile/exposure and hard
source RGB. The two independently rerendered baselines are byte-identical.

Fixed **train forearm skin ROI**, never held-out face or full-frame quality:

| Variant | PSNR | SSIM | LPIPS | Depth misses | Black RGB |
|---|---:|---:|---:|---:|---:|
| Matched protected baseline | 22.169533 | .760377 | .356042 | 910 | 911 |
| Three-depth prior | 22.232553 | .763148 | .353475 | 893 | 894 |
| One-depth prior | 22.232553 | .763148 | .353475 | 893 | 894 |

Do not compare these directly with the older protected table: the matched
baseline now includes the subsequently corrected native texture footprint.
GT uses the renderer's exact mean-centered log-gain and fixed exposure.
The face campaign CSV and its protocol are unchanged.

The three-depth proposal adds 12,187 newly covered moving-view pixels with zero
new misses over the baseline, but visual inspection shows that **most gains
are clothing/cuff completion**, not hand recovery. Geometry-valid black RGB
pixels rise from 740 to 911 in that moving image (907 for the weaker variant).
Thus increased geometry coverage alone is insufficient for image coverage.
Only 1 moving-view / 4 train-view RGB pixels change in the upper 1,400 rows;
this is not a cheek/crown repair.

Main-agent inspection covered both added-surface clay images, all eight native
RGB hand/torso panels, and both six-stage clay strips. Clothing gaps shrink,
but broken fingers, wrist/forearm holes, polygonal skin/color seams and a
cloth-colored patch above the train-view hand remain. The real hand is visibly
motion-blurred; that does not excuse invented seams and missing surface.
Both variants **fail artifact-free hand/video acceptance**. No full movie rerun.

![Moving hand, three-depth proposal](/mnt/data/dec5_body_neighborhood_completion/001037/review/moving_hand.png)
![Native train reference and candidate](/mnt/data/dec5_body_neighborhood_completion/001037/review/H004_A005_1210M6_hand.png)
![One-depth control](/mnt/data/dec5_body_single_depth_seed/001037/review/moving_hand.png)

### Where candidate coverage is lost

`diagnose_body_prior_admission.py` uses the unchanged train forearm polygon:

| Stage | Baseline's 910 missing rays covered |
|---|---:|
| Full raw Poisson, not accepted geometry | 910 |
| Local append-only proposal | 690 |
| After semantic masks | 687 |
| After confidence admission | 17 |
| After final native guard | 17 |

The full raw surface is not a valid solution: the clay strip exposes broad
invented closure outside the bounded local proposal. Its coverage is diagnostic.

`explain_body_seed_gate.py` separates shape certification and the **initial
sampled free-space veto**, which occurs before final ray pruning. Among 687
nearest semantic-triangle hits, 320 pass sampled free-space checks. Three-depth
shape certification accepts 15 hits, of which only two also pass sampled free
space. One-depth shape certification accepts 91, but only three also pass free
space; 88 of its 91 collide with that veto. The independent direct-support arm
accepts 15 hits. These counts describe the nearest semantic triangle, not an
exact counterfactual after removing occluders; the final mesh covers 17 rays.

The bottleneck is not just the number of seed views. Lowering three to one
does not improve this fixed ROI. Some additions lack a stable local shape,
while others conflict with independently corroborated depth-map samples.
Whether those vetoes are appropriate on this blurred hand needs inspection
against the existing color-qualified body evidence, not another blind threshold
relaxation or a claim that missing data proves empty space.

## Insights

This branch gives a visible clothing gain but fails its main hand target.
Do not roll it out to 150 times or select the weaker recipe per frame. The
next useful diagnostic is the actual RGB/depth evidence behind the rejected
hand patches and the difference between depth-only and color-qualified vetoes;
otherwise a looser prior may simply replace black holes with incorrect skin.
The camera-path workaround and mesh-repair requirement remain separate.

Eleven focused tests pass. All workers ended normally, except an initial audit
checker indexing error: corrected and rerun to completion without changing or
regenerating geometry; its failed log is retained. Artifacts:
`/mnt/data/dec5_body_neighborhood_completion/001037` and
`/mnt/data/dec5_body_single_depth_seed/001037`.
`freeze_body_neighborhood_completion.py --check` verifies retained artifacts,
external evidence and the explicit failed visual verdicts. The user objective
is unfinished; the published dynamic actor/camera movie is unchanged.
