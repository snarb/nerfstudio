# Lower-row FoundationStereo forearm canary

## What was tested

The previous [anchor-bias study](dec5_foundation_anchor_bias.md) mostly validated
already reconstructed fingers and palm. Test new observations that actually
cover the missing wrist/lower forearm at actor time `001037`:
`E004_D–F004_D` and `E004_E–F004_E` (four distinct train cameras).

Physical calibration and fixed camera color profiles/exposure are unchanged.
The existing rough wrist and middle-knuckle landmarks set a lower local crop
focus only; they do **not** supply surface depth. Local rectification retains
native focal scale in `768×768` patches. Existing research-only FoundationStereo
weights run forward/reverse, with the same LR and source-domain checks. Model
RGB inputs remain unmasked; coarse warm-object regions constrain post-hoc
geometry, not an asserted anatomical ground truth.

Reuse the previous spatial-anchor offset estimator, dual-pair agreement,
maximum edge length and 62-camera native free-space guards **without changing
their thresholds**. Test empty-ray-only additions and missing-foreground-layer
additions separately. All original vertex/triangle arrays remain unchanged.
The current texture pipeline renders five new controls/candidates in three
disjoint workers; the matching published moving-view control is hash-verified.
No actor retiming, camera change, RGB averaging, new exposure or video rollout.

## Results

**Partial geometry recovery, but the RGB candidate fails visual acceptance.**

| Pair | Warm-region common pixels | LR ≤2 px | Anchor offset | CV median normalized depth error, before → after |
|---|---:|---:|---:|---:|
| E/D–F/D | 75,735 | 73,272 | +0.5107 px | .0003924 → .0003243 |
| E/E–F/E | 78,132 | 76,864 | −0.6920 px | .0004427 → .0003870 |

Both pass the previously frozen **aggregate anchor** criterion. Spatial tails
still contain large disagreements: E/E–F/E fold 0 p90 is about `.0235` after
correction. Agreement/anchor scores do not establish missing-surface accuracy.
Forward+reverse inference takes 1.68/1.08 seconds per pair, excluding loading,
staging and validation; these are not complete pipeline runtimes.

| Addition rule | Proposed triangles | Retained after native guards |
|---|---:|---:|
| Empty mesh ray | 9,050 | 8,972 |
| Missing nearer foreground layer | 40,925 | 34,926 |

The lower pairs provide 57,191 mutually agreed pixels. The foreground candidate
has 21,196 eligible pixels; pruning removes 5,998 triangles in the first pass and
one in the second. A final 124 native checks have no remaining corroborated
free-space contradiction. This does not certify the inferred shape.

The final foreground mesh exposes 4,726 previously missing depth pixels in H/A,
4,589 in E/D and 5,368 in the actual moving view. These are visibility diagnostics,
not independent anatomical coverage or image-quality metrics.

Actual RGB inspection:

- H/A: part of the lower arm is recovered, but wrist cutouts, black peppering and
  a hard texture boundary remain.
- E/D: additional arm coverage, but some scattered dark/blue surface conflicts
  become worse.
- Moving view: some lower forearm returns; the wrist is still visibly damaged.

All three candidate RGB views are **fail** for clean-video acceptance. The main
agent inspected both rectification panels, both disparity panels, three geometry
comparisons, three RGB comparisons, three native veto panels and three corrected
near/far layer panels: 16 explicitly recorded panels. Other saved panels are not
claimed as visually reviewed. No new face or full-frame PSNR/SSIM/LPIPS was computed.

- [H/A RGB versus train GT](/mnt/data/dec5_foundation_lower_forearm/rgb_review/H004_A005_1210M6.png)
- [E/D RGB versus train GT](/mnt/data/dec5_foundation_lower_forearm/rgb_review/E004_D005_1210L4.png)
- [Actual moving-view comparison](/mnt/data/dec5_foundation_lower_forearm/rgb_review/moving.png)
- [Visual verdict](/mnt/data/dec5_foundation_lower_forearm/visual_review.json)
- [Artifact audit](/mnt/data/dec5_foundation_lower_forearm/artifact_manifest.json)

### Why the guard removes part of the prior

Inspected five deterministic vertical-quantile samples in each of three strong
veto cameras, not a hand-selected collection of successes. Of their veto pixels,
98.53% in B/E and 100% in E/E and F/E have a warm RGB color. Warm color alone is
not segmentation or proof of correct depth. The native crops nevertheless show
many vetoes on the real arm, not solely outside its silhouette.

For four inspected **forearm** B/E samples, the farther PatchMatch point
reprojects onto blue clothing in all four lower train images, while the proposed
near point stays on the arm. This is evidence of erroneous correspondence/layer
assignment in those PM observations, despite multiple geometric votes. The fifth
B/E sample is on the hand and projects its farther point toward the upper torso;
it is not counted as a forearm/clothing example.

Other conflicts are genuinely unresolved by this check: all five sampled F/E
farther points stay inside all four coarse arm regions; many projections are
skin-to-skin at a different position along the forearm. Thus these 15 samples do
not justify globally disabling the measured-depth guard or declaring the learned
depth universally correct.

- [Native B/E veto pixels](/mnt/data/dec5_foundation_lower_forearm/veto_diagnosis/B004_E005_1210VE.png)
- [B/E near/far points in four lower views](/mnt/data/dec5_foundation_lower_forearm/layer_diagnosis/B004_E005_1210VE.png)
- [Ambiguous F/E skin-to-skin conflicts](/mnt/data/dec5_foundation_lower_forearm/layer_diagnosis/F004_E005_1210FP.png)

The first layer montage had a visualization bug: very distant points could draw
markers outside their tile. It is retained in `layer_diagnosis_crop_unsafe` with
its producer. Corrected crops expand to contain both points and draw locally;
all three corrected panels were reinspected. Some expanded tiles are rescaled,
and their native crop size/projection coordinates are recorded.

## Insights

The missing region needs observations of the **missing part**, not simply more
agreement on adjacent good geometry. Lower-row sources make a material coverage
difference under the same admission rules, but coverage is not clean appearance.

The next confidence-fusion step must distinguish a clearly wrong background-layer
match from ambiguous low-texture skin-to-skin matches. It should preserve reliable
existing surfaces and avoid creating peppered gaps by treating every correlated
depth vote as equally decisive. Current evidence supports investigating that
distinction; it does not yet establish a safe replacement rule.

All jobs are terminal. Production source data, mesh inventory, camera path and
150-frame movie remain unchanged. Opt-in staging/build/diagnostic adapters and
seven focused tests are separate from existing defaults. Recheck the retained
artifacts with `scripts/freeze_foundation_lower_forearm.py --check`.
