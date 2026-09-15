# FoundationStereo measured-anchor bias and foreground-layer completion

## What was tested

Follow the [calibrated stereo hand canary](dec5_foundation_hand_stereo.md) with
actual high-confidence PatchMatch depth, keeping physical calibration fixed.
Test whether the two strong stereo pairs' disagreement is partly a constant
disparity offset. This uses the existing research-only model predictions, not
new inference, neural training, camera optimization, or held-out RGB.

At actor time `001037`, left-camera native PatchMatch points need at least three
**other** real depth-map votes (`0.001` normalized depth tolerance, `1.5 px`
roundtrip, `1°` parallax). Matching learned depth cannot establish anchor trust.
Learned LR/domain confidence only establishes an available prediction at an
already measured anchor. Coarse train warm-object masks include hand/lipstick;
they are not exact anatomical ground truth.

Fit a scalar median disparity residual on three spatial fold groups and evaluate
on the fourth: `128×128 px` blocks, an `8 px` excluded margin, four folds, at least
100 train/test anchors each. The frozen rule limits corrections to `±4 px` and
fold spread to `1.5 px`; pooled held-out-anchor median must improve at least 10%
and p90 must not worsen more than 5%. These are held-out **spatial depth anchors**,
not the held-out RGB evaluation camera. No missing-surface ground truth is inferred
from their accuracy.

Then propose additions from two disjoint pairs / four physical cameras only:
corrected depths agree within `0.001`, roundtrip within `2 px`, both pairs' eroded
left/right masks pass, grid edges stay below `0.002`. Select the reference by
largest calibrated baseline, not appearance. Protect original mesh arrays and
run the existing 62-camera measured-free-space guard at both native ray offsets.

## Results

**Calibration improved; useful mesh completion did not. Production unchanged.**
All depth numbers below use the existing normalized scene coordinates, not meters.

| Pair | Available anchors | CV anchors | Offset (px) | CV median error before → after | CV p90 before → after |
|---|---:|---:|---:|---:|---:|
| F/A–G/A | 2,003 | 1,579 | −1.2569 | .0008411 → .0004670 | .0018531 → .0015203 |
| E/C–F/C | 3,837 | 2,955 | +1.6635 | .0005854 → .0002571 | .0011285 → .0007208 |
| G/A–H/A | 606 | insufficient per-fold coverage | not fitted | N/A | N/A |

Median CV errors improve approximately 44% and 56%. Not every region improves:
fold 3 p90 worsens from `.0024416` to `.0029139` for F/A–G/A and from `.0016224`
to `.0021187` for E/C–F/C. Thus a scalar correction removes a bias but does not
remove local shape errors. The aggregate anchor gate is not a surface-admission gate.

Strong-pair median cross-reprojection difference falls from `.0013323` to
`.0003092` in one direction and `.0016847` to `.0003754` in the reverse direction.
Overlap changes slightly after reprojection; these are agreement diagnostics,
**not independent ground-truth accuracy**. The weak third pair remains unreliable.

| Geometry admission stage | Result |
|---|---:|
| Strong-pair agreed pixels | 27,579 |
| Agreed pixels with no old mesh ray hit | 0 |
| Agreed foreground ≥.003 in front of old mesh | 218 |
| Proposed foreground grid triangles | 290 |
| Removed by corroborated native PatchMatch free-space evidence | 284 |
| Final retained faces | 6 |

The empty-ray control serializes a byte-identical production mesh. Its no-change
result exposes an important distinction: a missing hand layer can reveal an
existing deeper body surface, not necessarily produce a ray miss. The second
variant corrects this eligibility criterion, but almost all its proposed patches
conflict with measured depth. Final 124 native guard checks report zero remaining
contradictions after pruning. That safety result does **not** make six faces a
successful hand reconstruction.

The main agent inspected three anchor-residual panels and the native H/A and
moving-camera original/proposed/guarded geometry comparisons. The large wrist
and forearm tear visibly remains. No new textured RGB or video was rendered;
no face PSNR/SSIM/LPIPS or full-frame metrics were recomputed.

- [F/A–G/A measured residuals](/mnt/data/dec5_foundation_anchor_bias/F004_A_G004_A/anchor_review.png)
- [E/C–F/C measured residuals](/mnt/data/dec5_foundation_anchor_bias/E004_C_F004_C/anchor_review.png)
- [Native hand/forearm geometry versus train GT](/mnt/data/dec5_foundation_foreground_patch/review/H004_A005_1210M6.png)
- [Actual moving-camera geometry comparison](/mnt/data/dec5_foundation_foreground_patch/review/moving.png)
- [Verdict](/mnt/data/dec5_foundation_anchor_bias/visual_review.json)
- [Frozen input/output hashes](/mnt/data/dec5_foundation_anchor_bias/artifact_manifest.json)

Five focused tests pass. All geometry jobs finished normally; source EXRs,
calibration, learned predictions, production meshes and movie remain unchanged.
The guard passes only after rejecting most proposals, not by relaxing its thresholds.

## Insights

Small learned disparity bias is real on the tested measured regions and can be
corrected without moving the calibrated cameras. Spatially held-out testing
prevents treating a fit to its own anchors as validation. Transfer to additional
actor times remains untested.

Two better-aligned learned surfaces largely corroborate already reconstructed
fingers/palm. Their trustworthy overlap does not reconstruct the missing wrist.
Better agreement therefore must not be presented as better hole coverage. The
next useful investigation needs observations spanning the missing wrist/lower
forearm, and must inspect the actual conflicting measured pixels before deciding
which depth evidence is wrong. Neither the learned model nor PatchMatch is
declared correct by a four-view count alone.

Opt-in helpers: `stereo_anchor_bias.py` for spatial validation and
`missing_foreground()` in `build_foundation_foreground_patch.py` for distinguishing
a missing front layer from a completely empty ray. All existing defaults remain
unchanged. Recheck with `scripts/freeze_foundation_bias_canary.py --check`.
