# DEC5 forearm: calibrated RGB qualification of free-space witnesses

## What was tested

The previous [annotation-domain fix](dec5_forearm_annotation_domain.md) removed
one artificial split but left serious lateral forearm holes. This experiment
asks whether chromaticity-only witnesses still falsely carve inferred skin.

`diagnose_forearm_qualified_witnesses.py` visualizes **only** observations with
three depth-and-chroma-compatible witnesses. Eighteen deterministic query points
across the three largest initial integer-grid veto cameras on 001037 produce
54 observed/candidate correspondence pairs. All three native sheets were directly
inspected. Candidate RGB comparisons are not visibility-certified evidence.

The distinction matters: many raw depth vetoes are already rejected by the
chroma filter. In the surviving examples, F004_E005 mostly matches other actual
forearm skin; do not declare those measurements false just because they lie
outside another manually drawn ROI. Some G004_E005 examples instead match a
much brighter tan background. B004_E005 includes cloth/skin-boundary samples
where vetoing an overextended patch may be correct. Blanket veto removal is
therefore not justified.

The opt-in `--witness-rgb-limit` adds absolute mean RGB error of calibrated 5x5
display patches to the existing depth, roundtrip, angle and chromaticity tests.
There is no newly fitted exposure, per-image normalization, texture averaging,
source-image edit or held-out input. Three **other** cameras must pass both
color tests. Existing defaults remain unchanged.

Before mesh evaluation, thresholds 0.04/0.06/0.08/0.10/0.12 were compared using
observed positive skin anchors at 001029/001033/001037. The selected 0.12 is the
smallest tested threshold retaining at least 98% of the previously qualified
samples in every one of nine camera/time groups. It retains 8583/8598 overall
(99.83%); these overlapping observations are not statistically independent.
The 0.10 control loses 35/941 samples in the worst group and is not selected.
This is a diagnostic threshold, not a universally calibrated probability.

All three mesh canaries use the identical pre-carve curved proposal and previous
annotation-domain correction. Only the additional RGB witness gate changes.

## Results

On the 001037 diagnostic samples, qualified veto pixels change:

| Query camera | Chroma-only | Chroma + RGB |
|---|---:|---:|
| F004_E005_1210FP | 276 | 276 |
| G004_E005_1211KO | 260 | 203 |
| B004_E005_1210VE | 258 | 254 |

These are sampled ray observations, not unique triangles or hole pixels.
The complete 62-camera guard retains 0/101/104 additional triangles on
001029/001033/001037. The previous retained faces and all vertices are preserved;
the pre-carve meshes are byte-identical to the matched control.

Fixed manual **train forearm-skin ROI**, H004_A005_1210M6:

| Frame | Guard | PSNR | SSIM | LPIPS | Black skin pixels |
|---|---|---:|---:|---:|---:|
| 001029 | Chroma | 31.5263 | 0.940505 | 0.079503 | 18 |
| 001029 | Chroma + RGB | 31.5263 | 0.940505 | 0.079503 | 18 |
| 001033 | Chroma | 21.0872 | 0.818142 | 0.288355 | 1433 |
| 001033 | Chroma + RGB | 21.1772 | 0.822946 | 0.284872 | 1400 |
| 001037 | Chroma | 19.3682 | 0.653651 | 0.521395 | 2121 |
| 001037 | Chroma + RGB | 19.4532 | 0.668844 | 0.453904 | 2080 |

These are train reprojection diagnostics, **not held-out face scores**, and do
not replace or modify the main face campaign CSV. No full-frame quality metrics.

The main agent inspected all six native moving/train comparisons. Small holes
remain on 001029; severe side cavities and coarse hand seams remain on
001033/001037. The 001037 lower-arm texture is more coherent, but the new
LPIPS cannot be equated with recovered geometry: only 39 skin depth pixels
change, while 307 RGB pixels change (268 at unchanged depth), with 286 source-ID
changes. The unchanged hard-source graph reacts to the repaired connectivity.
For 001033 the corresponding depth/RGB/source changes are 34/100/91 pixels.
The native head strip y=500..1300 is byte-identical in all six comparisons.

All three final meshes pass fresh two-lattice, 62-camera checks under the new
RGB-qualified guard. This is **not** a pass of the old depth-only or chroma-only
guard. Prefix/face-set audits and render receipt hashes also pass; 25 focused
tests pass. Preparation, RGB rendering and audits ran in disjoint supervised
workers on clever-shadow and terminated normally. No new PatchMatch or training
job was needed. Source data, production meshes and the published movie are intact.

Artifacts:

- [Qualified correspondence sheets](/mnt/data/dec5_forearm_qualified_witnesses_001037)
- [Three-time positive-anchor controls](/mnt/data/dec5_forearm_rgb_witness_control)
- [Candidate meshes and fresh audits](/mnt/data/dec5_forearm_rgb_qualified_curve)
- [Metrics, native panels and visual verdict](/mnt/data/dec5_forearm_rgb_guard_review)

![001037 matched moving view](/mnt/data/dec5_forearm_rgb_guard_review/001037/moving_comparison.png)

## Insights

Discarding brightness is useful for gain invariance but can turn similarly
colored background into apparently valid foreground evidence. Once exposure and
camera profiles are fixed, retaining an absolute-color check removes a concrete
class of false free-space evidence. The measured improvement is partial; the
candidate is not promoted into the 150-frame video.

Remaining F004_E005 matches show that some disagreements are between two nearby
skin-surface hypotheses, not skin versus background. The next useful test is to
constrain the inferred surface with trustworthy nearby depth observations rather
than keep carving an unconstrained quadratic proposal. Preserve the distinction
between a shape prior, observed evidence and final visual acceptance; additional
color-threshold relaxation alone is not a justified route to an intact arm.
