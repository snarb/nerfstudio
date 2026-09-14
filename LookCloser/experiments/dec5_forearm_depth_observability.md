# DEC5 forearm: local shape fit versus depth-discriminating evidence

## What was tested

The starting point is the [RGB-qualified curved control](dec5_forearm_rgb_confidence.md),
not the production video mesh. All experiments preserve original production
geometry, fixed camera calibration/profiles/exposure and hard-source rendering.
No new PatchMatch run, source edit, held-out input or full-video replacement.

**Local measured-anchor residual.** Fit a smooth scalar correction along each
reference ray, using one median per camera/node and then across cameras. Graph
smoothness 0.2, prior weight 0.001, displacement bound 0.006, nearest assignment
distance 0.75 px. Original/boundary vertices are pinned. Fit even-indexed source
cameras and check odd-indexed cameras before the all-source fit. The baseline
prior itself uses all cameras, so this is conditional residual validation, not
independent held-out scene evaluation. Recheck semantic limits and the complete
RGB-qualified free-space guard after deformation.

**Depth-hypothesis comparison.** A compatible measured depth is not automatically
more photometrically plausible than the nearer inferred patch. Compare their
calibrated 5x5 RGB patches in the same previously qualified witnesses. A margin
of 0.01 mean absolute RGB is fixed for the experiment. The pilot carved only
when three witnesses preferred the observed depth. It incorrectly abstained
when alternative projections were unavailable; two inspected cloth-boundary
samples lost their existing veto for that reason. The supported variant
preserves the previous RGB rule whenever fewer than three alternatives can be
compared. Only a genuinely comparable ambiguity can now suppress a veto.
The pilot is retained separately, with its original source snapshots.

The supported rule is opt-in `--witness-comparison-margin 0.01`, requiring the
existing `--witness-rgb-limit 0.12`. Defaults are unchanged. It does not certify
the inferred surface as measured or recover a statistically calibrated depth
probability.

## Results

### Shape fit: rejected on the first canary

On 001037, conditional camera-partition P90 depth error improved
0.004883 -> 0.001116; median error 0.001870 -> 0.000339, over 1677 samples.
The full fit has 625 observed nodes, 82 fixed boundary nodes and two clipped
displacements. The fresh semantic, bounded-ray deformation and 124-ray audits
pass. Nevertheless, native moving and train RGB show a larger upper side cavity.
The train-skin black-pixel count worsens 2080 -> 2194; PSNR 19.4532 -> 19.0459,
SSIM 0.668844 -> 0.659412. LPIPS improves slightly, 0.453904 -> 0.445091, which
does not override the visual failure. **Rejected; no neighboring-frame runs.**

Artifacts: [shape control](/mnt/data/dec5_forearm_anchor_residual/001037),
[native comparison and metrics](/mnt/data/dec5_forearm_anchor_residual_review).

### Confidence comparison: partial improvement, not production acceptance

The three-time supported variant keeps the pre-carve proposals byte-identical.
It preserves every previous retained vertex/face and adds 7/291/135 surviving
triangles on 001029/001033/001037. Each passes fresh 62-camera, two-offset ray
checks under the **new** rule; this is not a pass of the former RGB-only rule.

All scores below use the same fixed manual **train forearm-skin ROI** in
H004_A005_1210M6. They are not held-out face scores or full-frame metrics, and
the main face campaign CSV is unchanged.

| Frame | Variant | PSNR | SSIM | LPIPS | Black skin pixels |
|---|---|---:|---:|---:|---:|
| 001029 | RGB guard | 31.52632 | 0.940505 | 0.079503 | 18 |
| 001029 | Supported comparison | 31.52622 | 0.940515 | 0.079498 | 18 |
| 001033 | RGB guard | 21.17724 | 0.822946 | 0.284872 | 1400 |
| 001033 | Supported comparison | 21.73409 | 0.835507 | 0.256374 | 1260 |
| 001037 | RGB guard | 19.45324 | 0.668844 | 0.453904 | 2080 |
| 001037 | Pilot, no availability fallback | 19.75074 | 0.680482 | 0.427273 | 1957 |
| 001037 | Supported comparison | 19.64523 | 0.675119 | 0.440426 | 1992 |

The supported variant reduces some side pinholes without a conspicuous new
failure in the inspected moving/train crops. Major lateral cavities, wrist
holes and coarse hand seams remain. The apparently better pilot scores do not
override its unavailable-comparison weakness. Main-agent review covers the
two shape RGB panels, two pilot/support panels and all six transfer panels;
none is declared artifact-free. No isolated canary is spliced into the movie.

Artifacts:

- [Witness votes and controls](/mnt/data/dec5_forearm_contrastive_witness_control)
- [Pilot and supported comparisons](/mnt/data/dec5_forearm_contrastive_review)
- [Three-time supported meshes/audits](/mnt/data/dec5_forearm_contrastive_guard_supported)
- [Three-time metrics and native review](/mnt/data/dec5_forearm_contrastive_transfer_review)

33 focused tests pass, including camera-balanced fitting, fixed pins, bounded
displacement, flat-patch ambiguity, discriminating patches and unavailable-view
fallback. All workers terminated normally; source and published outputs remain
unchanged. These are local geometry/confidence experiments, not new model defaults.

## Insights

Lower residual error on a subset of measured depths can coexist with worse
rendered coverage. Hard carving then exposes conflicts that the local shape fit
did not reconcile. Comparable alternative hypotheses provide a better reason to
abstain than simple missing evidence, but this only fixes part of the problem.

The updated [residual-stage diagnosis](/mnt/data/dec5_forearm_supported_residual_topology/001037)
was inspected in all four views. In the H004_A005 diagnostic plane envelope,
1951 pixels remain uncovered: 947 fall outside the reference skin polygon,
300 were rejected before initial triangulation, 185 lack an initial triangle,
46 are lost at transfer and 417 at later carving (56 hit old reference geometry).
The grey outside-reference band is mainly a boundary limitation; this does not
mean all major wrist holes are caused by that band. Plane-envelope counts are
diagnostics, not measured surface quality or a new RGB metric.

A further read-only [admission-order probe](/mnt/data/dec5_forearm_admission_order_probe)
isolates a larger next hypothesis: semantic admission happens on the flat
proposal **before** its shape is refined. Applying the existing global quadratic
prior to the original candidate points before the same semantic checks yields:

| Frame | Previously accepted | Newly eligible points | Previously accepted points lost |
|---|---:|---:|---:|
| 001029 | 1136 | 0 | 0 |
| 001033 | 7585 | 52 | 40 |
| 001037 | 8861 | 1319 | 0 |

All three native reference overlays were inspected; the 001037 newly eligible
points occupy actual forearm skin near the remaining side gap. This is only
point-level evidence, before boundary feathering, triangle extent or depth guards.
It is **not** a reconstructed mesh improvement. Removing the initial depth-only
veto is not the explanation: only one of 1744 rejected 001037 candidates has
that veto, whereas 1743 have a semantic disagreement. The next geometry test
should correct the ordering of shape inference and irreversible admission,
retaining a common policy and all final guards, rather than keep tuning a
post-filter on an already truncated proposal.
