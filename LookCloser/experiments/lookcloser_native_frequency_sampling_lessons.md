# LookCloser frequency and sampling lessons from Marshawn1660

## What was tested

This is the source and experiment interpretation at the user's stop request on2026-11-09. The [scene report](marshawn_native63_24h.md) owns the full chronological results. These broader lessons use the local [paper](../Paper%20LookCloser.md), sealed D873 source, and strict63 TRAIN/DC-HC-KA validation. The held-out views were repeatedly used to select settings; they are not an untouched test. No cross-scene superiority is established.

## Results

**Measured:** the selected uniform16/P15/core.01/hash25 field reached140000 updates with PSNR27.228975, SSIM.590570 and actor LPIPS.098840536. The learned2D maps were not necessary for this selected recipe; constant16 is a preprocessing input, while its3D frequency grid still evolved independently. Larger field capacity, head proposals, core prior, Adam and rendering policy all contribute to the final result. This is not a pure estimate of native-crop benefit. The older native-versus-HD19.5463% LPIPS comparison used all66 views for training and monitoring; the fresh strict63 HD control had not completed scientific training at this review.

**Source-supported:** the effective frequency rule is more constrained than simply “more detail gets more samples.” With16 levels and resolutions16…8192,

`N(l) = 16*(8192/16)^(l/15)`

Here `N(l)` is the target spatial resolution, not the actual number of samples on a ray.

`dt = clamp(1 / (2*N(l)*||ray_direction/AABB_size||), 0.0001, 0.1)`.

Coarse occupancy traversal first creates intervals using step.0015625. Their requested counts are rounded upward, then the corrected allocator shares at most1024 samples over the full ray. It preserves full-interval coverage rather than discarding its far end. Minimum step, coarse intervals and the ray budget can therefore override the nominal Nyquist rule. [Sampler source](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code/nerfstudio/model_components/lookcloser_samplers.py:303).

**Measured and inferred separately:** the terminal hash25 grid's active-cell centres contain9,519 level0 and20,230 level15 values, with none between. This is not a termination-weight histogram. The legacy projection uses2D frequency times raw pixel focal divided by rendered depth, rounds/clamps to levels and only increases stored values. This supplies a mechanism for saturation and retaining zeros in unupdated cells. A saved zero does not distinguish unvisited support from genuinely low frequency, and neither class is a physical-geometry certificate. The paper's units and implementation's normalization must be audited before treating a map as calibrated scene frequency. [Exact saved-grid audit](/fsx/tmp/marshawn_native63_24h_20261108/research/final_grid_audit/P15_corehash25_140k_CPU_v1/REPORT.md), [projection](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code/nerfstudio/model_components/lookcloser_grid.py:325).

Frequency updates really run every1024 steps: a separate no-gradient2048-ray TRAIN render estimates depths, then max-stamps cells. These points are auxiliary work, excluded from packed training-point totals. Consequently an observer must distinguish training forwards from grid-probe forwards. Updating more often also changes random-number consumption under the historical shared RNG stream. [Callback](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code/nerfstudio/pipelines/lookcloser_pipeline.py:821).

| Actual control | Observation | What it does not establish |
|---|---|---|
| Old grid cadence1024→512/256 |32k global LPIPS improved3.35/3.23%; HC grey-face worsened5.03/5.99%; packed points increased29.4/55.4% |A clean detail or speed win on current strong hash25 parent |
| Early FR switch20k, strengths0/.3/.6 |Small mixed32k changes; FR0 at50k was1.92% worse than old reference |That late strength.3 is optimal on every scene |
| Finer minimum step, max3 interval lookup, larger sample cap |Existing negative or mixed controls, with recipe-specific cost/opacity caveats |That increasing all sample counts universally improves detail |
| Constant training cap12 and later uncapping |Reduced point work, but training-quality tradeoffs; separate cold trajectories |That inference-only cap behavior predicts training gradients |
| Frozen terminal inference cap10 |Global LPIPS.095144829 versus.098840398, lower PSNR/SSIM, some face/boot regressions |A new trained model, physical geometry correctness or video stability |
| Frozen cap9 |Lower aggregate LPIPS than10, but native-face/boot regressions caused rejection as a detail upgrade |That the numerically lowest aggregate is the preferred renderer |

FR changes feature suppression independently of ray sampling. Its strength blends one-sided high-level feature weights toward1; strength0 leaves all feature weights at1 without disabling ARM or the grid. Early FR0/.6 controls share the20k switch with FR20k; against the old75941 reference they also move the historical boundary seed reset. Comparisons must identify which complete event changed. [Feature weights](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code/nerfstudio/fields/lookcloser_field.py:370), [bound audit and exact captures](/fsx/tmp/marshawn_native63_24h_20261108/research/frequency_to_samples_audit_CPU/REPORT.md).

## Insights and remaining limits

- Audit actual call paths before screening a knob. Alpha-threshold and early-stop variants were rejected as inference no-ops because the eval sigma/visibility branch was inactive. Cap14 was min-step-floor-bound for the audited geometry. FAS level weights can also cancel when every2D map has the same level. A hardcoded sampling multiplier is not an exposed worker parameter.
- Proposal changes require the actual full-mixture normalization, including actor/body/head/fallback quotas and raster support. Variable resolution and batch4096 must use their real counts, not an8192 assumption. CPU draws and actual ray/gradient canaries test different failure modes; neither substitutes for the other.
- Keep single cold trajectories and own-resume provenance distinct. Field, Adam, learned occupancy,3D frequency, counters and RNG must continue together. Backtracked work consumes the physical budget. Persistent packed counters are read once; resumed endpoint counters are not summed.
- More points and more capacity can help, but cost and local detail must be measured. Model-loop clocks exclude setup, queue and2D fitting; mixed-host timing cannot establish a causal speedup. Ordered inference timing has cache effects. One seed and selected observations do not provide uncertainty estimates or a complete campaign Pareto frontier.
- Masked validation scores omit exterior floaters and can trade faces against clothing. Native images, alpha and novel views remain necessary. Camera-motion rendering with one fixed model tests view-dependent shimmer; it would not test independent-frame video flicker. The motion-path draft was stopped before qualification and produced no render or result.

At stop, current-parent cadence512 remained an unrun hypothesis, not an adopted improvement. The old100/120 comparison under policy10 was a fixed-policy protocol, not a measured result in this audit. No further training, source modification or GPU work is authorized by this report.


## Occupancy and geometry lessons

**Measured:** the terminal R0 grid retains every previously covered nominal actor-contour ray;32804 heldout rays have zero misses. Its510 TRAIN misses exactly reproduce known static-prop annotation contaminants. Nevertheless raw renders retain substantial floor sheets and hair halos. Thus contour-ray coverage is necessary but does not certify surface depth, empty space or ideal geometry. [Terminal audit](/fsx/tmp/marshawn_native63_24h_20261108/research/final_grid_audit/P15_corehash25_140k_CPU_v1/REPORT.md).

**Measured:**8358 active cells have two source-clear votes and234 robust low cells have such contradictions. These are useful inspection candidates, not safe deletion labels. Occluded/out-of-FOV/uncertain volume remains UNKNOWN. Earlier floor clipping damaged boots; an earlier R1 control added27089 cells without recovering the historical510. Neither aggressive pruning nor one-voxel dilation is a universal repair.

**Source-supported:** occupancy and frequency use different indexing conventions in this implementation; replay the exact lookup and its float32 boundary behavior. Saved aliases/capture/core must agree, and capture age must match the update period. EMA maxima and disconnected components are historical diagnostics, not direct measured alpha or proof of false occupancy. A clean2D overlay alone cannot establish containment: preserve the rigid world↔floor-box transforms and inspect ray entry/exit/depth or inside-box views. The numeric AABB bounds alone are insufficient.

**Recommendations inferred from these failures:** use a conservative multi-view grid with explicit UNKNOWN and protected native hair/foot margins; audit representative contradictory cells against source photos before any deletion rule. Re-run native heldout/novel views after changing traversal. These procedures reduce known failure risks; this session does not prove they yield a perfect grid on another scene.

## Data and operational lessons

**Measured limitations:** older native-versus-HD crop gains were in-sample; the strong strict63 resolution control was interrupted. The selected soft-confidence/core-prior model still has exterior artifacts; claiming that the soft band caused them would require a separate causal control. Precise masks, native crops, head quotas, geometry and capacity must be isolated in matched comparisons rather than attributing a combined leader to one change.

**Recommended practice:** freeze verified transfer functions, exposure and calibration before geometry/quality comparisons; preserve exact crop principal-point shifts and ray equivalence. Validate the actual host UID's ability to write nested output/journal/stop receipts, not merely read a spec. A readback that retains a prior worker PID is a setup transition until fresh progress proves otherwise. Persistent admission STOP and exact-owned process identity are separate safeguards. Raw exit143 after requested shutdown is not experimental evidence of a bad model.

**Final status:** all four hosts' training was independently verified stopped. [Authoritative final report](marshawn_native63_final_results.md) and [frozen recipe](marshawn_native63_frozen_recipe.md) supersede the earlier chronological live status. No remaining hypothesis in this file authorizes further work.
