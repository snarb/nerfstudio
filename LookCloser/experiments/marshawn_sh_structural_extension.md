# Marshawn: structural factors after corrected SH

## What was tested

Two cold experiments use the corrected-SH full/body80/R0 recipe as their fixed
parent. Each changes exactly one configuration field. Data, accepted AABB,
original actor-mask scoring protocol, seed42, leader trajectory through106316
and subsequent plateau policy are preserved.

| Arm | Change | Scope |
|---|---|---|
| `structural_exp_hplus1` | `density_activation: softplus → trunc_exp` | Existing FP32 exp(h+1), with clipped-gradient backward; not historical native exp(h) |
| `structural_normalized_jacobian` | `frequency_projection_mode: legacy → normalized_jacobian` | Existing mode bundles normalized Jacobian, half-pixel centers and opacity admission; not a formula-only comparison |

The source is copied from the frozen SH snapshot, with a strict override
whitelist and a guard rejecting simultaneous changes in this first generation.
Original campaigns are unchanged. Historical Luster/DEC5 recipes motivate these
factors but do not prove either works on Marshawn.

## Results

Production was launched at18:44:16UTC on2026-11-06 after root review; controller
PID2050975, root `/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_extension_controller`.
Both workers entered training on silver-sable. No quality result is claimed yet.

Seven CPU checks pass. The exp technical run performed30 successful Adam updates
in32 attempts with finite combined gradients after native AMP backoff. The
Jacobian technical run performed1025 updates in1026 attempts and executed the
actual frequency-grid update at1024:1237 admitted samples,1222 nonzero cells,
all values finite and in0–15. Probe instrumentation did not change the recipe.

- [Exact config differences and source bindings](/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_extension/verification.json)
- [Technical release review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_extension/release_review.json)
- [Root release receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_extension/root_release.json)
- [Conditional follow-up plan and historical recipe references](/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_extension/extension_plan.md)

## Insights and next steps

Corrected SH already improved matched15188 actor LPIPS.080374→.069681 and
PSNR24.8495→26.4216, with better support at independently reserved local shirt
points. It still contains false foreground density; see
[the matched depth audit](marshawn_sh_prefix_matched_depth.md).

Compare new factors against that SH parent, including actual renders and depth
mass, rather than only against the weaker original recipe. Frequency-map
threshold tuning must ultimately use the best qualified structural recipe
across campaigns. The original fixed DAG remains a historical comparison;
its winner is not automatically the parent of later experiments.

The user requires investigation through at least2026-11-07 14:30:16UTC.
Individual runs still stop at plateau. Further combinations, geometry-source
experiments, frequency variants and paired-seed checks depend on measured
results; no unchanged run is extended merely to consume that duration.
