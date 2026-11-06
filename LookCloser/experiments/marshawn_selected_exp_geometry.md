# Completed native-exp parent: actual selected-checkpoint geometry

## What was tested

The structural exp(h+1) parent completed at126316 and selected **60752**, rather than its final checkpoint. Its v1 held-out actor PSNR/SSIM/LPIPS are **26.87955 / .792158 / .0540711**. This separate read-only audit preserves the actual selected checkpoint, completion receipt and authoritative occupancy sidecar. Completion SHA256: `738f3c1f909fd6ebeb821ce9fbee2d326e6bafae22764fb4a7b2c7e30d246229`.

The same103 predeclared TRAIN pixel rays were traced through actual native occupancy, a diagnostic single halo, all cells and dense8192/16384 integration. Actual corrected SH, exp(h+1), legacy frequency projection and **FR1 at60752** were checked against the loaded field. No optimizer, callback, model or active training state changed. The source group names below describe prospective auxiliary roles; this exp parent fitted only RGB and its original recipe, not those auxiliary labels.

## Results

All103 rays passed the declared8192→16384 RGB/opacity convergence checks. The existing C producer remained alive and advanced84956→86182 during the authorized≤10GiB read-only overlap. A CPU report assembly initially omitted the trace-hash dictionary; that bookkeeping was repaired and the existing immutable arrays reused without another GPU replay.

| Source group | Rays | Native ±.02 mass | Native front mass | Median signed t50 residual | Median10–90 width | Mean10–90 width |
|---|---:|---:|---:|---:|---:|---:|
| Old18 source rays | 18 | 87.24% | 2.90% | +.000649 | .012026 | .042149 |
| New66 source rays | 66 | 89.31% | 6.54% | +.000946 | .018887 | .061157 |
| Old nearby reserves | 8 | 97.57% | 0.00% | −.000656 | .006332 | .007252 |
| New whole-patch reserves | 9 | 80.52% | 18.55% | +.000302 | .006501 | .089180 |

Signed t50 is native median termination depth minus the recovered witness projection. Quantiles refer to original ray probability, with exact exponential inversion inside stored intervals; no opacity renormalization is applied. Distances are prepared-world units, not certified meters.

The small median reserve residual conceals two severe HA outliers:

| Reserved source | t50 residual |10–90 width |
|---|---:|---:|
|5568, inherited black-sleeve label | −.375399 | .401330 |
|5556, grey hip/trouser | −.324155 | .352981 |

[Observed source crops and actual native cumulative termination](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_exp_selected_000060752/selected_HA_depth_outliers.jpg) were visually inspected. Point5556 lies clearly on the grey garment. Rechecking the complete HA/IE/JB triplet shows5568 on a hanging braid or its boundary against chest fabric, so its inherited sleeve label is inaccurate. A separate material annotation marks these three rays as unsuitable for a pure opaque-garment acceptance test. The sealed103 rays and original measurements remain unchanged;5568 is absent from the auxiliary training archive. The5556 result alone provides a clear garment-ray example of premature termination. [Triplet recheck](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_exp_selected_000060752/reserved5568_source_triplet.png) and `reserved5568_material_annotation.json` preserve this qualification separately.

Local receipts and per-ray summaries: [/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_exp_selected_000060752](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_exp_selected_000060752). Protected checkpoint and all raw profile arrays remain at the same campaign-relative path on silver. The original15188 evidence is unchanged.

## Insights

The selected exp model has strong numerical RGB metrics and good nearby-shirt depth, but substantial foreground geometry remains in a spatially separate reserved garment region. Group medians and ±.02 containment alone conceal this failure. Continue to report signed residuals, distribution width and individual outliers alongside RGB metrics.

These measurements do not certify occupancy in skipped space and do not promote a model or change any conditional experiment-launch permission. The full-objective P/PB arms remain a controlled test of whether expanded geometry and the positive band improve these failures; the separately queued actor-objective experiment tests incompatible background supervision.
