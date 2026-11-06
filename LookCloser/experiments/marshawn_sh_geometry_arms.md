# Marshawn SH-corrected geometry A/B/C: release preparation

## What was tested

Three separate cold-seed42 arms preserve the frozen SH-corrected full/body80/R0 recipe. A adds the previously tested18-ray prefix at0.003. B changes only its source coverage to84 rays. C adds an opaque-clothing band at0.001 to B. Running campaign code and jobs were not modified.

Candidate root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/sh_geometry_candidate_20261106`.

Silver root: `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/sh_geometry_candidate_20261106`.

The new worker requires both configuration and actual field `correct_sh_directions=True` and native `density_activation='softplus'`. Full resolved config comparison against the frozen SH reference differs only in custom parser module identity; parser behavior and bounds are identical. The training loop, historical evaluation gates,75941 frequency/seed transition, optimizer and validation plateau policy remain unchanged. Production must train through106316 and continue until the existing plateau rule is met.

## Results

Full4096-ray actual-model probes used128 auxiliary rays,64 prefix samples and16 band samples. All separate and actual combined backward passes were finite at the original native AMP scale65536. No optimizer update was made, no parameter `.grad` was accumulated, and full model state hashes matched before and after. The norm diagnostic covers the hash encoding and geometry MLP; it does not claim to measure total color-head gradients or quality.

| Arm/source | Prefix0.003 / RGB geometry gradient norm | Band0.001 / RGB | Band0.003 / RGB |
|---|---:|---:|---:|
| A: original18 | 13.61% | 23.29% (diagnostic only) | 69.86% (diagnostic only) |
| B: expanded84 | 3.26% | 10.56% | 31.67% |
| C: expanded84 | 3.26% | 10.55% | 31.65% |

C therefore uses0.001. Its actual combined geometry-gradient norm was1.0114 times the RGB norm. A/B retain prefix0.003 for controlled source attribution. Small B/C probe differences are measured GPU behavior; identical seeds are not a guarantee of bit-identical fused-kernel arithmetic.

The band queries the actual native density function, integrates optical thickness in FP32, and evaluates stable `-log(-expm1(-tau))`. It does not infer an exponential from raw logits, replace derivatives, or introduce MVS/floor/occupancy targets. Prefix packets are bit-identical with and without band sampling. The band has an independent deterministic private seed stream; it preserves global RNG and source identity selection.

The policy halfwidth is `max(.005,.01*depth)` with strict AABB clipping/insets. Old9 shirt-point SIFT graphs were additionally recovered: maximum actual fitted residual0.543HD pixels and maximum available leave-one-view-out XYZ shift0.000650 world units. New source maxima are0.966HD pixels and0.001011 world units. Every available old/new leave-one-out shift is smaller than its band halfwidth; all clipped bands retain their witness. Three-view points have no fourth-view leave-one-out certificate. These are conservative soft priors, not calibrated confidence intervals. All old/new reserved points remain absent from both objectives.

Each arm separately trained32 technical attempts, paused after15, and resumed from its actual checkpoint:

| Arm | Successful Adam steps before → after | AMP scale before/after | Actual rendered views |
|---|---|---:|---|
| A | 15 →31 | 32768 | DC, HC, KA |
| B | 15 →31 | 32768 | DC, HC, KA |
| C | 15 →31 | 32768 | DC, HC, KA |

Adam states were nonempty, moments/model parameters finite, both private RNG streams reproduced the next packet, and actual RGB/opacity outputs were finite at full1920×1012. All nine images were visually checked in [technical overview](/fsx/tmp/marshawn_001660_hd/tuning_20261106/sh_geometry_candidate_20261106/technical32_overview.jpg). They are untrained fog after32 attempts and support no reconstruction-quality claim.

The technical suite initially treated the deliberate pause exit143 as a failure. This orchestration-only handling was corrected; the verified step15 checkpoints were resumed without changing worker/objective source. The logs preserve this event. GPU3/5/7 shared global locks covered each active suite and both sides of resume; all technical leases are now released.

## Insights and launch gate

`config_parity.json`, `old18_band_uncertainty.json`, `probe_A/B/C/probe.json`, and `verification_A/B/C.json` contain exact evidence. The frozen code inventory has1494 files and matched every remote SHA256. Geometry manifests, review files and arrays remain sealed. The hash-bound release wrapper verifies source and evidence before it starts a worker.

`controller_plan.json` defines three autonomous jobs on silverGPU5/6/7, checks every30 seconds, and uses shared per-user GPU locks. Root reviewed and approved actual `release_A/B/C.json`; all remote source/spec/evidence gates passed. The durable controller launched as PID276766, with state under the candidate root `controller/state.json`. The `.pending.json` files remain preparation history. This separate branch does not join or alter the existing campaign DAG. Root retains ongoing supervision and the required visual gates.

The next quality decision must compare all three held-out actor renders/LPIPS/PSNR/SSIM and matched geometry diagnostics, including the newly reserved whole garment patches. Better fitted mass on the supervised sources alone is insufficient evidence of improved generalization or correct occupancy.
