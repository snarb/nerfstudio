# Marshawn: larger uniform batch and 22k decay

## What was tested

One cold local RTX PRO6000 quality trial, seed42, maximum22000 attempts, default plateau stop and mandatory2k in-place gate. Relative to the frozen original PNG linear-distortion control, only the optimization settings change: target samples262144→1048576 and scheduler horizon200000→22000. LR endpoints remain0.01→0.0001. NN, PNG targets, AABB, LOS, surface and distortion losses, uniform sampler and render step remain unchanged. This combines two optimization changes and a different host; it is not a pure causal batch comparison.

[Runnable launch plan](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/launch_plan.json), [five-field config diff including bookkeeping](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/configuration_diff.json), [official scheduler curves](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/scheduler_curve_cpu.json). Original driver and pipeline source are copied byte-for-byte. Config SHA: `4afe885d97a90b048ad452d140767dd4bd2255d1375730f31134a3bd5bab424a`.

## Results

[Actual cold CUDA preflight](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/cold_probe_local/cold_probe.json) made zero optimizer objects/updates. Initial64 rays produced37643 samples; native adaptation then drew1782 unique pixels across all63 train cameras and1039071 packed samples. Combined AMP gradients were finite at scale65536. Peak adapted-probe allocation/reservation was2.11/2.45GiB. NN/schema/groups, original grid and RNG checks passed.

Root launched one controller1953585 and worker1953643. The first64 attempts had2 initial AMP skips, then stable finite updates; native backoff was preserved. The2k checkpoint paused before2001 and passed root's actual three-view HD/floor review for continuation only. Source images, opacity/depth buffers and receipts are retained under `gate2000_HD` and `gate2000_floor_native_HC/KA`.

| Cold run / step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original /2000 | 27.0117 | 0.754104 | 0.424410 |
| New /2000 | 28.2825 | 0.785626 | 0.345541 |
| Original /4000 | 27.7503 | 0.778196 | 0.359429 |
| New /4000 | 28.8731 | 0.800482 | 0.301447 |
| New /6000 | 29.0220 | 0.805989 | 0.286736 |
| New selected /8000, formal HD | 29.128387 | 0.809287 | 0.280373 |

Exact TensorBoard dynamic-ray counts correct the earlier approximately500-ray premise for this particular baseline: original at1990/7990 used next-batch counts2615/3464. New at1990/3990 used13062/14740, with roughly one million actual packed samples. Generic CSV ray-count fields are blank; the TensorBoard `Train Metrics Dict/num_rays_per_batch` value is the **next** batch count, not the batch already rendered.

## Insights

The first2k images show coherent opaque actors without a gross brown replacement cloud. Floor/mat softness and mottling remain; continuation is not clean-quality acceptance. Training remains supervised with controller/worker, compact progress, AMP and GPU checks. No old best or download ZIP is overwritten. Selected-best final images and all-view metrics are required before any quality win is accepted.

The run subsequently stopped normally under the default plateau rule at the8k eval boundary, with last compact training step8320. All checkpoints remain below22000 attempts. Strict all-view PSNR with LPIPS tie-break selected8000; native final HD evaluation succeeded. Only2 initial AMP attempts were skipped; the8k trace records7999 Adam updates, finite weights and scale131072. Controller/worker exited; GPU memory returned to14MiB.

Actual final HD images were manually inspected. Despite improved aggregate metrics, KA lower-left contains a dark smudge absent from GT. The model is therefore **not accepted as a clean-floor win**. This quality failure overrides the aggregate metric improvement; the existing safe13k guarded model is retained by root.

## Black KA region diagnosis

[Nine pixels selected before probing](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/selection.json) cover the smudge and neighboring floor. Exact actual native rays reproduce saved RGB within floating-point tolerance. The same frozen field is tested with source core0, exactly one halo, all admitted cells, and grid-bypassing uniform16384/32768 integration. Actual small ROI renders use the same masks and step0.003. NN/EMA/checkpoint/config state is unchanged and the original mask is restored.

At the darkest selected pixel `(242,747)`:

| Traversal | Opacity | RGB red channel | Expected ray distance |
|---|---:|---:|---:|
| Native core0 | 0.160541 | 0.052943 | 1.23042 |
| One halo1 | 0.511731 | 0.162648 | 1.18905 |
| All cells, step0.003 | 0.511731 | 0.162648 | 1.18905 |
| Uniform16384 | 0.525395 | 0.163932 | 1.15310 |
| Uniform32768 | 0.525427 | 0.163934 | 1.15299 |
| Source GT | not supplied | 0.301961 | not supplied |

The native termination-conditioned RGB codes are approximately `(0.330,0.307,0.283)`, not intrinsically black; lost opacity over the model's black background is the dominant darkening mechanism. This division is descriptive, not a rendered correction or physical albedo estimate.

Grid culling amplifies the deficit, but all-cells traversal does not restore an opaque surface. Uniform16k/32k convergence is tight: maximum opacity difference3.16e-5 and RGB difference2.67e-5 across the nine rays. A missed ultra-thin peak from the fixed0.003 step cannot explain the remaining deficit. The full learned field itself still has an opacity gap along the failing ray.

The exact same nine rays on the safe uniform-LR13k field have opacity approximately1 for native, halo, all-cells and dense traversal. At `(242,747)`, native RGB is approximately `(0.281,0.245,0.229)`; the black smudge is absent, although broader gray residuals remain. These comparisons do not establish exact physical depth or certify the whole volume.

Evidence: [cold8k numeric audit and actual ROI renders](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/cold8k/audit.json), [uniform13k same-ray audit](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/uniform13k/audit.json), [comparison](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/comparison.json). Raw per-sample density, RGB, weights, world XYZ and source-cell membership are retained. Both GT/native/halo/all-cells contacts were manually viewed. No training restart or image compositing was performed.
