# MVS conditional-depth continuation: preparation

The 52 manually reviewed train-only plane-sweep candidates expose a remaining floor-depth discrepancy in the native PNG 14k field. This staged trial adds conditional expected first-termination depth on 128 extra rays (32 per actual train camera MA/NA/NB/LB). Graph-connected native packed weights produce Σwt/Σw; the residual is converted to normal height and uses Charbonnier sqrt(error²+.005²)−.005. The robustness scale is an optimization choice, not a calibrated MVS uncertainty interval. Extra ray selection and stratified marching use private/restored RNG. No RGB targets, carving, grid edits or NN parameters are added.

The existing model, RGB data, random-background objective, original SfM/free-space/surface observations, finite floor prior, distortion objective and loaded optimizer/scheduler remain. The LOS accessor additionally receives the independently tested same-forward true-exponential derivative fix; this is an explicit second change, so this trial would not isolate the depth auxiliary alone.

Actual local GPU zero-optimizer proof passed using the complete immutable 14k checkpoint e92e525f…98ae9. The first main packet had 64 rays/4068 samples; the adapted packet had 16496 rays/1206604 samples, plus 128 depth rays/2884 samples. Only field.mlp_base.model.params receives the new auxiliary gradient; the color head receives none. All combined AMP gradients were finite at scaler 131072. Auxiliary/RGB geometry gradient ratios for λ=.001/.01/.1 were .02196/.21963/2.19969 on the adapted packet. Selected λ=.01. Minimum sampled opacity was .999901. NN, Adam, scheduler, scaler, RNG and grid states remained exact, with zero optimizer updates. Native main-forward parity was ≤6.56e−7 RGB and ≤3.34e−6 depth.

The exact tested configuration is sealed in zeroopt_LOCAL_scan/config_TESTED.yml. Controller dry-run passed. The reused runner pauses the same process before 16001, after 2000 new attempts, and requires root/peer/zeroopt gates before training. Conservatively charging the rejected head-only branch gives 17170 prior attempts, global start14001/end-exclusive18830, at most4829 additional attempts and family upper bound21999. Checkpoints/all-view evaluation remain every2000; unscheduled terminal18829 receives actual all-view evaluation before final ranking. Crash capture records failed attempts and state without treating failure dumps as resumable checkpoints.

**Status: prepared, zeroopt passed; no training launched by geometry agent.** MVS photometric depths remain limited, overlapping four-camera evidence; third-view LB selection validation is not an unseen test. No improvement in color or scene quality is claimed.

Artifacts: [/fsx/tmp/marshawn_001660_hd/native_png_MVSdepth14k_STAGED](/fsx/tmp/marshawn_001660_hd/native_png_MVSdepth14k_STAGED), especially zeroopt_LOCAL_scan/MVS_depth_zeroopt_probe.json, configuration_review.json, launch_plan.json and MVS_manifest.json.

## Actual 16k visual gate

ROOT subsequently authorized and launched the same-process continuation. Worker1979124 reached16000 with exact2000 attempted/2000 successful updates, no AMP skips, family attempt upper bound19170. It paused before16001; geometry did not send SIGCONT. Checkpoint SHA5aa08c7cc725cb38e1de33141acc9a5dfd77041d9e808d7d9d97c3832a4c11f2.

| Actual all-three-HD result | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Parent14k | 29.23558 | .813125 | .269608 |
| MVS16k | 29.1929 | .813430 | .268048 |

All208 source rays were remeasured with unchanged actual checkpoint grid and independent uniform8192/16384 quadrature, plus32768 on16 worst convergence cases. Native core contains all52 candidate cells. Dense expected-depth normal absolute error median/p90 improves .003951/.011982→.000133/.000372. Majority termination mass within normal±.005 rises125→192/208, within±.01 rises182→208. However, median10–90%normal thickness .00835→.00808 barely changes; p90 thickness .01331→.01608 worsens. This objective centers the conditional mean; it does not certify a thin surface. Worst16 uniform16384→32768 opacity/band-mass difference ≤.000212.

Actually inspected all-three overview, full native KA, HC/KA floor crops, KA left finite-prior margin and DC corner. Bodies remain coherent, but KA left gray/brown waves and HC/KA floor mottling are essentially unchanged from14k. This is **not accepted as a clean-floor fix**. Cropped metrics are mixed: HC floor31.965/.86452/.22456→32.248/.86633/.21914, KA floor31.736/.82777/.32249→31.660/.82818/.31854, DC corner28.787/.82412/.35417→27.429/.81996/.35555. Better geometry-mean agreement alone does not demonstrate that floor appearance is solved.

Evidence in stage: gate16k_HD, gate16k_all3_GT_parent_MVS_safe.jpg, gate16k_manual_review_geometry.json, gate16k_crop_metrics.json, gate16000_MVS208/{audit.json,comparison14k.json}. Contact source targets are byte-identical float RGB; 13k comparison is original native-core .003 ns-eval PNG for visual reference, explicitly quantized8bit, not the protected-grid download.
