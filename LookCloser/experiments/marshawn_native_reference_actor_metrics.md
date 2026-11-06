# Immutable native-NGP reference under current actor metrics

What was tested: On 2026-11-06, dev3 rendered the existing native-NGP 20,000-step checkpoint without optimization, using the accepted oriented floor box and the current campaign's frozen DC/HC/KA actor masks/crops. This is an auxiliary historical reference, not a LookCloser campaign candidate or replacement for its dependency graph.

Checkpoint: `/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_R1_DEV3_STAGED/immutable_selected_20000/step-000020000.ckpt`; SHA256 `fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2`.

Results: Float32 inside-box RGB was scored with `marshawn_tuning_metrics.py`, using neutral RGB 0.5 outside each fixed actor mask and AlexNet LPIPS without resize. Aggregate weights are equal actors within each view, then equal views. PNG quantization did not affect scores.

| Domain | PSNR ↑ | SSIM ↑ | Masked LPIPS ↓ |
|---|---:|---:|---:|
| All actors/views | 30.14774 | 0.827527 | 0.054466 |
| DC | 30.09301 | 0.831563 | 0.062503 |
| HC | 29.65774 | 0.821725 | 0.054993 |
| KA | 30.69247 | 0.829292 | 0.045904 |
| Black actor | 29.27595 | 0.813543 | 0.054716 |
| Grey actor | 31.01953 | 0.841511 | 0.054217 |

All six eroded actor masks have less than 1% of pixels with opacity below 0.9; the maximum is HC/black, 0.9491%. This diagnostic does not certify clean exterior geometry.

- [Source/native actor crop comparisons](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_reference_20000/metrics/actor_comparison.jpg)
- [Unmasked inside-box views](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_reference_20000/metrics/three_views_inside.jpg)
- [Machine-readable metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_reference_20000/metrics/metrics.json)
- [Source/runtime/checkpoint receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_reference_20000/receipt.json)
- [Historical HC repeatability check](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_reference_20000/historical_parity.json)

Insights: All three source/render actor pairs were visually inspected. Native renders retain clothing folds and hair structure, with some softer face/hand detail. All three unmasked views retain a third person and substantial floor/mat geometry; the mat has large missing regions. These images are not an artifact-free two-actor isolate. The actor-only metric deliberately excludes those exterior defects.

Source RGB SHA256s, full-size raster and intrinsics match the current campaign. Current campaign poses equal the accepted `world_to_box` transform multiplied by native source poses. Native runtime poses were checked against its original source. Rendering preserves native field normalization and occupancy grid, zeros density for sample midpoints outside the accepted box, and reintegrates. Model-state and checkpoint hashes remain unchanged across all three renders; native-compositor parity is checked in each view receipt.

Cross-runtime caveat: this uses the historical native nerfstudio/model snapshot with the current parser module on dev3 (Python 3.8.20, Torch 2.4.1+cu121, TorchMetrics 1.5.2). Its architecture, sampling and full-scene training history differ from the current campaign. Fresh HC rendering agrees with the historical unchanged-model receipt but is not bitwise identical: maximum raw RGBA difference is 0.002874. Current box JSON metadata has changed since the historical receipt; numeric bounds and rigid transform are identical. All reported metrics use fresh renders with one consistent current box file.
