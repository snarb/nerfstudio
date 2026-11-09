# Marshawn1660: frozen selected recipe

## What was tested

Frozen at the user's explicit stop on 2026-11-09. **No training, evaluation or final66 launch is pending authorization in this document.** This is the selected detail/LPIPS compromise, not a claim of universally optimal parameters. [Final results](marshawn_native63_final_results.md), [complete chronological record](marshawn_native63_24h.md), [machine-readable selection](recipes/marshawn1660_native63_hash25/selection.json).

Training: `P_B8beta99_SAM15_core001_hash25`, actual140000 updates, terminal checkpoint index139999. Inference-only override: `adaptive_max_frequency_level=10`. The field was trained with the maximum-frequency cap unset; do not silently train it at10 or12.

## Results and exact recipe

| Component | Frozen setting |
|---|---|
| Source frame / images |1660; original8192×4320 TIFF source; unresized native actor crops, per-camera integer crop; no training-image resize |
| Photometry |Accepted RAW Log3G10/RWG → linear Rec709, gain4/camera EV/radial correction → Reinhard+sRGB PNG; inherited preparation, not a new PQ conversion |
| Split |63 training cameras; DC/HC/KA held out; repeatedly used for validation, not an untouched test |
| Coordinate frame |Floor-aligned rigid box frame; both4×4 transforms in [geometry.json](recipes/marshawn1660_native63_hash25/geometry.json) are mandatory |
| AABB in that box frame |[-.91,-.26,-.08] → [.15,.50,.63]; these are not original-world coordinates |
| Independent initialization |Fresh field, Adam, learned occupancy/EMA and3D frequency; own-resume only; no cross-frame or cross-arm transfer |
| Training rays / seed |8192; seed42; fixed ray batch (`target_num_samples_per_batch=0`) |
| Sampling |`body80`, W64 confidence raster, precise native SAM head proposal15%; actual mixture normalization; no1/q correction |
| Full-ramp8192 quota |2116 core-uniform,2116 person-FAS,1092 protective-band,1639 whole-valid,1229 heads; head ramp4096→8192 |
| Input frequency map |Constant scalar `uniform16`; no learned2D fitting required;3D grid still starts empty and updates |
| Field |16 levels,2 features/level, resolution16→8192, log2 hash size25, hidden width64, SH4/corrected directions, trunc_exp density, sigmoid RGB |
| Reconstruction |Source-bound Charbonnier RGB objective with confidence normalization; no LPIPS training term |
| Optimizer |Adam LR.01, betas(.9,.99), epsilon1e-15, weight decay0; mixed precision |
| LR schedule |Exponential .01→.0001 over200000 steps, no warmup; do not shorten horizon to140k merely because training stops there |
| Original fixed-sampler warmup |4096 steps, then corrected adaptive ray marching |
| ARM training |Coarse.0015625; min step1e-4; max step.1; midpoint interval estimate; min frequency0/max unset; max1024 samples over the whole ray |
| Feature reweighting |Strength1 then.3 at75941, including the inherited seed-reset event; preserve the complete event |
|3D frequency updates |Every1024 steps, separate2048-ray no-gradient TRAIN depth probe; max-stamping/legacy projection |
| Occupancy |128³,1 level, EMA.95, threshold.01, update16, warmup4096; semantic static admission mapped hash b084919b… |
| Occupancy dilation |R0 in training and evaluation; no adopted R1, pruning or floor clipping |
| Opaque-material prior |Coefficient.01, fixed TRAIN-only32-native-pixel-eroded core assets |
| Distortion schedule |.01→.1 over4096→12288; preserve source schedule rather than the serialized initial scalar alone |
| Steps |140000 actual attempted optimizer updates maximum; replay consumes budget; terminal model reached the limit |
| Selected inference |Same140k weights, cap10; other ARM/FR/grid settings unchanged; no SSAA, image cleanup or masking in the scored raw renders |

Selected inference: PSNR27.100272, SSIM.579104, actor AlexLPIPS.095144829. Default inference on the same weights:27.228975/.590570/.098840536. Training loop297.730558 Silver minutes excludes preparation/setup/queue; this is not an end-to-end per-frame clock. Cap9 has lower aggregate LPIPS.094086626 but worsens HC face/boot detail, so it is retained as a numeric candidate rather than the selected detail renderer.

## Reproduction and artifacts

The [native training spec](recipes/marshawn1660_native63_hash25/training_spec_native_PROVENANCE.json) and [terminal config](recipes/marshawn1660_native63_hash25/terminal_config_native_PROVENANCE_ONLY.yml) are byte-exact provenance copies. The terminal config contains its old own-resume pointer: **it is not a cold-training launcher**. Reproduction requires a new output/stage path and the validated worker's cold `build_config(spec,resume=None)` path, with all load fields None and initial field/Adam/grid audits. The archived paths are native to Silver; this is not a portable runtime bundle. No existing output should be reused for a new frame.

- [Durable140k archive](/fsx/tmp/marshawn_native63_24h_20261108/delivery/model_archive/P_B8beta99_SAM15_core001_hash25_140k_native63/ARCHIVE_COMPLETE.json): exact checkpoint468bbf38…/config4f74fbe4…/native spec4241952e…/source receipts.
- [Selected raw HC cap10](/fsx/tmp/marshawn_native63_24h_20261108/ops/frozen_hash25_140_caps10to14_finite/actual_native/maxfreq10/HC.png), [default HC](/fsx/tmp/marshawn_native63_24h_20261108/review/arms/P_B8beta99_SAM15_core001_hash25/renders/step-000139999/HC.png), [matching GT](/fsx/tmp/marshawn_native63_24h_20261108/data/uniform16/images/HC.png). These are1464×2360 native crops, not enlarged4K files.
- [Terminal model review](/fsx/tmp/marshawn_native63_24h_20261108/review/P15_hash25_final140_CPU/REPORT.md), [cap10 review](/fsx/tmp/marshawn_native63_24h_20261108/review/frozen_H25_140_cap10_CPU/REPORT.md), [cap8/9 tradeoffs](/fsx/tmp/marshawn_native63_24h_20261108/review/frozen_H25_140_caps89_CPU/REPORT.md).

## Insights and limits

There is no completed all66 final field, final video or camera-motion certificate from this stop. The existing final66 data are preparation only. The chosen settings do not remove the visible hair halos/floor sheets and do not establish ideal physical occupancy. Keep measured numeric winners, visual detail selection and untested future ideas separate.
