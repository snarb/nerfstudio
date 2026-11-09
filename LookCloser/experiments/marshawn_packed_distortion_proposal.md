# Marshawn remaining floor patches: distortion proposal, not trained

## What was tested

The unchanged accepted PNG R0 8k field/data/scene.loose and existing LOS/surface
terms were probed with an optional native packed-weight distortion objective,
starting at paper weight .01. Actual floor ray audits establish that broad
termination persists when occupancy is bypassed; the local approximate panel
plane remains diagnostic and supplies no new target. Two explicit coordinates
were compared on the same packed graph: normalized bounded distance (linear,
legacy adaptation) and normalized inverse depth (paper-like disparity).

## Results

Ten CPU tests passed against independent quadratic values/gradients, empty rays,
FP64 prefixes, source corruption, ROI handling and hook cleanup/RNG behavior.
The actual DEV3 L40S/Torch2.4 probe loaded the exact baseline checkpoint into the
optional pipeline subclass, with zero optimizer updates. Native field/grid hash,
parameter groups, checkpoint keys and hooks were preserved. Global random-packet
RNG verification covers TorchCPU/CUDA; the supplemental floor probe additionally
preserved Python and NumPy RNG.

At .01, two random train packets give linear gradients at 6.87–7.07% of RGB total
and 26.5–29.2% of RGB base-network gradients. Inverse-depth global norms are
similar, 6.88–7.86% and 28.4–30.4%, so global magnitude alone hides the floor issue.

| Vetted train floor pixel | Linear gradient/RGB | Inverse-depth gradient/RGB |
|---|---:|---:|
| AA550,650 |16.53%|.1507%|
| GA1350,500 |6.37%|.0443%|
| GA1450,550 |4.96%|.0422%|
| OA1450,750 |24.47%|8.77%|

The first three far-floor inverse-depth gradients are 110–145× weaker. The
single-ray diagnostic uses equivalent per-ray AMP scale 32 instead of the
2048-ray packet scale 65536; the initial single-ray overflow is preserved in the
staging logs. No production AMP setting changed. All successful gradients are
finite. This probe alone measures no trained quality improvement.

- [Actual source, normalization, probe receipts and review proposal](/fsx/tmp/marshawn_001660_hd/packed_distortion_STAGED/README_REVIEW.md)
- [Random-packet CUDA probe](/fsx/tmp/marshawn_001660_hd/packed_distortion_STAGED/actual_probe_dev3/actual_cuda_probe.json)
- [Four-train-floor CUDA probe](/fsx/tmp/marshawn_001660_hd/packed_distortion_STAGED/actual_floor_probe_dev3_scale32/actual_floor_probe.json)
- [Independent actual floor termination audit](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/PNG/audit.json)

## Insights and next steps

Review linear .01 as an explicitly bounded-world/legacy coordinate adaptation,
not a claim of paper disparity fidelity. No coefficient increase is justified by
these measurements; compensating inverse-depth far attenuation with 100× weight
would also amplify already sizable global/near-ray effects. Distortion cannot
identify correct depth by itself, repair an already thin wrong sheet, or
regularize skipped/unqueried cells. Preserve genuine wrinkles/reflections.

Root review was required before training. A causal cold control must keep the
original DEV3 runtime, PNG data, scene.loose, LOS/surface, seed 42, max 22000 and
default stopping. Actual all-view and floor-crop/profile gates remain required:
visible floor patches mean the user's quality objective is still incomplete.

## Matched cold control: actual result

After independent production-config review and the actual cold-config AMP probe
both passed, one new cold seed42 model ran on the same DEV3 L40S/Torch2.4 runtime.
It retained PNG data, native field/schema, scene.loose, LOS .01, surface .01,
radius0, native dynamic batching, optimizer/scaler and default plateau stopping.
Only bounded-linear packed distortion .01 was added. This is a scene adaptation,
not the paper's disparity coordinate.

The run stopped normally after the 8k evaluation. Latest compact step8600 gives a
conservative upper bound of 8610 optimizer attempts, below the 22000 cap; training
was not restarted. The selected checkpoint is 8000, selected by all-eval PSNR
with LPIPS tie-break within .07dB. Final evaluation rendered all three native
1920×1012 views from that checkpoint.

| Evaluation domain, three-view mean | Original PNG8k | Linear .01 PNG8k |
|---|---:|---:|
| Original scalar PNG PSNR / SSIM / LPIPS |27.532591 / .780458 / .359162|28.709667 / .796946 / .317703|
| Common canonical fixed-display PSNR / SSIM / LPIPS |27.415559 / .781585 / .358962|28.598103 / .796725 / .319146|

The common comparison scores quantized PNG predictions unchanged against the
same previously validated canonical common GT. It never PQ-decodes or applies
exposure again to these PNG predictions. The common-domain original baseline
uses its released inference guard; the new model uses its trained radius0 mask.
The original-domain row instead compares the native checkpoint evaluations.

Manual inspection covered all three whole views and three identical HD floor
crops, including the new model at2k/4k/8k. People, floor seams and background
features are clearer. Gray mottling remains visible on the HC/KA floor, so
the requested absence of floor/mat spots is **not achieved**. Grid correctness,
novel-view behavior and depth-profile changes require new checkpoint-specific
audits; old baseline occupancy certificates do not transfer to this model.

- [Immutable source/config/probe/launch and final-stop receipts](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/finished_receipt.json)
- [Selected native HD evaluation](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/run/marshawn_001660_png_distortion_linear001/instant-ngp-bounded/cold_seed42/eval_best_step-000008000.json)
- [Common canonical comparison](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/common_domain_best8000_vs_PNG/comparison.json)
- [GT / original8k / new8k whole views](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/matched_best8k_GT_original_distortion.jpg)
- [DC floor crop](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/matched_best8k_floor_native_DC.jpg), [HC floor crop](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/matched_best8k_floor_native_HC.jpg), [KA floor crop](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/matched_best8k_floor_native_KA.jpg)

Controller55684 and train worker55709 exited; DEV3 GPU was verified idle after
final evaluation. The previously accepted PNG package remains unchanged.
