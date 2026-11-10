# Marshawn908–1268 production export checks

## What was tested

Independent export verification and visual review after the six-hour tuning campaign. Frozen recipe: [selected recipe](marshawn_908_1268_tuned_recipe.md). Each final model trains independently on all72cameras, using native crops and the accepted common AABB. Metrics below are in-sample actor metrics, not held-out generalization.

## Results

Verified snapshot: 2026-11-10T07:19:47.223622+00:00. Eleven verified exports; the361-frame campaign remains incomplete.

| Frame | Selected iterations | PSNR(dB) | SSIM | Actor LPIPS |
|---|---:|---:|---:|---:|
| 908 | 144000 | 32.535930 | 0.799172 | 0.108058 |
| 909 | 150000 | 32.572324 | 0.801283 | 0.112001 |
| 910 | 150000 | 32.611592 | 0.801385 | 0.115309 |
| 911 | 150000 | 32.592926 | 0.799519 | 0.116112 |
| 912 | 150000 | 32.609488 | 0.800391 | 0.116158 |
| 914 | 150000 | 32.669783 | 0.802720 | 0.112677 |
| 915 | 144000 | 32.631765 | 0.801563 | 0.113487 |
| 916 | 144000 | 32.661553 | 0.801708 | 0.115050 |
| 918 | 150000 | 32.712914 | 0.803937 | 0.112944 |
| 1102 | 150000 | 32.668340 | 0.798791 | 0.140155 |
| 1112 | 144000 | 32.774396 | 0.798674 | 0.137925 |

The eight Silver models completed150000iterations. Frames915/916 selected144000; the other six selected150000. All eight model files passed streamed SHA verification, cold/native/split checks, and input-frame identity checks. [Batch audit](marshawn_908_1268_diagnostics/silver_first8_completed_model_audit.json). Earlier export receipts: [908](marshawn_908_1268_diagnostics/frame908_completed_model_audit.json), [1112](marshawn_908_1268_diagnostics/first_completed_model_audit.json), [1102](marshawn_908_1268_diagnostics/frame1102_completed_model_audit.json).

[Eight-page raw Full HD and novel-view contact PDF on Drive](https://drive.google.com/file/d/1zyLbyRfg_rgaYcr5VwDCOXKA76T3Z8xT/view). All48 views were visually inspected; each frame has CB/GC/NB calibrated views and0.25SLERP neighbour views,1920x1080 letterboxed output. Both render receipts for every model matched the independently verified selected model SHA. No GT masking was applied to these full-FOV field renders.

## Insights

Four performers remain represented, with source-FOV occlusion/cutoff where applicable. Significant contour halos and unsupported background fragments persist; these results do not establish a clean matte. Native head/source-mask diagnostic patches remain linked in the selected-recipe notes. Continue the frozen sequence campaign and audit each new export; choosing144000 uniformly would miss improvements observed at150000 in six of these eight frames.

Silver artifacts: `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/frames/<frame>/model/inference.ckpt`; full-resolution review assets: `final_reviews/<frame>/{fullhd,novel}/`. The relocated Clever1102 model is under `external_models/clever/001102/model/`. All eight Silver queues advanced automatically to their next frames; use live controller state for current progress rather than this historical snapshot.
