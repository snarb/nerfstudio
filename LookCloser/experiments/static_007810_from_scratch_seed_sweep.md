# Static LookCloser from scratch on 007810: three-seed sweep

## What was tested

Three independent LookCloser campaigns start from random initialization on
`/home/brans/temporal_perframe_stride7_45f/007810`. They run concurrently on GPU 0 with seeds
42, 43 and 44. No checkpoint from 007740, another frame, or another seed is loaded.

Each campaign uses the frozen static-leader weight recipe:

- 66 filename-split train views and 3 eval views;
- standard `lookcloser_frequencies` maps;
- B4096, hash23, max-res8192;
- adaptive ARM with 4096-update warmup, cap1024 and coarse step0.00625;
- FAS1.0 and frequency grid from update zero;
- Charbonnier RGB, distortion0.01 and early depth0.001;
- Adam `0.01→0.0001` over 200000 scheduler steps;
- Stage A with FR1.0 through step75940;
- same-campaign full-state continuation with FR0.3;
- checkpoint and full three-view eval every15188 updates.

After the fixed leader stages, each seed is extended one interval at a time until its last two
full-eval intervals both satisfy the numeric plateau thresholds: PSNR gain below0.03 dB, SSIM gain
below0.001 and LPIPS improvement below0.003. The final cross-seed selector maximizes full-eval
PSNR, then minimizes LPIPS within the inclusive0.07-dB window. Loss is not used in reporting.

Controller worktree:
`/home/brans/repos/nerfstudio_007810_scratch3`, branch
`scratch-007810-seed-sweep`.

Campaign root:
`/mnt/data/lookcloser_007810_from_scratch_seed_sweep`.

## Results

All three campaigns completed without OOM or CUDA errors. The supervisor ran from
2026-07-28 16:14 UTC through 2026-07-29 00:34 UTC. Each seed stopped independently only after two
trailing numeric plateau intervals; the selected seed's final two intervals were also reviewed
side by side and recorded as no visible improvement.

| Seed | Latest step | Selected step | PSNR | SSIM | LPIPS | Full serious artifacts | Plateau |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 151880 | 151880 | 29.631721 | 0.676035 | 0.218873 | 1 | yes |
| 43 | 212632 | **182256** | **29.697031** | 0.671247 | **0.209829** | **0** | yes |
| 44 | 167068 | 167068 | 29.614494 | **0.682954** | 0.220858 | 1 | yes |

The cross-seed trajectory maximum is seed43 step121504 at PSNR29.715626. The inclusive0.07-dB
window contains seed43 step182256, whose PSNR is only0.018595 dB lower and whose LPIPS improves by
0.005302. The frozen selector therefore chooses seed43 step182256. No loss value participates in
the report or selection.

Selected artifacts:

- checkpoint:
  `/mnt/data/lookcloser_007810_from_scratch_seed_sweep/007810_leader_recipe_seed43_tail_s182256/lookcloser/20260728_161435/nerfstudio_models/step-000182256.ckpt`;
- checkpoint SHA-256:
  `d7772a5cb9901a08fd4138e384f32bffa0b47aaa502ab741de41061717e9b7e2`;
- fresh eval:
  `/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/eval.json`;
- fresh renders:
  `/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/renders`;
- numeric selection:
  `/mnt/data/lookcloser_007810_from_scratch_seed_sweep/selection_numeric.json`;
- supervision log:
  `/mnt/data/lookcloser_007810_from_scratch_seed_sweep/supervision.jsonl`.

Selected contact crop:

![seed43 step182256 contact crop](/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/roi/contact_hands_chain_2x2.png)

Selected full views:

![eval0 GT and render](/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/renders/eval_img_0000.png)

![eval1 GT and render](/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/renders/eval_img_0001.png)

![eval2 GT and render](/mnt/data/lookcloser_007810_from_scratch_seed_sweep/campaigns/007810_leader_recipe_seed43/evaluations/step-000182256/renders/eval_img_0002.png)

The selected candidate has zero serious full-view artifacts and a non-serious fixed ROI. Manual
review passes all three views: raised-hand fingers remain distinct to the extent present in the
motion-blurred GT, cables and thin structures remain continuous, and there is no new structural
hole or obvious local blur. Steps197444 and212632 do not add visible detail and form the final
confirmed visual plateau.

## Insights

- The three-seed run was necessary. Seed43 was weak at the first boundary
  (PSNR26.9542 at15188), but later became the best trajectory; an early seed decision would have
  selected seed44 incorrectly.
- Seed44 retained the best SSIM, while seed43 won the declared PSNR-window/LPIPS selector and was
  the only selected-per-seed candidate with zero automatic serious full-view artifacts.
- Seed43 was non-monotonic: it reached PSNR29.715626 at121504, fell sharply, recovered to
  29.697031/0.209829 at182256, then plateaued. Extending by measured intervals rather than stopping
  at the first regression materially improved the final LPIPS.
- The selected target result is0.143112 dB below the static 007740 leader PSNR gate
  (29.840143), but improves its LPIPS by0.009626 and SSIM by0.002044. The strict static-leader
  all-metric gate therefore remains a visible miss rather than being relabeled as acceptance on a
  different frame.
- The main worktree was not modified. All controller changes and documentation are committed only
  on `scratch-007810-seed-sweep`.
