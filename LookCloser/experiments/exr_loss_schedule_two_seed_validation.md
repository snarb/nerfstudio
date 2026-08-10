# EXR loss-schedule two-seed validation

## What was tested

New seeds 43 and 44; native linear EXR, frozen knee maps/geometry guard, matched cumulative point exposure, and dense4 corrected evaluation. Seed 42 is historical context only.

## Results

| Strategy | Mean PSNR | Mean SSIM | Mean LPIPS | Cable gaps | Mean lineage train s |
|---|---:|---:|---:|---:|---:|
| direct_pqmse | 34.793732 | 0.901144 | 0.213592 | 0 | 5916.2 |
| eag_continue | 34.763405 | 0.901025 | 0.212974 | 0 | 5912.5 |
| direct_pql1 | 34.747658 | 0.901082 | 0.213803 | 0 | 5917.6 |
| mature_lpips_to_pql1 | 34.737003 | 0.900355 | 0.205244 | 0 | 5936.6 |
| mature_lpips_to_pqmse | 34.657064 | 0.898698 | 0.196580 | 0 | 5936.7 |
| mature_lpips_continue | 34.342108 | 0.896699 | 0.190733 | 0 | 5946.6 |

Per-seed measurements:

| Strategy | Seed | PSNR | SSIM | LPIPS | Cable gaps | Train s |
|---|---:|---:|---:|---:|---:|---:|
| eag_continue | 43 | 34.993431 | 0.901516 | 0.209626 | 0 | 5938.8 |
| eag_continue | 44 | 34.533379 | 0.900534 | 0.216323 | 0 | 5886.2 |
| direct_pql1 | 43 | 34.971119 | 0.901578 | 0.210617 | 0 | 5938.8 |
| direct_pql1 | 44 | 34.524197 | 0.900587 | 0.216989 | 0 | 5896.3 |
| direct_pqmse | 43 | 35.016891 | 0.901659 | 0.209936 | 0 | 5938.8 |
| direct_pqmse | 44 | 34.570572 | 0.900629 | 0.217247 | 0 | 5893.5 |
| mature_lpips_continue | 43 | 34.591877 | 0.897425 | 0.187113 | 0 | 5969.4 |
| mature_lpips_continue | 44 | 34.092339 | 0.895972 | 0.194352 | 0 | 5923.9 |
| mature_lpips_to_pql1 | 43 | 34.934059 | 0.900587 | 0.200320 | 0 | 5959.3 |
| mature_lpips_to_pql1 | 44 | 34.539948 | 0.900123 | 0.210167 | 0 | 5913.8 |
| mature_lpips_to_pqmse | 43 | 34.722874 | 0.898436 | 0.190629 | 0 | 5959.4 |
| mature_lpips_to_pqmse | 44 | 34.591255 | 0.898960 | 0.202531 | 0 | 5914.0 |

Early-rejected controls:

| Run | Reason | Last PSNR | Last SSIM | Last LPIPS | Train s |
|---|---|---:|---:|---:|---:|
| s43_prefix_lpips | stable_material_degradation_after_2_eval_boundaries | 27.082300 | 0.766785 | 0.472112 | 456.1 |
| s43_prefix_pqmse | stable_material_degradation_after_11_eval_boundaries | 19.539200 | 0.559606 | 0.764756 | 5731.6 |
| s44_prefix_lpips | stable_material_degradation_after_2_eval_boundaries | 26.995400 | 0.771952 | 0.498232 | 387.3 |
| s44_prefix_pql1 | stable_relative_degradation_after_3_eval_boundaries | 33.783900 | 0.888701 | 0.279277 | 1350.8 |
| s44_prefix_pqmse | stable_material_degradation_after_2_eval_boundaries | 21.203200 | 0.682988 | 0.792566 | 885.8 |

Visual review: **pass**.

- Inspected full-frame GT/EAG/mature-PQ-L1/mature-PQ-MSE sheets for all 3 eval views and both seeds; no loss-specific structural regression was visible.
- All six selected-candidate cable crops are continuous; detector reports total_gap_pixels=0 for every evaluated terminal arm.
- Caveat: a small bottom-right color artifact is shared by all seed-44 eval0 variants, including EAG, so it is inherited from the common parent rather than caused by the selected loss schedule.
- Artifact: `/mnt/data/lookcloser_loss_schedule_validation/campaigns/exr_loss_schedule_two_seed_v1/visual_review`

## Insights

`direct_*` branches share the frozen EAG prefix and switch only for the final matched-exposure tail. `mature_lpips_*` branches add the same short 64x64 PQ-L1+LPIPS phase before their final tail.

Quality winner: `mature_lpips_to_pql1`. Selected after variance/time/simplicity rules: `mature_lpips_to_pql1`. Visual review status: `pass`.

Against the highest-PSNR branch (`direct_pqmse`), the selected schedule changes PSNR by -0.056728 dB, LPIPS by -0.008348, and mean end-to-end lineage time by +20.4 s. It is the only strategy inside the frozen paired-seed equivalence bands.
