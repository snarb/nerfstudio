# 007740 → 007747 LookCloser transfer

## What was tested

The goal was to recover the accepted frame `007747` quality without random model
initialization, starting from the canonical frame `007740` leader:

- checkpoint: `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt`;
- source metrics: `29.840143 / 0.669203 / 0.219455`;
- target dataset: `/home/brans/temporal_perframe_stride7_45f/007747`;
- fixed visual ROI: eval0 `(700, 100, 1120, 480)`, containing the contacting
  hands, separated fingers and chain.

The initial low-LR/full-resume experiments were rejected. They kept too much
frame-dependent `007740` state and did not match the accepted `007747`
architecture/data recipe. The corrected experiment transferred only a
function-preserving hash24 version of the leader fields into fresh target-frame
training state.

## Correct recipe

1. Expand the leader hash grid from log2 size 23 to 24:

   ```bash
   /home/brans/repos/nerfstudio/.venv/bin/python \
     LookCloser/scripts/expand_lookcloser_hash_checkpoint.py \
     --source-checkpoint \
       /home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt \
     --output-checkpoint \
       /home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/expanded_leader_step-000091128.ckpt
   ```

   The conversion repeats each saturated per-level table into every new modulus
   partition. Copying the old table only into the first half is wrong because
   changing the hash-table modulus changes which rows are queried. Adam moments
   are expanded by the same mapping for completeness.

2. Before target training, evaluate the expanded source on `007740`. The
   conversion must be function-preserving. The measured result was
   `29.840143 / 0.669203 / 0.219455`, equal to the canonical source evaluation
   within evaluator floating-point noise; render maximum pixel difference was
   `1/255`.

3. Start frame `007747` with `checkpoint_load_mode=model_parameters_only` from
   the expanded checkpoint. This copies fields but creates fresh Adam,
   scheduler, scaler, RNG, occupancy grid, frequency grid, FAS counter and
   frame point telemetry. Do not full-resume target training directly from the
   `007740` checkpoint.

4. Freeze these target recipe coordinates:

   - seed `42`, B4096, stable occupancy reduction;
   - hash log2 size `24`, 16 levels × 2 features, max resolution `8192`;
   - frequency maps `lookcloser_frequencies_chroma422`;
   - FAS strength `0.75`;
   - Adam LR `0.01`;
   - exponential scheduler from `0.01` to `0.0001` over `200000` local steps;
   - fixed traversal/fresh occupancy warmup for local updates `0..4095`;
   - FR `0.3` through checkpoint `60752`;
   - full same-frame resume from `60752` with FR `0.2`; retain target
     model/Adam/scheduler/scaler/RNG and target grids thereafter.

5. Full-evaluate at every `15188` local updates. Crash-safety saves may occur
   more often with `save_only_latest_checkpoint`, but they are not additional
   evaluation boundaries. Continue one interval at a time after `60752`.

6. Confirm plateau only after two consecutive intervals satisfy all metric
   thresholds and the fixed moving-detail ROI does not visibly improve. Select
   max PSNR, then minimum LPIPS among checkpoints within an inclusive `0.07 dB`
   of that maximum. Keep the visual ROI decision explicit rather than allowing
   a negligible global tie-break to hide a crop regression.

The exact campaign record is:

`/home/brans/lookcloser_temporal_finetune_runs/campaigns/007747_hash24_transplant_freshadam_v1/campaign.json`

## Results

Fresh full-eval trajectory:

| Local step | FR | PSNR | SSIM | LPIPS | ROI PSNR | ROI SSIM | ROI LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | 0.3 | 29.218811 | 0.676875 | 0.354465 | 28.728535 | 0.769734 | 0.194799 |
| 30376 | 0.3 | 29.465122 | 0.686334 | 0.306088 | 29.287648 | 0.781251 | 0.162609 |
| 45564 | 0.3 | 29.658146 | 0.688436 | 0.276877 | 29.489539 | 0.783510 | 0.140840 |
| 60752 | 0.3 | 29.782728 | 0.691371 | 0.260494 | 29.639132 | 0.784593 | 0.131245 |
| 75940 | 0.2 | 29.736145 | 0.693306 | 0.247179 | 29.680964 | 0.784639 | 0.126738 |
| 91128 | 0.2 | **29.819077** | 0.694944 | 0.238740 | 29.677431 | 0.783883 | 0.122619 |
| 106316 | 0.2 | 29.786087 | 0.695211 | 0.234586 | 29.704659 | 0.783537 | 0.119911 |
| 121504 | 0.2 | 29.768623 | 0.691754 | 0.231089 | 29.651444 | 0.783437 | 0.118796 |
| 136692 | 0.2 | 29.773516 | 0.695678 | 0.229076 | 29.749597 | 0.783384 | 0.116511 |
| 151880 | 0.2 | 29.791567 | **0.697209** | 0.227115 | 29.708475 | 0.783057 | 0.116433 |
| 167068 | 0.2 | 29.783972 | 0.696873 | 0.225269 | 29.701001 | 0.782640 | 0.115542 |
| 182256 | 0.2 | 29.766191 | 0.696109 | **0.225243** | 29.653533 | 0.782179 | 0.117388 |

Step `91128` is the maximum-PSNR checkpoint. The inclusive selector window is
therefore PSNR `>=29.749077`; step `182256` is the formal global selection
because its LPIPS is only `0.000026` below step `167068`.

The fixed ROI regressed at `182256`, however. Step `167068` is the visual
selection: all three full views and the ROI have zero detected serious
artifacts; fingers remain separated and the chain remains continuous. Direct
side-by-side inspection against the accepted scratch step `197444` found no
meaningful loss of sharpness.

| Candidate | PSNR | SSIM | LPIPS | ROI PSNR | ROI SSIM | ROI LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Canonical `007740` leader | 29.840143 | 0.669203 | 0.219455 | 29.735380 | 0.773583 | 0.112038 |
| Accepted `007747` scratch, step 197444 | 29.753056 | 0.673878 | **0.203921** | 29.776102 | 0.783162 | **0.112835** |
| Transfer max-PSNR, step 91128 | **29.819077** | 0.694944 | 0.238740 | 29.677431 | 0.783883 | 0.122619 |
| Transfer visual selection, step 167068 | 29.783972 | **0.696873** | 0.225269 | 29.701001 | 0.782640 | 0.115542 |
| Transfer formal selector, step 182256 | 29.766191 | 0.696109 | 0.225243 | 29.653533 | 0.782179 | 0.117388 |

The transfer visual selection improves scratch PSNR by `0.030916 dB` and SSIM
by `0.022995`; LPIPS is `0.021348` worse. Against the canonical leader it is
`0.056171 dB` lower in PSNR, `0.027670` higher in SSIM and `0.005814` worse in
LPIPS. In the priority crop its LPIPS is only `0.002708` worse than accepted
scratch and the visual gate passes.

The last two intervals confirm plateau:

| Interval | PSNR growth | SSIM growth | LPIPS improvement | Moving-detail review |
|---|---:|---:|---:|---|
| 151880 → 167068 | -0.007595 | -0.000336 | 0.001846 | no visible improvement |
| 167068 → 182256 | -0.017781 | -0.000764 | 0.000026 | no improvement; ROI regressed |

## Why the earlier transfer was worse

- It compared a hash23/FAS1/regular-map/FR0.3 run with the accepted
  hash24/FAS0.75/chroma422-map/FR0.2 scratch recipe. That was not a controlled
  transfer comparison.
- Naively changing hash23 to hash24 is not a capacity extension: the changed
  modulus redirects lookups. The leader must be expanded by per-level modulus
  partition repetition and source-equivalence tested before use.
- Occupancy is only one target-dependent state. Reusing the `007740` frequency
  grid, FAS exposure, dynamic-ray telemetry and low-LR Adam trajectory anchored
  the old moving silhouette. At step `15188`, the rejected full-resume branch
  measured `29.1298 / 0.677986 / 0.3774`; its hand ROI was only
  `26.8369 / 0.7315 / 0.26484`.
- The assumption that transfer required a lower LR was wrong for this motion.
  Fresh target Adam at the scratch LR `0.01` was needed to move geometry while
  the inherited field still supplied useful static-scene structure. At matched
  step `15188`, the corrected transfer reached
  `29.218811 / 0.676875 / 0.354465` and ROI
  `28.728535 / 0.769734 / 0.194799`.

## How to avoid this failure

- Treat JPEG revision, frequency-map directory, FAS, hash capacity and FR
  schedule as checkpoint-compatible recipe coordinates; assert all of them in
  the run manifest.
- Never reuse source-frame occupancy/frequency/FAS state across moving frames.
  Use model-only fields transfer into a fresh target pipeline. Use full resume
  only inside the same frame.
- Never prefix-copy a TCNN hash table when changing its modulus. Run the
  function-preserving converter and source eval before any target experiment.
- Screen LR from the local target schedule, not from the small LR stored late in
  the source checkpoint.
- Compare candidates at identical local updates and with the same three-view
  eval plus the fixed moving-detail crop.
- Preserve both the formal global selection and the visual selection when a
  negligible LPIPS tie-break conflicts with the priority ROI.

## Outputs

- Visual checkpoint:
  `/home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/freshadam_runs/007747/lookcloser/007747_hash24_transplant_freshadam_lr1e-02_exp_fr02_s167068/nerfstudio_models/step-000167068.ckpt`
- Visual renders:
  `/home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/freshadam_runs/007747/lookcloser/007747_hash24_transplant_freshadam_lr1e-02_exp_fr02_s167068/evaluations/step-000167068/renders`
- Fixed crop contact sheet:
  `/home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/freshadam_runs/007747/lookcloser/007747_hash24_transplant_freshadam_lr1e-02_exp_fr02_s167068/evaluations/step-000167068/priority_roi/contact_hands_chain_2x2.png`
- Scratch/transfer comparison:
  `/home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/final_comparison_007747_s167068.png`
- Formal-selector checkpoint:
  `/home/brans/lookcloser_temporal_finetune_runs/hash24_transplant_v1/freshadam_runs/007747/lookcloser/007747_hash24_transplant_freshadam_lr1e-02_exp_fr02_s182256/nerfstudio_models/step-000182256.ckpt`
