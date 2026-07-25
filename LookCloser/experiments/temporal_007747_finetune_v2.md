# 007740 → 007747 hash23 fine-tuning v2

## What was tested

The v2 controller implements direct model-only transfer from the canonical
007740 hash23 leader into fresh target state for the canonical 4:2:2 revision
of 007747. It freezes standard maps, FAS1.0, FR0.3, B4096, warmup4096 and all
leader architecture/runtime coordinates while screening only initial LR and
the exponential-decay horizon.

The original staged screen was:

| Phase | Initial LR | Final LR | Horizon |
|---|---:|---:|---:|
| Wave A | 0.0075 | 0.0001 | 200000 |
| Wave A, leader control | 0.0100 | 0.0001 | 200000 |
| Wave A | 0.0150 | 0.0001 | 200000 |
| Wave B | selected Wave-A LR | 0.0001 | 100000 |
| Wave B | selected Wave-A LR | 0.0001 | 150000 |

The winning schedule is replayed solo from the original leader checkpoint for
the official `time_to_leader`, then continued one interval at a time to a
two-interval metric and visual plateau.

The original `LR=0.01`, 200k-horizon campaign reached a numeric/visual
plateau without passing the LPIPS gate (`0.231206` at step212632 versus the
required `<=0.219455`). A follow-up screen therefore changed only the two
permitted scheduler coordinates:

| Arm | Initial LR | Final LR | Horizon | Step60752 PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| R-L125-H400 | 0.0125 | 0.0001 | 400000 | 29.934767 | 0.678393 | 0.248865 |
| R-L150-H300 | 0.0150 | 0.0001 | 300000 | 29.933397 | 0.680863 | 0.250484 |
| R-L150-H400 | 0.0150 | 0.0001 | 400000 | 29.823717 | 0.678141 | 0.249133 |

`R-L150-H300` was selected after the native crop gate and replayed alone as
`authoritative-R-L150-H300`. The screen artifacts are discovery evidence only;
all timing and final selection below come from the solo replay.

## Results

Implementation validation:

| Check | Result |
|---|---|
| Leader checkpoint SHA-256 | `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930` |
| Leader config SHA-256 | `a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb` |
| Dataset revision SHA-256 | `5983bc94168ded04ec6b8fe10ec01f0703417ba903115a01ced4d2b280e996e0` |
| JPEG/map file hashes | 69 JPEG and 132 map/sidecar files verified |
| Dataset split | 66 train + 3 eval |
| Runner tests before production | 119 passed |
| Native crop smoke | Complete 3×2 leader/scratch/target sheet; zero serious artifacts |
| Production preflight | Frozen provenance/runtime/reference/data checks passed; storage guard stopped at 112.1 GiB free versus 180 GiB required |
| Alternate storage preflight | Passed with 1659.0 GiB free on `/mnt/data` |
| First baseline attempt | Infrastructure stop before eval/update/checkpoint: controller incorrectly assumed the fresh occupancy binary mask was all-true |
| Production campaign | Corrected rerun uses a new `_r2` directory; the failed directory is preserved unchanged |

The recorded regression command was:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python -m pytest -q -o addopts='' \
  LookCloser/tests tests/engine/test_trainer_checkpoint_load_modes.py
```

The clean-main production preflight command was:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python \
  LookCloser/scripts/run_lookcloser_007747_finetune_v2.py --preflight-only
```

It exited with infrastructure code 3 at the final storage check, as intended. No data was deleted
and the floor was not weakened.

The first alternate-storage baseline proved every other startup invariant, including direct field
hash equality, fresh Adam/LR/scheduler/scaler/RNG, and zero occupancy values, frequency grid, FAS
counter and point telemetry. Nerfacc's fresh binary occupancy mask is not all-true. The corrected
audit compares its true-count with the constructor count captured by the trainer, retaining a
fail-closed freshness check without imposing the reset-only all-true representation. No evaluation,
optimizer update or checkpoint occurred in the failed attempt.

The crop smoke used a canonical target-GT-bound composite and the accepted scratch render. The
accepted scratch artifact has a historical GT panel that is not byte-identical to the active
4:2:2 revision; the protocol records this fact while continuing to require every new candidate's
GT panel to match the canonical target JPEG exactly.

The authoritative solo replay completed through step167068:

| Boundary | PSNR | SSIM | LPIPS | Numeric gate | Reviewed crop |
|---:|---:|---:|---:|---|---|
| 60752 | 29.897165 | 0.677997 | 0.249663 | fail | pass |
| 75940 | 29.946390 | 0.679668 | 0.237836 | fail | pass |
| 91128 | 29.882019 | 0.677806 | 0.230007 | fail | pass |
| 106316 | 29.873976 | 0.678924 | 0.224729 | fail | pass |
| 121504 | 29.859125 | 0.677917 | 0.222631 | fail | rejected |
| 136692 | 29.895269 | 0.677203 | 0.217243 | pass | pass |
| 151880 | 29.880142 | 0.675660 | 0.214533 | pass | pass |
| 167068 | 29.849859 | 0.675603 | 0.214825 | pass | no improvement; not selected |

The first complete numeric and visual leader pass is step136692.
`time_to_leader` is `9003.521066166 s` (`2:30:03.521`) from immediately
before the authoritative trainer start through that evaluation.

The two terminal changes, 136692→151880 and 151880→167068, simultaneously
satisfy the declared plateau bounds and have no reviewed crop improvement.
Selection over eligible checkpoints first maximizes PSNR and then minimizes
LPIPS within the inclusive 0.07-dB window. It selects step151880:
`29.880142 / 0.675660 / 0.214533`. A fresh evaluation reproduced
`29.880142 / 0.675660 / 0.214533` with maximum metric drift
`3.73e-7`.

Selected artifacts:

- checkpoint:
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3/authoritative/authoritative-R-L150-H300/lookcloser/run/nerfstudio_models/step-000151880.ckpt`;
- checkpoint SHA-256:
  `000fbc9144505fe4041d61ba71f0f9f804c78de19517b70cd0584d519ae6a358`;
- fresh renders:
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3/final_confirmation/step-000151880/renders`;
- complete campaign:
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3`.

The completed campaign preserves every 15188-step checkpoint, metrics,
three-view renders, native crop comparisons, exact configs, hashes and
separate train/evaluation wall timings. Evaluation loss is intentionally
excluded.

## Insights

The late LR stored in the leader checkpoint is irrelevant to cross-frame
startup: `model_parameters_only` creates a new target optimizer and scheduler
from the candidate config. Full optimizer/scheduler/scaler/RNG/grid state is
loaded only for later same-frame continuation, without an LR override.

The no-update baseline must not be represented by Nerfstudio checkpoint
`step-000000000`, because that checkpoint is written after optimizer update
zero. The v2 runner evaluates the transplanted pipeline directly before
calling `trainer.train()`.

`scripts/run_lookcloser_temporal_finetune.py` is now the production
reproduction entrypoint for the selected result, not a scheduler sweep. Its
defaults freeze direct step91128 hash23 model-only transfer, LR0.015,
300000-step decay horizon and the selected local step151880. One invocation
runs the pre-update baseline, direct training through step60752 and the same
per-interval full-resume sequence through step151880. It creates a new
timestamped v2 directory and never reuses the completed campaign.
