# 007740 → 007747 hash23 fine-tuning v2

## What was tested

The v2 controller implements direct model-only transfer from the canonical
007740 hash23 leader into fresh target state for the canonical 4:2:2 revision
of 007747. It freezes standard maps, FAS1.0, FR0.3, B4096, warmup4096 and all
leader architecture/runtime coordinates while screening only initial LR and
the exponential-decay horizon.

The staged screen is:

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

## Results

Implementation validation:

| Check | Result |
|---|---|
| Leader checkpoint SHA-256 | `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930` |
| Leader config SHA-256 | `a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb` |
| Dataset revision SHA-256 | `5983bc94168ded04ec6b8fe10ec01f0703417ba903115a01ced4d2b280e996e0` |
| JPEG/map file hashes | 69 JPEG and 132 map/sidecar files verified |
| Dataset split | 66 train + 3 eval |
| Runner tests | 117 passed |
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

Training metrics, native crop comparisons, first leader pass,
`time_to_leader`, plateau selection and any separate visual selection will be
written after the production campaign. Evaluation loss is intentionally
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
