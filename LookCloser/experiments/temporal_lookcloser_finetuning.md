# Temporal LookCloser fine-tuning

## What was tested

Implemented the safe temporal fine-tuning infrastructure before starting the 45-dataset campaign:

- exact `fields`-only cross-frame initialization with fresh local state;
- full same-frame resume with a state-preserving fields LR override;
- atomic controller dry-run/preflight/resume guards and promotion gates;
- optical-flow tracked and temporal-difference ROI tooling;
- a real two-stage GPU smoke on frame `007747`.

The long `5e-4 / 1e-3 / 2e-3` screen and sequential chain were deliberately not started from the modified,
uncommitted worktree. Campaign preflight requires a clean `main` so the recorded commit/source fingerprint is
meaningful.

## Results

Preflight inputs validated locally:

| Check | Result |
|---|---|
| Canonical checkpoint | step `91128`, SHA-256 `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930` |
| Canonical config | SHA-256 `a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb` |
| Datasets | 45/45; each `66 train + 3 eval`, transforms SHA matched, 66 frequency maps |
| Runtime | Python `3.10.20`, Torch `2.7.1+cu128`, CUDA `12.8`, canonical TCNN binding matched |
| Capacity guards | about `689.5 GiB` disk free; `97,239 MiB` VRAM free, three-job requirement `81,920 MiB` |
| Deterministic dry-run | exactly three model-only commands at local step `60752` for the declared LRs |
| Targeted and related regression tests | 62 passed |

GPU smoke assertions:

| Frame / phase | Local step | Result |
|---|---:|---|
| `007747`, model-only | 0 | source/copied field hashes equal; fresh Adam; zero pre-update occupancy/frequency/FAS/point state; occupancy updated at step0; fixed warmup active |
| `007747`, full resume + LR÷4 | 1 | pipeline buffers, optimizer, scheduler, scaler and RNG loaded; occupancy/frequency state preserved; FAS and exact point exposure continued; LR override persisted |

The ROI protocol also completed against an existing three-view render set: 3/3 views, 14 crops across all
four required categories, zero serious full-view/ROI artifacts, and a readable 768-pixel-wide contact sheet.

Per-frame campaign results (PSNR, SSIM, and LPIPS only) will be written here by the controller:

| Frame | Parent | LR | Selected local step | PSNR | SSIM | LPIPS | Gate | Frame points | Temporal points |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|

## Insights

- Raw parent step `91128` no longer leaks into a new frame's warmup or checkpoint schedule.
- Model-only loading cannot transfer LPIPS weights, scene bounds, occupancy/frequency grids or telemetry because
  it copies parameter objects belonging to optimizer group `fields`, not a filtered whole-pipeline state dict.
- Changing only Adam's current LR is insufficient with a loaded scheduler; updating scheduler base LRs prevents
  the next scheduler step from restoring the old value while leaving moments and scheduler progress intact.
- Manual contact-sheet approval is intentionally required for promotion; a numerically selected checkpoint is
  rejected rather than silently replaced when its visual gate fails.

## Rejected/failed frames

- None. No long campaign frame has been evaluated or promoted yet.

## Next steps

- Commit the reviewed implementation on `main`, rerun `--preflight-only`, then run `--smoke-test` through the
  production controller so its provenance is stored in the campaign manifest.
- Start/resume the LR screen with the visual-decision file prepared for transfer, seed43 repeat, and scratch
  controls. Continue the chain only from fully accepted parents.
