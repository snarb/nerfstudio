# Occupancy Grid Improvement Ideas

## What was tested

This v2 report collects implementation findings and next experiments for
LookCloser occupancy-grid artifacts. No new training run was launched for this
report.

Primary local context:

- Current artifact notes: `occupancy_grid_tests.md`, `holes_inspection.md`,
  `architecture.md`.
- LookCloser code: `nerfstudio/models/lookcloser.py`,
  `nerfstudio/model_components/lookcloser_samplers.py`.
- Nerfstudio Instant-NGP code: `nerfstudio/models/instant_ngp.py`,
  `nerfstudio/configs/method_configs.py`.
- Installed nerfacc: `0.5.2` in the nerfstudio conda environment.
- Local instant-ngp checkout: `/home/ubuntu/repos/instant-ngp`.

The v1 report was updated because nerfacc 0.5.2 has two mean-clamp behaviors
that can make naive `occ_thre`, update-step-size, and `alpha_thre` sweeps no-ops
unless `occs.mean()` is measured first.

## Metrics and decision protocol

Every follow-up run must compare:

| Signal | Direction | Source |
|---|---:|---|
| PSNR | higher better | `ns-eval` JSON |
| SSIM | higher better | `ns-eval` JSON |
| LPIPS | lower better | `ns-eval` JSON |
| Eval loss | lower better | `metrics_compact.csv` selected checkpoint row |
| Artifact score | lower better | `scripts/detect_structural_artifacts.py` on saved eval renders |
| Train seconds | lower better | quiet runner `run_summary.json` |
| Eval seconds | lower better | quiet runner `run_summary.json` |
| Artifact detector seconds | lower better | quiet runner `run_summary.json` |
| Total seconds | lower better | quiet runner `run_summary.json` |

Per-run provenance is required before sweeps: full resolved params, git branch,
git SHA, dirty flag, dataset fingerprint, and frequency-map fingerprint.

Noise-floor policy: run the pinned baseline three times with seeds `42`, `43`,
and `44`. Report sweep deltas in units of baseline standard deviation. Do not
draw conclusions from deltas below one sigma.

Acceptance gate for a candidate config:

- artifact score improves by at least 10% and more than one baseline sigma;
- SSIM and LPIPS are no worse than baseline minus one sigma;
- PSNR drop is no more than `0.2 dB`;
- total seconds are no more than `+25%`;
- ties are ordered by artifact score, SSIM, LPIPS, PSNR, eval loss, runtime.

## Verified code facts

### Nerfstudio and nerfacc

- Nerfstudio Instant-NGP defaults use `grid_resolution=128`, `grid_levels=4`,
  `alpha_thre=0.01`, `cone_angle=0.004`, and default `render_step_size =
  scene_aabb_diag / 1000`.
- The `instant-ngp-bounded` preset changes the bounded-scene behavior:
  `grid_levels=1`, `alpha_thre=0.0`, `cone_angle=0.0`, scene contraction
  disabled, and `near_plane=0.01`.
- LookCloser uses `OccGridEstimator(..., levels=1)`. Its default
  `render_step_size` is `scene_aabb_diag / 1000 * render_step_size_mult`, with
  `render_step_size_mult=0.75`.
- In LookCloser ARM, `render_step_size` is the occupancy update scale through
  `density_fn(x) * render_step_size`. Adaptive traversal instead uses
  `adaptive_coarse_step_size` when set, otherwise `adaptive_max_step_size`.
- Nerfstudio currently calls `nerfacc.update_every_n_steps()` without exposing
  nerfacc update knobs. Installed defaults are `occ_thre=1e-2`,
  `ema_decay=0.95`, `warmup_steps=256`, and `n=16`.
- Nerfacc samples all cells during its warmup, then samples
  `cells_per_lvl // 4` random cells plus currently occupied cells.
- Occupancy values update as `max(old * ema_decay, new_occ)`, then binary
  occupancy uses `occs > min(mean(occs), occ_thre)`.
- Mean-clamp trap: when `mean(occs) < occ_thre`, changing `occ_thre` above that
  mean is a no-op for binary occupancy. In the same regime,
  `occupancy_update_step_size` is scale-invariant for the binary grid because it
  scales both `occs` and `mean(occs)`.
- Alpha clamp trap: nerfacc sampling clamps `alpha_thre` to
  `min(alpha_thre, occs.mean())` before density-based train-time pruning. Values
  above `occs.mean()` are equivalent even during training.
- Eval caveat: nerfstudio samplers pass `sigma_fn` to nerfacc only during
  training. In normal eval, `sigma_fn=None`, so `alpha_thre` does not act as
  density-based eval pruning. Eval still uses the binary occupancy grid and
  deterministic traversal (`stratified=False`).
- ARM handoff caveat: `adaptive_warmup_steps` only gates the LookCloser training
  path. Eval runs adaptive whenever ARM is enabled.

Useful GitHub issue signal:

- nerfacc issue #213: scale `render_step_size` linearly with AABB to keep roughly
  the same sample count; `occ_thre=0.01` is reasonable around 1000-2000 samples.
- nerfacc issue #181: multi-level grids are primarily for expanded AABBs /
  unbounded-like coverage. This bounded indoor scene should start with a
  single-level grid.

### Instant-NGP behavior to borrow

Local instant-ngp uses a dense `128^3` density grid per cascade and packs a
bitfield for traversal. For `aabb_scale=4`, it uses three active cascades.

Transferable details:

- Occupancy values are optical-thickness-like: `density * MIN_CONE_STEPSIZE`.
- Updates are max-with-decay: `max(prev * decay, new)`, explicitly to preserve
  very thin features.
- Threshold is `min(0.01, mean_density)`.
- Occupancy bitfields are conservatively max-pooled into coarser levels.
- Cells unseen by any training camera are initialized as untrained (`-1`), while
  visible cells start at `0`.
- Recent local/upstream instant-ngp commits did not materially change the
  density-grid algorithm.

Useful issue signal:

- NVlabs/instant-ngp#123: update index sequence is a cheap broad-coverage
  permutation.
- NVlabs/instant-ngp#360: initial empty-grid concerns are mitigated by
  visible-cell initialization and early positive density.
- NVlabs/instant-ngp#661: threshold and mean-density behavior are intentional;
  tune the optical-thickness threshold rather than assuming a bug.

## Implemented hooks for this phase

LookCloser should expose these knobs with defaults preserving current behavior:

```python
occupancy_occ_thre: float = 1e-2
occupancy_ema_decay: float = 0.95
occupancy_warmup_steps: int = 256
occupancy_update_interval: int = 16
occupancy_update_step_size: Optional[float] = None
occupancy_thre_clamp_mult: float = 1.0
occupancy_dilation_radius: int = 0
```

Post-update logic must run after every actual nerfacc update because `_update`
recomputes `binaries`:

```python
if step % occupancy_update_interval == 0:
    if occupancy_thre_clamp_mult != 1.0:
        threshold = min(occs.mean() * occupancy_thre_clamp_mult, occupancy_occ_thre)
        binaries = occs > threshold
    if occupancy_dilation_radius > 0:
        binaries = dilate(binaries, occupancy_dilation_radius)
```

Dilation:

```python
import torch.nn.functional as F

def dilate_occ_binaries(occupancy_grid, radius: int) -> None:
    if radius <= 0:
        return
    binaries = occupancy_grid.binaries.float()[:, None]  # [levels, 1, D, H, W]
    occupancy_grid.binaries = (
        F.max_pool3d(binaries, kernel_size=2 * radius + 1, stride=1, padding=radius)[:, 0] > 0
    )
```

Log per update:

- occupancy ratio per level;
- `occs.mean()` and `occs.max()`;
- default and effective binarization threshold;
- effective alpha threshold after nerfacc mean clamp;
- cells flipped on/off since the previous update;
- mean samples per ray and zero-sample ray rate where available.

Artifact-to-grid debugging is handled by
`scripts/debug_artifact_occupancy_grid.py`. It loads a run/checkpoint, detects
the largest artifact bbox in a side-by-side eval render, projects sampled
artifact pixels through the eval camera, and reports whether the surface-depth
voxels and along-ray voxels are occupied in the nerfacc grid. Its output is a
JSON/Markdown pair plus a candidate-panel overlay, and it classifies the failure
as `grid_miss_likely` or `field_issue_likely`.

## Stage 0 diagnostics

Run these before any training sweep.

| # | Test | Read |
|---|---|---|
| 0.1 | Dense-render override on current best checkpoint and artifact views | Artifact disappears means traversal/grid miss; artifact persists means the field did not learn it during training. |
| 0.2 | Dump occupancy stats from existing checkpoints: ratio, `occs.mean()`, effective threshold, effective alpha threshold | If `mean(occs) < 1e-3`, drop naive `occ_thre` / update-step-size sweeps and use clamp multiplier, resolution, decay, and dilation. |
| 0.3 | Run `scripts/debug_artifact_occupancy_grid.py` on the largest artifact bbox | Confirms whether the hole is an off-cell / voxel-boundary miss and whether tuning grid resolution/decay/dilation is relevant. |
| 0.4 | Eval same checkpoint with `alpha_thre=0`, `0.0025`, `0.01`; compare per-pixel max abs diff and PSNR between renders | Expected near-identical eval renders. If not, investigate eval path first. |
| 0.5 | Eval pre-handoff checkpoints with ARM force-disabled vs enabled | Measures trained-fixed/eval-adaptive mismatch. |

## Stage 1 observability and noise floor

Implement exposed occupancy knobs, per-update occupancy stats, provenance, and
multi-view artifact scoring. Then run the pinned baseline for seeds `42`, `43`,
and `44`.

Artifact scoring must run on a fixed set of eval views including the known
thin-stand view. Report per-view scores and aggregate with max score as the
primary safety metric and mean score as a secondary metric. Spot-check about 10
render/crop pairs once; if score disagrees with visual judgment, fix the
detector before trusting selection.

## Stage 2 screening matrix

Use a screen-then-confirm protocol:

1. Screen each candidate with seed `42`.
2. Confirm the best 1-2 candidates with seeds `42`, `43`, `44`.
3. Save final renders from the selected checkpoint and run artifact detection.
4. Skip any row Stage 0 marked dead.

Use a shortened schedule for screening, then full-length confirmation in Stage 3.

| # | Factor | Values | Notes |
|---|---|---|---|
| 2.1 | `adaptive_warmup_steps` / ARM handoff step | `0`, `8192`, `12288`, `16384`, `20000` | `4096` intentionally removed; pair with Stage 0.5 control. |
| 2.2 | `grid_resolution` | `128`, `192`, `256` | Direct thin-geometry volume-fraction knob. |
| 2.3 | `occupancy_ema_decay` | `0.95`, `0.99` | Principal thin-structure survival knob after resolution. |
| 2.4 | `occupancy_update_interval` | `16`, `4` | Hold per-step decay constant: for `n=4`, use `0.95 ** (4/16) ~= 0.987`. |
| 2.5 | `occupancy_dilation_radius` | `0`, `1` | Price via occupancy ratio, samples/ray, and total seconds. |
| 2.6 | `occupancy_thre_clamp_mult` | `1.0`, `0.5` | Main threshold knob when mean-clamp regime is active. |
| 2.7 | `occupancy_occ_thre` | `1e-2`, `1e-3` | Only if Stage 0.2 shows `mean(occs) > occ_thre`; do not also sweep update step size. |
| 2.8 | `adaptive_coarse_step_size` and matched `adaptive_max_step_size` | `0.0125`, `0.00625`, `0.003125` | `max_steps_per_ray=2048`; use `4096` only for dense finalists. |
| 2.9 | Handoff refresh | off, on | Prefer public-path temporary dense updates; private `_update` only as diagnostic. |
| 2.10 | nerfacc `occupancy_warmup_steps` | `256`, `1024` | Report dense-update count `warmup_steps / n`, not raw steps. |

## Stage 3 confirmation

- Combine the 2-4 screening winners, expected among resolution + decay,
  resolution + dilation, handoff-step winner + refresh.
- Run full schedule, two seeds each first; expand to three seeds for candidates
  near the acceptance threshold.
- Controls on the best candidate:
  - dense-render override for traversal-miss residual;
  - eval-loss-selected checkpoint vs artifact-score-selected checkpoint.
- Apply the acceptance gate. Failing candidates are reported, not shipped.

## Contingency

Run only if Stage 3 fails the artifact gate:

- Proposal-assisted hybrid intervals before frequency-aware subdivision. This is
  not a pure proposal replacement and is time-boxed.
- Instant-NGP-style camera-visibility cell initialization only if Stage 0.3 shows
  floaters rather than holes.
- Zip-NeRF/PyNeRF multi-scale ideas, VDB/HDDA, and learned occupancy remain out
  of scope for this phase.

## Insights and next steps

- The likely failure is still the interaction between ARM handoff, coarse
  occupancy traversal, hidden nerfacc update policy, and thin-geometry voxel
  coverage.
- `mean(occs)` is the single most informative new scalar. It gates the usefulness
  of `occ_thre`, update-step-size, and `alpha_thre` sweeps.
- The thin-structure survival model links grid resolution, decay, and dilation:
  hit probability per update is roughly the structure's cell volume fraction,
  misses decay the cell, and threshold crossing creates flicker/hole risk.
- `adaptive_warmup_steps` needs a real sweep. Existing evidence mostly compares
  immediate handoff vs `12288`, not a proper schedule.
- `alpha_thre` is a training-path knob and is additionally clamped by
  `mean(occs)` even in training.
- Keep `render_step_size` (occupancy update scale) and
  `adaptive_coarse_step_size` (traversal) separated in all reports.
- Artifact score plus saved crops remain the acceptance gate. Global
  PSNR/SSIM/LPIPS can improve while the thin-stand artifact worsens.
