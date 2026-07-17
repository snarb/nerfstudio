# Static LookCloser leader: end-to-end reproduction on clever-shadow

Frozen operational recipe: `experiments/static_leader_reproduction_recipe.md`. This file retains
the full forensic trajectory, rejected controls and variance evidence; the frozen recipe is the
short source of truth for future speed comparisons.

Status: active. The paired stable-occupancy S0/S1 ablation is complete. Both runs pass all three
numeric gates at step 91128; S1 also passes automatic full-view/ROI gates and becomes the first
accepted stable candidate, while S0 is rejected for one serious full-view artifact. S1 preserves
cable holes, thin pipe and fingers versus the archive but fails the deliberately strict five-ROI
all-metric aggregate on stand and label. On 2026-07-15 the user accepted measured same-seed ranges
of at most `0.06 dB PSNR / 0.01 SSIM / 0.005 LPIPS`; S0/S1 passes that gate. The paired
stable-occupancy + FP32 TCNN grid-gradient F0/F1 control also passes it at step 91128. Both FP32
checkpoints passed the numeric and automatic clean gates, but the mode is slower and has weaker
strict cable/detail consistency than accepted stable-FP16 S1. Independent-RNG work is parked unless a quality failure
reopens variance. End-to-end speed tuning is now the active phase.

## Technical summary

The archived LPIPS leader was trained through `A@75940 → A_fw03@106316`, not through the later
geometry-first controller. The initial local control therefore fixes the historical algorithm,
seed, 4096-ray batch, FAS/FR history, Adam state, and exponential scheduler. Training progress is
compared by cumulative point samples; dev3 wall-clock is not used as the stopping criterion.

The earlier local LPIPS gap is now explained and numerically overcome. That controller accumulated
only `31–48 B` points, introduced FAS late, ramped FR only to `0.3`, used a corrected ARM allocator,
a `2^19` dynamic point target and piecewise constant LR. The leader accumulated about `250 B`
points with FAS/grid active from zero, FR `1.0` through the first `~167 B`, fixed 4096-ray batches,
and one restored exponential Adam/scheduler trajectory. Exact local H0 and R2 each produce a
scheduled checkpoint that beats all three archived full-frame metrics simultaneously; the
remaining problem is reproducibility and strict detail, not the original LPIPS descent itself.

The checkpoint-matched historical comparison uses step 106316. Campaign acceptance follows the
predeclared selector rule: the first scheduled checkpoint that passes all numeric and automatic
artifact gates is retained. In both cases the metric thresholds are:

| Metric | Gate | Archived local full-precision reference |
|---|---:|---:|
| PSNR | `≥ 29.617964` | 29.617964 |
| SSIM | `≥ 0.668450` | 0.668450 |
| LPIPS | `≤ 0.231135` | 0.231135 |

Full renders, significant-artifact counts and archived detail crops remain guardrails. The archived
checkpoint itself has significant artifacts in two eval views, so the first control may reproduce
the numeric leader without yet satisfying the later clean-production target.

## Scope, sources, and metric definitions

- Scene: `007740`, filename split, 66 train and 3 eval images.
- Local dataset: `/home/brans/temporal_perframe_stride7_45f/007740`.
- Dataset comparison: byte-for-byte manifest against read-only dev3 before every campaign.
- Historical executable worktree: `/home/brans/repos/nerfstudio_leader_repro`, commit `85818149`.
  This is the last executable code change before Run A; the following `fe9dd951` changed only docs.
- Compatibility patch: `pil_to_numpy` uses public `np.asarray(PIL.Image)` because Pillow 12 removed
  the historical private encoder signature. The controller rejects any other dirty historical file
  and records the exact compatibility-diff SHA-256. This patch changes image loading only; decoded
  uint8 pixels are parity-checked and model/sampler/optimizer code remains historical.
- Compatibility patch: historical checkpoint reads explicitly use `weights_only=False`, which is
  the pre-PyTorch-2.6 default. This is required to restore trusted Adam/scaler NumPy values and does
  not modify checkpoint contents or restored state.
- The same explicit trusted-checkpoint load is required in historical `eval_utils.py`. Its omission
  was discovered only after the first complete training run: training and full-eval rows succeeded,
  but post-training `ns-eval` failed under the PyTorch-2.6 default. The evaluator compatibility
  patch was added to the strict whitelist, and only final evaluation/renders were retried; no model
  or optimizer update was repeated.
- Blackwell extension target: CUDA 12.6 with `TORCH_CUDA_ARCH_LIST=9.0+PTX` and the validated
  `/home/brans/.cache/torch_extensions_lookcloser` cache. System nvcc cannot emit native `sm_120`;
  compute-90 PTX is JIT-compiled by the Blackwell driver, matching the prior local static setup.
- Canonical environment: `/home/brans/repos/nerfstudio/.venv`, PyTorch 2.7.1/CUDA 12.8,
  RTX PRO 6000 Blackwell 98 GB on `clever-shadow`.
- Historical tiny-cuda-nn provenance was recovered after the first diagnostic boundary. The dev3
  binary was built on 2025-12-31, three minutes after cloning clean commit
  `2e757bbe781db59c4980d389d7dccbf5edc09669`; its CUTLASS submodule is `1eb63551` and the binding
  targets Ada `sm_89`. The initial local environment used newer tiny-cuda-nn `749dd70c` with
  CUTLASS `82f50759`. A separate Blackwell overlay now builds the historical source/CUTLASS for
  native `sm_120`; the only compatibility diff changes C++/CUDA language standard 14 to 17 for
  PyTorch 2.7/CUDA 12.8. The controller validates and records both source revision and diff hash.
- Source of metric truth: three-view full evaluation. Training-batch metrics are diagnostic only.
- Exposure has two explicit forms. `Legacy ARM points` reproduces the archive convention,
  `sum(logged train_num_samples_per_batch × median logging interval)`, and is the quantity directly
  comparable with the historical `~250.035B`. `Total points` adds the known fixed-march warm-up,
  `4096 updates × 4096 rays × 256 samples = 4,294,967,296` points. The old marcher did not emit a
  sample-count scalar during that warm-up, so extrapolating the first ARM count backwards would be
  incorrect.

The exact historical source did not record a git hash. Commit `85818149` is therefore a strongly
supported reconstruction from git chronology, not a cryptographically embedded run provenance.

## Exact control design

### Stage A: from scratch through step 75940

- seed 42;
- no checkpoint load;
- fixed 4096 train rays per batch;
- adaptive ARM, cap 1024, coarse step 0.00625, adaptive warmup 4096;
- frequency grid on from the beginning;
- FAS strength 1.0 from the beginning, no warmup/ramp;
- FR strength 1.0 from the beginning;
- hash23, 16 levels × 2 features, max resolution 8192;
- Charbonnier RGB, distortion 0.01, early depth 0.001;
- Adam LR 0.01 with exponential decay to 0.0001 over 200000 scheduler steps;
- all step-15188 checkpoints retained.

The trainer limit is `75941`, which includes and saves step 75940. The optimizer scheduler horizon
remains 200000, so the model trajectory through that checkpoint matches the historical schedule.

### Stage A_fw03: continuation through step 106316

- load the exact step-75940 model;
- load Adam and scheduler state;
- preserve FAS strength 1.0 and the complete model/sampler configuration;
- change only FR strength from 1.0 to 0.3;
- continue the same exponential LR schedule;
- retain step-91128 and step-106316 checkpoints;
- final evaluation uses the latest exact checkpoint.

The controller is:

`/home/brans/repos/nerfstudio/LookCloser/scripts/run_static_leader_e2e.py`.

Its one-command defaults now select the accepted stable-occupancy worktree and historical FP16 TCNN
overlay. It records the two commands, selected/generated seed, dataset provenance, GPU snapshots,
checkpoint SHA-256, full-eval trajectory, estimated point exposure, and stage wall-clock in
`campaign.json`. After training it fresh-evaluates scheduled numeric candidates in order and
records the first automatic clean pass, without requiring manual checkpoint choice.

## Variance and local-minimum protocol

Variance is measured before changing the recipe.

1. `C0`: exact seed42 end-to-end control.
2. `C1`: an independent exact seed42 repeat. Compare every common checkpoint, not only the final row.
3. `R1–R3`: each run generates a new random seed once, writes it to `campaign.json`, and then keeps
   that seed fixed across both ancestry stages. Failed random seeds are not substituted.
4. Attribute divergence at the earliest checkpoint where it exceeds the accepted same-seed range:
   `0.06 dB PSNR / 0.01 SSIM / 0.005 LPIPS`. Earlier experiments below retain references to the
   original stricter diagnostic gate (`0.01 / 0.001 / 0.002`) so their contemporaneous decisions
   remain auditable; promotion from 2026-07-15 onward uses the accepted range here.
5. Compare trajectory spread at common step and common cumulative-point exposure. Report final
   range, standard deviation, and clean-pass rate; do not report only the best seed.

Potential local minima are diagnosed from retained checkpoints. A branch is considered stalled only
after two full-eval windows show less than `0.03 dB` PSNR, `0.001` SSIM and `0.003` LPIPS improvement,
while remaining materially behind the archived trajectory at similar point exposure.

From one stalled checkpoint, paired forks keep the model and seed fixed:

- continue unchanged for a longer window;
- temporary LR increase (`2×`, then `4×` only if stable) with scheduler preserved;
- LR decrease (`¼×`);
- preserve versus reset Adam at the winning LR;
- preserve the historical scheduler versus a documented restart.

`scripts/fork_static_checkpoint_optimizer.py` creates these branches without modifying the source
checkpoint. It records LR multiplier, Adam/scheduler/scaler reset policy and before/after optimizer
state in a SHA-256 sidecar. Its dry-run also makes the historical limitation explicit: these old
checkpoints contain model, Adam, scheduler and scaler state, but no RNG state, so resumed pixel/FAS
streams restart from the runner's recorded seed.

The fork decision is based on all three metrics plus render/artifact gates. A short metric spike is
not promotion evidence; the branch must remain improved at the next boundary. These forks diagnose
the optimizer basin and are not folded into the frozen recipe until confirmed by another seed.

## Speed phase — active after numeric reproduction

The seed42 control now meets the numeric gate and its repeat is within the user-approved range.
Speed experiments therefore change one factor at a time:

1. batch size / rays per update at equal cumulative points;
2. LR and scheduler compression;
3. ARM, occupancy and FAS warmup lengths;
4. checkpoint/evaluation cadence and finalization overhead.

Every accelerated candidate must match or beat the reproduced control with one automatically
selected checkpoint. The sequence of milestones is ≤60 minutes, then ≤30, then ≤15.

## Results

The exact-command dry-run and dataset checksum gate pass. Decoded uint8 data are identical for all
69 images after the Pillow compatibility patch; the SHA-256 over decoded image bytes is
`6b9b9b5c09e700ed5a82e896226d236528a546a2e2711b15c6375c2d107c44ff`.

The Blackwell smoke now passes end to end:

- Stage A trained through step 16, evaluated, and wrote `step-000000016.ckpt`.
- Stage A_fw03 loaded that checkpoint with model, Adam and scheduler state, changed FR to 0.3,
  trained through step 32, evaluated, and wrote `step-000000032.ckpt`.
- The small-step metrics are correctness probes only and are not reproduction evidence.

Parallelism was measured before launching the long controls. Both profiles used the exact early
training configuration, 1000 optimizer steps, no full eval, and one final save:

| Concurrent runs | Seconds per run | Aggregate steps/s | GPU memory | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 45.052 | 22.20 | about 8 GB | Reference |
| 2 | 80.09 | 24.97 | about 16 GB | 12.5% aggregate gain |

During two-way training the GPU sustained roughly 85–90% SM utilization and 340–355 W. A third run
is therefore not launched: memory permits it, but compute is already saturated enough that it is
unlikely to improve aggregate point throughput. This decision will be checked again after ARM
warmup because the mature sample distribution differs from the first 1000 steps. The post-warm-up
check gives the same conclusion: roughly 85–89% SM, 57–62% memory-controller utilization and about
17 GB framebuffer usage for two runs while each processes about 1.06–1.07M points/update at the
sampled boundary.

Two initial seed-42 diagnostics, `leader_exact_C0_seed42` and `leader_exact_C1_seed42`, started
concurrently after separate successful provenance checks using the then-current local tiny-cuda-nn.
Their manifests are under `/home/brans/lookcloser_leader_repro_runs/campaigns/`. The paired
concurrency kept hardware load matched for the first same-seed nondeterminism estimate; their
replacement by dependency-faithful H0/H1 is recorded below.

The written C0 Stage-A `config.yml` was diffed directly against the archived parent config on
read-only dev3. The complete diff contains only experiment name, output/data paths, timestamp, and
trainer `max_num_iterations` (`200000` historically versus `75941` for an exact stop after saving
step 75940). All model, ARM, FAS, FR, optimizer and scheduler fields match. The scheduler retains
its independent `max_steps=200000`; shortening the trainer horizon therefore does not compress LR.

Early variance is already measurable but is not yet a quality conclusion. C0 and C1 are exactly
equal at logged step 0 and begin to differ slightly by step 10. Across 260 common ARM telemetry
rows through step 6690, the absolute relative difference in points/update has mean `1.14%`, p95
`2.27%`, and maximum `3.01%`; training-batch PSNR has mean absolute difference `0.102 dB` and p95
`0.235 dB`. These are stochastic-batch diagnostics, not the three-view gate.

A separate seed42 1000-step run and a seed42 member of the two-way throughput profile had identical
configs except output names. At step 1000 only five nonempty floating model tensors differed, but
the hash-encoding tensor had maximum absolute difference `1.968` and L2 difference `1584.6` over
171,739,264 values. The first logged loss was identical and the mean absolute training-loss gap was
only `1.70e-4`. This is consistent with small CUDA-level numerical nondeterminism being amplified
through hash-grid optimization; the full-eval checkpoint spread is needed before deciding whether
it is operationally significant. A matched pair of isolated solo repeats remains scheduled to
separate intrinsic kernel variance from concurrent-load effects.

AMP introduces another count that cannot be replaced by trainer step. The archived parent
checkpoint says `step=75940` but its scheduler has `last_epoch=75904` (gap 36); the final leader says
`step=106316`, `last_epoch=106267` (gap 49). A clean local step-1000 checkpoint has
`last_epoch=997`, so roughly three positions are normal initialization/ordering offset and most of
the remaining historical gap represents GradScaler-decrease iterations where optimizer and
scheduler do not step. The continuation accumulated 13 additional gaps. From rounded live LR, both
current controls have an inferred gap near 7 at trainer step 11420. Exact scheduler/scaler state is
therefore read from every saved checkpoint and treated as a separate optimizer-update exposure;
trainer step and point samples alone are insufficient to establish trajectory parity.

The archived and local raw CSVs also differ before accumulated optimization can explain quality:
at step 0 the training loss differs by `3.84e-4`. The archived early LR is approximately one
scheduler tick ahead of the local reconstruction. This is evidence of the known unrecorded
historical environment/source gap (GPU, PyTorch/CUDA and architecture-specific tiny-cuda-nn binary
generation, plus possibly unrecorded dirty Nerfstudio source), not a config mismatch. Read-only
dev3 reflog places HEAD at `fe9dd951` before training; its only difference from the reproduced
`85818149` is a Markdown file, so the committed executable tree is resolved. The written configs
and tiny-cuda-nn source revision also match as described above.

### First full-eval boundary: step 15188

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 17.565 B | 15,179 | 8,192 | 28.5960 | 0.651726 | 0.371653 |
| C0 seed42 | 17.843 B | 15,179 | 8,192 | 28.6843 | 0.649331 | 0.361993 |
| C1 seed42 | 17.914 B | 15,179 | 8,192 | 28.7115 | 0.651333 | 0.360247 |
| H0 seed42, historical TCNN | 17.868 B | 15,180 | 8,192 | 28.8026 | 0.647827 | 0.359616 |
| H1 seed42, historical TCNN | 17.782 B | 15,180 | 8,192 | 28.6632 | 0.646578 | 0.364161 |

Both local runs are ahead of the archive in PSNR and LPIPS. C1 is `0.000393` below archive SSIM;
C0 is `0.002395` below. Optimizer-update count, scheduler epoch, LR (`0.007050339`) and scaler state
match the archived checkpoint exactly. The local ARM exposure is `+1.58%` and `+1.99%`, so the
quality comparison is close but not exactly exposure-matched.

The same-seed ranges are `0.0272 dB / 0.002002 / 0.001746`; LPIPS passes the provisional repeat
tolerance while PSNR and SSIM do not. Because optimizer/scaler state is identical, this first
boundary localizes the spread to the CUDA/model/sampler numerical trajectory rather than a
different number of successful Adam updates.

These two current-tiny-cuda-nn diagnostics were intentionally stopped after retaining checkpoint
15188 (training had reached steps 20290/20310) once the dependency mismatch was proven. They are
not promoted as the exact controls. `leader_histtcnn_H0_seed42` and
`leader_histtcnn_H1_seed42` restarted from zero in parallel with the historical tiny-cuda-nn
overlay and separate successful dataset provenance checks.

A two-step historical-overlay smoke produced the same local step-0 loss `0.209601` as the newer
local build, versus `0.209217` on dev3. Thus the initial numeric offset is not caused by the CUTLASS
revision; it remains attributable to Blackwell/PyTorch/CUDA or unrecorded dirty leader source. The
historical dependency is nevertheless retained because later fused-MLP accumulation can diverge.

At its first full eval, historical-TCNN H0/H1 are both ahead of the archive in PSNR and LPIPS but
behind it in SSIM by `0.003899` and `0.005148`. They have `+1.73%/+1.24%` ARM exposure and exactly
one more successful Adam update than the archive; LR is correspondingly one scheduler tick ahead.
Their same-seed range is `0.1394 dB / 0.001249 / 0.004545`, failing every provisional repeat
tolerance. Historical CUTLASS therefore does not reduce early Blackwell variance. This is not a
plateau signal: both quality trajectories exceed the archived PSNR/LPIPS point and continue
unchanged to the next full-eval boundary.

### Second full-eval boundary: step 30376

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 50.671 B | 30,360 | 8,192 | 29.2098 | 0.676160 | 0.305969 |
| H0 seed42 | 51.488 B | 30,360 | 8,192 | 29.3767 | 0.665064 | 0.290918 |
| H1 seed42 | 51.351 B | 30,360 | 8,192 | 29.3033 | 0.664370 | 0.295757 |

Both local controls remain ahead in PSNR and LPIPS, but are now behind archive SSIM by
`0.011096/0.011790`. Their successful Adam-update count, scheduler epoch, LR (`0.0049704991`) and
AMP scale match the archive exactly; ARM exposure is only `+1.61%/+1.34%`. This localizes the SSIM
trade-off to the Blackwell numerical/model-sampling trajectory rather than scheduler or scaler
drift. The local same-seed range narrows to `0.0734 dB / 0.000694 / 0.004839`: SSIM repeat tolerance
passes, PSNR and LPIPS do not. This is not a plateau because both local runs improved substantially
in all three metrics from step 15188; exact training therefore continues unchanged.

### Third full-eval boundary: step 45564

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 87.566 B | 45,541 | 16,384 | 29.3952 | 0.673553 | 0.279821 |
| H0 seed42 | 88.943 B | 45,541 | 8,192 | 29.6932 | 0.668872 | 0.260269 |
| H1 seed42 | 88.620 B | 45,541 | 8,192 | 29.5451 | 0.669836 | 0.264000 |

The local runs are `+0.2980/+0.1499 dB` in PSNR and `-0.019552/-0.015821` in LPIPS versus the
archive. Their SSIM gap narrows sharply to `-0.004681/-0.003717`; both are already above the final
leader SSIM gate `0.668450`. Adam-update count, scheduler epoch and LR (`0.003504209`) match exactly;
the lower local scaler value did not create a skipped-update difference. ARM exposure is
`+1.57%/+1.20%`. All three metrics improved from step 30376 and the SSIM deficit is recovering, so
there is no plateau/local-minimum evidence and no justification for an LR fork.

### Fourth full-eval boundary: step 60752

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 126.588 B | 60,722 | 8,192 | 29.5279 | 0.677030 | 0.262007 |
| H0 seed42 | 128.371 B | 60,722 | 8,192 | 29.8168 | 0.672429 | 0.243675 |
| H1 seed42 | 127.891 B | 60,722 | 8,192 | 29.5736 | 0.667859 | 0.245053 |

H0 stays ahead of the archive by `+0.2889 dB` PSNR and `-0.018332` LPIPS while narrowing its SSIM
deficit slightly to `-0.004601`. H1 is still ahead by `+0.0457 dB` and `-0.016954` LPIPS, but its
SSIM deficit widens to `-0.009171`. Both checkpoints exactly match the archived `60,722` successful
Adam updates, scheduler epoch, LR (`0.00247047236`) and AMP scale. Their ARM exposure is
`+1.41%/+1.03%` versus the reconstructed archive count.

The same-seed spread is now `0.2432 dB / 0.004570 / 0.001378`: only LPIPS meets the provisional
repeat tolerance. This confirms that the Blackwell trajectory variance is materially amplified by
this point even with identical seed and optimizer exposure. It is not a plateau trigger. H0 improves
by `0.1236 dB / 0.003557 / 0.016594` from step 45564, while H1's PSNR improvement is only
`0.0285 dB` and SSIM regresses by `0.001977`, but LPIPS still improves by `0.018947`; the prescribed
stall condition requires two consecutive windows with small improvements in all three metrics.
The exact historical schedule therefore remains unchanged through step 75940.

### Fifth full-eval boundary and FR transition: step 75940

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 166.919 B | 75,904 | 8,192 | 29.6217 | 0.675272 | 0.252857 |
| H0 seed42 | 169.036 B | 75,903 | 8,192 | 29.8330 | 0.671334 | 0.231886 |
| H1 seed42 | 168.380 B | 75,905 | 8,192 | 29.5692 | 0.669769 | 0.234464 |

Before the historical FR switch, H0 is already within `0.000751` LPIPS of the final archived gate
and passes its final PSNR/SSIM gates. Relative to the Stage-A archive, it is `+0.2113 dB` PSNR and
`-0.020971` LPIPS, with SSIM `-0.003938`. H1 is `-0.0525 dB` PSNR and `-0.018393` LPIPS, with SSIM
`-0.005503`. The same-seed spread grows to `0.2638 dB / 0.001565 / 0.002578`, so no metric meets
all repeat tolerances jointly. H0/H1 have one fewer/one more successful Adam update than the
archive; the corresponding LRs are `0.00174168656/0.00174160635` around the archived
`0.00174164645`. All three scaler values are 8,192. ARM exposure is `+1.27%/+0.88%`.

This boundary still does not satisfy the two-window plateau rule. H0 gains only `0.0162 dB` and
loses `0.001095` SSIM, while H1 loses `0.0044 dB` and gains `0.001910` SSIM, but LPIPS continues to
improve substantially by `0.011789/0.010589`. Both controls therefore transition exactly as the
leader did: load their own step-75940 model, Adam, scheduler and scaler state, change only FR from
1.0 to 0.3, and continue to step 106316.

The written H0 continuation config was diffed against the archived A_fw03 config. Differences are
limited to experiment/output/data/load paths, timestamp, trainer stop at `106317` instead of
`200000`, and retaining all checkpoints rather than only the latest; the model, sampler, FR/FAS,
optimizer and scheduler fields match. `/proc` maps for both active trainer processes confirm that
the historical tiny-cuda-nn `sm_120` overlay remains loaded after the restart.

### Sixth full-eval boundary: step 91128

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 208.140 B | — | — | 29.6920 | 0.672744 | 0.240396 |
| H0 seed42 | 210.510 B | 91,086 | 16,384 | 29.8285 | 0.669689 | 0.218186 |
| H1 seed42 | 209.673 B | 91,088 | 16,384 | 29.5830 | 0.668016 | 0.221218 |

H0 is the first end-to-end checkpoint in this campaign to pass all three final numeric gates at
once: `+0.210536 dB` PSNR, `+0.001239` SSIM and `-0.012949` LPIPS relative to the final archived
leader gate. Against the same-step archive it is `+0.1365 dB / -0.003055 / -0.022210`. H1 passes
the final LPIPS gate by `0.009917`, but misses final PSNR by `0.034964 dB` and SSIM by `0.000434`;
against the same-step archive it is `-0.1090 dB / -0.004728 / -0.019178`.

The FR=0.3 continuation improves LPIPS from each run's own step-75940 checkpoint by
`0.013700/0.013246`. H0 PSNR/SSIM change by `-0.0045/-0.001645`, while H1 changes by
`+0.0138/-0.001753`. Thus the historical switch is reproducing its intended perceptual movement,
but it does not collapse the pre-existing Blackwell PSNR variance. The same-seed spread is
`0.2455 dB / 0.001673 / 0.003032`, outside every repeat tolerance. Local ARM exposure is
`+1.14%/+0.74%` relative to the reconstructed archive count. The historical intermediate
checkpoint was deleted by its latest-only save policy, so its exact Adam/scaler state is not
available for a direct row comparison; both local states are retained and recorded.

Numeric reproduction at H0@91128 is not treated as final acceptance or best-of-two selection. Both
exact controls continue on schedule to step 106316, including renders and artifact gates. H1 is
reported as a miss rather than substituted, and the planned random-seed and isolated solo-repeat
campaigns remain required to quantify and reduce variance.

### Seventh full-eval boundary: step 106316

| Run | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | 250.035 B | 106,267 | 16,384 | 29.617964 | 0.668450 | 0.231135 |
| H0 seed42 | 252.674 B | 106,267 | 16,384 | 29.865969 | 0.665502 | 0.209431 |
| H1 seed42 | 251.603 B | 106,269 | 16,384 | 29.590485 | 0.667013 | 0.213473 |

Late continuation preserves and improves the large LPIPS gain, but moves SSIM below the final gate.
H0 is `+0.248005 dB / -0.002947 / -0.021703` versus the archived final; H1 is
`-0.027479 dB / -0.001437 / -0.017662`. H0 exactly matches the archived number of successful Adam
updates and LR (`0.000865625417`); H1 has two more updates and LR `0.000865585554`. Local ARM
exposure is `+1.06%/+0.63%`. The final same-seed spread is
`0.275484 dB / 0.001510 / 0.004041`, failing all repeat tolerances.

From step 91128 to 106316, H0 changes by `+0.037468 dB / -0.004187 / -0.008731`, and H1 by
`+0.007485 dB / -0.001004 / -0.007745`. The archive's corresponding move was
`-0.074036 dB / -0.004294 / -0.009261`. Thus the late SSIM-for-LPIPS trade-off is reproduced almost
exactly in magnitude, while Blackwell PSNR follows a different, seed-sensitive trajectory. This is
why the first scheduled all-metric checkpoint, H0@91128, is retained rather than automatically
assuming the last checkpoint is best.

The post-training evaluator initially failed only because the historical `eval_utils.py` still used
the new `torch.load(weights_only=True)` default. After applying the same trusted-local-checkpoint
compatibility argument already used by the trainer, both final full-precision evals and renders
completed. `finalize_static_leader_campaign.py` records the original wrapper return code, retry,
checkpoint hashes, point exposure and artifact outputs without changing weights. The H0 campaign
manifest closed `3 h 27 min 50 s` after controller launch, including provenance, concurrent
training, retry, renders, artifact/detail scoring and manifest finalization; the two training stages
themselves took `7983.6 + 3673.1 s` per concurrent run.

### Accepted H0 checkpoint: numeric, artifacts and detail

Fresh full-precision evaluation of H0@91128 is:

| PSNR | SSIM | LPIPS | Significant artifacts | Serious ROI | Result |
|---:|---:|---:|---:|---:|---|
| 29.828501 | 0.669689 | 0.218162 | 0/3 views | 0/9 ROIs | numeric + automatic artifact pass |

Visual inspection of all three side-by-side full renders found no black/missing regions, broken
stand geometry, melted cable loops or missing fingers. The thin pipes, cable holes and labels remain
continuous. H0 final@106316 is also artifact-clean (`0/3`, `0/9`) but fails SSIM; H1 final has two
significant full-view components (eval0 and eval2) while its nine selected ROIs remain non-serious.

The strict archived detail comparison is mixed rather than a blanket pass:

| ROI | ΔPSNR | ΔSSIM | ΔLPIPS | Per-ROI all-metric gate |
|---|---:|---:|---:|---|
| stand eval0 | -0.197441 | +0.000001 | -0.004665 | FAIL |
| thin pipe eval1 | -0.058031 | -0.000251 | -0.000644 | FAIL |
| stand label eval2 | +0.164022 | +0.003950 | +0.000448 | FAIL |
| tangled cable holes eval2 | +0.123291 | +0.001936 | -0.003563 | pass |
| fingers eval2 | +0.432241 | +0.004630 | -0.002639 | pass |

Therefore the specifically requested cable-hole and finger micro-detail is measurably no worse and
is visually intact, but the aggregate `all five ROIs × all three metrics` reference flag is false.
The small losses are metric trade-offs, not detected structural holes; they remain a guardrail for
the random-seed and later speed-selection stages rather than being rounded away.

### Random-but-recorded controls

R1 and R2 were assigned seeds `1145319261` and `1973976890` before training. Both passed an
independent dataset-provenance check and run concurrently with the same historical dependency
overlay. At their first boundary:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Archive seed42 | 17.565 B | 15,179 | 28.5960 | 0.651726 | 0.371653 |
| R1 seed1145319261 | 17.502 B | 15,179 | 28.6235 | 0.649549 | 0.365350 |
| R2 seed1973976890 | 17.167 B | 15,179 | 28.7030 | 0.653878 | 0.365637 |

The two random seeds have identical optimizer exposure and a range of
`0.0795 dB / 0.004329 / 0.000287`. LPIPS is already tightly clustered, while SSIM remains much
more variable than the `0.001` target.

At the second boundary the optimizer and scheduler states remain exact: both have `30,360` Adam
updates, scheduler epoch `30,360`, LR `0.00497049911` and AMP scale `8192`.

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Archive seed42 | 50.671 B | 30,360 | 29.2098 | 0.676160 | 0.305969 |
| R1 seed1145319261 | 50.810 B | 30,360 | 29.2562 | 0.662178 | 0.296328 |
| R2 seed1973976890 | 50.054 B | 30,360 | 29.3939 | 0.666494 | 0.295738 |

The R1/R2 range is now `0.1377 dB / 0.004316 / 0.000590`. Both random trajectories beat the
archive in PSNR and LPIPS at comparable cumulative exposure, but remain `0.0097–0.0140` below it
in SSIM. This repeats the dependency-faithful seed-42 pattern: the LPIPS discrepancy is not a
failure to reproduce the leader's perceptual descent (the new runs descend faster), whereas SSIM
is the systematic unresolved mismatch. Every metric improved materially from step 15188, so the
predeclared plateau criterion is not met and an LR fork here would confound the exact-recipe
control. Both runs continue unchanged; no seed is substituted.

At step 45564 the two trajectories are:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Archive seed42 | 87.566 B | 45,541 | 29.3952 | 0.673553 | 0.279821 |
| R1 seed1145319261 | 88.031 B | 45,541 | 29.3496 | 0.668321 | 0.266416 |
| R2 seed1973976890 | 86.886 B | 45,540 | 29.5463 | 0.671423 | 0.263342 |

The R1/R2 range widened to `0.1967 dB / 0.003102 / 0.003074`, while both remain substantially
better than the archive in LPIPS. The SSIM deficit has shrunk to `0.0021–0.0052`. R2 incurred one
additional skipped Adam update (AMP overflow) after the previous boundary and its scheduler epoch
is consequently one behind (`45,540` versus `45,541`; LR differs by `8.1e-8`). This exposes an
additional variance amplifier: a small numerical divergence can cross the GradScaler overflow
threshold, after which optimizer and scheduler exposure are no longer identical. The interval
still improves every metric by a wide margin, so the no-plateau decision remains unchanged.

At step 60752:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Archive seed42 | 126.588 B | 60,722 | 29.5279 | 0.677030 | 0.262007 |
| R1 seed1145319261 | 127.290 B | 60,722 | 29.5131 | 0.670019 | 0.249894 |
| R2 seed1973976890 | 125.702 B | 60,721 | 29.6394 | 0.669345 | 0.247255 |

The random-seed range is `0.1263 dB / 0.000674 / 0.002639`. Both continue to beat the archived
LPIPS trajectory by `0.0121–0.0148`, but SSIM is `0.0070–0.0077` lower. R1 improved all three
metrics from step 45564; R2 improved PSNR and LPIPS substantially while SSIM regressed by
`0.002078`. Because PSNR/LPIPS improvements remain far above the plateau thresholds, neither run
qualifies for an optimizer fork. R2 remains exactly one successful Adam/scheduler update behind;
both scalers are `8192`.

The exact Stage-A parent boundary is:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Archive seed42 | 166.919 B | 75,904 | 29.6217 | 0.675272 | 0.252857 |
| R1 seed1145319261 | 167.892 B | 75,904 | 29.4422 | 0.669408 | 0.240838 |
| R2 seed1973976890 | 165.800 B | 75,902 | 29.6269 | 0.669619 | 0.238800 |

The random-seed range is `0.1847 dB / 0.000211 / 0.002038`. Both improved LPIPS by more than
`0.008` since step 60752 but PSNR did not improve; this is still not a three-metric two-window
plateau. The controller then loaded each run's own immutable step-75940 checkpoint, including Adam,
scheduler and scaler, and changed only FR strength `1.0→0.3`. Logs confirm the exact source paths;
the first continuation LRs (`≈0.0017416`) continue the parent schedules rather than restarting.
R1 exactly matches the archive's `75,904` successful updates; R2 has two additional AMP skips and
starts continuation two scheduler epochs behind.

At the first continuation boundary:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS | Numeric gate |
|---|---:|---:|---:|---:|---:|---|
| Archive seed42 | 208.140 B | — | 29.6920 | 0.672744 | 0.240396 | archive trajectory only |
| R1 seed1145319261 | 209.326 B | 91,087 | 29.5020 | 0.669728 | 0.227905 | FAIL: PSNR |
| R2 seed1973976890 | 206.769 B | 91,085 | 29.6296 | 0.675218 | 0.226455 | pass |

R2@91128 is a second independent numeric reproduction: the same checkpoint simultaneously exceeds
the final archived leader's PSNR/SSIM and improves LPIPS. R1 passes SSIM and LPIPS but misses PSNR
by `0.1160 dB`. The random-seed range is `0.1276 dB / 0.005490 / 0.001450`. Both LPIPS curves
continue to improve by more than `0.012`, so neither satisfies the plateau criterion. Both AMP
scales grew to `16384`; R2 remains two successful updates behind R1. R2's checkpoint is retained
for fresh full-precision render/artifact/detail finalization after training, avoiding an extra GPU
evaluator that could perturb the still-running variance controls.

At the final scheduled boundary:

| Run | Legacy ARM points | Adam updates | PSNR | SSIM | LPIPS | Numeric gate |
|---|---:|---:|---:|---:|---:|---|
| Archive seed42 | 250.035 B | 106,267 | 29.617964 | 0.668450 | 0.231135 | reference |
| R1 seed1145319261 | 251.444 B | 106,267 | 29.5120 | 0.668512 | 0.218864 | FAIL: PSNR |
| R2 seed1973976890 | 248.424 B | 106,267 | 29.6579 | 0.666923 | 0.217414 | FAIL: SSIM |

Both runs converge back to exactly the archive's successful Adam/scheduler count despite their
different intermediate AMP skips. LPIPS continues improving, but R2 loses `0.008295` SSIM after
its passing step-91128 checkpoint. This is direct evidence that simply waiting longer is not a
monotonic solution: late optimization trades structural similarity for perceptual distance. The
selector therefore retains R2@91128; final R2 is not promoted by its better PSNR/LPIPS, and R1
never reaches the joint numeric gate.

Fresh full-precision finalization of R2@91128 reproduced
`29.629644 / 0.675218 / 0.226430`. All three full views have zero significant components, all nine
problem ROIs are non-serious, and visual inspection found intact stand, thin pipes, fingers and
cable loops. The strict archived five-ROI comparison nevertheless fails:

| ROI | ΔPSNR | ΔSSIM | ΔLPIPS | Strict all-metric gate |
|---|---:|---:|---:|---|
| stand eval0 | -0.137371 | -0.001437 | -0.004358 | FAIL |
| thin pipe eval1 | -0.091379 | +0.000624 | +0.001242 | FAIL |
| stand label eval2 | +0.040220 | +0.003437 | +0.008648 | FAIL |
| tangled cable holes eval2 | +0.068739 | +0.001655 | +0.005125 | FAIL |
| fingers eval2 | +0.475132 | +0.006778 | -0.000982 | pass |

Thus R2 is a clean numeric reproduction but not an all-metric cable-detail reproduction: cable
geometry is visually present and its PSNR/SSIM improve, while local LPIPS is worse. The campaign
manifest records this distinction rather than collapsing detail into the structural artifact gate.
The parallel R1/R2 controller wall from launch through final renders was about `3 h 14 min 52 s`;
each run spent `7938.7 + 3751.1 s` in the two training wrappers.

R3 then started from scratch with recorded random seed `404519541` as an intentional solo run.
Its early post-start iteration time is about `0.032–0.034 s`, versus roughly `0.10–0.11 s` per
run in the concurrent pair. That transient ratio is not the full-stage speedup: R3 Stage A took
`4389.8 s`, versus `7938.7 s` per run in the concurrent R1/R2 pair, so solo is `1.81x` faster per
trajectory while the pair gives about `10.6%` more aggregate completed-run throughput. R3 remains
solo because adding a second training workload would defeat its role in separating load-amplified
variance from the paired controls.

At its first boundary R3 has `17.581 B` adaptive points and
`28.7307 / 0.644257 / 0.370762`. It has `15,180` successful Adam updates and AMP scale `16384`,
versus `15,179` and `8192` for the archived/R1/R2 states. The solo run therefore does not collapse
variance by itself; its early SSIM is lower than both concurrent random seeds. Because R3 also has
a different seed, causal attribution to load still requires the planned solo seed42 repeat.
An automatic mutually-exclusive queue will launch `leader_histtcnn_H2_seed42_solo` immediately
after R3 reaches `complete`; it uses the same historical worktree, dependency overlay and seed42
recipe without sharing the GPU with another training process.

At the second boundary R3 has `50.954 B` adaptive points and
`29.2492 / 0.661222 / 0.309685`. Its `30,360` successful Adam updates, scheduler epoch,
LR (`0.00497049911`) and AMP scale (`8192`) exactly match the archived state, while point exposure
is only `+0.56%` above the reconstructed archive count. R3 is `+0.0394 dB` in PSNR but
`-0.014938` in SSIM and `+0.003716` in LPIPS versus the archive. This is the first local random
trajectory that has not yet beaten the archive LPIPS curve at this boundary, but it is not a
plateau: from step 15188 it improves by `+0.5185 dB / +0.016965 / -0.061077`. Exact training
therefore continues without an LR or optimizer fork.

At step 45564 R3 reaches `88.246 B` adaptive points and
`29.5676 / 0.666914 / 0.280620`. Exposure is `+0.78%` versus the archive. Its `45,541` Adam
updates, LR (`0.00350420899`) and AMP scale (`8192`) match exactly; relative quality is
`+0.1724 dB / -0.006639 / +0.000799`. The initially poor perceptual trajectory has therefore
nearly recovered to the archive LPIPS curve without intervention. From step 30376 this window
improves by `+0.3184 dB / +0.005692 / -0.029065`, again rejecting the plateau condition and any
LR fork at this checkpoint.

Separating the three `random_recorded` seeds from the explicit seed42 repeats gives, at step 45564,
a between-seed range of `0.218000 dB / 0.004509 / 0.017278` and population standard deviation of
`0.098132 dB / 0.001884 / 0.007526`. This is already above the eventual production limits in PSNR
and SSIM (and close in LPIPS), so the problem is not only same-seed CUDA repeat noise. Both the
optimizer basin selected by the seed and the nondeterministic execution path need to be controlled.

At step 60752 R3 has `127.483 B` adaptive points and
`29.6592 / 0.667266 / 0.260907`. It is now `+0.1313 dB / -0.009764 / -0.001100` versus the
archive: LPIPS has crossed below the historical curve without an intervention, while SSIM remains
the main deficit. R3 has `60,723` successful Adam updates, one more than the archived/R1 state;
its LR is correspondingly lower by about `5.7e-8`, and AMP scale remains `8192`. From step 45564
the run improves by `+0.0916 dB / +0.000352 / -0.019713`. PSNR and LPIPS are far outside the
plateau thresholds, so exact training continues. Across random seeds the range/std contract to
`0.146100/0.064712 dB`, `0.002753/0.001172` SSIM and `0.013652/0.005913` LPIPS at this boundary,
still above the eventual reproducibility limits.

The Stage-A parent checkpoint is `29.7023 / 0.669252 / 0.251222` at `167.986 B` adaptive points.
It is `+0.0806 dB / -0.006020 / -0.001635` versus the archive and exactly matches all archived
optimizer exposure fields: `75,904` Adam/scheduler updates, LR `0.001741646454` and AMP scale
`8192`. From step 60752 it still improves by `+0.0431 dB / +0.001986 / -0.009685`; the complete
Stage A therefore never meets the plateau condition. Between random seeds, the Stage-A range/std
is `0.260100/0.109266 dB`, `0.000367/0.000150` SSIM and `0.012422/0.005439` LPIPS: SSIM happens
to cluster tightly here while PSNR and perceptual basin selection remain wide.

The continuation starts from the immutable R3 parent and changes FR only from `1.0` to `0.3`.
Its written config retains base LR `0.01`, final LR `0.0001` and scheduler horizon `200000`; the
first logged continuation LR is `0.00174125` after the expected first few restored scheduler ticks,
not a restarted `0.01`. FAS remains enabled at strength `1.0` and the recorded seed remains
`404519541`.

At step 91128 R3 is `29.7198 / 0.666346 / 0.239873` at `209.359 B` adaptive points. Against the
same-step archive it is `+0.0278 dB / -0.006398 / -0.000523`; against the final acceptance row it
passes PSNR but misses SSIM by `0.002104` and LPIPS by `0.008738`. The FR switch changes R3 from
its parent by `+0.0175 dB / -0.002906 / -0.011349`, reproducing the intended perceptual trade-off
without yielding a joint pass. The checkpoint has `91,086` Adam/scheduler updates, LR
`0.00122783497` and AMP scale `16384`. Its random-seed range/std is still
`0.217800/0.089352 dB`, `0.008872/0.003656` SSIM and `0.013418/0.006013` LPIPS. R3 is reported as
a miss and continues to the final scheduled boundary; no seed is substituted.

R3 finishes at `29.7067 / 0.665516 / 0.231663` with `251.452 B` adaptive points. It passes PSNR
but misses the archived SSIM gate by `0.002934` and LPIPS gate by only `0.000528`; no scheduled R3
checkpoint passes all three metrics. From step 91128, waiting longer changes quality by
`-0.0131 dB / -0.000830 / -0.008210`: perceptual distance improves while PSNR and SSIM regress,
so longer training is again not a monotonic joint solution. R3 has `106,266` Adam/scheduler
updates, one fewer than the archive, LR `0.000865645349` and AMP scale `16384`; the one-update LR
difference is about `2e-8` and cannot account for the quality gap. Final random-seed range/std is
`0.194700/0.082715 dB`, `0.002996/0.001224` SSIM and `0.014249/0.006403` LPIPS. R3 is retained as
a full miss rather than replaced by R2.

Fresh final evaluation is `29.706665 / 0.665516 / 0.231635`, confirming the online result and the
numeric miss. All three GT/pred renders were visually inspected: no large black/missing region or
collapsed cable bundle is visible, and all nine selected problem ROIs score non-serious. The
significant full-frame detector nevertheless flags one `311 px` component in eval1 around the thin
vertical stand/cable next to the left performer (`artifact score 0.13`), while eval0/eval2 are zero.
R3 therefore also fails the automatic artifact gate (`1/3` significant, `0/9` serious ROI); the
detector result is retained even though the full-frame visual defect is subtle.

### Isolated same-seed control H2

H2 uses the exact H0/H1 historical recipe and seed `42`, but runs alone on the GPU. Its first
full-eval boundary is:

| Run | Load | Legacy ARM points | Adam updates | AMP scale | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Archive seed42 | original L40S | 17.565 B | 15,179 | 8,192 | 28.5960 | 0.651726 | 0.371653 |
| H0 seed42 | paired Blackwell | 17.868 B | 15,180 | 8,192 | 28.8026 | 0.647827 | 0.359616 |
| H1 seed42 | paired Blackwell | 17.782 B | 15,180 | 8,192 | 28.6632 | 0.646578 | 0.364161 |
| H2 seed42 | solo Blackwell | 17.788 B | 15,179 | 8,192 | 28.6694 | 0.650107 | 0.363670 |

H2 matches the archive's successful-update count, scheduler epoch, LR (`0.00705033901`) and scaler,
and its point exposure is only `+1.27%` above the reconstructed archive. Nevertheless it does not
reproduce H0: the H2–H0 difference is `-0.1332 dB / +0.002280 / +0.004054`. H2 is much closer to
paired H1 in PSNR and LPIPS (`+0.0062 dB / -0.000491`) while its SSIM is `+0.003529` higher.
Across all three exact seed42 repeats the range is therefore
`0.1394 dB / 0.003529 / 0.004545`, failing every repeat tolerance. The solo result does not expand
the prior PSNR or LPIPS envelope, but it expands SSIM spread and proves that concurrent training
load is not required for a materially different same-seed trajectory. Later H2 boundaries are
still required to measure whether load changes amplification magnitude rather than its existence.
No LR fork is triggered at this boundary: H2 is ahead of the archive in PSNR/LPIPS, optimizer
exposure is exact, and there is no two-window plateau.

At step 30376, H2 reaches `51.354 B` adaptive points and
`29.2997 / 0.666809 / 0.297535`. It exactly matches the archive and H0/H1 at `30,360` successful
Adam/scheduler updates, LR `0.00497049911` and AMP scale `8192`; point exposure is `+1.35%` versus
the archive and differs from H1 by only `0.003 B`. H2 remains H1-like in PSNR (`-0.0036 dB`) but
differs from it by `+0.002439` SSIM and `+0.001778` LPIPS. Relative to H0 it is
`-0.0770 dB / +0.001745 / +0.006617`. The three-run same-seed range is now
`0.0770 dB / 0.002439 / 0.006617`, still failing every tolerance; LPIPS spread is wider than in
the concurrent H0/H1 pair. Consequently matched concurrent load is neither necessary nor the
dominant explanation for the observed variance, although later checkpoints will still quantify
its effect on amplification. H2 improves from its first boundary by
`+0.6303 dB / +0.016702 / -0.066135`, so it is emphatically not stalled and continues without an
LR/Adam/scheduler fork.

Immediately after the already-saved step-30376 boundary, a one-second provenance probe imported
the historical TCNN binding and constructed (but did not run forward/backward through) a matching
encoding to read its selected precision. Construction initializes parameters and may briefly open
a second CUDA context, so H2 is claimed as strictly workload-isolated only through step 30376;
later checkpoints remain useful quality controls but are not used to strengthen the load-causality
claim. The first two boundaries—and their material divergence—were completed before this probe.

At step 45564, H2 is `29.6230 / 0.671590 / 0.268447` at `88.594 B` adaptive points. It exactly
matches the archive at `45,541` successful Adam/scheduler updates and LR `0.00350420899`; its AMP
scale is `8192` rather than the archive's `16384`, as in H0/H1, without an optimizer-update gap.
Exposure is `+1.17%`. Relative to the archive, H2 is
`+0.2278 dB / -0.001963 / -0.011374`: perceptual descent is ahead and the SSIM gap has nearly
closed. H2 lies between H0 and H1 in PSNR, above both in SSIM and behind both in LPIPS. The
three-repeat range is `0.1481 dB / 0.002718 / 0.008178`, failing every tolerance and widening in
all metrics from step 30376. From its own previous boundary H2 improves by
`+0.3233 dB / +0.004781 / -0.029088`, so no local-minimum fork is justified and exact Stage A
continues.

At step 60752, H2 is `29.6655 / 0.668091 / 0.249763` at `127.832 B` adaptive points. Its
`60,722` successful Adam/scheduler updates, LR `0.00247047236` and AMP scale `8192` exactly match
the archive; exposure is `+0.98%`. H2 is `+0.1376 dB / -0.008939 / -0.012244` versus the archive.
It remains inside the H0/H1 PSNR and SSIM envelope, but has the worst LPIPS of the three seed42
repeats at this boundary. Consequently the same-seed range is
`0.2432 dB / 0.004570 / 0.006088`: PSNR/SSIM spread is already set by the concurrent pair, while
the solo trajectory expands LPIPS variance substantially. From H2@45564 the change is
`+0.0425 dB / -0.003499 / -0.018684`. The SSIM regression does not meet the declared plateau rule
because PSNR and LPIPS still improve beyond their thresholds; no LR fork is launched.

The H2 Stage-A parent at step 75940 is `29.8040 / 0.666388 / 0.238982` with `168.319 B` adaptive
points. It is `+0.1823 dB / -0.008884 / -0.013875` versus the Stage-A archive at only `+0.84%`
exposure. Most importantly, checkpoint state is an exact optimizer match: `75,904` Adam/scheduler
updates, LR `0.001741646454` and AMP scale `16384`. H2 is within the H0/H1 PSNR envelope but below
both in SSIM and above both in LPIPS; the three-repeat range becomes
`0.2638 dB / 0.004946 / 0.007096`. From step 60752 H2 changes by
`+0.1385 dB / -0.001703 / -0.010781`, so Stage A again does not meet the plateau rule.

The immutable parent SHA-256 is `f49852fe…680f`. Stage A completed in `4389.84 s`, essentially
the same solo wall as R3, and the controller automatically started A_fw03 from that checkpoint.
The written continuation changes only FR `1.0→0.3`; FAS remains `1.0`. Its first logged LR is
`0.00174125` after the expected restored scheduler ticks, proving that Adam/scheduler/scaler were
continued rather than restarted. Direct raw-YAML inspection confirms base LR `0.01`, final LR
`0.0001` and scheduler horizon `200000` are unchanged; the operational config differences are the
experiment/load path, trainer stop `75941→106317` and FR strength `1.0→0.3`.

At step 91128, H2 is `29.7960 / 0.664110 / 0.224061` at `209.667 B` adaptive points. It passes
the final archived PSNR and LPIPS gates by `+0.178036 dB` and `0.007074`, but misses SSIM by
`0.004340`, so it is not a numeric candidate. Against the same-step archive it is
`+0.1040 dB / -0.008634 / -0.016335`. The FR switch changes H2 from its own parent by
`-0.0080 dB / -0.002278 / -0.014921`: the intended perceptual gain is reproduced, but the
structural trade-off is stronger than the available SSIM margin.

H2 has `91,085` successful Adam/scheduler updates, LR `0.00122786325` and AMP scale `8192`; it is
one update behind H0/R3 and three behind H1. Its point exposure differs from H1 by only `0.006 B`,
yet SSIM differs by `-0.003906`, so neither point count nor the tiny LR offset explains the basin.
Across H0/H1/H2 the same-seed range is now
`0.2455 dB / 0.005579 / 0.005875`, failing every tolerance. LPIPS improved far beyond the plateau
threshold, so H2 continues to the scheduled final checkpoint without an optimizer fork.

H2 finishes at online `29.7641 / 0.661509 / 0.217408` and fresh full-precision
`29.764133 / 0.661509 / 0.217380`, with `251.710 B` adaptive points. Relative to the final archived
leader it is `+0.146169 dB / -0.006941 / -0.013755`: PSNR and LPIPS pass comfortably, but SSIM
does not. From step 91128 to the final boundary, additional training changes quality by
`-0.0319 dB / -0.002601 / -0.006653`; simply waiting longer improves perceptual distance while
making the joint objective worse. The checkpoint has `106,266` Adam/scheduler updates, one fewer
than the archive, LR `0.000865645349` and AMP scale `8192`. The one-update/LR difference is far too
small to explain the `0.006941` SSIM gap.

Fresh H0/H1/H2 final values give a same-seed range of approximately
`0.275484 dB / 0.005504 / 0.007949`, failing every production tolerance. H2 remains inside the
H0/H1 PSNR envelope but expands both SSIM and LPIPS spread. It has no scheduled numeric candidate;
there is therefore no best-of-three substitution.

H2 finalization is structurally clean: `0/3` significant full-view artifacts and `0/9` serious
problem ROIs. Visual inspection of all three GT|prediction renders found no black/missing regions,
broken stands, collapsed cable bundle or missing fingers. Fresh archived-detail deltas are:

| ROI | ΔPSNR | ΔSSIM | ΔLPIPS | Strict all-metric gate |
|---|---:|---:|---:|---|
| stand eval0 | -0.073154 | -0.002434 | -0.008063 | FAIL |
| thin pipe eval1 | -0.011902 | +0.000459 | -0.001041 | FAIL |
| stand label eval2 | +0.160721 | +0.003185 | -0.003511 | pass |
| tangled cable holes eval2 | +0.046618 | -0.000132 | -0.002392 | FAIL |
| fingers eval2 | +0.386936 | +0.005240 | -0.002319 | pass |

Cable-hole geometry is visually intact and improves PSNR/LPIPS, but its `0.000132` SSIM deficit is
retained as a strict micro-detail fail rather than rounded away. The controller closed `complete`
after `4389.84 + 2049.11 s` stage walls; from campaign manifest creation through fresh eval,
renders, artifact gates and closure it took about `6441.81 s` (`1 h 47 min 22 s`).

### Variance mechanism audit

The same-seed H0/H1 spread cannot be explained by seed assignment: both processes start with the
same Python, NumPy and Torch seed, and their saved optimizer/scheduler exposure matches. Source
inspection instead found two concrete nondeterministic/amplifying mechanisms in the preserved
historical stack:

- historical tiny-cuda-nn hash-grid backward accumulates colliding feature gradients with CUDA
  `atomicAdd` (`include/tiny-cuda-nn/encodings/grid.h` via `atomic_add_gmem`). Floating-point
  accumulation order is scheduling-dependent, so equal seeds need not give bitwise-equal updates;
- after occupancy warm-up, nerfacc samples uniform and occupied cell IDs with replacement and
  writes their EMA values using advanced indexed assignment. Duplicate cell IDs can therefore be
  updated in an order that is not a stable reduction. Once the field differs slightly, occupancy
  masks and point counts differ too.

This occupancy mechanism is frequent enough to be consequential. R3@30376 has a `128^3` grid with
`1,092,577 / 2,097,152` cells marked occupied. A post-warm-up update draws `524,288` uniform and
`524,288` occupied IDs with replacement. Five distribution-matched draws contained
`260,315–260,820` duplicate entries, or a mean duplicate fraction of `24.85%`. The legacy indexed
assignment therefore resolves roughly one quarter of its writes through duplicate destinations;
this is not a rare collision. A stable per-cell max reduction is the intended nerfacc operation
(the vendored source retains the commented `scatter_max` alternative) and is the first isolated
occupancy ablation after the exact solo controls. Because it changes which repeated candidate wins,
it will be labelled an algorithmic correctness ablation rather than silently treated as the
historical recipe.

The TCNN mechanism is now resolved down to its actual arithmetic precision. The historical Python
binding reports `Precision.Fp16`, and the matching hash encoding selects FP16 internally. For
`n_features_per_level=2`, `GridEncodingTemplated` defines `grad_t=T`, so the grid backward uses
vectorized `atomicAdd(half2)` rather than accumulating into FP32; the FP32 tensor visible in a
checkpoint is the Python master parameter and does not change that kernel reduction. A
`grad_t=float` build is therefore a well-defined second precision ablation because stable occupancy
alone left repeat variance high. It can reduce half-precision order sensitivity, but its FP32
atomics are still ordering-dependent, so it is not pre-labelled deterministic and must pass paired
repeat plus full quality gates before promotion.

The reviewed FP32 worktree is at
`/home/brans/deps/tiny-cuda-nn-2e757-gradfp32`, detached at the exact historical commit. Its only
changes are the already-required C++/CUDA-17 compatibility edits plus `grad_t=float` in
`grid.h`. It was compiled with CUDA `12.8.93` for `sm_120` into the isolated overlay
`/home/brans/deps/tcnn_2e757_gradfp32_py310`; source diff, header, binding and build-log SHA-256s
are preserved in `build_provenance.json`. A CUDA smoke test imported that exact binding and passed
finite hash-grid forward/backward with FP32 parameter gradients. The controller exposes this as
`--tcnn-grid-grad-fp32`, requires an isolated overlay, validates the reviewed `grid.h` SHA-256 and
records it separately from stable occupancy. Default exact commands remain on the original overlay.
After adding the flag, a pure-Python reconstruction against the active H2 manifest confirms exact
list equality for both default Stage-A and Stage-A_fw03 commands; neither ablation flag is present.

The paired `F0/F1` seed42 campaigns started concurrently at `2026-07-15 04:34 UTC` after two
independent local-versus-dev3 dataset provenance matches. They retain stable occupancy and change
only TCNN grid-gradient accumulation from FP16 to FP32. Initial load is approximately `90%` GPU and
`16.75 GiB` combined model memory, so two processes again use the Blackwell efficiently without
memory pressure. Scheduled checkpoints and evals remain at every `15188` steps, and both runs will
be reported even if one becomes worse.

At the first full eval, F0 is `28.7597 / 0.650683 / 0.360250` and F1 is
`28.7667 / 0.647161 / 0.359092`. Their same-seed range is
`0.0070 dB / 0.003522 / 0.001158`, which passes the accepted repeat gate. This is a material change
from stable-FP16 S0/S1
(`0.0339 dB / 0.001696 / 0.003957`): FP32 strongly reduces PSNR/LPIPS spread but does not yet make
the full metric vector reproducible. F0/F1 cumulative ARM exposure at step 15188 is
`18.206–18.252 B`, a `0.046 B` pair spread versus `0.176 B` for S0/S1 and `17.565 B` in the archive.
At the same earlier step 11160, the F0/F1 exposure spread was `0.0170 B` versus `0.1491 B` for
S0/S1, so the precision change clearly stabilizes the occupancy/point trajectory even though its
first-boundary SSIM spread remains above tolerance. Both runs continue unchanged; one early
boundary is not used to promote or reject the ablation.

At step 30376, F0 is `29.4026 / 0.665068 / 0.294471` and F1 is
`29.4332 / 0.675217 / 0.295494`. The repeated-seed range is
`0.0306 dB / 0.010149 / 0.001023`: SSIM narrowly exceeds the accepted gate by `0.000149`. Relative
to stable-FP16 S0/S1 (`0.0350 dB / 0.000033 / 0.003597`), the precision ablation again improves
LPIPS and modestly improves PSNR spread but makes SSIM spread substantially worse. Cumulative ARM
exposure is tightly grouped at `52.526–52.635 B`, versus `52.211–52.633 B` for S0/S1 and
`50.671 B` in the archive. Both FP32 runs are ahead of the archive in PSNR and LPIPS, so there is
no collapse or plateau; the pair continues to measure the late FR=0.3 trajectory. Two independent
boundaries already show that FP32 grid gradients alone do not satisfy the production repeat gate.

At step 45564, F0 is `29.7291 / 0.670683 / 0.262234` and F1 is
`29.6826 / 0.679534 / 0.260583`. Their range is
`0.0465 dB / 0.008851 / 0.001651`, which passes the accepted repeat gate.
Stable-FP16 S0/S1 had `0.0321 dB / 0.001386 / 0.002619`, so FP32 consistently improves LPIPS
repeatability but does not control the SSIM branch. ARM exposure is exceptionally close at
`90.827–90.861 B` (only `0.034 B` spread) versus `90.218–91.056 B` for S0/S1 and `87.566 B` in the
archive. Thus point-trajectory stabilization is real but insufficient to stabilize image metrics.
Both runs still improve strongly over step 30376 and beat the archive in PSNR/LPIPS; no plateau or
collapse criterion is met.

At step 60752, F0 is `29.8275 / 0.671679 / 0.245760` and F1 is
`29.7904 / 0.678962 / 0.244733`. The range is
`0.0371 dB / 0.007283 / 0.001027`, which passes the accepted repeat gate. Stable-FP16 S0/S1 had
`0.0388 dB / 0.000502 / 0.001427`, so FP32 leaves PSNR spread essentially unchanged, improves
LPIPS, and again worsens SSIM repeatability. ARM exposure remains close at `131.089–131.216 B`,
compared with `130.340–131.589 B` for S0/S1 and `126.588 B` in the archive. Both runs continue to
improve PSNR/LPIPS and neither satisfies the two-window plateau rule.

At the Stage-A parent step 75940, F0 is `29.9012 / 0.673394 / 0.234071` and F1 is
`29.8704 / 0.684880 / 0.232027`. Their range is
`0.0308 dB / 0.011486 / 0.002044`; only SSIM exceeds the accepted repeat gate. Neither checkpoint
yet meets the leader LPIPS gate. ARM exposure is tightly grouped at
`172.578–172.931 B`, compared with `171.770–173.479 B` for S0/S1 and `166.919 B` in the archive,
again separating image-metric variance from point-count variance. Each FP32 Stage A took
`9261.6–9261.7 s`, versus `8104.0 s` for stable-FP16, a roughly `14.3%` wall-time penalty. Both
controllers automatically restored model, Adam, scheduler and scaler into A_fw03 and changed only
FR `1.0→0.3`.

At step 91128, both FP32 runs simultaneously pass all three archived numeric gates: F0 is
`29.8888 / 0.675644 / 0.220255` and F1 is `29.8312 / 0.681454 / 0.218846`. Their range is
`0.0576 dB / 0.005810 / 0.001409`, which passes the accepted repeat gate. Stable-FP16 S0/S1 had
`0.0150 dB / 0.003384 / 0.000710`, so FP32 gives excellent absolute LPIPS but worse late-checkpoint
PSNR/SSIM repeatability. ARM exposure is `214.945–215.493 B`, much more tightly grouped than
`214.088–216.287 B` for S0/S1 but still above archive `208.140 B`. Fresh evaluation and visual
gates were deferred until both full trajectories finished, avoiding a third concurrent GPU
evaluator during training.

At step 106316, F0 is `29.8583 / 0.670751 / 0.211213` and F1 is
`29.7786 / 0.679843 / 0.211387`. Both still beat the archived leader on all three metrics, but their
repeat range `0.0797 dB / 0.009092 / 0.000174` misses the accepted PSNR range. This confirms the
predeclared first-checkpoint selector: step 91128 is the reproducible boundary, while training
longer is not automatically better for repeatability.

Fresh full-precision step-91128 finalization reproduced F0 as
`29.888805 / 0.675644 / 0.220226` and F1 as
`29.831200 / 0.681454 / 0.218816`. Each scores `0/3` significant full-view artifacts and `0/10`
serious ROI artifacts. Both pass eval1 thin pipe; F0 narrowly misses the cable aggregate only on
SSIM (`−0.000597`) while improving cable PSNR/LPIPS, whereas F1 misses cable PSNR/LPIPS. The strict
five-ROI aggregate fails for both. Accepted stable-FP16 S1 therefore remains the speed baseline:
it is clean, passes the complete cable/pipe/fingers comparator, and avoids FP32's wall penalty.

F0/F1 Stage B took `4296.5 s` each and their full two-stage training took about `13558 s` per
contended run, versus about `11933 s` for S0/S1. FP32 is roughly `13.6%` slower over the full
trajectory (`14.3%` in Stage A), so it is rejected as a speed default despite excellent LPIPS.

A CUDA RNG/reducer unit suite was inadvertently executed for `0.94 s` after the first boundary,
at approximately step 15700. It did not overlap the step-15188 render and allocated only a tiny
third CUDA context, but it is conservatively recorded as external-load contamination for all later
F0/F1 boundaries. Those boundaries remain useful diagnostic evidence; if the pair meets the repeat
tolerance, it is retained as diagnostic evidence until the automatic quality finalization is
complete. That finalization is now complete and clean for both checkpoints; the contamination note
remains provenance, not a reason for rejection. FP32 is not the speed baseline because of measured
wall cost and weaker strict detail consistency.

The ablation is complete. A detached worktree at
`/home/brans/repos/nerfstudio_leader_stable_occ` starts from the same `85818149` commit and contains
only the three required PyTorch/Pillow compatibility patches plus the stable occupancy reducer,
its runner flag and tests. The exact controller now accepts this mode explicitly and records every
allowed source hash. The one-command operational default now selects the stable worktree and
historical FP16 TCNN overlay; legacy racing occupancy is available only as an explicit forensic
control. Stable mode has fingerprint `69d4f36c…b252` after adding the reviewed randomized
slow-reference property test. Historical reducer tests pass `5/5` with the real CUDA parity case
enabled.

A resume smoke loaded the exact H2 Stage-A checkpoint at step 75940, crossed two post-warm-up
occupancy update boundaries and exited cleanly at step 75956. Adam/scheduler exposure continued
from `75904` to `75920`, LR continued from `0.00174164645` to `0.00174100493`, the Adam-to-trainer
gap stayed `36`, and AMP scale stayed `16384`; model/optimizer/scheduler state was therefore loaded,
not reset. The smoke took `82.1 s` including two full evals. Dry runs for S0/S1 then matched in all
recipe and provenance fields except experiment paths, using the original FP16 TCNN binding and
stable-mode fingerprint `69d4f36c…b252`.

The stable experiment consists of two complete concurrent seed42 campaigns, `S0` and `S1`,
using the same original historical TCNN FP16 overlay and changing only
`stable_occupancy_reduction=false→true`. A CUDA CPU/GPU parity test and a short end-to-end smoke
passed before launch. Both campaigns run the full A→A_fw03 ancestry and all scheduled evals;
neither is stopped or substituted because the other looks better. They are reported as a pair,
including same-seed range at every boundary and separate numeric/artifact/detail outcomes. No
third run shares the GPU because the measured two-way workload is already compute-bound and two
runs provide the best observed aggregate point throughput. They started together after independent
dataset-provenance matches; initial steady load is about `87%` GPU utilization and `16.0 GiB`
combined model memory, leaving ample memory headroom but little reason for a third compute-bound job.

The first S0/S1 boundary at step 15188 is:

| Step | Campaign | ARM points | PSNR | SSIM | LPIPS |
|---:|---|---:|---:|---:|---:|
| 15188 | S0 seed42 | 18.211 B | 28.7819 | 0.647596 | 0.357641 |
| 15188 | S1 seed42 | 18.035 B | 28.7480 | 0.645900 | 0.361598 |
| 30376 | S0 seed42 | 52.633 B | 29.4196 | 0.660633 | 0.291537 |
| 30376 | S1 seed42 | 52.211 B | 29.4546 | 0.660666 | 0.295134 |
| 45564 | S0 seed42 | 91.056 B | 29.7248 | 0.668729 | 0.260964 |
| 45564 | S1 seed42 | 90.218 B | 29.6927 | 0.667343 | 0.263583 |
| 60752 | S0 seed42 | 131.589 B | 29.8183 | 0.669892 | 0.243945 |
| 60752 | S1 seed42 | 130.340 B | 29.7795 | 0.669390 | 0.245372 |
| 75940 | S0 seed42 | 173.479 B | 29.8855 | 0.673041 | 0.233830 |
| 75940 | S1 seed42 | 171.770 B | 29.8455 | 0.671617 | 0.232661 |
| 91128 | S0 seed42 | 216.287 B | 29.8551 | 0.672587 | 0.220189 |
| 91128 | S1 seed42 | 214.088 B | 29.8401 | 0.669203 | 0.219479 |
| 106316 | S0 seed42 | 259.835 B | 29.8666 | 0.667140 | 0.212544 |
| 106316 | S1 seed42 | 257.053 B | 29.8175 | 0.667386 | 0.213476 |

The repeated-seed range is `0.0339 dB / 0.001696 / 0.003957`, failing all three production
tolerances. Relative to the three exact H0/H1/H2 repeats at the same boundary
(`0.1394 dB / 0.003529 / 0.004545`), stable occupancy substantially reduces early PSNR/SSIM spread
and slightly reduces LPIPS spread, but does not make the trajectory reproducible. Both stable runs
continue unchanged; one early boundary is not used to promote or reject the ablation.

At step 30376, the stable range is `0.0350 dB / 0.000033 / 0.003597`: SSIM passes its repeat
tolerance, while PSNR and LPIPS still fail. The corresponding exact H0/H1/H2 range was
`0.0770 dB / 0.002439 / 0.006617`, so the reducer again improves all three variance measures but
does not eliminate the residual. Stable cumulative ARM exposure is `52.211–52.633 B`, about
`3.0–3.9%` above the archive's `50.671 B`; this is recorded rather than normalized away because
the full historical step trajectory, not a new point-capped recipe, is the declared control.

At step 45564, the stable range is `0.0321 dB / 0.001386 / 0.002619`, narrowly failing all three
tolerances. This remains a large reduction from exact H0/H1/H2
(`0.1481 dB / 0.002718 / 0.008178`) but establishes at a third boundary that stable occupancy alone
is insufficient. Cumulative ARM exposure is `90.218–91.056 B` versus archive `87.566 B`. Both
trajectories continue because quality improved strongly from step 30376 and no plateau condition
is present.

At step 60752, the stable range is `0.0388 dB / 0.000502 / 0.001427`: SSIM and LPIPS now satisfy
their repeat tolerances, while PSNR still fails. The exact H0/H1/H2 range at this boundary was
`0.2432 dB / 0.004570 / 0.006088`, so the correctness fix remains a clear variance improvement.
Cumulative ARM exposure is `130.340–131.589 B` versus archive `126.588 B`. Both runs are also well
ahead of the archive in PSNR/LPIPS but below it in SSIM; the planned continuation is required to
measure the joint late-FR objective.

At the Stage-A parent step 75940, the stable range is
`0.0400 dB / 0.001424 / 0.001169`: LPIPS passes, PSNR and SSIM do not. Exact H0/H1/H2 had
`0.2638 dB / 0.004946 / 0.007096`. Stage A took `8104.0 s` for each concurrent job; their ARM
exposures are `173.479 B` and `171.770 B` versus archive `166.919 B`. Both checkpoints have exactly
`75,905` Adam/scheduler updates, LR `0.00174160635`, trainer-to-Adam gap `35`, and AMP scale `8192`.
Thus the pair's remaining metric spread cannot be attributed to different update counts or LR.
The archived parent had one fewer successful update and AMP scale `16384`; this small exposure/scaler
difference is retained in provenance but cannot explain S0 versus S1 because their states agree.
Controllers loaded these checkpoints automatically and entered A_fw03 without user intervention.

At step 91128, both stable runs simultaneously pass the archived leader's three numeric gates:
S0 is `29.8551 / 0.672587 / 0.220189` and S1 is
`29.8401 / 0.669203 / 0.219479`. Their range is
`0.0150 dB / 0.003384 / 0.000710`: LPIPS passes the repeat tolerance, PSNR misses by only
`0.005 dB`, and SSIM still fails. Cumulative ARM exposure is `214.088–216.287 B` versus archive
`208.140 B`. Both checkpoints have exactly `91,087` Adam/scheduler updates, trainer-to-Adam gap
`41`, LR `0.0012278067`, and AMP scale `16384`, again ruling out optimizer exposure as the source
of their metric difference.

Fresh candidate evaluation confirms S0 at
`29.855082 / 0.672587 / 0.220167` and S1 at
`29.840143 / 0.669203 / 0.219455`. S0 is rejected: eval1 contains one serious detected component
with score `0.104` and area `254 px`, although all ten ROI gates are non-serious. S1 has `0/3`
significant full-view artifacts and `0/10` serious ROI artifacts, so it is recorded as the first
stable accepted candidate; no seed substitution occurred because S0's failure remains a failure.
Visual inspection of all three S1 GT|prediction renders and its blind detail contact sheet found no
missing stand, collapsed cable bundle, lost cable holes or missing fingers.

The strict archived-detail comparison for S1 is:

| ROI | ΔPSNR | ΔSSIM | ΔLPIPS | Strict all-metric gate |
|---|---:|---:|---:|---|
| stand eval0 | -0.038811 | -0.001083 | -0.005838 | FAIL |
| thin pipe eval1 | +0.222586 | +0.002273 | -0.006823 | pass |
| stand label eval2 | +0.206326 | +0.004421 | +0.002525 | FAIL |
| tangled cable holes eval2 | +0.048704 | +0.002008 | -0.001946 | pass |
| fingers eval2 | +0.342773 | +0.005809 | -0.005851 | pass |

Thus the user-prioritized micro/cable-hole gate passes, but the deliberately stronger all-five
aggregate remains false. S0 also passes cable holes and thin pipe, but its automatic full-view
failure already disqualifies it.

At final step 106316, fresh S0 is
`29.866632 / 0.667140 / 0.212521` and fresh S1 is
`29.817513 / 0.667386 / 0.213453`. Both lose enough SSIM to fail the joint numeric gate despite
further LPIPS improvement. Their fresh final range is approximately
`0.049120 dB / 0.000246 / 0.000932`: SSIM/LPIPS repeat tolerances pass and PSNR does not. S0 final
is automatically clean (`0/3`, `0/9` serious ROI); S1 final has one serious eval0 component
(score `0.132`, area `321 px`) and `0/9` serious ROI. This independently supports selecting the
earlier 91128 checkpoint rather than simply training longer.

Stage A took `8104.0 s` and A_fw03 took `3828.4–3828.8 s` per concurrent run. Manifest creation
through final eval, renders, artifact gates and closure took about `3 h 18 min 56 s` for the pair.
Aggregate exposure throughput is about `44.0 M point-samples/s`, roughly `10.7%` above the isolated
H2 trajectory's stage throughput; parallel execution therefore saved aggregate experiment time
while preserving matched load. The machine-readable live/final summary is
`experiments/artifacts/stable_occupancy_pair_monitor.json`.

A read-only environment check on the original host also closes an important provenance gap. The
leader trained on an NVIDIA L40S (`sm_89`, 46 GB), PyTorch `2.4.1+cu121` and CUDA runtime `12.1`.
The reproduction uses RTX PRO 6000 Blackwell (`sm_120`), PyTorch `2.7.1+cu128` and CUDA `12.8`;
the historical tiny-cuda-nn source is compiled as `compute_90` PTX and driver-JITed for Blackwell.
The dev3 source checkout is exactly `2e757bbe781db59c4980d389d7dccbf5edc09669`, matching the
source used by the local overlay, so this is an architecture/toolchain difference rather than an
unresolved tiny-cuda-nn revision mismatch.
Both environments report matmul TF32 off, cuDNN TF32 on and deterministic algorithms off. Thus
TF32 policy is not the observed difference, while kernel generation, launch scheduling and atomic
reduction order necessarily differ between the two architectures. This can explain a shifted
same-seed trajectory, but not by itself the large variance between two runs on the same Blackwell.

The second mechanism also amplifies the first: occupancy sampling, traversal jitter and FAS offsets
consume CUDA RNG; a changed occupancy branch/count changes subsequent sampling and eventually the
training data sequence. This is consistent with H0/H1/H2 having nearly identical Adam counts but
different point exposure and a final `0.2755 dB / 0.005504 / 0.007951` same-seed range. It is a source-based causal
hypothesis, not yet an isolated ablation.

The clean static worktree contains the corresponding stream-isolation control. Pixel/FAS,
occupancy and frequency-grid updates use separate step-derived seeds inside `torch.random.fork_rng`,
so a branch or count change in one subsystem cannot advance another subsystem's stream. Unit tests
verify stable and distinct stream seeds plus restoration of the enclosing CPU/CUDA RNG state; the
combined RNG/reducer suite passes `8/8` tests with CUDA parity enabled. A minimal historical control
is now prepared separately at `/home/brans/repos/nerfstudio_leader_stable_rng`: it starts from the
same `85818149` commit, reproduces the stable worktree byte-for-byte, and adds only the RNG helper,
pipeline/model gating, runner seed propagation and tests. Its controller dry-run fingerprint is
`fad3a418…e125a`; `machine.seed`, `pipeline.training_seed` and `model.training_seed` are all `42`,
and both ancestry stages contain the explicit independent-stream flag. This change is deliberately
absent from S0/S1 and F0/F1. If FP32 precision remains insufficient, stable-FP16 plus independent
streams is the next separate ablation before combining fixes or changing LR/FR.

Historical Nerfstudio checkpoints preserve model, Adam, scheduler and AMP scaler but not Python,
NumPy, CPU-Torch or CUDA RNG states. Thus future LR forks from a common historical checkpoint can
be paired fairly by resetting every branch to the same recorded seed, but cannot be claimed as the
bitwise continuation that would have occurred in the original process. The clean static worktree's
post-reproduction scheduler already adds Python, NumPy, CPU-Torch and all-device CUDA RNG state to
new checkpoints; variance-minimization and LR-policy promotion will use those stateful checkpoints,
while the historical controls remain explicitly labelled as reset-seed forks.

## Limitations and open questions

- PyTorch/CUDA and GPU differ from the historical dev3 environment, so same-seed bitwise identity is
  not expected even with the reconstructed historical source.
- The historical CSV does not contain an exact cumulative counter; exposure is reconstructed from
  sample-count telemetry. The new manifest preserves that estimate consistently for comparisons.
- Same-seed variance is within the user-approved `0.06 / 0.01 / 0.005` range for stable-FP16 S0/S1
  and FP32 F0/F1 at step 91128. Independent RNG streams remain prepared but are not promoted while
  speed is the active phase. The accepted range is a repeat-noise criterion, not a relaxation of
  the numeric leader or artifact/detail quality gates.
- The accepted checkpoint passes cable-hole/finger detail but not the deliberately strict all-ROI,
  all-metric reference aggregate; later selectors must report this separately from artifact gates.
