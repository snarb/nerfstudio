# LookCloser architecture

## Key files

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/scripts/lookcloser_debug_preprocess.py` — focused standalone preprocessing debug checks.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.
- `scripts/render_temporal_*.py` — read-only temporal snapshot, camera-path, metric dolly, lossless-video, and validation utilities.

## Git and worktree layout

- `/home/brans/repos/nerfstudio` is the clean canonical `main` worktree. Its no-argument
  defaults and `scripts/run_static_leader_e2e.py` reproduce the accepted quality recipe.
- `/home/brans/repos/nerfstudio_leader_speed` is the clean named branch
  `nerfstudio_leader_speed`. Every current speed/variance change is committed there; unpromoted
  controls stay default-off and are not treated as accepted results.
- Speed provenance is fail-closed on branch name, exact committed HEAD, an empty worktree and hashes
  of all 35 reviewed source/test files. The previous uncommitted-patch convention is retired.
- The older freqmap/temporal work is preserved on branch `freqmap-speed` at commit `74cb6d1d`; it is
  not a benchmark source and does not occupy a worktree.

## Training monitoring additions

Baseline runs can enable `--logging.csv-writer.enable True` to write compact `metrics_compact.csv` rows for train/eval trends, `best_eval_*`, plateau and overfit status, which is useful because recent 3k baselines plateau early and best checkpoint metrics are more informative than final-step metrics.

Use `scripts/detect_structural_artifacts.py` as the automatic detector for serious structural artifacts in rendered crops or triptychs, especially broken/dislocated thin structures, holes, and floaters. Include its `artifact_score` alongside PSNR, SSIM, and LPIPS when comparing candidate checkpoints; lower `artifact_score` is better and `0.0` means no qualifying severe local-SSIM artifact blobs. Evaluation loss is internal checkpoint bookkeeping and is not a reported quality metric. When needed, the script also saves bbox overlays, heatmaps, and suspicion maps for visual analysis of problematic artifacts.

## Static leader reproduction controller

`scripts/run_static_leader_e2e.py` reproduces the archived 007740 leader through its actual two-stage
ancestry: FR/FAS `1.0` from scratch to step 75940, then checkpoint/Adam/scheduler continuation with
FR `0.3` to step 106316. It runs the frozen historical worktree at commit `85818149`, validates the
dataset against read-only dev3, and can load the isolated historical tiny-cuda-nn overlay. Progress
is compared by reconstructed cumulative point samples and trainer step, not only wall time;
successful Adam/scheduler exposure can be audited separately from checkpoints.

The one-command defaults are the accepted stable-FP16 recipe: worktree
`/home/brans/repos/nerfstudio_leader_stable_occ`, stable occupancy reduction enabled, and the
historical FP16 TCNN overlay `/home/brans/deps/tcnn_2e757_py310`. Legacy racing occupancy remains an
explicit forensic control rather than the operational default. The frozen recipe and exact gates
are documented in `experiments/static_leader_reproduction_recipe.md`.
The main `ns-train lookcloser` preset and the no-argument quiet runner now resolve the exact Stage-A
part of this recipe: seed42, local 007740 data, filename split, scene scale1.5, B4096, hash23,
FR/FAS1.0, max-res8192, 4096-update fixed/occupancy warmup, stable occupancy, historical Adam and
15188-update eval/save cadence through step75940. End-to-end reproduction still uses the controller,
because a static method preset cannot express the checkpointed FR `1.0→0.3` continuation and
fail-closed selector in one `ns-train` process.

These are the promoted quality defaults. Dynamic point budgets, corrected ARM, cached rays, CPU FAS
prefetch, fused Adam, TCNN JIT, fixed GradScaler and independent RNG streams remain explicit
default-off controls: none has yet passed the complete wall, numeric, artifact and priority-detail
protocol more reliably than the accepted trajectory. Occupancy diagnostics remain enabled until
the measured same-parent opt-out branch passes its predeclared wall and quality gates.

The accepted source fingerprint is `69d4f36cc1e06256a8dcd5a1e9dd6c4a465bb81e8cee09a3d8b188358857b252`;
the current controller/gate protocol fingerprint is
`156a73bf475771e357af73afe298f88421502387f8fcda6b24d689c8d50550ad`.

All scheduled checkpoints are retained. The controller fresh-evaluates numeric candidates in
chronological order, renders/scores the first automatic clean pass, records it in `campaign.json`,
and avoids a duplicate unconditional render of the latest checkpoint. Its controller wall-clock
includes provenance, both stages and finalization. Finalization is fail-closed: source, TCNN and
PyTorch/CUDA fingerprints are checked; each candidate must return a fresh complete summary with
exactly three artifact views and ten ROI scores. Exit 0 means accepted, exit 2 means a fully
evaluated quality failure, and exit 3 means unfinalized or infrastructure failure. The selector
stops at the first infrastructure error instead of skipping an unevaluated scheduled checkpoint.
Training-source provenance and selection-protocol provenance are independent: the latter hashes the
normalized controller plus evaluator/recorder/finalizer, dataset checker, detail scorer and frozen
detail reference. This avoids calling a weight recipe reproducible while silently changing its
promotion rules.
`scripts/analyze_static_leader_campaigns.py`
quantifies
same-seed/random-seed spread and reports the first scheduled checkpoint that passes PSNR, SSIM,
LPIPS and automatic artifact gates. Its all-run range includes both effects; a separate
`random_recorded` ensemble reports between-seed range/std, while the same-seed tolerance is
computed only inside groups with the same recorded seed. The accepted repeat ranges are
`0.06 dB PSNR / 0.01 SSIM / 0.005 LPIPS` (user decision, 2026-07-15); these do not relax the
leader-quality or artifact gates. Singleton random seeds are never
mislabeled as nondeterministic repeats. `scripts/inspect_static_checkpoint_state.py` records Adam,
scheduler, LR and scaler exposure; `scripts/fork_static_checkpoint_optimizer.py` creates immutable
diagnostic LR/Adam forks only after a confirmed two-window plateau.

Historical compatibility patches are restricted to public Pillow decoding and explicit
`torch.load(weights_only=False)` for trusted local checkpoints in trainer/evaluator paths. They do
not change model or optimizer semantics. `scripts/finalize_static_leader_campaign.py` records a
post-training evaluator retry without rewriting weights if finalization alone fails.

The accepted default `--stable-occupancy-reduction` remains an explicitly fingerprinted algorithmic
change. It runs from `/home/brans/repos/nerfstudio_leader_stable_occ` at the same historical commit
and changes duplicate nerfacc occupancy updates from racing indexed assignment to one per-cell
max-with-decay update. Forensic legacy behavior is available only by combining
`--no-stable-occupancy-reduction` with
`--historical-worktree /home/brans/repos/nerfstudio_leader_repro`; its different fingerprint and
artifact/variance outcome must be reported rather than silently treated as the operational recipe.

The second explicit variance ablation is `--tcnn-grid-grad-fp32`. It requires a separately built
overlay from the exact historical TCNN commit and a reviewed source hash that changes hash-grid
gradient accumulation from FP16 `atomicAdd(half2)` to an FP32 temporary buffer before casting back
to the master parameter representation. The controller rejects the flag without that overlay and
records it independently from stable occupancy. FP32 atomics remain order-dependent, so this mode
is a precision experiment rather than a deterministic-mode claim.

The next isolated control is `--independent-rng-streams`, prepared in
`/home/brans/repos/nerfstudio_leader_stable_rng` from the same historical commit and stable reducer.
It derives separate step-addressed seeds for pixel/FAS, occupancy and frequency-grid sampling with
`torch.random.fork_rng`, restoring the enclosing CPU/CUDA state after each subsystem. The runner
passes the campaign seed explicitly to both pipeline and model configs. The exact controller
fingerprints the additional source/test paths and currently requires stable occupancy so the
ablation cannot be launched accidentally against an unreviewed worktree.

The same default-off control is now ported to `/home/brans/repos/nerfstudio_leader_speed` for a
predeclared speed-prefix screen. With CPU FAS prefetch enabled, pixel sampling uses a private
generator seeded for the requested trainer step and queues the separately seeded next step; it
does not borrow or commit process-global CPU RNG. Pipeline/model enable flags and seeds must match
or initialization fails. This port is not part of the frozen E2E recipe unless its sequential
same-seed prefix and subsequent solo wall/quality campaign both pass.

The predeclared speed `A/B/B` prefix rejected this mode. B1/B2 restored identical Python, NumPy,
Torch CPU and CUDA RNG snapshots and had the same Adam/skipped-update counts, but their scaler
trackers differed and their metric range was `0.007658 / 0.002257 / 0.003767`. Exact points spread
by`0.337111%`; field/occupancy-`occs` drift was `0.856276/0.516133`; one B render set had significant
components in all three views. Stream isolation is therefore functioning but insufficient against
TCNN atomic→occupancy→AMP feedback, and it remains outside the E2E/default recipe.

The selector keeps numeric/artifact acceptance distinct from the stricter archived five-ROI detail
aggregate. Promotion nevertheless requires the frozen priority micro-detail crops—eval1 thin pipe,
eval2 tangled cable holes and eval2 fingers—to be no worse than the archive on every measured
metric. Stand and label remain reported diagnostics rather than promotion gates. Detail-scorer
exit 2 with a valid result is recorded as a quality failure; only missing, malformed or inconsistent
output is an infrastructure failure. Candidate recording accepts checkpoints from either Stage A
or Stage A_fw03, and an already accepted earlier scheduled checkpoint cannot be replaced by a later
one.
The retry finalizer applies the same numeric, 3-view/10-ROI and mandatory priority-detail gates,
requires the candidate checkpoint to belong to campaign ancestry, and derives its worktree from the
campaign manifest. Its detail JSON must contain the exact frozen five-crop set and name the same
render directory being promoted, so stale crops from another checkpoint cannot be substituted.
The retry also requires the campaign-recorded protocol fingerprint to equal the current one; a
legacy manifest with no fingerprint needs the explicit reviewed
`--allow-legacy-protocol-migration` path, while a recorded mismatch is never overridden. Thus retry
cannot provide a weaker promotion path.

`scripts/evaluate_static_leader_candidate.py --campaign ...` restores the historical worktree,
CUDA settings and isolated TCNN overlay recorded by that campaign before loading an intermediate
checkpoint. It hashes the actually imported TCNN binding and rejects an evaluator/runtime mismatch
before rendering. This prevents an FP32-grid or historical-overlay checkpoint from being silently
evaluated with the environment's default binding.

The geometry-first `scripts/run_static_metric_rate.py` controller in the clean static worktree is a
separate algorithmic research path. Its dynamic p19 point budget, phase-delayed FR/FAS, corrected
ARM and stricter original composite targets are not defaults for reproducing or timing the accepted
historical leader. Speed work begins from the frozen stable-FP16 recipe above and changes batch,
warmup, LR or scheduler as named ablations at reported cumulative point exposure.

`scripts/run_static_leader_speed_e2e.py` is the speed entrypoint for the frozen leader lineage. It
selects the fingerprinted `/home/brans/repos/nerfstudio_leader_speed` worktree and derives every
point-normalized batch field from one `--batch-scale {1,2,4}`. The worktree adds a persistent
`cumulative_point_samples` pipeline buffer and compact CSV field; a legacy checkpoint initializes
the telemetry-only counter to zero, while speed Stage B restores and continues it. The runner also
exposes the exponential scheduler horizon so LR progress remains matched in point space. These
telemetry/config additions are isolated from the accepted reproduction fingerprint.

The same speed worktree also has a faithful-leader dynamic point-budget mode. Passing
`--target-points N --corrected-arm-allocator` keeps the historical 75940/91128 optimizer-update
boundaries, LR horizon, FR/FAS trajectory, hash23, loss, stable occupancy and shared RNG policy,
while changing the live pixel-sampler ray batch from an EMA of the observed field points per ray.
`target_num_samples_per_batch=0` and `corrected_arm_allocator=False` are the defaults, so frozen and
fixed-batch runs are unchanged. The corrected allocator gives every retained occupancy interval at
least one subdivision, distributes the remaining per-ray cap by largest remainder, and
deterministically merges closest intervals when interval count itself exceeds the cap; it never
drops an internal interval or the far tail. Checkpoints persist controller EMA/current rays and
exact cumulative points. CSV logs distinguish actual rays for the current update from next-update
rays. Dynamic campaigns use the exact checkpointed cumulative counter rather than reconstructing
warmup exposure from a nominal fixed batch.

The canonical speed controller delays dynamic ray changes until the historical ARM/occupancy
warmup boundary (`dynamic_rays_start_step=4096`). Rays therefore remain exactly 4096 through the
fixed 256-sample warmup; EMA point control begins only once adaptive traversal is active. The config
default is zero for backward compatibility and is inert whenever the point target is disabled.

An optional live point-budget boundary is expressed by
`target_num_samples_switch_step/target_num_samples_after_switch`. It changes only the target used
to calculate the next ray batch after the named update; model, Adam, scaler, scheduler, RNG,
occupancy, EMA and cumulative samples remain continuous. Both fields default to `None`, the active
target is logged in compact CSV, and checkpoints reconstruct the schedule from config plus trainer
step while restoring EMA/current rays. That schedule generation used fingerprint
`021ff4df89a77a12e716d27ffd4e1b7e4095f1cea9499b7ee42b0d313478a6f6`; the current reviewed speed
worktree, including the later semantics-preserving hot-path patch set, opt-in fused Adam and
default-off TCNN network JIT hook, is committed with source fingerprint
`6cf7eb9560403ed05da27b2eb7ce732585e930b2d13a0ccfbfb9dd1766e4c258`.

Intermediate evaluation writes both the exact `load_step` and exact checkpoint path. This is
required because the historical Nerfstudio evaluator reconstructs the run checkpoint directory and
ignores `TrainerConfig.load_checkpoint`; leaving `load_step=None` silently selects the latest file.

The reviewed continuous-p21 speed path keeps the point target and optimizer trajectory fixed while
removing tensor-plumbing overhead. Feature reweighting uses a device LUT only when levels came from
the enabled nearest-neighbor discrete frequency grid; explicit or fractional levels retain the
analytical Eq. 6 path. FAS builds a dense device height/width LUT with legacy defaults for absent or
invalid image IDs, avoiding per-ray CUDA scalar reads. Corrected ARM uses the equivalence between
iteratively removing the first smallest adjacent gap and one stable ascending gap selection, then
reduces each merged group's target step by `amin`. Known subdivision sizes are passed to
`repeat_interleave`, and sampler `packed_info` is reused by rendering.

For black-background training, RGB, accumulation and expected depth share one packed accumulation;
evaluation retains the general renderer path. The dynamic controller obtains exact point count from
packed tensor shape and mirrors EMA on the host while preserving checkpoint tensors. Checkpoints
also persist the FAS `sample_count`, so warmup/ramp/decay schedules resume rather than restarting;
the canonical full-FAS recipe is constant but the resume rule is general. Focused properties require
exact discrete FR/ARM/count parity. Packed RGB/depth CUDA reductions are accepted only within the
separately measured reference-repeat atomic floor, while packed weights, indices and spacing remain
bit exact.
FAS bucket-count weighting remains an opt-in research field with
`fas_level_count_alpha=0.0` by default. The reviewed `0.5` diagnostic over-weighted the common high
frequency buckets and failed aggregate LPIPS plus every frozen crop LPIPS comparison, so it is not
part of either the reproduction or speed defaults.
The regression test requires the generated config to bind both values before an offline metric can
be reported.

The same speed entrypoint exposes a bounded `--lr-scale` ablation. It scales both the historical
fields LR and the exponential endpoint together, records the resolved values in the campaign, and
is rejected outside point-normalized speed mode. The frozen reproduction default remains
`0.01 -> 0.0001` (`lr_scale=1`).

PyTorch fused CUDA Adam is exposed only through the explicit `--fused-adam` speed flag. The base
`AdamOptimizerConfig.fused` default is `None`, and the frozen reproduction resolves it to the
historical non-fused optimizer. When a legacy foreach-Adam checkpoint is loaded into an explicit
fused run, scalar step tensors are migrated to the parameter device while weights, moments and
step counts are retained. A scaled one-update CUDA parity test bounds parameter differences at
`1.2e-7`; enabling fused Adam still changes the speed source fingerprint and is never inferred on
hardware capability alone.

TCNN runtime JIT for the geometry/color MLPs is likewise explicit through
`--tcnn-network-jit`; `LookCloserModelConfig.tcnn_network_jit=False` preserves every frozen and
ordinary speed command. The canonical historical overlay lacked the five CUDA RTC headers copied
by upstream `setup.py`, so its earlier JIT attempt correctly fell back. The separate overlay
`/home/brans/deps/tcnn_2e757_py310_jit_rtc` adds only those headers, retains the bit-identical
binding, and records their hashes in `rtc_overlay_provenance.json`. Both networks compile and keep
JIT enabled on Blackwell; performance and training parity remain separate promotion gates.
The long-horizon fixed-B4096 seed-42 feasibility control reached its full gates in about 56:36 but
failed aggregate LPIPS and all three priority detail crops. Its step75940 divergence from the
historical trajectory was already `-0.0735 dB / +0.000695 SSIM / +0.003087 LPIPS`, despite the
same-parent 500-update fused/JIT smoke staying inside the repeat bounds. Consequently fused Adam,
TCNN JIT and the live FR boundary remain explicit default-off experiment controls; none is promoted
into the frozen reproduction command.

Historical train-time evaluation is part of the stochastic training trajectory in the shared-RNG
implementation: the train and eval LookCloser pixel samplers consume the same global Torch stream,
so an eval at step15188 changes later training rays/FAS samples. The frozen reproduction therefore
retains batch, image and all-image eval cadence 15188. Disabling those callbacks is not a
semantics-preserving speed change unless independent per-subsystem RNG streams are enabled and
validated.

Restoring cadence did not make the combined fused-Adam+TCNN-JIT speed bundle reproduce S1. The
seed-42 fixed-B4096 screen already differed before the first cadence intervention and reached
step75940 at `29.3260 / 0.669644 / 0.239800`, versus S1
`29.8455 / 0.671617 / 0.232661`, at essentially equal cumulative point exposure. Its sole step80000
candidate finished the full gates in approximately 58:36 but failed at
`29.374962 / 0.669688 / 0.239359`, one significant full-view artifact and two of three priority
detail crops. Thus historical cadence remains mandatory for reproduction, while fused Adam and
network JIT remain separate default-off long-horizon ablations. They must be isolated one at a time
before any LR/batch/warmup optimization is layered on top.

The step15188 from-scratch factorial completed that isolation. Current-source historical Adam with
JIT off matches S1 within `-0.0074 dB / +0.002712 SSIM / -0.002125 LPIPS`, exonerating the common
semantic-fast hot path. Relative to that control, fused-only changes metrics by
`-0.0648 / +0.001693 / -0.001834` and JIT-only by
`-0.1313 / -0.001427 / +0.005662`; fused fails the relaxed PSNR repeat limit, while JIT fails PSNR
and LPIPS. Fixed-warmup cumulative exposure is bit-identical through step4090, then density and
occupancy feedback changes adaptive point exposure by as much as 0.7% at step15180. The primitives
therefore alter the closed training system, not merely final rounding. Neither may be enabled from
initialization in a faithful or promoted speed recipe. Future use requires delayed activation from
a proven historical checkpoint plus a paired historical-resume control.

That delayed common-parent pair showed why the activation boundary cannot be implemented as a
process restart. From the saved current-source H step15188 checkpoint, the historical-Adam/JIT-off
resume reached step30376 at `29.3432 / 0.662563 / 0.296550`, a
`-0.1114 dB / +0.001897 / +0.001416` delta from uninterrupted S1 and therefore a PSNR repeat
failure. From the same parent, fused Adam plus JIT reached
`29.3246 / 0.664055 / 0.297543`, only `-0.0186 dB / +0.001492 / +0.000993` from the paired resume
and inside all relaxed repeat limits. The resumed training segments took `761.132 s` historical
versus `670.955 s` fused+JIT, while cumulative point exposure differed by only `0.0248%`
(`56.8173 B` versus `56.8314 B`). Thus the fast bundle is locally safe after the historical prefix,
but the restart is not trajectory-faithful and the pair cannot be promoted.

The follow-up architecture uses opt-in in-process activation controls, default-off through
`TrainerConfig.fused_adam_switch_step=None` and
`LookCloserPipelineConfig.tcnn_network_jit_switch_step=None`. Core validation rejects combining an
initially enabled primitive with its scheduled switch. TCNN activation changes the existing
geometry/color modules' JIT flags while retaining their parameters; the callback is prepended
before occupancy callbacks so the boundary is applied before that update's training forward. Adam
activation rejects foreach and differentiable modes, moves scalar `step` state to the parameter
device, enables fused execution and synchronizes PyTorch's private `_step_supports_amp_scaling`
dispatch flag while retaining parameter groups, weights, moments and step counts. The optimizer manager
resynchronizes these invariants after checkpoint load. Scheduler, AMP scaler, RNG streams and
cumulative-point telemetry remain untouched. This behavior passed 66 focused tests and real CUDA
integration/checkpoint-resume smokes.

The first two scratch attempts reached approximately step4096 but were infrastructure-only: `v1`
used a fresh extension cache, while `v2` used the canonical cache without `CUDA_HOME` and
`TORCH_CUDA_ARCH_LIST`; both attempted an unsupported `compute_120` build. `v3` used the complete
canonical environment but was intentionally stopped before activation when review found the
missing `_step_supports_amp_scaling` synchronization. Corrected `v4` ran seed42 uninterrupted,
kept historical Adam/JIT-off through step15188, logged both activations exactly at step15189 and
reached step30376 at `29.437275 / 0.665461 / 0.298865`. Its delta from uninterrupted S1 is
`-0.017325 dB / +0.004795 / +0.003731`, inside the relaxed repeat bounds. The checkpoint retains
57,053,993,419 cumulative points, 30360 Adam/scheduler updates, LR
`0.004970499105762349`, AMP scale8192 and fused execution active. This accepts in-process activation
at step15189 for a longer diagnostic, but not yet as a final `<=60`-minute end-to-end candidate;
frozen defaults are unchanged.

Intermediate evaluation can be accelerated only through the opt-in
`TrainerConfig.replay_eval_trajectory=False` control. When enabled, the trainer preserves the
historical shared-RNG trajectory in its exact original order: it samples one eval batch, selects
one random eval image, then iterates the three fixed eval images, with the same eval/train mode
transitions. It skips field forward passes, metrics and renders at those intermediate boundaries.
Final candidate evaluation and all promotion renders/gates remain full. A same-checkpoint CUDA
preflight showed exact equality of Python/NumPy/Torch CPU/all-CUDA RNG state, eval sampler counters
and the complete following training batch/ray bundle between full evaluation and replay. The
frozen reproduction default stays `False`; replay is authorized only for the reviewed joint live
activation speed recipe and is recorded in controller provenance.

The first predeclared long candidate using that path is solo seed42, historical kernels through
step15188, live fused Adam plus TCNN JIT from step15189, the exact Stage-A restart/FR1->0.3 boundary
at step75940, and one hard final checkpoint at step80000. No extension or alternate checkpoint is
eligible, and promotion additionally requires controller wall `<=3600 s`. This is a speed
experiment, not a change to the canonical no-argument recipe.

That candidate completed the full controller protocol in `3511.836 s`, proving that replay plus
delayed fast kernels can fit the first wall milestone. Step80000 passed aggregate metrics at
`29.741262 / 0.676903 / 0.228587`, full-view artifacts `0/3` and serious ROIs `0/10`, but it was not
promoted: thin-pipe and tangled-cable priority detail did not simultaneously equal the archived
reference. The Stage-A-to-step80000 FR0.3 tail improved aggregate and thin-pipe LPIPS while cable
LPIPS worsened by `0.000848`. Consequently more iterations of the failed checkpoint are not a
semantics-preserving fix; the unresolved issue is long-horizon kernel/basin sensitivity in local
perceptual detail. Frozen reproduction defaults remain unchanged.

Forensic audit found that this failed treatment also restored the new speed checkpoint's persisted
RNG snapshot at the Stage-A-to-FR0.3 process boundary. The archived checkpoint has no RNG state, so
the actual leader used a deterministic seed-42 post-setup stream reset. The hard-speed controller
now requires `historical_stage_boundary_rng_reset`: it creates a provenance-bound checkpoint fork
that removes only `rng_state`, verifies all optimizer/scheduler/scaler/trainer fields and hashes,
and uses that fork as the Stage-B input. The operation costs about `3.35 s` for the 2-GB checkpoint.
This corrects process-boundary semantics without changing the canonical no-argument path, whose old
checkpoints already have no RNG snapshot.

Final evaluation uses a separately profiled chunk. On the same immutable checkpoint, chunks
2048/4096/8192 took `27.337/23.675/22.061 s`; PSNR and SSIM were identical, LPIPS differed by less
than `3e-7`, and all artifact/detail decisions agreed. Chunk8192 is used by the next explicit speed
candidate but is not a training default; chunk16384 remains excluded after a prior hash23 OOM.

The faithful Stage-B-reset hard run completed in `3495.525 s` and improved aggregate quality to
`29.765554 / 0.669484 / 0.225159`; thin pipe passed. It still failed one eval1 full-view artifact
and cable/fingers LPIPS. The next reviewed switch pair is therefore narrowly expanded from the
joint `(fused,JIT)=(15189,15189)` boundary to the staggered `(15189,30377)` boundary. This keeps
the faster optimizer after the proven historical prefix while withholding the larger early JIT
perturbation through the second cadence replay. Arbitrary switch pairs remain rejected by the
controller, and frozen defaults remain `None`.

The staggered hard run completed in `3510.713 s` (`58:30.7`) at `188.420 B` estimated point samples.
It passed aggregate metrics at `29.758568 / 0.678173 / 0.224903`, full-view artifacts `0/3`, and
serious ROIs `0/10`. It is nevertheless rejected: thin pipe passed, while cable and fingers missed
the archive only in LPIPS by `0.001164` and `0.006197`. Relative to the joint faithful-reset run,
delayed JIT removed its eval1 artifact and improved cable LPIPS by `0.004509`, but worsened fingers
LPIPS by `0.003478`. This demonstrates a crop-specific long-horizon JIT/basin trade rather than
uniform undertraining. The staggered pair is therefore not a default; no speed switch is promoted
until one hard checkpoint passes every priority crop.

A same-parent parallel Stage-B diagnostic then tested late LR multipliers `{2x,4x}` with loaded or
reset Adam. All four branches were clean automatically, but none passed priority detail. The 2x
branches retained aggregate gates while cable LPIPS worsened from `+0.001164` to
`+0.002608/+0.002983` and fingers remained near `+0.0062`; 4x degraded aggregates and every crop.
Late LR/Adam reset is therefore not promoted. The next reviewed causal axis is the existing live FR
schedule: switch `1.0 -> 0.3` before update64813 so hard step80000 receives the full historical
15188-update low-FR tail without adding optimizer updates or wall work.

The hardened controller accepts that FR schedule only as exact `(step,strength)=(64813,0.3)` and
only with staggered fused15189/JIT30377, eval-trajectory replay, historical Stage-B RNG reset and
hard step80000. The live FR arguments are emitted only for Stage A; Stage B begins at FR0.3 and does
not replay an already-completed switch after checkpoint load. The reviewed controller/gate protocol
for this candidate is `fef10e50e641ada1f7c3387529f4e28b70354b474ba4183dbf6c14a8a540a1b4`.

The early-FR run finished in `3556.836 s` (`59:16.8`) and remained automatically clean, but failed
at `29.757217 / 0.669999 / 0.231657` plus all three priority-crop LPIPS comparisons. Against the
matched staggered hard candidate it worsened aggregate LPIPS by `0.006754` and SSIM by `0.008174`.
The live FR mechanism and controller plumbing are correct, but `(64813,0.3)` is a rejected quality
schedule and is not a default. More low-FR updates do not substitute for the accepted late basin.

Legacy checkpoints persist model, Adam, scheduler, scaler and cumulative-point telemetry, but not
Python/NumPy/Torch RNG state. New speed-worktree checkpoints additionally persist Python, NumPy,
Torch CPU and every CUDA RNG state. Restore is deferred until immediately before the first resumed
iteration, after model/data/callback/writer setup has consumed initialization randomness. This
removes restart-stream drift from future LR/scheduler forks. Exact tensor identity is still not
promised: identical resumes retain a small CUDA/TCNN/occupancy atomic numerical floor, which is
measured at full-metric scale against the frozen reproducibility bounds. Legacy-parent forks still
require a same-parent restart control because their missing pre-boundary RNG state cannot be
reconstructed. Merely saving an intermediate model-only checkpoint does not restart the live process.
`scripts/fork_static_checkpoint_optimizer.py --drop-rng-state` is the explicit provenance-recorded
way to reproduce that historical new-process post-setup stream while leaving weights, Adam,
scheduler and scaler intact. The reviewed p21 diagnostic failed aggregate LPIPS and all priority
detail crops, so arbitrary RNG dropping within a phase is not a reproduction or speed default. The
one explicit exception is the historical Stage-A-to-Stage-B process boundary described above,
where absence of RNG state is part of the recovered leader recipe.

The speed worktree also has a default-off static training-ray cache. It is valid only for immutable,
uniform perspective-camera batches and fails closed otherwise. It precomputes direction, pixel
area and direction norm in bounded chunks, retains per-camera origins/times/metadata, consumes no
RNG, and registers all derived buffers as non-persistent. The canonical 66x1920x1080 cache is
`2.55 GiB`, builds in about `2.31 s`, and reduces mature B4096 iteration time from a
`48.63--48.80 ms` control median to `43.97 ms`. RayBundle fields are byte-identical before training;
500-update checkpoint drift is no larger than the measured control-repeat CUDA/TCNN floor.

The follow-up pipeline can stage JIT independently per TCNN subnetwork. The first live switch uses
`model.tcnn_network_jit_scope`; an optional later step/scope pair enables the second subset. Both
pairs default off, must be complete, and the second step must be later. Checkpoint load derives the
exact geometry/color JIT state from the loaded trainer step, then asserts both flags. This permits
the reviewed schedule color@15189 followed by geometry@30377 without reconstructing networks,
changing parameters, or resetting optimizer/RNG state.

Combined with the static-ray cache, fused Adam@15189, eval replay and the historical Stage-B RNG
reset, that schedule reaches hard step91128 at `29.802864 / 0.675499 / 0.222623`, clean `0/3` and
`0/10`, with pipe, cable holes and fingers all passing the archive comparator. It is the first
fully clean speed-worktree quality pass. End-to-end controller wall is nevertheless
`3617.973 s`, missing the 60-minute milestone by `17.973 s`; consequently all canonical defaults
remain historical/cache-off/JIT-off. Only semantics-free checkpoint/orchestration overhead may be
removed before this recipe is rerun for promotion.

Hard single-candidate speed campaigns now separate eval cadence from save cadence. They keep the
historical `15188` eval/RNG replay interval but use `final_step + 1` for scheduled saves, relying on
the trainer's unconditional `_after_train()` save for exact Stage-A and Stage-B endpoints. This
removes intermediate and duplicate checkpoint writes without changing model, optimizer, scheduler
or RNG execution. Canonical multi-candidate/default campaigns retain scheduled checkpoints.

Final checkpoint hashes are bound to a stable local-file identity recorded before candidate
evaluation. The recorder reuses the already computed digest only when resolved stage path,
successful return code, target step, SHA format and the complete device/inode/size/mtime/ctime
identity match; otherwise it hashes a non-final candidate or fails closed for a malformed/changed
authoritative stage. Candidate eval also suppresses the unused training-ray cache, runs exactly
three gate-only full-view artifact detector subprocesses concurrently, and skips ROI diagnostic
image writes while preserving full renders, detector logs, five detail crops and the contact sheet.

The second solo staged/cache seed42 campaign measured those changes end to end. Finalization fell
from `47.149` to `35.301 s`, but Stage A/B runtime rose from `2917.588/645.004` to
`3006.216/655.050 s`; total wall became `3704.831 s`. Its metric delta from the first same-seed run
passes the `0.06/0.01/0.005` repeat limits, but strict cable/fingers archive-detail parity failed by
small LPIPS deltas. Thus neither speed defaults nor the leader-quality gate are promoted. The
measured +3.2% mature-iteration variation means the next <=60 recipe needs roughly two minutes of
training margin rather than relying on orchestration savings alone.

The paired repeat first differs in train loss at step10 despite matching recorded CPU/CUDA RNG
hashes. Point exposure is identical through the fixed warmup and begins to differ only after
adaptive occupancy at step4100. By the Stage-A endpoint the slower repeat has one fewer successful
Adam/scheduler update from an additional AMP overflow skip and a different scaler history. This
places the repeat floor in nondeterministic CUDA/mixed-precision execution amplified by occupancy
feedback, not in checkpoint cadence. Exact deterministic arithmetic is not claimed; promotion uses
the frozen metric repeat bounds plus an independent strict archive-detail gate.

A dense same-parent Stage-B replay saved/evaluated steps84000/86000/88000/90000 and found no
earlier strict all-detail pass. Step90000 passes aggregate, automatic, pipe and cable gates but
still misses fingers LPIPS by `0.001171`; replayed step91128 retains tiny cable/fingers misses.
Therefore a shorter hard tail is an algorithmic quality trade and is not the next speed default.

The next default-off hot-path design prefetches one CPU FAS pixel batch on a single thread while
the main thread executes the current CUDA update. Only CPU indices/collated RGB may cross the
queue; cached ray generation remains synchronous on the main thread. Queue depth is exactly one,
FAS count commits at dequeue, and the queue is derived/non-persistent. Prefetch is barred across
frequency-grid CPU-RNG updates, eval/replay, save/final/phase boundaries and live sampler-size
switches; cancellation must restore the queued pre-sample Torch-CPU RNG state. Unknown CPU-RNG
callbacks fail closed. This feature cannot be enabled in the controller until byte-exact batch/ray
and RNG/state tests plus a solo ABA profile show at least `2.8 ms/update` saving.

The fingerprinted speed worktree additionally supports an opt-in live ray-batch boundary through
`LookCloserPipelineConfig.train_rays_switch_step` and `train_rays_after_switch`. At the boundary it
changes the existing training pixel sampler in-process and logs `train_rays_per_batch`; it does not
reload a checkpoint or reset RNG/Adam/scaler state. Both fields default to `None`, so the frozen
reproduction and fixed-batch speed recipes are unchanged. Scheduler and maintenance cadence do not
change implicitly with the ray batch: any such schedule is a separate named ablation.

The same experimental pipeline has an opt-in live FR boundary through
`feature_reweighting_switch_step` and `feature_reweighting_after_switch`. It updates the live
field/config strength before the scheduled forward pass and logs the active value, without a
checkpoint reload. Defaults are `None`; the accepted reproduction continues to use its explicit
historical Stage-A/Stage-A_fw03 boundary until a live schedule passes every quality gate.

The derived 2×/4× recipes scale rays per batch up and warmups, maintenance intervals, depth window,
scheduler horizon, checkpoint cadence and stage boundaries down by the same factor. Fixed warmup
exposure stays exactly 4,294,967,296 points. Full speed campaigns stop at the normalized historical
step-91128 acceptance boundary, then use the same first-numeric-clean selector. Mature batch
profiles and decisions are recorded in `experiments/static_leader_speed_optimization.md`.

## Static temporal from-scratch campaign

`scripts/run_static_target_from_scratch.py` is isolated from the frozen 007740 reproduction
controller. It trains one requested temporal frame from random initialization on the expected
branch (canonical production default: `main`), uses the same 75940-update FR1.0 ancestry and a
configurable continuation (default FR0.3), and rejects any Stage-A config containing a load
checkpoint or load directory. `--frame` defaults to the dataset directory name and must match it.
Stage B and optional one-interval tails may load only a checkpoint produced earlier by the same
campaign. The default data path remains the historical `007747` dataset.

`--stage-b-feature-reweighting` is an explicit, resume-validated recipe coordinate. The 007747
corrected-map sweep selects `0.2`; `0.1`, `0.2`, and `0.3` were compared from the same
random-initialized Stage-A parent. Seed is likewise explicit and uint32-validated so controlled
seed sweeps remain possible without weakening checkpoint ancestry checks. Default values preserve
the historical seed42/FR0.3 behavior.

The controller retains and fresh-evaluates every 15188-update checkpoint, then applies the normal
PSNR-first/LPIPS-within-0.07-dB selector. `scripts/static_target_roi_protocol.py` additionally
scores and saves the fixed eval0 crop `(700, 100, 1120, 480)` containing the contacting hands,
individual fingers and chain. Its contact sheet compares leader GT/render with target GT/render at
native crop resolution; automatic ROI metrics and artifact checks remain subordinate to an
explicit visual verdict. Campaign manifests, evaluation JSON, three-view renders and crops live
under `/home/brans/lookcloser_007747_from_scratch_runs/campaigns/<name>`.

Tail resumptions advance from the highest completed campaign-local checkpoint, so repeated
one-interval requests cannot replay an older boundary. Numeric plateau state and explicit
`improved`/`no_improvement` moving-detail reviews are stored per consecutive interval; plateau is
confirmed only when the last two intervals satisfy both gates.

### Default from-scratch quality budget: step 121504

The production default is now one complete 15188-update tail after Stage B:
Stage A `0→75940` at FR1.0, Stage B `75940→106316` at FR0.3, then the
campaign-local continuation `106316→121504` at FR0.3. Accordingly,
omitting `--tail-intervals` activates the absolute step121504 budget (one
interval from Stage B), and dry-run output includes the `tail_121504` command.
The absolute cap is resume-safe: resuming a campaign already at step121504 does
not silently extend it. `--tail-intervals 0` is a shorter diagnostic override;
an explicit positive interval count is plateau/research work rather than the
normal quality budget.

This boundary is backed by the three-seed, from-scratch `007810` sweep. Seed43
step121504 was the first and only checkpoint to pass that target's declared
aggregate PSNR/SSIM/LPIPS hard gates together with full-view, fixed-ROI and
manual visual gates (`29.715626 / 0.672032 / 0.215131`). Direct user inspection
confirms a visible quality improvement from step75940 to step121504. Continuing
seed43 to step182256 and finally step212632 produced no visible quality
improvement; it only reduced LPIPS slightly while PSNR and SSIM weakened.
Step121504 is therefore the reasonable default **quality** budget, not merely a
speed-versus-quality compromise. The budget change does not relax the
controller's existing acceptance gates; a checkpoint that misses its configured
quality threshold still fails closed at the default boundary.

The adaptive research controller may still extend one interval at a time until
two reviewed plateau intervals are confirmed. This default does not shorten
the separate cross-frame temporal fine-tuning controller: that pipeline keeps
its own measured budgets because earlier frames have required later boundaries.

The campaign preflight binds 66 train images, 3 filename eval images, 66 frequency maps, transforms,
JPEG profile, source hashes and the canonical leader reference. The leader checkpoint is hashed and
used only as a read-only reference; it is never passed to training.

### Temporal EXR-to-JPEG canonicalization

`scripts/convert_temporal_exr_to_leader_jpeg.py` is the frozen dev3 conversion path for temporal
frames. It stages below its script-local `temp` directory, pins decoder/encoder versions and JPEG
profile, proves the recipe 69/69 byte-exact on protected 007740, and permits a backed-up atomic
apply only to frames newer than 007740. Canonicalizing JPEGs creates a new dataset revision: do it
before frequency preprocessing, then regenerate maps from the new image bytes.

### Frequency-map preparation contract

Frequency maps are comparable only when their estimator-input contract is identical: decoded color
representation and chroma bandwidth, resolution, patch/SSIM settings, seed, and preprocessing
source must be frozen and recorded with image/map hashes. Generate one finite tensor plus a
filename-bound sidecar per train image into a new directory, and audit level histograms and scalar
resolution before training. If maps shift strongly while luminance detail is stable, first inspect
decoder/JPEG/chroma differences rather than tuning the model.

Keep train/eval images immutable within a map/training campaign. When a dataset family mixes export
profiles, either canonicalize the dataset as an explicit new revision before map generation or
normalize only the temporary estimator tensor to one declared profile and validate it with a
representative same-seed A/B; estimator-only normalization is not a universal default. For the
historical pre-conversion 007747 dataset, `scripts/build_chroma_normalized_frequency_maps.py`
applied horizontal 2× Cb/Cr low-pass to match the 007740 4:2:2-like reference while preserving
full-resolution luminance. Its `lookcloser_frequencies_chroma422` output belongs only to those old
4:4:4 JPEG hashes; after EXR-to-JPEG canonicalization, regenerate maps instead of reusing it.
The active 007747 temporal revision is therefore generated from canonical 4:2:2-like JPEGs and uses
only the ordinary `lookcloser_frequencies` directory. Its pre-conversion 4:4:4 JPEGs, direct maps
and estimator-normalized maps are forensic inputs archived outside the dataset root at
`/home/brans/007747_4_4_4`; they must not be discovered by new training jobs.

### Canonical 007747 hash23 fine-tuning v2

`scripts/run_lookcloser_007747_finetune_v2.py` is the dedicated controller for the active
canonicalized dataset. It loads the original step91128 hash23 leader directly with
`model_parameters_only`, verifies the revision manifest plus every JPEG/map hash, and compares the
effective target config against an exhaustive leader whitelist. Cross-frame startup must prove
copied field hashes and fresh Adam/scheduler/scaler/RNG, occupancy, frequency grid, FAS counter and
point telemetry. Same-frame continuation uses full resume with no grid or LR reset.

The pre-update baseline is evaluated in a separate process after trainer setup but before any
optimizer update; no fake step0 checkpoint is created. During training, the controller wraps the
existing scheduled all-image evaluation so the same three-view forward pass also saves renders and
wall timings. Checkpoint labels remain the repository's zero-based Nerfstudio labels
15188/30376/45564/60752. Each boundary gets a native 3×2 hands/fingers/chain sheet containing
leader, accepted scratch and candidate GT/render crops. New candidate GT is required to match the
active revision byte-for-byte. The accepted scratch GT is retained as historical visual context
and its revision mismatch is recorded rather than confused with target provenance.

Only initial LR and exponential-decay horizon vary. The first LR0.01/H200 campaign plateaued
without passing LPIPS, so an extended screen compared LR0.0125/H400, LR0.015/H300 and
LR0.015/H400. Final LR stays0.0001, warmup stays zero, and
FR0.3/FAS1/hash23/warmup4096 remain frozen. Discovery timing may be contended, but the official
`time_to_leader` comes from a new solo replay. Numeric success requires all three leader
thresholds plus an explicit visual pass; plateau additionally requires two reviewed consecutive
no-improvement intervals. The controller stops before production when source/data/config,
free-space, or visual evidence is incomplete.

The solo LR0.015/H300 replay first passed all gates at step136692
(`29.895269 / 0.677203 / 0.217243`) in `9003.521066166 s`. Two terminal
no-improvement intervals confirmed the plateau through step167068. The PSNR-first,
LPIPS-within-0.07-dB selector and the visual selector both choose step151880
(`29.880142 / 0.675660 / 0.214533`), checkpoint SHA-256
`000fbc9144505fe4041d61ba71f0f9f804c78de19517b70cd0584d519ae6a358`.

## Scene bounds / AABB

LookCloser now replaces the default `NearFarCollider(near=2, far=6)` with `AABBBoxCollider(scene_box)` when `pipeline.model.enable_collider=True`. This is important for fixed-step ablations because the fixed marcher should sample only the nerfstudio scene box instead of a hand-picked near/far slab.

For the 3k `007740` split use `nerfstudio-data --scene-scale 2.5` for current 3k LookCloser runs unless a later full validation contradicts it.

## Shared hash-grid defaults

With `scene_scale=2.5`, a short 3k sweep over LookCloser `max_res_base` found `2048` to be the best early eval-PSNR setting among `1024`, `2048`, and `4096`. Keep `pipeline.model.max_res_base=2048` as the current quality-first default; `1024` is close and slightly faster, so it is useful for fast debugging runs.

## Nerfstudio instant-ngp comparison hooks

For raw instant-ngp parity experiments, `nerfstudio.models.instant_ngp.InstantNGPModelConfig` exposes the underlying `NerfactoField` hash-grid and MLP shape: `base_res`, `num_levels`, `features_per_level`, `num_layers`, `hidden_dim`, `num_layers_color`, and `hidden_dim_color`. This allows testing raw-like settings such as 8 hash levels with 4 features per level without changing the default nerfstudio `instant-ngp` behavior.

The same comparison path also exposes `rgb_output_activation`, `loss_type`, and `raw_no_appearance_embedding`. These are for ablations against raw instant-ngp only: raw-like Huber loss and removing the appearance embedding were tested separately from the default `instant-ngp-big` baseline because they changed optimization behavior substantially.

`nerfstudio.data.dataparsers.instant_ngp_dataparser.InstantNGP` now reads `fl_y` directly when it is present in `transforms.json`. This avoids silently falling back to `fl_x` for non-square intrinsics in instant-ngp formatted transform files.

## Configurable LookCloser modules

The paper-level modules can be ablated independently through config flags.

- Frequency Grid: `pipeline.model.enable_frequency_grid` controls grid queries in the model; `pipeline.enable_frequency_grid` controls loading 2D maps and periodic grid updates. When disabled, the grid returns `fallback_frequency_level` and update steps are skipped.
- The legacy processed 3k dataset used by older ablations does not include `lookcloser_frequencies`, so those runs log a missing-map warning. This does not apply to the fingerprinted 007740/007747 datasets above.
- Feature Re-weighting: `pipeline.model.enable_feature_reweighting` controls Eq. 6 weighting in `LookCloserField`. When disabled, raw hash-grid features are passed to the MLP.
- FAS: `pipeline.datamanager.pixel_sampler.enable_fas` controls frequency-averaged pixel sampling. When disabled, `LookCloserPixelSampler` falls back to uniform `PixelSampler` behavior.
- Adaptive RM: `pipeline.model.enable_adaptive_ray_marching` controls adaptive ray marching. When disabled, `LookCloserModel` uses a fixed-step renderer with `fixed_num_samples_per_ray`.

Generic preprocessing prefers `train_steps_per_level` over the legacy `steps_per_image`, so every frequency level receives enough optimization before SSIM assignment. Its CLI entrypoint is `ns-process-lookcloser-freqs`; do not mix its outputs with a normalized-map campaign unless the estimator-input contract and provenance match.

## Temporal per-frame transfer

`scripts/run_lookcloser_temporal_finetune.py` is the production single-frame,
single-seed trajectory runner. It requires a target frame, the immediately
preceding accepted snapshot, and seed42/43/44. It evaluates the pre-update
transplant and freezes LR0.015, final LR0.0001, scheduler horizon300000 and the
15188-step evaluation/save cadence. The default horizon remains ten boundaries
through step151880. Process boundaries mirror the accepted 007747 run: one
direct model-only process through step60752 followed by one full-resume process
per interval. When the inherited 130% cap is below151880, the controller passes
that complete eval boundary as the initial horizon so the runner does not train
past the authorized budget; the same model-only start, optimizer recipe and
cadence are retained. Tail mode can resume exactly one additional interval
without changing the recipe.

`scripts/run_lookcloser_temporal_campaign.py` owns the sequential
007754--008048 chain. The active campaign runs one seed43 trajectory at a time
from the immediately preceding accepted snapshot; GPU-parallel seeds were
retired after measured contention made three concurrent trajectories take
6.44 hours instead of roughly 2.1 hours for one. Before every launch the
controller freezes the exact JPEG/map hashes, checks disk and VRAM, builds
native eval0 crop comparisons, and pauses fail-closed for explicit visual
decisions.

The preferred gates remain PSNR29.7, SSIM0.668 and LPIPS0.217, with
PSNR29.88, SSIM0.676 and LPIPS0.215 as the target tier. To prevent a difficult
frame from blocking the chain indefinitely, each frame has a user-authorized
training cap of 130% of its parent's selected local step, rounded down to the
last complete 15188-step evaluation boundary. If no checkpoint clears every
numeric gate by that boundary, the budget fallback considers only explicit
visual passes inside the cap, prefers PSNR+SSIM passes, and then orders by
maximum PSNR followed by minimum LPIPS inside the inclusive0.07-dB window,
earliest step and seed. If no boundary passes PSNR+SSIM, the same
PSNR-window/LPIPS selector is applied to all visual passes; SSIM never rescues
a bad PSNR/LPIPS candidate. A numeric miss can be promoted only through this
explicit fallback; the cap, missed gates and selection policy are recorded in
`selection.json` and `provenance.json`.

If every in-budget boundary fails the visual gate, the default remains
fail-closed and no child frame can start. Recovery requires both an explicit
frame and a complete final eval boundary on the controller CLI. It never
changes or resumes the failed trajectory with a different source revision:
the controller preserves that run, performs fresh GPU/storage preflight, and
creates a new attempt from the same accepted parent with model-only transfer
and fresh local-step0 state. The recovery cap and attempt lineage are recorded
in campaign state and, if promotion eventually succeeds, in the snapshot
budget override.

The detached controller remains responsible for hourly supervision: every
check verifies the controller and worker processes, compact progress,
checkpoint state, GPU memory and OOM evidence and appends the result to the
frame's campaign logs. Existing comparison sources carry size, mtime and
SHA-256 fingerprints so resume can reuse a proven render without repeatedly
hashing large images.

Promotion copies only the selected checkpoint into the target dataset, rewrites
the snapshot config to its final in-tree checkpoint and target dataparser, and
runs a fresh `ns-eval` using only that config. The controller then requires a
second explicit visual pass, writes provenance and the unique metrics row, and
only then exposes the snapshot as the next parent. Complete metrics, renders,
crops, configs, hashes, logs and timings remain under `/mnt/data`. To fit the
full campaign, the explicitly authorized retention policy removes only
nonselected intermediate checkpoint files after acceptance; their paths,
sizes and hashes remain in the pruning manifest and the selected source
checkpoint is retained.

`TrainerConfig.checkpoint_load_mode` defines the important boundary:
cross-frame `model_parameters_only` copies the exact `fields` parameter set but
not LPIPS, AABB, occupancy/frequency grids, FAS/point state, Adam, scheduler,
scaler or RNG. The target therefore begins at local step0 with a fresh pipeline
and fixed traversal for updates `0--4095`. Full `resume` is used only within
one frame and retains the complete target state.

If hash capacity changes across the boundary, parameter names and shapes no
longer match. `scripts/expand_lookcloser_hash_checkpoint.py` provides the one
supported hash23→hash24 conversion: it repeats each saturated TCNN level into
all new modulus partitions and applies the identical mapping to Adam moments.
A prefix copy is invalid because increasing the table modulus changes queried
rows. Every converted checkpoint must reproduce the source eval before it is
used; the canonical conversion reproduced `29.840143 / 0.669203 / 0.219455`
with render maximum pixel difference `1/255`.

The active validated `007740→007747` treatment loads the original hash23
step91128 leader without conversion, uses max-res8192, standard
`lookcloser_frequencies`, FAS1.0 and FR0.3 throughout. The fresh target Adam
starts at LR0.015 and decays exponentially to0.0001 over300000 local steps.
Reusing the late source LR or only resetting occupancy is rejected:
moving-frame transfer must also discard the source
frequency/FAS/dynamic-sampling state and optimizer trajectory. The
optional `resume_reset_frequency_grid` and `resume_reset_occupancy_grid`
controls exist for isolated same-checkpoint diagnostics; normal cross-frame
transfer obtains both resets structurally through model-only loading.

The hash24, chroma422-map, FAS0.75 and FR0.3→0.2 recipe was an accepted
historical pre-conversion treatment only. It is not the default, fallback or
map path for the active canonicalized revision.

Checkpoint filenames use the zero-based local Nerfstudio step, so step60752
contains 60753 completed updates. Full eval remains every15188 local updates.
Crash-recovery saves may be more frequent with `save_only_latest_checkpoint`
but do not create extra evaluation boundaries. Plateau requires two consecutive
complete intervals satisfying every numeric threshold and no visible
moving-detail improvement. Selection is maximum PSNR, then minimum LPIPS in
the inclusive0.07-dB window; SSIM is reported only. The fixed eval0
hands/fingers/chain crop remains a separate visual gate, and a formal global
tie-break is recorded separately when its crop is worse.

For the active canonical run, the first all-gate pass is step136692 and the
formal plus visual selection is step151880. The final two intervals through
step167068 satisfy the declared plateau rule. Exact recipe, rejected scheduler
controls and paths are recorded in
`experiments/temporal_007747_finetune_v2.md`. Historical pre-conversion
results remain in `experiments/temporal_lookcloser_finetuning.md`.

`scripts/temporal_roi_protocol.py` keeps permanent pipe/cable crops, propagates 007747 hand/chain/finger seed
boxes with forward/backward pyramidal LK confidence, discovers broad-motion and possible-hole crops from an
exposure-compensated adjacent-GT difference, and writes low-resolution `GT | render | residual` contact
sheets. Promotion requires a fresh exact-checkpoint three-view eval, all ROI categories, zero serious full/ROI
artifacts, confident tracking, critical-ROI LPIPS regression at most0.01, and an explicit visual decision.
Regression 0.01--0.02 or low confidence is ambiguous; regression above0.02 fails. A failed/ambiguous frame is
never substituted or forwarded. Every promoted frame must also remain within the declared canonical-leader
envelope: PSNR no more than0.20 dB lower, SSIM no more than0.010 lower, and LPIPS no more than0.015 higher.
The controller writes the isolated LR×0.5/LR×2/alternate-warmup/extra-FR1/
conditional-extra-tail diagnostic matrix and exits2; infrastructure/OOM/eval failures exit3 and are not
reinterpreted as quality experiments. Complete chain success exits0.

## Static-leader CPU FAS prefetch

The implemented speed experiment is transactional and intentionally narrow: one private-generator
CPU FAS batch, queue depth one, fixed B4096, one process, static cached rays, homogeneous fully
cached images, and no dynamic ray/point schedule. Step/count/config/tensor identity and mutation
version, plus the global CPU RNG pre-state, are validated before commit; otherwise the batch is
discarded and sampled synchronously. Frequency-grid, scheduled eval/replay, checkpoint, and
terminal boundaries explicitly drain the queue. The worker uses the same dense per-image bounds
LUT as the canonical sampler; this is required both for byte parity and to avoid Python/GIL
contention.

The feature is exposed as `--cpu-fas-prefetch` but remains default-off in both the quiet runner and
the E2E controller. It may only extend the exact reviewed staged cache/fused/JIT/replay recipe. The
reviewed committed-source fingerprint is
`6cf7eb9560403ed05da27b2eb7ce732585e930b2d13a0ccfbfb9dd1766e4c258` and controller protocol
fingerprint is
`156a73bf475771e357af73afe298f88421502387f8fcda6b24d689c8d50550ad`.

The first full solo E2E measurement with this path completed in `3501.901 s` (`58:21.9`), proving
that the implementation can move the reviewed staged recipe under the first wall milestone. It is
not a promoted default: the candidate produced `29.848965 / 0.667368 / 0.219900`, with clean
automatic artifacts but an absolute SSIM miss of `0.001082` and a priority cable-hole PSNR miss of
`0.088606 dB`. The transactional prefetch architecture remains available only behind the explicit
flag until one solo from-scratch run passes wall, numeric, automatic-artifact, ROI and all priority
detail gates simultaneously. The canonical reproduction path therefore remains unchanged.

The mature same-parent controls measured 42.8264 and 42.3648 ms/update; final LUT-prefetch measured
40.4975 ms/update with exact CPU/CUDA RNG, Adam, scheduler, LR and scaler parity. The observed
2.0981 ms/update saving is below the earlier 2.8 ms robustness screen. The subsequent full run
confirmed the wall projection but missed quality, so no default is promoted.

## Static-leader GradScaler variance control

`TrainerConfig.grad_scaler_init_scale` and `grad_scaler_growth_interval` make the two existing
PyTorch GradScaler constructor values explicit. Defaults remain `65536.0` and `2000`, so old
commands, configs and checkpoints retain historical behavior. The quiet runner exposes matching
optional flags and omits them by default; values must be finite/positive.

The initial variance experiment uses `8192/1000000`. It keeps FP16 autocast, unscale, finite checks,
safe skipped updates and backoff behavior, while preventing normal scale growth over the complete
static schedule. It does not make TCNN/atomic CUDA kernels deterministic and is therefore marked as
an algorithmic variance control, not a semantics-preserving optimization. Its purpose is narrower:
remove the discontinuous grow→overflow→skipped-update feedback that currently differs by one Adam
update between the all-quality v1 and failed-quality v2/v3 runs. The feature cannot become a
default from state correlation; paired same-seed prefixes and then a full quality/wall run are
required.

The predeclared solo `A/B/B` prefix rejected the initial policy. Both fixed arms made one safe
backoff to4096 and matched Adam, scheduler, LR and scaler state, yet their fresh step15188 metrics
differed by `0.100383 / 0.001135 / 0.003312` and cumulative points by `0.150640%`. Their field and
occupancy-`occs` symmetric relative L2 values were `0.841501` and `0.508255`. Fixed scaler growth
therefore neither removed TCNN/occupancy divergence nor preserved the default basin. The
`8192/1000000` mode remains diagnostic-only and is not wired into the E2E recipe or defaults.

`scripts/compare_static_checkpoint_drift.py` performs the reusable read-only checkpoint audit. It
uses mmap where supported, chunked FP64 norm accumulation, exact key/shape/dtype/finite checks and
reports per-key plus aggregate symmetric relative L2 for field and duplicate occupancy state.

## Native linear-EXR and adaptive frequency maps

The EXR path keeps scene-linear float RGB from decode through ray targets, field output, linear
volume compositing and EXR evaluation export. OpenCV's EXR decoder is enabled before import;
floating values are never divided by255, clipped to `[0,1]`, tone-mapped or passed through the
legacy sigmoid. Dataset calibration is deterministic and train-split-only: robust luminance
statistics provide the softplus output scale/initial value and one scene-to-nits constant shared
by training, PQ metrics and preprocessing. PNGs are neutral fixed-exposure review proxies; the
linear EXRs remain the masters.

`LookCloserModelConfig.rgb_output_parameterization` supports `linear_softplus` and `pq_code` in
addition to the legacy default. All HDR heads are decoded to linear radiance before compositing.
The implemented comparison set is linear L1, stop-gradient RawNeRF weighted L2, Linear-PQ
(linear head with PQ-domain L1), PQ-L1 (PQ-parameterized head decoded before compositing), and an
EAG-PT-inspired PQ-L1 plus patch DSSIM variant. ST2084 operations run in float32. Native eval
reports PQ-domain PSNR/SSIM/LPIPS and exports paired prediction/GT EXRs; the separate evaluator
adds clipping/finite diagnostics and fixed `-2/0/+2 EV` review sheets.

The HDR loss screen also exposes PQ-MSE and opt-in PQ-L1+LPIPS. LPIPS training uses genuine
row-major spatial patches sampled by FAS; the historical EAG-PQ-DSSIM path keeps independent-ray
sampling for checkpoint/recipe compatibility. Both losses still predict and composite linear RGB.

The promoted EXR quality path stages 64×64 PQ-L1+LPIPS training into a short PQ-MSE recovery and
uses dense 4× corrected adaptive rendering. Step107008 measures `34.369545 / 0.899050 / 0.199267`
PQ PSNR/SSIM/LPIPS with zero detected cable gaps; exact provenance is in
`experiments/exr_lpips_pareto.md`.
`run_hdr_sampling_ablation.py --checkpoint ... --variant adaptive_dense4x_corrected` reproduces
the selected renderer from an exact checkpoint instead of relying on a run directory's latest step.

A matched-point-exposure validation on the two new seeds43/44 now recommends EAG PQ-DSSIM followed
by a short64×64 PQ-L1+LPIPS phase and a PQ-L1 recovery tail for future EXR runs. Its dense4 corrected
mean is `34.737003 / 0.900355 / 0.205244`, with zero cable-gap pixels on all six eval renders. It is
only0.0567dB below the highest-PSNR branch while improving LPIPS by0.00835, and is the only branch
inside the frozen paired-seed equivalence bands. Scratch PQ-MSE and LPIPS were rejected after stable
large degradation. A later scratch-primary-loss audit invalidated the early relative rejection of
pure PQ-L1: small early metric gaps are not a safe stopping rule for an otherwise healthy loss. The
full two-seed audit at equal point exposure measured pure PQ-L1 at
`34.555962 / 0.899941 / 0.222590`, versus `34.763405 / 0.901025 / 0.212974` for EAG PQ-L1+DSSIM;
training time was equal within0.1%. Pure PQ-L1 also produced one visually confirmed12px cable gap,
while EAG produced none. Scratch PQ-MSE remained catastrophically bad on both seeds after two eval
boundaries (`~21.08` mean PSNR) and was correctly stopped. DSSIM0.3 is therefore now validated as
part of the primary recipe, rather than merely inherited from the earlier leader. Details are in
`experiments/exr_primary_loss_scratch_validation.md`. The existing seed42 step107008 checkpoint
above remains the historical retained artifact rather than being relabelled from cross-seed
evidence. The reusable controller,
selector and exact compact-checkpoint retention are in `scripts/run_exr_loss_schedule_validation.py`;
results and visual-review provenance are in `experiments/exr_loss_schedule_two_seed_validation.md`.

`scripts/build_adaptive_exr_frequency_maps.py` replaces the scene-specific SSIM constant. A single
progressive 2D HashGrid fit per training image produces the complete per-level PQ-SSIM recovery
cube. Three automatic map families reuse it: a scene-empirical calibrated crossing, a three-level
relative recovery ensemble, and a threshold-free knee. Structural Sobel/high-pass agreement,
rank agreement, high-detail recall, entropy/effective-bin coverage, top-bin collapse and unresolved
rate form the selection criteria; robust scaling, balanced TOPSIS and scene bootstrap stability
choose the winner. The proxy is measured in PQ so bright linear highlights do not suppress shadow
detail. Raw and guided-3×3-median candidates compete; the guided candidate preserves the strongest
20% structural patches. A family must also pass positive rank/detail gates before entropy can win
the global automatic selection. Every family and the global winner are saved with map/recovery hashes,
calibration, exact candidate parameters, quality values and preview images.

`scripts/run_exr_hdr_campaign.py` runs the map build, five-way HDR screen, three map-family screen,
loss-specific three-point tuning and one capped Stage-A-length final run. Candidate selection is
maximum PQ PSNR with LPIPS as tie-breaker inside0.07dB. `run_lookcloser_quiet.py` writes process,
GPU-memory/utilization and OOM checks at launch, at least hourly and at termination, so detached
execution does not bypass supervision. Exact map/tuning duplicates alias the already measured
same-seed run. After the first15188-step boundary, a candidate more than0.5dB PSNR,0.02 SSIM or
0.04 LPIPS outside the current leader envelope is stopped and retained as an explicit rejected
one-boundary result; candidates inside the envelope must reach a second boundary before selection.

The completed EXR campaign selected the threshold-free knee maps, EAG-inspired PQ-L1 plus11×11
patch DSSIM, and DSSIM weight0.3. The full75941-update run selected step75940 by the frozen
PSNR-first/LPIPS-inside0.07dB rule and measured `33.8176 / 0.8984 / 0.2218` in the dataset-calibrated
PQ domain. Native prediction/GT EXRs and fixed-exposure sheets are retained beside the checkpoint;
the independent evaluator found no non-finite or over-peak prediction channels.

The three-day HDR quality campaign adds two diagnostics/ablation layers without changing the
authoritative metrics. `score_hdr_edge_continuity.py` measures tolerant PQ-luminance edge recall
and long unsupported skeleton runs in declared cable ROIs, and writes GT/prediction/missed-edge
review sheets. `run_hdr_sampling_ablation.py` changes only ray-integration parameters on one frozen
checkpoint, allowing rendering artifacts to be separated from learned geometry errors before any
retraining. The optional EAG `eag_edge_weight` adds PQ horizontal/vertical finite-difference
consistency on the same contiguous training patches; zero preserves the completed leader exactly.

Broad edge averages are not a sufficient gate for the priority long black cable. Use
`score_thin_cable_gaps.py` after rendering native EXRs: coarse per-view waypoints define only a
corridor, anchors snap to the GT dark-ridge response, and a minimum-cost ordered centerline is traced
in PQ luminance. Prediction support is searched within ±3px; contiguous unsupported, brighter runs
of at least10px are veto failures. The script saves separate GT/prediction crops, a GT route mask,
red gap overlays and JSON lengths/fractions. This target gate invalidated the prior visual acceptance
of step98722 (longest gaps 60/67/39px across eval0/1/2); aggregate metrics cannot override it.

The corresponding renderer repair is eval-only selective occupancy dilation. With
`occupancy_eval_dilation_radius=1` and
`occupancy_eval_dilation_frequency_quantile=0.75`, the model temporarily expands the loaded binary
occupancy grid only inside the top quartile of that scene's nonzero frequency levels. The original
grid is cloned before dilation and restored on `train(True)`, so checkpoint state and resumed
optimization are unchanged. On step98722 this changes target cable gap pixels from246 to0 and the
longest gap from67px to0; PQ metrics are `34.0342 / 0.899406 / 0.212446`. Full-frame significant
artifact scores are zero on all eval views. The quantile is scene-adaptive and replaces the
diagnostic fixed-level15 control.

### Geometry-aware occupancy guard

New EXR training builds scene-relative PQ edge/dark-ridge maps and periodically projects them with
occupancy-independent density probes into a persistent 3D support grid. The selected q80 radius1
cross guard only augments binary traversal, keeps cosine/EAG-PQ-DSSIM unchanged, and at step106316
measures `34.136342 / 0.899861 / 0.211813` with zero cable gaps and zero significant artifacts.
Details and artifacts are in `experiments/exr_geometry_occupancy_guard.md`.

`build_edge_aware_frequency_variants.py` cheaply derives conservative candidates from the cached
EXR recovery results: knee+1, a scene-quantile structural floor, and knee/calibrated unions limited
either globally or to dilated high-structure cells. These candidates retain16-level scalar-
resolution maps and record hashes, changed fractions and bin statistics; they are experimental
until downstream training and visual gates accept one.

The cable campaign established that the budget-aware corrected ARM allocator improves aggregate EXR
metrics but does not by itself repair cable continuity. Corrected ARM plus selective q75 eval
dilation remains the historical frozen-checkpoint repair. New training uses corrected ARM plus the
geometry-aware q80 cross guard and requires no eval dilation.
Unlike the historical cap, which can truncate late occupied intervals after ceiling/scale rounding,
the corrected path merges excess intervals and allocates at least one sample before distributing
the remaining per-ray budget by largest remainder. An equal-state continuation showed that this is
causal rather than an update-count effect. Historical reproduction keeps the model/runner default
off; the recommended EXR recipe explicitly passes `--corrected-arm-allocator`, retains coarse step
`0.00625`, cap `1024`, knee maps, batch `3993`, EAG PQ-DSSIM weight `0.3`, and the cosine scheduler.
The EXR controller's default campaign namespace is consequently
`exr_hdr_auto_frequency_v2_corrected_arm`; the historical v1 artifacts remain immutable.
Dense `0.003125`/`2048`, edge-loss `0.1`, FR `0.3`, and structural map-floor candidates were rejected.
The selected step98722 measures `34.0497 / 0.8993 / 0.2134`; all native outputs are finite and the
five-ROI cable long-gap fraction is `0.07238` versus `0.07790` at the prior selected checkpoint.
