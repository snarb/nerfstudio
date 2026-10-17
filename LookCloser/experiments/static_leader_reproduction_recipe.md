# Frozen end-to-end recipe: static LookCloser leader on clever-shadow

Status: frozen reproduction baseline before speed optimization. The training recipe below is the
accepted stable-FP16 trajectory. It reproduces or beats the archived 007740 LPIPS leader with one
scheduled checkpoint and has a measured same-seed repeat range inside the user-approved
`0.06 dB PSNR / 0.01 SSIM / 0.005 LPIPS` gate.

This document is intentionally self-contained. The broader forensic history and rejected controls
remain in `experiments/static_leader_e2e_reproduction.md`.

## Additional-dataset findings (original leader unchanged)

For Luster person-focused sampling and the final measured configuration, see
[Luster adaptation and reusable findings](#luster-dataset-adaptation-and-reusable-findings-2026-10-17).
**Check one-cell occupancy support early when investigating spots, grid-aligned steps or missing
thin geometry.** This repeatedly helped in Luster/EXR controls; the section distinguishes that
mechanism from extra density layers and specifies noncumulative dilation inside the unchanged AABB.

## One-command run

From the clean main worktree `/home/brans/repos/nerfstudio/LookCloser`:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python \
  /home/brans/repos/nerfstudio/LookCloser/scripts/run_static_leader_e2e.py \
  --campaign-name <unique-name>
```

No seed or checkpoint choice is required. The default seed is 42. The controller checks the local
dataset against read-only `ubuntu@dev3`, trains both ancestry stages, retains scheduled
checkpoints, fresh-evaluates numeric candidates in chronological order, renders and scores the
first automatic clean pass, and records the result in:

`/home/brans/lookcloser_leader_repro_runs/campaigns/<unique-name>/campaign.json`.

Exit code 0 means that one checkpoint met all numeric and automatic artifact gates. Exit code 2
means training and the complete selector finished correctly but no scheduled checkpoint passed.
Exit code 3 means the result was not fully evaluated or finalization/provenance infrastructure
failed; it is never interpreted as a quality result. A failed seed is never replaced with another
seed. Disabling automatic finalization is allowed only for forensic use and cannot return success;
speed mode requires finalization.

Use `--random-seed` to generate one random seed once and record it in the manifest; the same seed is
then used across both stages. Use `--seed N` for an explicit non-default seed. Speed experiments
must keep this policy and must not use best-of-N selection.

## Frozen environment and provenance

| Item | Frozen value |
|---|---|
| Host/GPU | `clever-shadow`, NVIDIA RTX PRO 6000 Blackwell 98 GB (`sm_120`) |
| Python environment | `/home/brans/repos/nerfstudio/.venv` |
| PyTorch/CUDA | `2.7.1+cu128`, CUDA runtime 12.8 |
| Python/toolchain | `3.10.20`; `CUDA_HOME=/usr/local/cuda-12.6`; `TORCH_CUDA_ARCH_LIST=9.0+PTX` |
| Training worktree | `/home/brans/repos/nerfstudio_leader_stable_occ` |
| Historical Nerfstudio commit | `85818149` |
| Accepted source fingerprint | `69d4f36cc1e06256a8dcd5a1e9dd6c4a465bb81e8cee09a3d8b188358857b252` |
| Controller/gate protocol fingerprint | `156a73bf475771e357af73afe298f88421502387f8fcda6b24d689c8d50550ad` |
| TCNN source | `/home/brans/deps/tiny-cuda-nn-2e757`, commit `2e757bbe…c09669` |
| TCNN FP16 overlay | `/home/brans/deps/tcnn_2e757_py310` |
| TCNN source diff SHA-256 | `441f8877df4bbcc665dd1072c23d4cec8063f18ed14c909b598fde3a95a41673` |
| TCNN build provenance SHA-256 | `566e6dd9caba605ab053408794c9bbc854dedd0d171c1b1f99e77abe95180b5f` |
| TCNN binding SHA-256 | `f2163346afd103c27e78b9f56f8d82b6eeb3317c1ce11caf57d45f0216aece36` |
| Dataset | `/home/brans/temporal_perframe_stride7_45f/007740` |
| Split | filename, 66 train + 3 eval images |
| `transforms.json` SHA-256 | `022f8748a1a039861a754e68ab3ef830beeb3e5dd94ccb00457a630d28f64aa1` |
| Numeric eval chunk | `2048` rays by default |
| Runtime math flags | matmul TF32 false; cuDNN TF32 true; deterministic algorithms false |

The worktree contains only the public-Pillow/trusted-checkpoint compatibility fixes and the
accepted stable occupancy reducer. The reducer combines duplicate occupancy candidates by cell
before applying one max-with-decay update. This removes a CUDA write race; it does not replace
adaptive ARM traversal. Historical FP32 TCNN gradient accumulation is not enabled because it costs
about 14.3% more wall time and is unnecessary under the accepted repeat gate. The controller
records Python, PyTorch/CUDA, GPU, TF32 and deterministic-algorithm state; it rejects a PyTorch/CUDA
version or accepted source/TCNN fingerprint mismatch.

The canonical end-to-end defaults live in `scripts/run_static_leader_e2e.py`. The main
`ns-train lookcloser` preset and no-argument `scripts/run_lookcloser_quiet.py` now match Stage A of
this recipe, including the local dataset, B4096, hash23, FR/FAS1.0, max-res8192, 4096-update warmup,
stable occupancy and historical optimizer/cadence. Reproduction still uses the controller above:
it alone performs the checkpointed FR `1.0→0.3` Stage-B continuation, chronological selector and
complete fail-closed artifact/detail protocol.

The no-argument controller defaults are frozen as follows. Every speed feature is opt-in, so a
plain one-command run cannot silently inherit an experimental optimizer or sampling policy.

| Controller field | Frozen default |
|---|---:|
| `seed` | `42` |
| `historical_worktree` | `/home/brans/repos/nerfstudio_leader_stable_occ` |
| `batch_scale` | `1` |
| `target_points` | `0` (fixed 4096-ray batch) |
| `corrected_arm_allocator` | `false` (historical allocator) |
| `fused_adam` | `false` (historical non-fused Adam) |
| `fused_adam_switch_step` | `None` |
| `tcnn_network_jit_switch_step` | `None` |
| `replay_eval_trajectory` | `false` |
| `historical_stage_boundary_rng_reset` | `false` (canonical checkpoint already has no RNG snapshot) |
| `speed_final_step` | `None` |
| `lr_scale` | `1.0` (`0.01 → 0.0001`) |
| `stable_occupancy_reduction` | `true` |
| `tcnn_grid_grad_fp32` | `false` |
| `independent_rng_streams` | `false` |
| `speed_stop_at_accepted_boundary` | `false` (run full scheduled ancestry) |
| `automatic_finalization` | `true` |

The accepted training-source fingerprint remains `69d4f36c…8857b252`. The protocol fingerprint
changes when controller or gate code changes even if the weight recipe does not; the value above
includes the reviewed opt-in speed plumbing, the canonical speed wrapper, and fail-closed retry
finalization with the same mandatory priority-detail gate as normal candidate recording.

The experimental speed source is no longer represented by a detached historical HEAD plus a dirty
patch set. It is committed on branch `nerfstudio_leader_speed` in
`/home/brans/repos/nerfstudio_leader_speed`. Speed campaigns require that exact named branch,
committed HEAD, clean status and reviewed file hashes; changing any of them blocks the run until a
new speed generation is explicitly reviewed and frozen. This does not change the accepted
no-argument reproduction recipe above.

## Promoted main defaults

As of 2026-07-17, main promotes the accepted quality trajectory rather than the fastest observed
near-pass. `LookCloserModelConfig` defaults to adaptive warmup4096 and stable occupancy reduction;
the method preset uses the exact Stage-A trainer/eval/save boundary. Rejected or not-yet-promoted
axes stay opt-in: corrected ARM, dynamic point budget, cached rays, CPU FAS prefetch, fused Adam,
TCNN JIT, fixed GradScaler, independent RNG streams and occupancy-diagnostic opt-out. This policy
chooses reproducible quality over a nominal wall win that misses SSIM or a priority cable/finger
gate. It does not claim that bare `ns-train` implements Stage B; use the one-command controller at
the top of this document for that.

The controller refuses to start if the dataset manifest, source commit, allowed dirty-file set,
TCNN source/build provenance, imported binding hash, PyTorch/CUDA runtime, or transforms hash
differs. Candidate finalization is fail-closed: every evaluated candidate must produce a fresh
summary and a complete artifact protocol with exactly three rendered views and ten scored ROIs;
missing checkpoints, evaluator errors and incomplete ROI output stop selection as infrastructure
failures rather than allowing a later checkpoint to hide them.
The separate protocol fingerprint covers the normalized controller, evaluator, recorder, retry
finalizer, dataset-provenance checker, detail scorer and immutable detail reference. A change to any
of those inputs blocks the campaign until explicitly reviewed and refrozen.

## Stage A — from scratch through step 75940

- seed 42 by default;
- 4096 training rays per optimizer update;
- adaptive ARM, maximum 1024 samples/ray, coarse step `0.00625`;
- fixed-march/adaptive and occupancy warmups: 4096 updates;
- frequency grid enabled from update zero;
- FAS enabled from update zero at strength `1.0`, distribution `1→3`, no ramp;
- feature reweighting enabled from update zero at strength `1.0`;
- hash23, 16 levels × 2 features, maximum resolution 8192;
- Charbonnier RGB, distortion multiplier `0.01`, early depth multiplier `0.001`;
- Adam LR `0.01`, exponential decay to `0.0001` over 200000 scheduler steps;
- checkpoints/full three-view eval at steps 15188, 30376, 45564, 60752 and 75940;
- trainer limit 75941, which saves `step-000075940.ckpt`.

The fixed warmup exposes exactly
`4096 updates × 4096 rays × 256 samples = 4,294,967,296` point samples. Stage progress is reported
both as optimizer updates and cumulative point samples; wall-clock is not used to decide the FR
transition in the frozen reproduction.

## Stage A_fw03 — restored continuation through step 106316

- load the exact Stage-A step-75940 model;
- restore Adam, exponential scheduler and AMP scaler rather than resetting them;
- keep adaptive ARM, frequency grid and FAS unchanged;
- change only feature-reweighting strength `1.0 → 0.3`;
- continue the same LR/scheduler trajectory;
- scheduled checkpoints/full eval at steps 91128 and 106316;
- trainer limit 106317.

The seed remains recorded as 42, but a historical checkpoint does not contain Python/NumPy/Torch
RNG state. The optimizer and scaler continuation is exact; the resumed sampling stream is a
documented seed reset, matching the recovered leader procedure.

## Automatic checkpoint selector and gates

The controller does not choose the lowest internal eval loss and does not choose the best seed. It
uses each scheduled online three-view evaluation as a numeric prescreen, then fresh-evaluates
prescreened checkpoints in chronological order. The first checkpoint whose authoritative fresh
three-view metric vector simultaneously meets these gates proceeds to promotion:

| Metric | Gate |
|---|---:|
| PSNR | `≥ 29.617964` |
| SSIM | `≥ 0.668450` |
| LPIPS | `≤ 0.231135` |

Promotion additionally requires all three full eval views to be scored, significant artifact
count 0, serious ROI artifact count 0, and archived-detail parity on three frozen priority crops:
eval1 thin pipe, eval2 tangled cable holes and eval2 fingers. Blind stand and label scores are also
always saved. The all-five-ROI aggregate is deliberately stricter than the promotion gate: a
candidate cannot hide a failed priority crop, but a tiny stand/label trade is not mislabeled as a
structural hole. Detail-scorer exit 2 with complete JSON is a measured quality failure; missing or
malformed output remains a fail-closed infrastructure failure.
Candidate recording and retry finalization additionally require the exact frozen five-crop set and
bind the detail JSON to the render directory being promoted. Retry must match the protocol
fingerprint already stored by the campaign; only a manifest old enough to have no such field can be
stamped through the explicit reviewed `--allow-legacy-protocol-migration` option.

The selected checkpoint receives a fresh full-precision evaluation and full renders. The latest
checkpoint is not rendered redundantly. Controller wall-clock includes provenance, both training
stages, scheduled evals, final full evaluation, renders, artifact gates and detail/contact-sheet
generation. Dataset preprocessing and any SSH data transfer are reported separately.

## Measured accepted evidence

The accepted run is:

`/home/brans/lookcloser_leader_repro_runs/campaigns/leader_stableocc_S1_seed42/campaign.json`.

Its first passing scheduled checkpoint is step 91128:

| PSNR | SSIM | LPIPS | Significant artifacts | ROI serious |
|---:|---:|---:|---:|---:|
| 29.840143 | 0.669203 | 0.219455 | 0/3 | 0/10 |

Relative to the archived leader, it is `+0.222179 dB PSNR`, `+0.000753 SSIM`, and
`−0.011680 LPIPS`. Cable holes, eval1 thin pipe and fingers pass the archived local comparator.
Stand and label miss the deliberately strict all-metric crop aggregate, so that limitation remains
visible in the candidate JSON and contact sheet.

Immutable accepted artifacts:

- checkpoint:
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt`,
  SHA-256 `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930`;
- config:
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/config.yml`,
  SHA-256 `a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb`.

This accepted campaign predates the current hardened controller schema. It is authoritative
evidence for the weights, metrics, renders and gates, but not evidence that the current hardened
controller itself completed a canonical solo wall-clock run. The hardened controller has a
verified dry-run and must produce the next milestone timing end to end before a new speed claim.

The paired seed-42 S0/S1 online range at step 91128 is
`0.0150 dB PSNR / 0.003384 SSIM / 0.000710 LPIPS`, passing the accepted repeat criterion. S0 is not
substituted for S1: it is retained as a failed automatic-artifact repeat because eval1 contains one
serious full-view component. This makes clean-pass reliability a separate requirement from metric
repeatability.

Measured point exposure for S1 is about 176.065 B through Stage A and 261.348 B through the full
trajectory, including the fixed warmup. The accepted step 91128 is earlier, at roughly 218–220 B
total point samples depending on the repeat. These quantities, rather than dev3 training time, are
the starting reference for batch/LR/warmup speed experiments.

The paired S0/S1 campaigns were intentionally concurrent and therefore are not the canonical solo
wall-clock baseline: each took about 8104 s for Stage A and 3829 s for Stage B under shared GPU
load. Every claimed speed milestone must be rerun solo from controller start through finalization.
Parallel runs may screen quality and seed robustness, but their per-run wall time is labelled
contended and never used for a milestone claim.

## Frozen versus tunable fields

Before the first speed candidate, the accepted source/data/TCNN, hash23 capacity, FAS/FR ancestry,
loss, leader gates, selector and artifact/detail protocol are frozen. Named speed ablations may
change batch/warmup/LR/scheduler, point target or ARM allocation semantics, optimizer kernel, and
reviewed semantics-preserving hot paths only when the controller records the changed source and
configuration fingerprint explicitly. None of those axes changes the no-argument frozen recipe.
Each candidate first compares cumulative point exposure and quality, then the winning recipe is
rerun solo for canonical end-to-end wall time.

The separate geometry-first `MetricRateScheduler` is not this reproduction recipe. Its dynamic
p19 point budget, phased late FAS/FR, corrected ARM and different targets make it an algorithmic
research branch; it must not be used as the baseline for leader speed claims.

The first reviewed speed candidate is also deliberately outside these defaults: it opts into
in-process fused-Adam and TCNN-JIT activation at step15189, exact intermediate eval-trajectory
replay, and a hard final step80000. Those flags are recorded per campaign and cannot affect the
one-command reproduction shown at the top of this file. A speed recipe is promoted into this
document only after it passes the same numeric, artifact and priority-detail gates end to end.

That first speed candidate finished in `3511.836 s` and passed aggregate metrics plus automatic
artifact gates, but failed thin-pipe and cable priority-detail parity. It is therefore recorded as
a wall-feasibility result only and has not changed any frozen default in this recipe.

Forensic audit additionally showed that its new speed-worktree checkpoint restored persisted RNG
at the Stage-B boundary, unlike the archived leader's documented new-process seed reset. The next
explicit speed candidate corrects that boundary automatically while preserving model, Adam,
scheduler and scaler. This is opt-in plumbing for new checkpoints; the frozen canonical checkpoint
already lacks RNG state and therefore retains the exact behavior described in Stage A_fw03 above.

The faithful-reset candidate finished in `3495.525 s` and improved aggregate metrics, but still
failed one full-view artifact plus cable/fingers perceptual-detail parity. It is not promoted. The
next reviewed speed-only control delays JIT to step30377 while keeping fused Adam at15189; the
controller accepts exactly that named switch pair and continues to reject arbitrary boundaries.

That staggered control finished in `3510.713 s` (`58:30.7`) and passed aggregate metrics
(`29.758568 / 0.678173 / 0.224903`), full-view artifacts (`0/3`) and serious ROIs (`0/10`). It is
also not promoted: thin pipe passed, but cable and fingers missed the archived LPIPS reference by
`0.001164` and `0.006197`. The result confirms the `<=60` wall feasibility of the controller while
showing a crop-specific JIT/basin trade. Canonical defaults therefore remain seed42, historical
non-fused Adam, JIT off, full cadence evaluation, Stage A75940 to Stage B91128, and chunk2048.

The subsequent same-parent LR/Adam diagnostic rejected late LR multipliers 2x/4x with either
loaded or reset Adam: none recovered cable and fingers detail, while 4x degraded aggregate quality.
The next explicit speed-only control therefore keeps LR/Adam unchanged and moves the historical FR
`1.0 -> 0.3` transition in-process to update64813, giving exactly 15188 low-FR updates through the
hard step80000 checkpoint. This schedule is accepted by the controller only as the exact tuple
`(64813,0.3)` together with fused15189/JIT30377, replay, historical Stage-B RNG reset and hard80000.
It remains an experiment and does not change the canonical defaults above.

The exact early-FR campaign completed in `3556.836 s` but was rejected: aggregate
`29.757217 / 0.669999 / 0.231657`, automatic artifacts `0/3`, serious ROIs `0/10`, and all three
priority crops failed LPIPS parity. Relative to the matched staggered hard candidate, the longer
FR0.3 tail worsened aggregate LPIPS by `0.006754`. It is not promoted; canonical FR ancestry and all
other frozen defaults remain unchanged.

The later staged-JIT/cache seed42 campaign finally passed all aggregate, artifact and priority
detail gates at step91128 (`29.802864 / 0.675499 / 0.222623`, `0/3`, `0/10`, pipe/cable/fingers
pass), but its complete controller wall was `3617.973 s`. Because it missed the first speed
milestone by `17.973 s`, it does not replace this frozen recipe or its no-argument defaults. Its
manifest is
`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_hard91128_seed42_v1/campaign.json`.

A second solo same-seed run removed only redundant hard-candidate checkpoint writes and
finalization work. It reduced finalization from `47.149` to `35.301 s`, but training itself varied
upward: Stage A/B took `3006.216 / 655.050 s`, and controller wall was `3704.831 s` (`61:44.8`).
Its global metrics `29.832731 / 0.676254 / 0.218988` are within the approved same-seed repeat bounds
relative to the first staged run, and automatic artifacts remain `0/3` and `0/10`. The strict
archive-detail gate failed cable holes and fingers by only `+0.000898/+0.001004 LPIPS` (cable PSNR
`-0.009834 dB`), but the global reproducibility tolerance does not relax that quality gate. The
campaign is therefore not promoted:
`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_hard91128_seed42_ioprune_v2/campaign.json`.

This repeat strengthens the reason to keep the defaults above unchanged: the frozen recipe is the
accepted reproduction anchor, while the staged/cache recipe remains an explicit speed experiment.
The next speed milestone needs enough training-time margin to tolerate at least the observed 3.2%
mature-iteration fluctuation, and it must still pass the unrelaxed priority-detail comparator.

The speed worktree now also contains a reviewed, explicit `--cpu-fas-prefetch` experiment path. It
is not part of this frozen reproduction command and remains default-off: mature profiling passed
state parity but saved 2.0981 ms/update, below its conservative predeclared robustness screen. A
diagnostic full E2E run may test the observed wall, but this recipe changes only after an actual
`<=3600 s` run passes every frozen numeric, artifact, ROI and priority-detail gate.

That diagnostic has now completed. It passed the wall milestone at `3501.901 s` with clean
automatic artifacts (`0/3`, `0/10`) and aggregate `29.848965 / 0.667368 / 0.219900`, but it missed
the absolute SSIM gate by `0.001082` and the priority cable-hole PSNR comparison by `0.088606 dB`.
Its aggregate delta versus accepted S1 is inside the separate repeat tolerance
`0.06 / 0.01 / 0.005`; neither that tolerance nor the wall pass relaxes a per-run quality gate.
Consequently `--cpu-fas-prefetch` remains default-off and the no-argument reproduction recipe in
this document is still the authoritative accepted default. Full campaign:
`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_prefetch_lut_seed42_v3/campaign.json`.

## Luster dataset adaptation and reusable findings (2026-10-17)

This is a measured adaptation on **Luster 000540**, not a replacement for the frozen 007740
leader, its selector, numeric gates or no-argument defaults above. The final runtime remains in
its isolated campaign worktree; this documentation does not install its experimental flags into
main. The [full adaptation report](luster_000540_body_optimization.md) records the comparisons,
checkpoint paths, source archive and limitations.

### Check one-cell occupancy support early when diagnosing artifacts

**A one-cell occupancy guard has repeatedly removed real artifacts in our investigations.**
On Luster, the face discontinuity followed an occupancy-cell boundary: shifting only the grid
lookup by 0.25 cell moved the discontinuity while field, camera, AABB and fixed sample positions
stayed unchanged. A radius-one dilation removed it. In the separate EXR cable experiment,
geometry-derived occupancy support also needed a one-cell neighborhood to avoid breaks.
See the [Luster causal controls](luster_000540_face_fix_occupancy.md) and
[EXR support experiment](exr_geometry_occupancy_guard.md). The support-building algorithms differ;
the repeated finding is that a valid surface can be clipped by the binary occupancy lookup.

For the Luster cold-training adaptation, `occupancy_dilation_radius=1` expands the **binary occupied
cells** with a 3x3x3 neighborhood after each fresh thresholding (every 16 updates). Expansion is
**noncumulative**. It changes neither AABB dimensions nor grid resolution. Keep one verified AABB
for the whole video and check actual field, collider and grid bounds, not just metadata.
The earlier frozen-grid intervention applied dilation once to the loaded grid; changing a config
alone cannot repair a frozen binary buffer. Never repeatedly dilate an already expanded buffer.

The implementation and CLI plumbing already exist in main. Append the appropriate option to
the rest of the training command:

```bash
# scripts/run_lookcloser_quiet.py
--occupancy-dilation-radius 1

# ns-train lookcloser (model option, before the dataparser subcommand)
--pipeline.model.occupancy-dilation-radius 1
```

The model configuration key is `occupancy_dilation_radius`; `0` disables dilation and remains
the global default. See [`_dilate_occ_binaries`](../../nerfstudio/models/lookcloser.py) and the
[quiet runner](../scripts/run_lookcloser_quiet.py). For an unexpanded, frozen checkpoint grid,
the explicit one-time operation is `model._dilate_occ_binaries(1)` after loading; this is a private
diagnostic method, not an automatic load hook. Do not apply it again to a checkpoint whose binary
grid already contains the halo. Ordinary training rebuilds the binary grid before applying the
configured dilation, preventing cumulative growth.

**Default policy:** explicitly use `1` in the measured Luster adaptation; keep `0` in the frozen
leader and global preset. Repeated support-clipping fixes justify testing radius one early, but
do not establish a universal improvement: extra occupied neighbors can expose unwanted density,
increase work and degrade quality (see [earlier occupancy controls](occupancy_improvements_repport.md)).
Changing the global default requires matched radius0/radius1 runs on the original leader and
additional scenes, with priority-detail/geometry review and measured runtime as well as metrics.
The leader's existing acceptance gates still apply; this documentation does not change defaults.

This is an early diagnostic control, not an instruction to increase dilation indefinitely.
Further expansion did not fix the remaining Luster bands, and disabling occupancy exposed
untrained density. Large dirty face bands had a **separate depth mechanism**: multiple density
layers along a ray, reduced by raising the existing distortion regularizer to 1. Near-unit total
opacity does not establish a single correct surface. Do not attribute the entire face improvement
to dilation alone, or generalize distortion=1 as optimal for every scene.

### Retained Luster recipe

| Component | Measured configuration |
|---|---|
| Data and selection | Long side 1920, aspect ratio preserved; all 162 TRAIN views; cam012/095/150 used for validation/selection only; cam007 is TRAIN |
| Targets | Original unmasked RGB and measured background plates; no new depth/semantic priors, black compositing or texture transfer |
| Mask preparation for sampling | Preserve original positive pixels; closing 8 px and fill interior holes; protective band 16 px, 32 px over upper 75% of silhouette; no component pruning; inspect all camera overlays |
| Camera / pixel sampling | Cameras uniform; at full FAS strength: 33.333% uniform person core, 33.333% person-FAS, 13.333% protective band, 20% global image |
| Reliable FAS | 8x8 patches with at least 90% original-mask coverage and selected pixel inside that mask; class weights 1 to 3; uncertain edges remain reachable uniformly |
| Frequency preparation | SSIM acceptance threshold .95; TRAIN-only maps and matching FAS pools from the same 2D fits used in the .97 control |
| 3D frequency grid | Legacy-zero initialization; updates every 1024 steps from reliable TRAIN person pixels, alpha >= .9 and valid geometry; preserve grid when resuming |
| Geometry support | Verified common video AABB; noncumulative occupancy radius-one guard, as described above |
| Field | Hash size 2^23; 16 levels x 2 features; resolutions 16 to 8192; width64, geometry1/color2 layers, SH4; trunc_exp density and corrected SH directions |
| Warmup | Steps 0–4095: fixed 256 samples/ray, batch4096 rays |
| Adaptive stage | ARM from4096, maximum4096 samples, minimum step1e-4; FAS ramps4096–8192; distortion .01 to1 by12288 |
| Frequency reweighting | Strength1 through75940; .3 from75941; one historical seed42 reset at that boundary, preserving model/Adam/scheduler/scaler |
| Optimizer | Initial LR .01 decaying to .0001 over200000; no late LR multiplier |
| Practical duration | Checkpoint91128; checkpoint106316 retained for its small additional measured improvement |

Here FR=1 applies the full implemented attenuation; .3 relaxes it using
`1 + strength * (weight - 1)`. FR=0 disables attenuation. Body80/FAS50 is a mixture
configuration, not a guarantee that 80% of rays hit the true body: expected original-mask coverage
was 71.43%, versus 24.72% with uniform image sampling. Expanded masks guide sampling only;
fixed evaluation support is separate. Pixel margins and step counts are dataset-specific.

### Results and lessons to test on other datasets

Person metrics use fixed GT support with neutral exterior and spatial AlexNet LPIPS. They are
means over validation012/095/150, which were used for selection, not an untouched test set.

| Configuration | Step | Person PSNR | Person SSIM | Person LPIPS down | Fixed TRAIN007 skin LPIPS down |
|---|---:|---:|---:|---:|---:|
| Previously accepted face fix |111216|29.5697|.905105|.08450255|.01616112|
| Practical .95 / zero-grid adaptation |91128|28.9682|.911996|.07282727|.02072238|
| Selected existing .95 / zero-grid checkpoint |106316|28.9386|.911730|.07260232|.01950717|
| Numerical person winner, unknown-max grid + late control |108316|29.0815|.908952|.07210023|.03786935|

- **Maximum frequency can mean unresolved fitting, not genuine fine detail.** The preprocessing
  fallback assigns the maximum level to patches that never pass its SSIM target. Preserve an
  unresolved flag separately and inspect its coverage before interpreting high labels as texture.
- **A stricter preprocessing threshold can hurt final quality.** With the same 2D fits, .95 versus
  .97 reduced maximum-level coverage on reliable person pixels from33.32% to10.03%; at91128,
  person LPIPS was .07282727 versus .07585370. Threshold changes affect FAS, ARM and feature
  weights together; this is not an isolated test of one mechanism or proof of a universal .95 optimum.
- **Conservative unknown=max initialization is not free.** It reached the lowest overall person
  LPIPS here but required more work and produced worse skin than zero initialization. Unknown-cell
  policy changes both marching and feature attenuation; zero is a measured candidate, not a
  universally correct representation of unknown space.
- **Track priority parts alongside the whole object.** The table shows why the numerical winner
  was not selected for visual quality. Even the selected model's skin metric is worse than the
  accepted face-fix baseline; the improvement is in whole-person LPIPS, not every region or PSNR.
- **Object-focused sampling has diminishing returns.** Four matched 2000-update branches from
  the selected106316 parent (body80/90 x LR1/2) produced .07293919/.07296228/.07273537/.07261136.
  None materially improved the parent. These late tests do not prove the same ranking from scratch.
- **Use dense same-checkpoint crop rendering to separate integration error from model error.**
  Skin LPIPS at106316 was .01950717 with native ARM, .01919099 at8192 samples and .01923333 at16384.
  Fine lace holes remained filled/merged. More accurate integration did not fix their geometry.
- **Budget by measured work and quality, not iteration count alone.** The selected trajectory took
  80.8/94.6 minutes of 3D training ancestry at91128/106316, including setup/save but excluding
  preparation, evaluation and review. The extra15188 updates gained only .00022495 LPIPS.
  Full paired 2D preparation separately cost about3h23m; do not hide that cost in speed claims.

This was one frame and one seed. A .001 LPIPS practical stopping threshold was a budget decision,
not a statistical significance bound. Large new cheek spots and obvious cord breaks were not
seen in reviewed views, but **fine lace topology, eye shape and pendant errors remain**. The result
is not an artifact-free geometry acceptance. Mild blur alone was not a rejection reason.

Source/configuration provenance and actual metrics are retained in the
[full report](luster_000540_body_optimization.md); the original
[face-fix report](luster_000540_face_fix_occupancy.md) preserves the earlier causal experiments.
