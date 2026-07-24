# Fine-tuning task v2: 007740 → 007747

## Goal

Fine-tune frame `007747` directly from the canonical `007740` leader and reach
all leader thresholds:

- PSNR `>= 29.840143`;
- SSIM `>= 0.669203`;
- LPIPS `<= 0.219455`;
- the eval0 hands/fingers/chain crop is visually no worse than the leader.

Measure `time_to_leader`: wall time from immediately before target trainer
startup through the first complete full evaluation that passes all three
numeric thresholds and the visual crop gate. Minimize this time without
weakening any gate. After the first pass, continue to the confirmed plateau and
select the best checkpoint.

## Frozen inputs

- Work from clean `main` in `/home/brans/repos/nerfstudio`.
- Leader checkpoint:
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt`
  (SHA-256 `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930`).
- Leader config:
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/config.yml`.
- Target dataset:
  `/home/brans/temporal_perframe_stride7_45f/007747`.
- Use exactly these standard maps:
  `/home/brans/temporal_perframe_stride7_45f/007747/lookcloser_frequencies`.
- Require `/home/brans/temporal_perframe_stride7_45f/007747/dataset_revision_422.json`
  and verify its JPEG/map hashes before training. If it is absent or fails,
  stop; never fall back to another map directory.
- Never use `lookcloser_frequencies_chroma422`, any `*_probe` directory, or
  anything under `/home/brans/007747_4_4_4`.

## Exact initial treatment

Load the original hash23 leader checkpoint directly. Do **not** create or load
a hash24 checkpoint.

Use `checkpoint_load_mode=model_parameters_only`: inherit only field/model
parameters. Start target local step 0 with fresh Adam, scheduler, scaler, RNG,
occupancy grid, frequency grid, FAS counter and point telemetry. Full resume is
allowed only later within the same 007747 run.

Freeze the initial recipe to the leader config:

- seed 42, batch 4096, mixed precision, stable occupancy reduction;
- hash log2 size **23**, 16 levels × 2 features, min/max resolution
  `16/8192`, `max_res_base=2048`;
- standard maps above, FAS enabled with strength `1.0`;
- feature reweighting `0.3` from the first target update; never switch to
  FR1.0;
- Adam `lr=0.01`, `eps=1e-15`, no weight decay;
- exponential LR `0.01 → 0.0001` over 200000 local updates;
- fixed traversal and fresh occupancy warmup for local updates `0..4095`;
- every other architecture, loss, marcher, grid, cadence and runtime field
  must equal the leader config. Explicitly keep fused Adam, TCNN JIT, cached
  rays, CPU FAS prefetch and independent RNG streams disabled.

The only recipe changes relative to the leader are the target dataset/maps,
model-only cross-frame load with fresh target state, and target-local
training/evaluation horizon. Model weights then train normally; no layer is
frozen.

The only permitted later hyperparameter change is one controlled same-frame
continuation from the common step60752 checkpoint with feature reweighting
`0.3 → 0.2`. Keep every other field and all target state unchanged. Do not
sweep hash size, FAS, map variants, batch, warmup, losses or low learning
rates. Compare FR0.2 against the continued FR0.3 control at identical local
updates before selecting it.

## Evaluation and stopping

- Before training, full-evaluate the model-only transplant on 007747 as the
  local-step-0 baseline.
- Save and full-evaluate every 15188 local updates. Initial horizon: 60752;
  then continue one interval at a time while metrics or the crop improve.
- Record full-eval wall time separately from training time; `time_to_leader`
  includes both and is the first passing boundary, not an interpolated guess.
- Plateau requires two consecutive intervals where all hold simultaneously:
  PSNR growth `<0.03 dB`, SSIM growth `<0.001`, LPIPS improvement `<0.003`,
  and no visible crop improvement.
- Select maximum full-eval PSNR; among checkpoints within inclusive `0.07 dB`
  of that maximum select minimum LPIPS. SSIM is reported but cannot rescue bad
  PSNR/LPIPS.
- Do not report eval loss.

For every boundary, crop `eval_img_0000.png` at
`(left=700, top=100, right=1120, bottom=480)` and save a native-resolution
comparison against:

- leader renders:
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/renders_candidate_step-000091128`;
- accepted 007747 scratch reference:
  `/home/brans/lookcloser_007747_from_scratch_runs/evaluations/007747_fromscratch_E8_fw02/step-000197444`.

The fingers must remain separated and sharp, and the chain continuous,
gap-free and not blurred. Numeric success without this visual gate is not a
success.

## Implementation guardrails and deliverables

Do not run `scripts/run_lookcloser_temporal_finetune.py` unmodified: its
current default low-LR/FR1.0 screen is not this recipe. Add a dedicated
single-frame v2 mode or runner with fail-closed assertions for every frozen
coordinate and a config diff against the leader whitelist before launching.

Write outputs to a new v2 directory; never overwrite leader, scratch,
historical transfer, dataset archive or previous run artifacts. Preserve every
15188 checkpoint, metrics, three-view renders, crop comparisons, exact config,
source/data hashes and wall timings. Report the first leader-pass checkpoint
and `time_to_leader`, the plateau-selected checkpoint, and any separate visual
selection.
