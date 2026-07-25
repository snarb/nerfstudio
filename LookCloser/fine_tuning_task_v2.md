# Fine-tuning task v2: build snapshots for the complete temporal sequence

Process all 45 frame datasets `007740..008048` (stride 7) under
`/home/brans/temporal_perframe_stride7_45f`, in order. Create the real snapshot
tree at `/mnt/data/lookcloser_temporal_snapshots_v2/<frame>` and expose it as
`/home/brans/temporal_perframe_stride7_45f/snapshots` via a symlink. Never put
the full checkpoint set on the nearly-full root filesystem. Work from clean
`main` in `/home/brans/repos/nerfstudio` and keep a source fingerprint.

Each accepted snapshot must be self-contained for later per-frame video
rendering:

```text
snapshots/<frame>/
  config.yml
  nerfstudio_models/step-XXXXXXXXX.ckpt
  selection.json
  renders/eval_img_000{0,1,2}.png
  provenance.json
```

`config.yml` must reference that frame's dataset and its snapshot-local
checkpoint, not the temporary training run. `selection.json` records metrics,
selected step, parent frame/checkpoint, visual verdict and timings;
`provenance.json` records checkpoint/config/source/JPEG/map hashes. Prove every
promoted snapshot with a fresh render loaded from the snapshot config.

Bootstrap by copying, never moving or modifying, these already accepted
results:

- `007740`: checkpoint
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt`
  (SHA-256 `3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930`)
  with
  `/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/20260715_005006/config.yml`
  and `renders_candidate_step-000091128`;
- `007747`: checkpoint
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3/authoritative/authoritative-R-L150-H300/lookcloser/run/nerfstudio_models/step-000151880.ckpt`
  (SHA-256 `000fbc9144505fe4041d61ba71f0f9f804c78de19517b70cd0584d519ae6a358`),
  config
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3/authoritative/authoritative-R-L150-H300/lookcloser/run/config.yml`,
  and fresh-confirmation renders from
  `/mnt/data/lookcloser_007747_finetune_v2_runs/hash23_extended_scheduler_seed42_v3/final_confirmation/step-000151880/renders`.

For every later frame, load only field/model parameters from the immediately
previous accepted snapshot with
`checkpoint_load_mode=model_parameters_only`; start local step 0 with fresh
Adam, scheduler, scaler, RNG, occupancy/frequency grids, FAS state and
telemetry. Before training, freeze and verify the exact JPEG and standard
`lookcloser_frequencies` file set/hashes for that frame. Use the fixed recipe in
`LookCloser/scripts/run_lookcloser_temporal_finetune.py`: hash23, batch4096,
mixed precision, standard `lookcloser_frequencies`, FAS1.0, FR0.3,
warmup4096, Adam `0.015` (`eps=1e-15`) with exponential decay to `0.0001`
over300000 updates, and full eval/save every15188 steps. Generalize that runner
to a frame/parent pair without changing the recipe; train through step151880,
then resume one interval at a time only if needed to confirm the same
two-interval metric-and-visual plateau. Select maximum PSNR, then minimum
LPIPS within inclusive0.07 dB; require PSNR≥29.840143, SSIM≥0.669203,
LPIPS≤0.219455 and a clean propagated temporal ROI review before promoting or
forwarding a frame.

Run training artifacts on `/mnt/data`, preserve every evaluated boundary and
an auditable campaign manifest, and stop on the first failed frame—never skip
it or forward an unaccepted checkpoint.

## Easy mistakes to avoid

Never full-resume across frames, reuse the late checkpoint LR, switch to
hash24/FR1.0/alternate maps, or fall back when images/maps are missing or their
hashes change. Copied Nerfstudio configs often retain old `load_dir` paths, so
rewrite and test them; do not delete source runs or start parallel jobs unless
VRAM and disk guards explicitly pass.
