# Native Marshawn sequence runner

Campaign root on Silver: `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109`.
The persisted tuning deadline is 2026-11-09 17:08:12 PST; restarting services
must preserve `campaign_state.json`. The coordinator then writes the measured
`selected_recipe.{json,md}` and queues all361 independent frames908–1268.

- `campaign.py`: immutable cold specs, matched experiments, six-hour deadline,
  LPIPS selection and weighted source-aware final allocation.
- `agent.py`: one systemd executor per GPU; process identity survives controller
  restarts. Production retries back off; failed frames remain explicit.
- `worker.py` / `runtime.py`: sealed native training, own-state recovery and
  conservative150k optimizer-attempt ceiling including reserved crash replay.
  Frequency-map fitting is separate preprocessing, not radiance-field updates.
- `worker_background.py`: explicitly bound fresh ablation adding a weak opacity
  prior only on conservative TRAIN background. The baseline files stay sealed.
- `production.py`: native source preparation, current-frame SAM/qualification,
  all72-camera cold training, exact model export and optional archive transfer.
- `inference.py`: exported model evaluation without a training-image cache.
- `render_fullhd.py`: original calibrated FOV on1920×1080, raw RGB and straight
  alpha; crop origins are undone and ray parity checked. No GT cutout.
- `alpha_review.py`: auxiliary sampled background-opacity diagnostic within the
  saved native render ROIs; unqueried exterior is excluded.

Three tuning heldouts: CB/GC/NB. Final training includes all72 cameras; checks
on former heldouts are explicitly in-sample. Person IDs are camera-local SAM
instances, not temporal identities. Main metric is equal-person/equal-view
masked AlexLPIPS; PSNR/SSIM are supplementary. Native input crops are never
resized. FHD images are render/review outputs only.

Canonical physical V3 bounds/rotation are in `box.json`. Apply `world_to_box`
before testing its AABB; common historical model units use an isotropic
0.2217203609534452 scale unless the selected measured arm explicitly differs.
Parser orientation/centering/auto-scaling remain disabled.

Monitor `campaign_state.json`, each agent's `agent_state.json` and
`supervision.jsonl`, each run's `progress.json`/committed metrics/checkpoints,
and final `sequence_progress.json`. Keep supervision active at least hourly;
a detached service alone is not a completed task. Automatic checks run every5min.

A final frame has `frame_complete.json` and `model/inference.ckpt` with a SHA
manifest. Exported pipeline tensors round-trip exactly; optimizer/RNG states
are intentionally omitted. Local recovery checkpoints are released only after
a durable completion or verified remote archive receipt. An inference export
cannot resume optimization. Source RGB16 files remain untouched.

Example on a qualified receiving host (use its `settings.json` and Python):

```bash
python tools/inference.py --root CAMPAIGN_ROOT --spec FRAME/train/spec.json \
  --data FRAME/data_model --checkpoint FRAME/model/inference.ckpt \
  --output FRAME/reloaded_eval --caps 10
```

Choose the actual selected data variant (`data_model`, `data_resolution`, or
`data_exr`) and selected inference cap from the recipe. Code/runtime dependencies
must match the qualified environment; copying model tensors is not an environment
installation. The frozen base's1531 source files were verified against the
historical seal; new scientific extensions are explicitly listed in run hashes.
