# Semantic late distortion: evaluated-intermediate fork proposal

## What is proposed

Compare abrupt distortion coefficient **0.01 control vs 1 treatment**, both forked from the protected semantic R0 **30376** checkpoint. This is an `EVALUATED_INTERMEDIATE` anchor while its producer continues, not a completed-parent selection. The earlier full-objective pair establishes a useful mechanism but does not establish causality for semantic training.

Keep actor/body80/W64, radius0, exact fixed W64pad1 semantic admission, exponential(h+1) density, corrected SH directions, legacy frequency projection, source RGB/masks/crops/AABB, seed42, FR and all optimizer/scheduler/AMP/sampling settings unchanged. The only model-config difference is `pipeline.model.distortion_loss_mult: .01 → 1`.

Anchor artifacts and CPU audit: [anchor_audit.json](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_late_distortion_proposal/anchor_audit.json). Remote checkpoint is `/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/independent_occupancy/semantic_actor_progress/step-000030376/checkpoint.ckpt`, SHA256 `1b08dc38c80ab089aa3a9923b802bb14104b668cab3d31b4d8cf2ca67aaf1013`. The audit binds checkpoint, config, evaluated metrics, schema2 JSON/NPZ, producer spec/source manifest/release, plus canonical hashes of all seven serialized state sections.

## CPU evidence

Anchor V1 metrics: PSNR **31.46039**, SSIM **0.843210**, LPIPS **0.0274122**. Actual Adam step30362, LR0.00497027021109, scaler4096. Actual grid capture is step30368: core, active, EMA and persistent `semantic_learned_core` agree with schema2 sidecars. Aliased pipeline occupancy tensors agree byte-for-byte.

The fixed `semantic_admission` buffer is nonpersistent. It must be reconstructed by the **same SemanticAdmissionModel** from the exact sealed geometry manifest and verified against mapped128³ SHA `d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f`; loading a generic LookCloser model is invalid. `semantic_learned_core` and actual captured grid must be preserved. Keep schema2 exporter and post-load admission checks unchanged.

The new standalone configuration factory is isolated from the original worker. Its CPU tests reject full-objective, radius1, W32, altered sampling/projection/density, incomplete state, fake completed-parent status, stale capture and identity admission. Ordinary resume validation remains unchanged. No model or optimizer has been constructed for this proposal, and no GPU update or diagnostic launch is authorized by these artifacts.

## Required technical gates before the pair

1. Reverify every frozen source/input/anchor hash at execution, never substitute the moving producer checkpoint. Construct the semantic subclass with original setup and resume optimizer/scheduler/scaler/pipeline/sampling state; restore RNG after setup. Zero-update snapshots of all serialized sections must match anchor hashes. Recreate actual capture and reload exact nonpersistent semantic support. Validate native step30376 RGB against protected three-view renders and verify schema2 core/active/EMA/admission equality.
2. Reconstruct the seed42 private cache identically in both branches, and bind camera IDs, pixel hashes and first pre-update ray digest. **Historical cache ordering was not serialized**, so this demonstrates mutual paired reconstruction, not independently proven historical bitwise continuation. Do not change sampling to force equality. Record later digests; legitimate adaptive-state RNG divergence is evidence to explain, not a reason to alter the algorithm.
3. Each branch runs 32 attempts from30376 to30408, saves complete state, then resumes for another32 to30440. Require nonempty Adam state, actual update counts, finite combined gradients after native AMP backoff, exact resume equality and no field/grid reset. If backoff prevents actual updates, report failure/inconclusive rather than counting attempts as updates. These are technical outputs only; discard their ancestry for the diagnostic pair.
4. A reviewed bounded overlap supervisor may be proposed separately: one child/session, explicit producer PID/start/boot, shared technical-overlap lock, 10GiB allocator/12GiB measured-process ceiling, 900s deadline, producer progress/OOM checks every5s, kill only owned process group on failure. Alternatively use an exclusively idle ordinary GPU lock. No resource slot is reserved or launched by this proposal.

## Paired diagnostic after technical review

Start **fresh from30376**, not30440. Save/evaluate both branches at +500/+1000/+2000/+4000 attempts, global steps30876/31376/32376/34376. All gates precede the unchanged FR switch75941, so FR stays1. LR follows the exact original200k scheduler and actual Adam count; no restart, override or decay intervention. Count attempts and successful Adam updates separately.

Use unchanged V1 three-view/two-actor metrics, parent/control/treatment/GT contact sheets for garments, hair, hands and feet, plus actual-grid schema2 audit and matched103 TRAIN cloth/12DC error rays at the diagnostic endpoint. Preserve all gates and report KA/per-actor tradeoffs; the full-objective treatment improved DC/HC but worsened KA. Apply the existing .001 LPIPS/PSNR tie selection and per-actor guard only as explicit provisional selection, never automatic geometry approval.

Four thousand attempts are a **diagnostic gate**, not optimization completion. A promising branch can continue under a separately reviewed matched plateau protocol. This proposal does not authorize its GPU technical gates, diagnostic training, promotion, or replacement of current semantic R0/R1/W32 runs. A runnable isolated fork entrypoint and hash-bound technical policy remain the next implementation/review step; the factory does not bypass normal resume checks.
