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

## Runnable preparation, 2026-11-06 22:24 UTC

Separate worker `train_marshawn_semantic_late_distortion.py`, paired sequence and resource supervisors are implemented. CPU suite: **47 passed**, including every full-state section, nonempty real Adam roundtrip, changed/nonpersistent support and learned-core rejection, missing shared selector, initial-ray parity, resource limits and unreleased execution rejection. Actual GPU zero/render/Adam gates remain **unrun**.

Generation: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_late_distortion_candidate_v1`; its1499-file code snapshot contains all1495 parent files byte-identical and four new files. Parent files are immutable hardlinks, never edited. The idle supervisor is separately bound outside this code snapshot. Source manifests, six concrete zero/technical/diagnostic specs, unreleased templates, CPU tests and launch instructions are present. Ordinary producer worker/resume and semantic/occupancy implementations are unchanged.

Primary technical option uses fresh-idle GPU5 under `/tmp/lookcloser_tuning_gpu_1000_5.lock`, no dispatcher handoff. It exits if occupied; no implicit fallback. Alternative separately released policy uses GPU3 with the exact W32 producer914736/start3142778, boot3cd2faca-15ef-4b7e-9767-20e0b80af902 and shared technical overlap lock, checking producer identity/progress/OOM every5s. Both cap candidate allocation10GiB, measured memory12GiB and duration900s; only own child process group may be terminated. GPU5 was idle during preparation but subsequently acquired by a frequency fit; do not interrupt it. Parent chooses/releases the exact feasible policy.

The worker requires all seven restored-state hashes, exact data/runtime source receipts, semantic support hash and schema2 active/core/EMA consistency. Both zero gates additionally require all six original RGB/alpha PNG hashes and unchanged serialized state after rendering. Each technical32-attempt segment records actual Adam updates and finite combined gradients on update steps. Saves export schema2 actual-grid sidecars; no inferred empty-space certificate. The isolated4000-attempt diagnostic specs exist but have no release and no launched controller.

## First technical zero gate and isolated replay proposal

Root released GPU3 bounded technical supervisor987186. It stopped safely after56.44s on the first RGB SHA mismatch, **before any optimizer update**. All seven state sections, exact sampler/cache, semantic support and actual grid checks had passed, including unchanged serialized state after rendering. W32 producer progressed31309→32918; no OOM. Original logs, proof and outputs are preserved in candidate_v1.

CPU pixel comparison against source30376: DC/HC/KA differ in only5/3/2 RGB channel values (of5,829,120 each), all exactly one uint8 code. Alpha differs in828/728/569 values (of1,943,040), all one uint16 code. LPIPS delta5.40e-8, PSNR delta−1.13e-7dB, SSIM delta9.93e-9. These are bounded observations, **not proof of their cause**. SHA criterion has not been silently relaxed.

A separately prepared read-only scope, `C/semantic_zero_replay_v1/readonly_policy.json`, compares original frozen `base.build_config/evaluate` against repeat fork-factory construction on the same GPU3. Exactly two zero-update processes, unchanged full-state/RNG/source, same health guards, ≤600s and10/12GiB limits. It records six per-view code deltas, paired metrics and nonsaved runtime flags. Its code has no training-iteration/backward/update path (CPU AST test). Source inspection finds `current_train_step` only gates training warmup, while eval ignores that branch; actual runtime values will be recorded rather than assumed. Root review/release is required before launching this new scope. No numerical tolerance is approved by this proposal.

Read-only replay completed in108.19s, peak3892MiB; producer44534→47158, no OOM or optimizer updates. Original/fork full states, cache and runtime flags matched exactly. Original factory itself differs from historical source in8/4/5 RGB code values; original-vs-fork differs in5/4/6, always≤1code. Paired LPIPS delta2.48e-7, PSNR−5.83e-8dB, SSIM0. This supports numerical parity but does not identify the low-level cause. Artifacts: `C/semantic_zero_replay_v1/readonly_complete.json` and `comparison_to_anchor.json`. First launch failed before GPU because root receipts omitted two explicit false flags; corrected receipts and the prelaunch failure are preserved separately.

Root authorized preparing v2 with an explicit zero-render numeric gate: RGB8/alpha16 max≤1code; changed-value fraction RGB≤1e-5, alpha≤.001; PSNR/SSIM/LPIPS absolute delta≤1e-5 for aggregate, per-view, per-actor and six actor/view records. Original source PNG hashes, checkpoint, inputs, seven full-state hashes and semantic grid/support remain strict. Record actual deltas and `bitwise_render_claim=false`; no automatic tolerance widening. `C/semantic_late_distortion_candidate_v2/source_from_v1.diff` contains only this gate change; all1495 original source files are identical. CPU suite51 PASS, including source mutation, excessive code/fraction and per-view metric rejection. v2 technical policy SHA `4ee12177121a85bb07d48c01ac67918e0df9d4421d875e4568704cee83142d0a`;1506 staged artifact hashes verified. Actual v2 updates still require its own root release.

## v2 technical proof and released diagnostic queue

Root-released v2 technical supervisor1050165 completed **PASS** in292.94s, peak6828MiB. W32 producer58488→65488 remained healthy; no OOM. Both branches restored all seven state sections exactly, passed the explicit numeric zero-render gate, and completed Adam30362→30394→30426: **32 + 32 actual updates**, all combined gradients finite. Initial pre-update rays and both complete segment digests matched without altering sampling. Actual schema2 sidecars exist for saved technical checkpoints. Historical cache-order uncertainty remains explicit.

Root then released the two fresh30376 diagnostic specs. Dispatcher2090776 uses exclusive idle-GPU global locks; no technical30440 state is reused. The four diagnostic gates are30876/31376/32376/34376. Metrics/visual outcome and geometry approval remain pending.

Detached visual/archive watcher2091453, frozen at `C/semantic_late_visuals_v2/watcher.py` (SHA `87d9833127074dcb93e84a48f207f41c94b1bb696222f75695bc4fbed7a75324`), fetches all four paired gates. It validates checkpoint↔actual schema2 JSON/NPZ hashes, hardlinks each checkpoint/grid/config/spec/release/source inventory into `SC/independent_occupancy/semantic_late_pair_v2/{control001,treatment1}/step-*`, and records the saved effective distortion coefficient. It verifies PNG transfer hashes and creates all three GT/parent/control/treatment full/actor contact sheets plus unchanged V1 per-view/per-actor metrics. It cannot mark visual or geometry approval. Three CPU tests cover contact completeness, metric filtering, immutable artifact preservation and rejection of changed archives. Final34376 geometry replay is separately prepared by the occupancy agent; GPU admission remains separately reviewed.

## Matched semantic diagnostic result: lambda1 is not the RGB winner

Both fresh30376 branches completed4000 attempts and all four visual gates. Control made3998 successful Adam updates (30362→34360); treatment made3999 (→34361), with native AMP behavior unchanged. Initial ray digest and the complete4000-attempt ray-stream digest match exactly between arms; no common-random-number modification was introduced.

| Attempts | Control PSNR | Control SSIM | Control LPIPS | Lambda1 PSNR | Lambda1 SSIM | Lambda1 LPIPS |
|---:|---:|---:|---:|---:|---:|---:|
|500|31.51833|.84391|.027356|31.62313|.84864|.028784|
|1000|31.53298|.84450|.027050|31.71316|.84855|.028132|
|2000|31.52584|.84441|.027203|31.65070|.84861|.028741|
|4000|31.51389|.84400|.026732|31.70173|.84940|.029091|

Treatment LPIPS is worse at every gate; final regression is8.82%. At4000, DC/HC/KA LPIPS change .032397→.037772, .024985→.025357, .022813→.024143. Black/grey change .028534→.032345 (+13.36%) and .024930→.025837 (+3.64%): both fail the3% guard **against the matched control**. This is separate from comparison against the earlier source30376 checkpoint.

Manual review of all six actor contacts at500 and4000 found sharper silhouettes and less diffuse halo, but persistent floating floor/background clumps. Black garments can become smoother and lose fine contrast; local grey-garment changes do not establish a general detail gain. PSNR/SSIM improvement does not override the user's actor-LPIPS objective. No promotion or ideal-grid claim.

Exact metrics, raw PNGs, nine GT/parent/control/treatment contacts and manual observations: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_late_visuals_v2/step-000034376/`. All four checkpoint gates and schema2 actual occupancy sidecars are preserved under `SC/independent_occupancy/semantic_late_pair_v2/`. The independent103 TRAIN +12 DC ray audit is a separate geometry gate;4000 attempts remain a diagnostic endpoint, not optimization completion. A smaller late coefficient is a new separately reviewed factor, not evidence that lambda1 won.
