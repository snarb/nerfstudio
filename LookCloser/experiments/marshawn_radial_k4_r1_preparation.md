# Matched radial-K4 native cold R1: DEV3 preparation

Separate stage: `/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_R1_DEV3_STAGED`. Matched to the finalized R0 configuration, with only training grid radius0→1 and independent output/export paths. Model/field,63/3 source data, seed42, optimizer/scheduler,1M target samples, native random-background RGB, near.052, true-LOS, observed SfM surface band, linear distortion.01 and distributional MVS.01 are unchanged. No assumed floor-plane prior or frequency maps. No cold checkpoint is loaded.

Actual DEV3 Python3.8/Torch2.4.1+cu121/L40S zero-optimizer proof passed. Same-seed NN/schema/state keys equal the native baseline on that host. Actual main64 rays/81864 samples adapt to819 rays/1021188 samples. MVS extra128 rays have86369 samples, minimum opacity.81363. No optimizer objects or updates were created. Original NN/grid and process RNG restored.

The actual native update→guard callbacks ran at0/16. Controlled core tests use the SAME callback: one occupied cell gives exactly27 cells; repeating the same step does not expand again; a fresh distant single-cell core replaces the old guard rather than accumulating it. Actual stored active equals one halo of cached GPU core. Constructor does not redilate loaded checkpoints. No CPU mean reconstructs an authoritative core.

At native initial AMP scale65536 the color-head RGB gradient overflows on both tested packets;32768 gives finite gradients for every loss and combined objective. This is recorded diagnostic backoff, not hidden success at65536. Production retains native65536, so skipped AMP attempts must be counted under the22000-attempt ceiling. Future controller pauses after checkpoint2000 before2001 and retains default stopping/terminal21999 ranking.

Actual DEV3 controller dry-run passed; UID1000 write access was granted only on the new stage/output tree. Configuration SHA60ba68b9b11b14b81113882f5c47d6adaabf7c51ed5486ec4d34f0d635dea9ba. R0 is a different hardware/runtime (RTX PRO6000, Torch2.7.1+cu128); matching source/config/seed does not guarantee bit-identical cross-hardware FP16 behavior.

Evidence: `actual_cold_zeroopt_DEV3/probe.json`, `configuration_review.json`, `controller_dryrun_DEV3.log`, `freeze_receipt.json`. All underlying NN helpers are byte-identical to R0; only the runner radius guard and read-only probe tests differ. **No R1 training launched by geometry; peer/root gates remain separate.**

## Actual 2k read-only gate

ROOT subsequently launched R1 on DEV3 (controller79757/worker79774). Exact2001 attempts produced2000 optimizer updates/one AMP skip, finite weights; the same worker paused before2001. Gate checkpoint SHA c05dc5f4c125389e92ad15271757c0b9d5ed5d0abb775edd9802a2775e92d202. Geometry sent no CONT.

One O_EXCL-reserved read-only reader80169 ran on DEV3 after verifying the actual worker T state and checkpoint hash, then finished with a complete report. It loaded the stored radius1 mask directly and did not dilate it again; model/active-grid hashes stayed unchanged. Combined reader+paused-worker GPU use was about10.4GB, no OOM evidence.

| Full-HD reader metric domain | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Own radial GT | 28.67461 | .777559 | .364275 |
| Inverse-mapped old scalar GT | 29.02609 | .787707 | .357415 |
| Actual MVS16k reference, old GT | 29.19285 | .819852 | .268069 |

Reader SSIM explicitly uses data_range1; native CSV SSIM omits that argument and reports.770521 for R1. These implementations must not be silently mixed. PSNR/LPIPS agree. R1 cold2k versus old16k is an actual-quality reference, not equal-budget or radial-only causality.

Actually viewed DC/HC/KA own-GT contacts plus KA left-wide, HC-floor and KA-floor contacts. Bodies are coherent/opaque and there is no gross DC brown camera veil. KA left floor is less wavy than old16k, but broad dark bias and coarse/soft/mottled mats remain. **Continuation gate only; no clean-floor or whole-grid acceptance.** Root retains the continuation decision.

Artifacts: `gate2000_HD_common/report.json`, `gate2000_HD_common/contacts`, `gate2000_manual_review_geometry.json`, `gate2000_reader_launch.json`, `run/geometry_supervision.jsonl`.

## Actual external 4k gate

ROOT authorized continuing the SAME worker only to4k. External watcher80506 verified worker birth, fully exported radius1 grid archive/JSON and checkpoint hash, then stopped79774 during native evaluation before4001. No config or NN-process restart. Exact4001 attempts/4000 successful updates/one AMP skip; scaler131072, finite weights. Checkpoint SHA3a94bff43eb8be042eff542a8e2f1858a784d1010e58e4dea5fd994006850018.

One O_EXCL read-only reader80732 finished using the stored active grid without additional dilation, with unchanged model/grid hashes. Full-HD own-radial metrics29.610088/.796615/.312740; common-old29.953197/.806400/.306166. All improve versus corresponding2k metrics. Actual same-GT 2k→4k contacts show sharper/coherent opaque bodies and no gross DC veil, but KA left broad dark bias and floor/mat mottling remain. Continuation gate only, no clean-floor/ideal-grid acceptance. Geometry sent no CONT beyond4k.

Evidence: `run/gate_4000_external.json`, `gate4000_reader_launch.json`, `gate4000_HD_common/report.json`, `gate4000_manual_review_geometry.json`, `gate4000_progression/{all3.jpg,KA_black_wide.png,HC_floor.png,KA_floor.png}`.

## Actual external 8k gate

ROOT actually reviewed all3 R1 common-old4k contacts and authorized the SAME79774 only to8k or earlier default plateau. Continuation recorded in root_authorized_continue_4k_to8k.json; budget4001→8001 attempts, no new NN/config. Native6k evaluation improved30.1394/.798671(auto-SSIM)/.291157, so no plateau stop. Watcher81075 verified PID birth and fully exported radius1 grid/checkpoint, then paused before8001. Exact8001 attempts/8000 updates/one AMP skip, finite8kweights, scaler524288. Checkpoint SHA128cd05b2ef952fdd867b40e586b2f8d52e4f03917264c9d95ff070b2560bd5f.

One reserved read-only reader81572 finished; actual field/stored active mask remained unchanged with no extra halo. Own-radial8k metrics30.436523/.810552/.279717; common-old30.791508/.820091/.273120. All improve relative to4k. Actually viewed all3 own contacts, KA left-wide/HC-floor/KA-floor and GT|4k|8k progression. Opaque/coherent people and sharper background/details; no gross DC veil, but KA left broad gray/brown bias and mat/floor mottling remain. No spot-free or ideal-grid acceptance. Geometry sent no CONT beyond8k; worker remains T awaiting root.

Evidence: run/gate_8000_external.json, gate8000_reader_launch.json, gate8000_HD_common/report.json, gate8000_progression/all3.jpg, gate8000_manual_review_geometry.json, root_continuation_4k_8k_budget_record.json. Controller alive, reader/watcher terminal, GPU paused-worker11.36GB, no OOM evidence.

## Exact checkpoint12k review and watcher failure

ROOT authorized the SAME worker only from8k to12k or earlier default plateau. Native10k improved30.6134/.806875(autoSSIM)/.271702; no early stop. **The external watcher had a real filename bug:** naive string substitution8000→12000 generated `step-0000012000_grid.json` instead of canonical nine-digit `step-000012000_grid.json`. It missed the intended gate. ROOT detected this; geometry terminated only the faulty watcher81968 and ran a corrected external checker, which verified canonical grid/checkpoint SHA and stopped the same79774. NN/config were not patched or restarted.

After actual T pause, CSV/controller logs proved the live worker had completed12070, so it was NOT merely paused during12k evaluation as initially inferred from sparse AMP logging. Synchronous per-loop CSV flush/every10-step rows bound attempted iterations to12071–12081 (12080 could be in progress before flush); conservatively charge12081,80 beyond the authorized12001 gate. The absolute22000 ceiling was not exceeded. Saved checkpoint12000 remains valid, SHA df719c5d1d969a07c1de9ade6439e76609c2a9e59b8d3a2c52a8bcfe3839cb41, with12001 attempts/11999 updates/two skips. These checkpoint counters must not be mislabeled as the unknown ahead-live counters. Original wrongly labeled watcher receipt was preserved separately; corrected metadata distinguishes checkpoint and live step.

One O_EXCL read-only reader82663 rendered **exact saved12000**, not the ahead-live field, and finished with unchanged field/stored grid hashes and no additional halo. Own metrics30.710227/.814936/.266911; common-old31.069455/.824385/.260447, all improved versus8k. All3 own-GT contacts, KA-left-wide, HC/KA-floor and GT|8k|12k progression were actually viewed. Coherent opaque people/no gross DC veil; pale/brown floor bias and mat mottling remain, so no clean/ideal-grid acceptance. Same live worker remains T; geometry sent no CONT after corrected pause.

Evidence: `external_gate12000_filename_bug_receipt.json`, `external_watcher_filename_validation.json`, `run/gate_12000_external_RAW_WRONGLABEL.json`, `run/gate_12000_external.json`, `gate12000_HD_common/report.json`, `gate12000_manual_review_geometry.json`, `gate12000_progression/all3.jpg`. Future names must be generated from integer `step-%09d`, never digit-substring replacement.

## Robust immutable16k gate

ROOT authorized SAME79774 from the already-ahead live state only to16k or earlier default plateau. Before CONT, current12k full container was copied read-only into immutable_gate_12000; its SHA df719…9cb41 is byte-identical to the previous reader container, proving model/state equivalence. No rewind, so the prior gate-overrun attempts are included within the same global trajectory, not added twice.

A fresh frozen integer-step monitor (`marshawn_external_checkpoint_gate.py --step16000 --metrics ...`) used canonical step-000016000, PID birth, checkpoint/grid hashes and an O_EXCL reservation.14k improved30.7800/.809800(autoSSIM)/.263747, so no default stop. Monitor83192 successfully paused the same worker and sealed immutable_gate_16000/step-000016000.ckpt SHA931269a653845c1ba4c62b6f41c596a71a1f278458819aaba5e306d076bcbfdb. CSV15990 plus completed AMP16000 and synchronous per-loop flush establish next16001 had not begun:16001 attempts,15997 updates,four AMP skips. A matching grid copy/receipt was also sealed. No NN/config hotpatch.

Reader83603 rendered that IMMUTABLE snapshot once and finished without altering field/stored active mask. Own metrics30.819967/.816823/.261827; common-old31.182826/.826223/.255375. Actual all3 own contacts, KA-left-wide, HC/KA-floor and GT|12k|16k progression were viewed. Modest detail improvement/coherent opaque people/no gross camera veil, but pale/brown floor bias/mat mottling persist. No clean-floor/ideal-grid acceptance. Worker remains T; no CONT beyond16k.

Actual GPU core1100217 cells and one-halo active1902112 cells (~90.7%volume) describe sampling admission, not physical occupancy truth. Evidence: immutable_gate_16000/{gate.json,sealed_grid_binding.json}, gate16000_HD_common/report.json, gate16000_manual_review_geometry.json, gate16000_progression/all3.jpg, root_authorized_continue_overrun12k_to16k.json. Reader/monitor terminal, no OOM, paused-worker GPU12.88GB.

## Terminal20k selection

The canonical20k monitor paused the same worker79774 and sealed checkpoint7cd58088…87e88f before CONT. ROOT actually reviewed all3 contacts and authorized only finishing the existing evaluation/default stop. Default plateau then terminated the worker and controller normally; no restart or disabled stopping. Last actual CSV20200 gives conservative total attempt upper20211 (<22000). The selected checkpoint itself contains20001 attempts/19996 Adam updates/five AMP skips; sparse logging does not establish exact final live update count.

Formal selection chose20k within0.07dB of18k because LPIPS was lower. Native formal metrics: PSNR30.808121, SSIM(auto-range).811104, LPIPS.260500. The immutable selected container SHA fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2 differs from the pre-eval container only outside pipeline tensors: every pipeline tensor is exactly equal. Thus existing20k HD renders remain valid. Explicit-range1 reader metrics: own30.808124/.817456/.260500; common-old31.170195/.826837/.254102. Keep native auto-range SSIM and explicit-range SSIM distinguished.

Actually inspected all3 GT|16k|20k panels: actors remain coherent/opaque with no gross brown camera veil. Floor pale/brown bias and mat mottling remain; change after16k is small. This is not cleanliness or whole-volume physical-grid acceptance. All owned R1 worker/controller/evaluator processes terminal. Evidence: finished_receipt.json, immutable_selected_20000/gate.json, gate20000_manual_review_geometry.json, gate20000_progression/all3.jpg.
