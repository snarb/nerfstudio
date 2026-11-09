# Marshawn1660: final experiment record at user stop

## What was tested

**STOPPED by explicit user instruction on2026-11-09. All11 campaign GPU slots across Olive, Silver8, DEV3 and Clever were verified free of our training/render workers; admissions and restarts are blocked.** No stage completion, GPU evaluation, new experiment or final66 training was awaited after the stop. [Independent four-host stop proof](/fsx/tmp/marshawn_native63_24h_20261108/ops/USER_STOP_20261109_1252/ROOT_ALL_HOSTS_STOP_VERIFIED.json). DEV3's final direct-stop readback is newer than the first aggregate helper's permission error; raw exit143 means requested stop, not a negative scientific result.

The original budget was Nov8 16:13:49→Nov9 16:13:49 UTC. The user canceled the remainder; final four-host exit verification was12:59:21 UTC, approximately20h46m after campaign authorization. **This is not a completed24-hour search or a convergence certificate.**

Frame1660, native unresized actor crops from8192×4320 source TIFFs. Photometry, inherited calibration and floor-aligned geometry were fixed. Each independent field was cold; only its own exact field/Adam/grid/RNG state could resume.63 TRAIN / DC-HC-KA validation, equal weighting of the two actors in each of the three cameras. These views were repeatedly used for tuning and are not an untouched test. Calibration/AABB were inherited preparation, not independently refitted blind validation. Earlier all66 comparisons are labeled in-sample.

The [chronological campaign report](marshawn_native63_24h.md) preserves source changes, actual controls, negative results, technical incidents and visual gates. This file is the authoritative final selection/status summary. [Frozen exact recipe](marshawn_native63_frozen_recipe.md), [cross-scene lessons](lookcloser_native_frequency_sampling_lessons.md), [old fight-scene recipe comparison](marshawn_vs_static_fight_recipe.md).

## Results

### Selected checkpoint and rendering policy

| Saved model / renderer | Updates | PSNR | SSIM | Actor LPIPS↓ | Interpretation |
|---|---:|---:|---:|---:|---|
| **P25/SAM15/core.01, cap10 inference** |140000|27.100272|.579104|**.095144829**|Selected detail/metric compromise; trained uncapped |
| Same weights, default inference |140000|27.228975|.590570|.098840536|Reference for inference contrasts |
| Same weights, cap9 inference |140000|26.964343|.567310|**.094086626**|Lowest aggregate score; HC faces+6.16/+9.05%, boot+7.08% versus10; not selected as a detail upgrade |
| Same weights, cap8 inference |140000|26.688938|.545071|.09465819|HC face softness worse; not selected |
| Hash26/SAM15/core.01, default |100000|27.303179|.591387|.098730880|Promising alternative; mixed faces and weaker cloth opacity; later continuation interrupted |
| Hash24/SAM15/core.01, default |140000|27.367060|.599782|.10361110|Earlier matched-capacity reference |

The selected field took297.730558 summed Silver model-loop minutes, including in-loop evaluation but excluding preparation/setup/queue/2D fitting. Hash25 versus hash24 at140k/default lowered LPIPS4.604% for26.552% more loop time. This is a single observed contrast, not a universal capacity rule. Hash26_100 was near hash25_140 in default aggregate score at284.358835min, but its KA-black-face score worsened5.58% against hash25_100 and cloth low-alpha tail was1.225% versus.568%; it is not an unqualified winner.

Cap10 versus default on the SAME140k weights lowered LPIPS3.739%, render points69.588%, ordered render time17.723%. PSNR/SSIM fell, and HC-grey-face/boot LPIPS worsened about2%. Three fixed native interpolated cameras showed no gross new structural hole but no convincing broad novel-detail upgrade. No temporal or independently trained video-frame stability certificate was produced. Cap9's better aggregate comes with materially softer HC face detail; the numeric minimum is preserved rather than hidden. [Cap10 native review](/fsx/tmp/marshawn_native63_24h_20261108/review/frozen_H25_140_cap10_CPU/REPORT.md), [cap8/9 review](/fsx/tmp/marshawn_native63_24h_20261108/review/frozen_H25_140_caps89_CPU/REPORT.md), [novel3 review](/fsx/tmp/marshawn_native63_24h_20261108/review/novel3_H25_140_default_cap10_CPU/REPORT.md), [hash26_100 review](/fsx/tmp/marshawn_native63_24h_20261108/review/P15_hash26_100_CPU/REPORT.md).

### Frequency-map comparisons

All rows below share hash25/SAM15/core.01, strict63 validation and cold initialization. The scalar50k gates have complete saved native images/journals, but their joint native-detail contact review was interrupted at user stop; they do not gain a visual-pass label by inference.

| Map / control |32k LPIPS↓|50k PSNR|50k SSIM|50k LPIPS↓|Status |
|---|---:|---:|---:|---:|---|
| Constant uniform16, SAM15 |.126402422|27.529367|.609089|**.114365472**|Matched reference |
| Learned shared threshold.90 |.127776700|27.453834|.609125|.116187641|1.59% worse than constant at50; stopped before80 |
| Learned shared threshold.95 |.132711335|—|—|—|Held32; cloth/detail regression |
| Learned shared threshold.98 |.128082650|27.495117|.609234|.117045941|2.34% worse at50; stopped before80 |
| Constant uniform16, SAM30 |.130522064|27.555365|.610927|.118687431|Aggregate3.78% worse at50 despite four improved face scores; stopped before80 |
| Constant uniform16, cap12→full50k |.126371858|27.481610|.608436|.113937621|50k marker is step49999, BEFORE uncap at50000; no mature post-switch result |

**Evidence:** constant16 wins both matched32k and saved50k scalar comparisons. **Unknown:** terminal80/140 ordering for the new.90/.98 trials; no extrapolation from their interrupted prefixes. The constant input map requires no learned2D fit; runtime3D frequency/occupancy still evolve from scratch. Three learned thresholds were derived from each of63 genuine16k single-image2D fits:1,008,000 auxiliary updates,7700.086s summed fit-plus-proof clocks, not exact GPU busy time or an unavoidable production cost. [Completed-map audit](/fsx/tmp/marshawn_native63_24h_20261108/research/independent_shared63_frequency_actual_CPU_v1/REPORT.md), [32k native review](/fsx/tmp/marshawn_native63_24h_20261108/review/sharedThresholds_SAM30_joint032_CPU/REPORT.md), [saved50k source bindings](/fsx/tmp/marshawn_native63_24h_20261108/review/STOP_intermediate50_joint_CPU/native_bindings).

### Adaptive rendering, sampling, priors and training length

| Actual contrast | Measured result | Conclusion / limit |
|---|---|---|
| Hash25 coarse step×½, cold32 |LPIPS.125247708 vs.126402422; all four faces improve1–6.6%; cloth low-alpha431 vs325pixels |Modest promising detail; more point work; own80 interrupted; mixed-host time not causal |
| Minimum step1e-4→5e-5, frozen100k |46.79% more render points, LPIPS slightly worse |More integration work alone did not help this field |
| Max1024→2048 samples/ray, cold32 |Only.164% global LPIPS improvement; grey cloth+5.82%; no speed benefit |No adoption; not every ray doubles its work |
| Midpoint→max3 interval estimate |Frozen100k LPIPS+.430%; training controls also recorded |No demonstrated broad gain |
| Normalized-Jacobian projection on P15/hash24, cold32 |LPIPS+27.69%, points−76.51%, all eight detail regions worse |Held; does not prove all normalized/retuned frequency recipes fail |
| Cadence1024→512/256, older parent32 |Global LPIPS−3.35/−3.23%; grey face+5.03/+5.99%; points+29.4/+55.4% |Real mixed controls, not a current-hash25 speed/detail win |
| Feature reweighting early switch20k, strengths0/.3/.6 |Small mixed32k effects; FR0 at50k1.92% worse than old reference |Switch also moves the historical seed reset; do not attribute solely to strength |
| Hash24 cap12→full50k,120k |LPIPS.104646955 vs.104222131;184.263vs198.758Silver min |7.29% shorter loop for.408% higher LPIPS; own140 final status in stop ledger |
| NN/hash24 FAS temperature.5,100k |LPIPS.102449823 vs.108228890; all four faces improve, boot+5.84% |Useful parent-specific tradeoff; later stage interrupted |
| P25 own120→140 |45.528 extra min, LPIPS.099572367→.098840536 |.735% gain under DEFAULT renderer; not proof of saturation or optimal early stopping |
| Density/core prior.01 and precise-head proposals |Included in selected model; matched early controls and all local tradeoffs preserved in chronological report |Not evidence that each component independently caused the full final gain or is universally optimal |

The current-parent old100/120 comparison under chosen cap10 was canceled at CPU inventory/protocol stage. Its120k original config blob was not found in examined captures; actual140k config could only have been reused with an explicit engine-provenance label and default-render parity. No candidate scores or optimal-step claim were fabricated.

Native crop benefit: old all66 A(native/legacy allocator) versus I(HD/legacy allocator) at32k lowered LPIPS19.5463%, with PSNR+.7151dB/SSIM+.05456. This is in-sample and cannot establish the same heldout gain on the present strong recipe. The new strict63 AREA4-control was interrupted; its saved prefix is retained, not a completed matched terminal ablation. [Attribution correction](/fsx/tmp/marshawn_native63_24h_20261108/research/crop_resolution_A_I_32k_clarification_CPU/PROOF.json).

### Occupancy and AABB

The selected box is an AABB only in its rigid floor-aligned frame. [Frozen geometry and both transforms](recipes/marshawn1660_native63_hash25/geometry.json) are mandatory; reusing the numeric bounds in the original world recreates a different box. The inherited review sampled1660…2020 every.5s; historical output-count48 metadata is a two-second queue field, not proof that this tuning campaign produced a video. Mat avoidance is not claimed over the whole interval: the later grey fighter lies on the mat.

The exact terminal140 R0 grid has29749 active cells within108699 allowed cells; zero outside allowance, capture step139984/age15, matching registered aliases/core.715529 nominal contour rays were replayed: heldout32804 has zero misses, TRAIN has exactly the SAME510 historical static-prop annotation contaminants, with no newly failed contour pixel. A ray meeting any active cell is not a correct-depth/surface certificate.

8358 active cells have at least two source-clear votes;234 robust low cells are candidates, not proven physical-free volume or safe deletion targets. Source witnesses show upper background/floor-support disagreements and novel renders show large halos/sheets. An earlier audited R1 control added27089 cells without recovering the historical510; no dilation or pruning was adopted. Saved EMA maxima are historical occupancy proxies, not measured alpha or physical density. **An ideal occupancy grid was not achieved or certified.** [Full140 grid audit](/fsx/tmp/marshawn_native63_24h_20261108/research/final_grid_audit/P15_corehash25_140k_CPU_v1/REPORT.md), [static-cap correction](/fsx/tmp/marshawn_native63_24h_20261108/research/final_grid_audit/P15_corehash24_100k_CPU_v1/interpretation_correction_v1/ERRATUM.md).


### Exact final stopped trajectories
Latest checkpoint index and last scored gate are separate. These checkpoints are saved prefixes, not interpolated target-stage results. [Full final per-output inventory](assets/marshawn_native63_STOP/FINAL_TRAJECTORY_INVENTORY.json) supersedes earlier live/snapshot progress; [final all-host stop](assets/marshawn_native63_STOP/FINAL_STOP_PROOF.json) and [dedicated-controller closure](assets/marshawn_native63_STOP/DEDICATED_CONTROLLERS_CLOSED.json) verify shutdown. All pending-evaluation flags are false.
|Output / host|Saved checkpoint index|Completed logical updates|Last committed evaluated updates|
|---|---:|---:|---:||P_B8beta99_SAM15_core001_hash25_coarseHalf / olive|52932|52933|50000||NN_shared90_SAM15_B8beta99_hash25_core001_t0 / silver|70855|70856|50000||NN_shared98_SAM15_B8beta99_hash25_core001_t0 / silver|71165|71166|50000||P_B8beta99_SAM15_core001_hash25_ARM12toFull50000 / silver|75352|75353|50000||NN_FAS0_seed43 / silver|139999|140000|140000||P_B8beta99_SAM30_core001_hash25 / silver|71758|71759|50000||P_B8beta99_SAM15_core001_hash26 / silver|102426|102427|100000||P_B8beta99_SAM15_core001_hash25_HD4 / silver|22383|22384|16000||P_B8beta99_SAM15_core001_hash24_ARM12toFull50000 / silver|139796|139797|120000||NN_B8beta99_hash24_core001_t05 / silver|99999|100000|100000||Q_seed43 / dev3|139999|140000|140000||P_B8beta99_SAM15_core001_hash24_ARMmax12_DEV3 / dev3|53074|53075|50000||P_B8beta99_SAM15_core001_hash25_ARMmax12 / clever|27646|27647|16000|
The stopped hash24 ARM50 checkpoint is index139796/139797 logical updates with its last committed score at120k: it did **not** complete or validate140k. The native HD control stopped at22384 and Clever constant-cap12 at27647 logical updates, each last scored at16k. t05 continuation stopped during setup without a proven new-worker training step. B4k has no committed scientific result. DEV3 retains its separate2000 replay debit. Exact source paths, hashes, return codes and atomic-checkpoint checks are in the inventory.

### Complete saved evidence and unfinished work

- [Snapshot of496 saved metric gates](assets/marshawn_native63_STOP/all_saved_metrics.csv), [117 saved arm-name summaries](assets/marshawn_native63_STOP/all117_saved_arms.csv), [227 topic-report/erratum paths](assets/marshawn_native63_STOP/all_report_artifacts.csv), [completeness checklist](assets/marshawn_native63_STOP/CHECKLIST.md), [inventory provenance](assets/marshawn_native63_STOP/inventory_provenance.json).This12:54UTC metric snapshot is supplemented by the later final per-output inventory above.589 source JSONs were deduplicated by arm/logical update.117 is not a count of all queued/failed/never-started jobs, and496 scalar records are not496 independently visually passed models. Logical iteration markers can differ from physical attempts after documented replay or technical-start markers; use the trajectory ledger.
- The saved50k frequency/header comparisons, latest checkpoint, metric-minimum step and interrupted-worker step are separate columns/states. A numeric minimum is not silently declared a retained checkpoint. Unfinished80k and120k endpoints are not filled by interpolation.
- User stop preserves pending-evaluation markers rather than launching an evaluation. Latest unscored checkpoints are not selected by assumed quality. DEV3's last saved index53074 is53075 logical completed updates with a separate2000 replay debit; it is not an80k result.
- B4k/hash25 was CPU/native-qualified and queued; no scientific result is claimed without an actual saved gate. Constant-cap12/Clever, HD4, new frequency-map and sampling/curriculum prefixes remain interrupted controls. Final all66 preparation is not an all66 trained model.
- Motion-path CPU work stopped before framing qualification and produced no render. A planned crop would have clipped hair, so it was not passed off as a valid test. No fixed-model shimmer or independent-frame flicker result exists.
- Evaluator alpha roundoff failures, a DEV3 telemetry timeout, backend/prebuilt qualification, immutable-stage-controller corrections, stop-helper permission errors and requested SIGTERM are operational events, not negative model-quality evidence. Raw receipts/errata remain linked in the historical record. The packed counter is persistent: read its terminal value once; sum only reset elapsed clocks.

## Insights: evidence versus assumptions

**Supported on this scene:** native source detail is useful in the older in-sample comparison; constant maps beat the new learned maps at matched32/50; capacity25 improves the matched140 default result; cap10 trades some PSNR/SSIM for better masked LPIPS; further reduction to9 hides face/boot regressions in the aggregate; more samples or more frequent grid updates are not monotonic quality improvements; conservative occupancy coverage does not remove phantom geometry.

**Source-supported explanations, not causal proofs:** min-step floors and a shared whole-ray budget can make frequency knobs ineffective; raw-focal/max-only frequency updates can saturate the grid; feature reweighting and sampling are distinct uses of the same grid; incomplete scalar maps/coverage and ray allocation can interact. The measured0/15 active-cell histogram does not give ray-weighted usage or prove oversampling everywhere.

**Not established:** cross-scene superiority, globally optimal hyperparameters or iteration count, a perfect physical grid, accurate empty-space labels, saturated validation, a heldout19.55% crop gain on the current recipe, a finished all66/video result, or flicker-free rendering. See [general lessons and source-backed checks](lookcloser_native_frequency_sampling_lessons.md).

## Saved delivery

The selected140k checkpoint/config/source receipts have a complete durable archive. The [frozen recipe](marshawn_native63_frozen_recipe.md) and [machine-readable selection](recipes/marshawn1660_native63_hash25/selection.json) preserve all main hyperparameters, phase schedules, geometry transforms, exact source/spec/config hashes, cold-initialization requirements and the inference-only override. Raw selected [HC cap10](/fsx/tmp/marshawn_native63_24h_20261108/ops/frozen_hash25_140_caps10to14_finite/actual_native/maxfreq10/HC.png) and [matching GT](/fsx/tmp/marshawn_native63_24h_20261108/data/uniform16/images/HC.png) remain native1464×2360, not upscaled4K. No new work is launched by this report.
