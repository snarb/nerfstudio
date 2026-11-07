# Matched frozen-source semantic refinement

What was tested: whether a zero256 residual on a frozen learned128 actor head separates boundaries while preserving feet. **The fine residual does not outperform matched coarse refinement, and neither produces a clean actor asset.** ROOT reviewed all three actor views and nine details and declined promotion: bright interleg/contact-floor density and outer shoe fringes remain.

Both arms start from the same learned FOUR-UNKNOWN128 checkpoint `7937ab3df744254ea4b4087f8efda9384888e3bccbf833719bc45945b8a81beb`. Coarse128 updates its existing logits. Residual256 freezes an independent copy and learns only a zero-initialized fine logit grid. Each explicitly resets Adam0, LR.03, seed42 and repeats the identical2064 camera history; neither resumes the technical-check optimizer. The original2064 base physical export is reused separately.

The residual is `sigmoid(trilerp(base128) + trilerp(residual256))`, using separate interpolations because the127/255 knot denominators are not nested. Initial equality passed bitwise on all12,993,330 cached sample positions and native rendered classes, plus109 source observations at41 positions. Native visibility weights,69,888 queries, targets and four qualified UNKNOWN confidences remain unchanged. No additional labels, optical alphaGT, whole-ray-empty targets or density/RGB training were introduced. Positions already use the identity floor frame, bounds `[-.84,-.21,-.07]..[.15,.42,.63]`.

This is an experimental semantic isolation stage around the frozen LookCloser source, not a feature-reweighting/sampling change from the local `Paper LookCloser.md`. Source step126316, checkpoint `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`, pipeline `c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636`, FR.3 and native legacy ARM1024 remain fixed.

Results: fixed actor crops, identical RGB.5 exterior, AlexNet LPIPS without resizing. Aggregation is unchanged from the base export; exact unrounded values and per-actor records remain in each metrics JSON.

| Head | Fixed selector | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Base7937 | hard.1 | 30.87165600 | 0.828192055 | 0.0221801096 |
| Base7937 | hard.5 | 30.71502342 | 0.827014585 | 0.0233414806 |
| Coarse128 | hard.1 | 30.89895187 | 0.828318129 | 0.0220101820 |
| Coarse128 | hard.5 | 30.81491597 | 0.827845116 | 0.0224600798 |
| Residual256 | hard.1 | 30.89303603 | 0.828252405 | 0.0221494446 |
| Residual256 | hard.5 | 30.78703764 | 0.827669342 | 0.0226772468 |

| View | Head | PSNR.1 | SSIM.1 | LPIPS.1 | PSNR.5 | SSIM.5 | LPIPS.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| DC | Base7937 | 31.617177 | 0.82771996 | 0.023044288 | 31.463177 | 0.82711762 | 0.023719775 |
| DC | Coarse128 | 31.660097 | 0.82784531 | 0.022812422 | 31.614469 | 0.82770312 | 0.023069953 |
| DC | Residual256 | 31.644715 | 0.82780632 | 0.022931494 | 31.565206 | 0.82751894 | 0.023257609 |
| HC | Base7937 | 31.192844 | 0.83129084 | 0.019538209 | 31.133371 | 0.83104026 | 0.020009150 |
| HC | Coarse128 | 31.204341 | 0.83137238 | 0.019547121 | 31.163934 | 0.83117259 | 0.019813110 |
| HC | Residual256 | 31.196031 | 0.83130565 | 0.019566462 | 31.155001 | 0.83113769 | 0.019840447 |
| KA | Base7937 | 29.804947 | 0.82556537 | 0.023957832 | 29.548522 | 0.82288587 | 0.026295517 |
| KA | Coarse128 | 29.832418 | 0.82573670 | 0.023671003 | 29.666344 | 0.82465965 | 0.024497177 |
| KA | Residual256 | 29.838362 | 0.82564524 | 0.023950377 | 29.640905 | 0.82435140 | 0.024933685 |

[Base metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_LA_two_UNKNOWN_physical_plan_v1/prior001/visual_gate/metrics.json), [coarse metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/coarse128/visual_gate/metrics.json), [residual metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/residual256/visual_gate/metrics.json).

All109 sparse observations retain both fixed thresholds. Sole6862 posterior rises from.51106739 to.85088474 coarse/.80828327 residual; that is not a whole-foot guarantee. Native-hit TRAIN positive retention at conditional actor posterior≥.5 is99.4856% base,99.6158% coarse and99.7721% residual; negative retention is1.0441%,.6553%,.3320%. Better TRAIN classification did not translate into cleaner physical extraction.

| Frozen-mask exterior diagnostic: alpha>.1 outside3px actor dilation | hard.1 | hard.5 |
|---|---:|---:|
| Base7937 | 57,863 | 44,685 |
| Coarse128 | 61,492 | 51,030 |
| Residual256 | 60,960 | 50,298 |

These counts are a fixed-mask diagnostic, not optical alphaGT; source bag-mask contamination remains. Both refinements increase exterior opacity despite modest actor-crop gains. The fine residual is marginally below coarse refinement on all aggregate PSNR/SSIM/LPIPS values at both fixed hard thresholds.

Visual inspection: all three bodies and all nine hair/hands-contact/feet comparisons were inspected by the worker and ROOT. Visible shoes remain represented, including the sole lost by the earlier independently initialized256 grid. Bright material between legs, floor/contact fragments and outer shoe fringes persist. Coarse hard.5 restores some contour mass while retaining more exterior material. The residual does not deliver a visible cleanup advantage over coarse. No threshold or head was automatically selected.

[All three actor comparisons](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/matched_review/all_actors_matched_refinement.jpg), [all nine details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/matched_review/all_details_matched_refinement.jpg), [ROOT's signed visual review and no-promotion decision](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/matched_review/root_manual_physical_review.json). Individual DC/HC/KA hair, feet and hands-contact PNGs are alongside these galleries.

Validation:35 CPU tests passed. Six GPU technical jobs passed exact resume START restoration of head/Adam/RNG/sampler. Independent64 versus32+32 maximum posterior drift over all13m cached samples is1.79e-7 coarse/1.19e-7 residual, with zero hard-threshold crossings. Both actual fresh heads then completed Adam0→2064; their camera histories and final RNG/sampler match each other and the original base. The frozen128 tensor remains byte-identical in the residual arm. Source/all60 caches post-verify unchanged.

All seven physical jobs completed0. Both three-view source MODEL RGB/alpha replays pass existing saved-code tolerances; no bitwise image identity is claimed. Eight full-source immutability checks and head digests pass after rendering. Explicit typed constructors/schema1 restore the actual heads; legacy128 archives cannot masquerade as residual heads. Original density selectors and native paths are unchanged, and physical transmittance is recomputed from selected sigma. The original-weight partition remains a separately labeled nonphysical diagnostic. Native1024 is capped and unchanged admission can omit geometry, so full visibility is not certified. Observed GPU usage is2596MiB, sampled RSS≤5.80GiB; no exact memory peaks are claimed, no OOM occurred and all expected processes terminated.347 runtime bindings post-verify unchanged.

[Initial full-cache identity](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_plan_v1/CPU_initial_function_qualification.json), [technical restart audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_plan_v1/CPU_actual_technical_audit.json), [actual2064 TRAIN/source audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_plan_v1/CPU_actual_fresh2064_audit.json), [typed actual-artifact qualification](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/CPU_actual_typed_artifact_qualification.json), [terminal physical audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/terminal_supervision.json).

Actual2064 head hashes: coarse `3859f5882cb032cb5d0ae8fd164b3469ad71f35851834e00e7b34fd8f24cbc84`; residual `88871d80a8f34284245a33972396112e4388078ca982a79cdc6b5c8588841c19`. Only the two genuine fresh checkpoints were fetched to olive, byte-verified, then exposed through explicit path-only completion adapters. The dispatcher runs locally on olive and sends fitting workers to silver by SSH; direct self-SSH dispatch on silver is rejected.

Insights: exact coarse-function preservation solves an initialization problem, but additional capacity and more epochs do not supply missing class evidence for newly visible floor. A continuous frozen base can preserve both real shoe support and unqualified floor support. The measured tradeoff does not justify cleanup promotion. Retain the full source/head lineage and images; original126316 remains the raw-field reference. The next experiment should target verified TRAIN floor/fringe evidence with real-shoe constraints, using a separate CPU-qualified packet and explicit release rather than silently changing the original69,888-query dataset.
