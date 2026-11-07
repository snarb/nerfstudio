# Isolated stationary-bag RGB-confidence correction

## What was tested

Six TRAIN views contain a manually qualified stationary bag mislabeled as the grey actor. MC/NA temporal source images independently show the bag remaining fixed after the shoe leaves. Of6874 qualified polygon-interior pixels, exactly6522 intersect the original grey label: MA1388, MB1229, MC1057, NA930, NB1141, ND777. The other352 pixels, unknown shells, all black positives and all other confidence pixels stay unchanged.

The isolated cold candidate retains the semantic W64/R0 actor/body80 recipe, original masks, sampling pools/FAS, frequency maps, fixed admission, field and schedule. It zeroes only the6522 verified false-grey RGB confidence values and recomputes the analytic proposal normalization. Proposal labels intentionally remain stale for a controlled single change; distortion still applies to these sampled rays. There is no alpha-zero or geometric-free-space assertion.

Frozen candidate: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106`. Source manifestSHA `847740fea7deb9112f6ae32518c67e3012358fb15ef8050005adfaafa7f82f08`; bag manifestSHA `fc338ac7c7353c9e04a40353894dbdba19b6d02810d809eb4ce064d99f092b3f`; production specSHA `9c811650c8440d3da98875ed69d50f23187a9bc27b6c15fd4f9c7fbf60107ff7`.

## Results

Seven CPU tests pass. The actual frozen63-camera adapter retains identical proposal pools, frequency arrays, sampled4096-ray batches and RNG states at steps0,4096,8192,30376. Exactly6522 effective weights change; their RGB gradient becomes zero. Every other RGB gradient differs only by the required normalization factor. Mature proposal expectation changes.6893199440→.6885764847; the warmup value changes.6852940238→.6843829117. Independent sparse expectation-delta errors are at most1.37e-16. Saved effective-weight/source bindings restore deterministically and reject tampering.

[CPU proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/runtime_cpu_proof.json), [actual config parity](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/config_parity_vs_R0.json), [local environment/data validation](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/local_port_validation.json). These checks created no CUDA context or model. Actual native Adam/resume and render validation remain pending a separately reviewed technical release; no production job has launched.

The proposed technical gate holds the global idle-GPU0 lock across warmup16+16 and ARM16+16 pause/resume, with900-second deadline,12GiB monitored process-memory ceiling,10GiB allocator cap and5-second resource checks. It requires nonempty finite Adam state on both sides, corrected-confidence input/state binding, exact RNG restoration, native core/EMA/admission checks and finite three-view renders. SIGTERM/SIGINT clean up only the owned child process group.

## Insights

The current frozen model uses a black render background. Actor RGB confidence remains positive in the W64 transition band, while target pixels retain the original scene background. A ray crossing the bounded actor volume can therefore be rewarded for reproducing a wall, floor or bag behind the volume with opacity inside it. The existing objective has no alpha/background-consistency term. This is a plausible pressure toward halo or sheets, separate from occupancy correctness; distortion can compress such opacity without resolving incompatible targets. The failed early W32 geometry gate shows that merely narrowing the confidence band is not guaranteed to help.

This candidate removes one verified source-label error only. Broader remedies require separate controls: qualified visible-background negative-opacity evidence outside a hair-safe uncertainty margin, or background compositing with an independently qualified clean plate. Unknown occlusions, feet and missed hair must remain protected. A source-coordinate density admission test is another separately prepared possibility; none is installed by this correction.

## Actual native technical gate

The separately root-released localGPU0 test completed in191.4seconds under one continuous exclusive global lease; monitored process peak7948MiB. Successful Adam counters are16→32 for warmup and15→30 for ARM. The latter legitimately skips one scaled update in each16-attempt segment. Both paths retain finite parameters/moments, exact sampler and RNG restoration, identical corrected-confidence/source binding, and valid actual core/EMA/admission state. The ARM path starts cold at4096 solely to exercise ARM; it is not a trained4096-step model.

[Technical receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/technical_complete.json), [warmup proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/verification_warmup.json), [ARM proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/verification_ARM.json), [personally inspected six technical renders](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_prepared_20261106/technical32_overview.jpg). Coarse actor silhouettes and broad floor/foreground support remain; these32-attempt images establish integration only. The GPU and lock were released afterward.

Production awaits an exact separate release. Its template binds25 artifacts and1496 source files, independently verified on silver-sable. The existing unchanged-field native/all-ones proof is explicitly inherited; successful Adam under the new objective is directly measured above. Prepared first15188 auditing retains the same103 TRAIN geometry witnesses and12 posthoc DC rays, reports groups/reserves/signed depth width separately, and requires convergence and saved-RGB parity. Ongoing per-view/actor renders and metrics remain mandatory; technical approval is not geometry promotion.

## Released local cold comparison

Root separately approved the path-relocated local production candidate `semantic_actor_bagloss_local_20261106`; remote silver proposal remains unreleased. The1496 source files and scientific recipe are identical. Actual resolved configuration differs only in five data/output/source path fields. Local releaseSHA `5cace9c07f66e61ef1a8d3f296b96e5771bec2642a8c486606b4ab2ca1a9a613` binds32 artifacts, including the original25 technical/source artifacts and local path/audit proofs.

Controller2090883 launched supervisor2090890 and trainer2090895 on exclusively locked localGPU0. First reviewed progress reached2039 without OOM; peak allocator3.73GB. Artifact watcher2090906 polls every10seconds and paired15188 auditor2090907 preserves actual schema2 state before read-only103TRAIN+12DC replay. The unchanged106316→plateau trajectory is active. Local root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_bagloss_local_20261106`. No quality improvement is claimed before the matched gate.

## Matched first15188 quality gate

Bag-corrected PSNR31.47280/SSIM.840701/LPIPS.03106445 is almost unchanged from original R0's31.41439/.841056/.03117262. Viewed both actors in all three GT/control/candidate crops: hair and shoes remain, while broad exterior haze and floor residuals persist. [Visual comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_matched15k/GT_R0_bagcorrected_15188_crops.jpg).

The sealed geometry gate regresses: native witness-band mass changes80.23→68.01% for old18,83.95→81.98% for new66,84.48→70.55% for old8 reserves,95.95→89.76% for new9 reserves. Old8 early mass rises9.76→21.96%. All206 clothing and24 DC replay/convergence checks pass, and the producer remains healthy. [Geometry report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_matched15k/comparison.json), [DC report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_DC12_matched15k/comparison.json).

The source-label correction is independently qualified, but this single cold trajectory does not establish better global geometry. It is not promoted; the released trajectory continues to the same plateau rule. Padding and other unqualified views can still provide incompatible background targets. A small confidence correction does not by itself solve broader opacity ambiguity.

## Natural completion and saved-alpha review

The local cold correction arm finished naturally at141316. Established selection is121316: PSNR31.54293/SSIM.838076/LPIPS.02221597. The separately protected raw LPIPS minimum is terminal141316:31.53088/.837431/.02193486. All actual checkpoint/grid identities are retained in [final choices](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_final_choices/complete.json). The original selector/plateau was not changed.

CPU saved16-bit alpha diagnostics use the fixed original actor masks at margins0/3/8/16/32. At16px beyond the masks, alpha>.1 pixel counts change from originalR0 selected111316 DC/HC/KA185261/107305/74830 to bag selected121316181926/105236/72535, and terminal141316180782/104980/72453. The comparison deliberately exposes different checkpoint ages; these are exterior-support diagnostics, not physical free-space labels.

Personally inspected DC black, HC grey and KA black source/RGB crops show only a small overall change. Broad hair/torso haze and floor/bag-like components remain, despite correct removal of the6522 qualified false-grey confidence pixels. The result does not establish that the geometry is clean. Its actor LPIPS is close to originalR0, with no decisive overall promotion. Final selected103+DC12 depth has not yet been replayed; the first15188 diagnostic remains an early result.

[Alpha receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_final_exterior/receipt.json), [DC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_final_exterior/DC_black_comparison.jpg), [HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_final_exterior/HC_grey_comparison.jpg), [KA](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_bagloss_final_exterior/KA_black_comparison.jpg).
