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
