# Independent semantic node-resolution experiment

## What was tested

Preparation and actual bounded GPU testing of a separate [128/256 worker](../scripts/probe_marshawn_LC_semantic_resolution_head.py), copied from the sealed head worker SHA `6f6bca9b69c97f0375d943a37f208a17407fcb64f046e920556e101382ed149d`. The existing worker, `FrozenActorGrid`, original capture, model, labels, boxes and renderer are unchanged. The class function remains sigmoid of trilinear XYZ logits, with Z/Y/X sampling coordinates, `align_corners=True`, inclusive accepted bounds and the identity floor transform. Native visibility weights and the existing conditional class objective are unchanged.

The scientific comparator is **FOUR-UNKNOWN128 versus FOUR-UNKNOWN256**, both independent fresh heads with p=.01, seed42, Adam LR=.03 and the same 2,064 camera packets. The four reviewed UNKNOWN records are IA40470/40622 and LA54411/54594; positive targets remain1 and confidence is0. The new worker reuses the exact root-qualified label file SHA `44e7734ed93a2c10742de6dc55886978d7c2097ee62779915dada93feda94df4` and matched128 spec SHA `ccfdd894d29d06ef17b8dffd8fa66b7db06b4a8383d7691246632d8fff7fba93`. The original TWO-UNKNOWN128 baseline is immutable and is a separate label ablation, not the resolution comparator.

The local [paper](../Paper%20LookCloser.md), §4.2, motivates matching feature resolution to observed detail; it does not propose or validate this independent semantic grid. Finer semantic support may separate floor/fringe/interleg/shoe attachments, but its ability to preserve unobserved body regions requires an experiment. The earlier [hierarchical residual helper](marshawn_semantic_residual_head.md) remains a separate CPU option.

## Results

The [materializer](../scripts/prepare_marshawn_LC_semantic_resolution_cpu.py) created the local false-release namespace:

`/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_resolution256_fourUNK_plan_v1`

Its specs reference the corresponding `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/` execution namespace. Inherited source bindings were verified against local bytes; the old control head uses an explicit same-SHA local archive evidence alias. Actual native source step126316, source SHA `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`, sealed capture SHA `fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d`, all 69,888 queries and 12,993,330 packed points remain unchanged.

The [CPU stager](../scripts/stage_marshawn_LC_semantic_resolution_silver.py) subsequently verified235 remote files, reusing216 existing files and uploading only19 new package files (152,099 bytes). [Staging receipt](assets/marshawn_LC_semantic_resolution_20261107/stage_receipt.json) and [remote CPU proof](assets/marshawn_LC_semantic_resolution_20261107/remote_CPU_proof.json) bind all225 execution inputs, source checkpoint/aliases/history, the four UNKNOWN qualification references and their IA/LA leaves, label bytes and controller SHA `2e7a1be53794882ffc3c0a6a82fff3cc24070b36fbf7593e65771772e0dadf39`. All nine sibling modules and the controller import on silver; all four worker dry-runs pass with CUDA uninitialized. At this CPU staging gate, no256 release/run or controller/GPU work had started; the existing FOUR-UNKNOWN128 run remained untouched.

The final false-template SHA is `b53ce6871a01a41f03c6b1a294724ae494dbb8ae5a9d64d810004faea15a19cb`; worker SHA is `8e9e62d86b5cde6bfb7b08839ee0fe6dcec71530e1283f91670efe1535c85102`. Exact technical/fresh DAG and all four spec hashes are in the staging receipt. Technical256 is intended for GPU1 only after the healthy128 run finishes and root grants a separate release.

[CPU proof](assets/marshawn_LC_semantic_resolution_20261107/CPU_proof.json) covers 19 new tests plus 17 existing head/LC contracts: **36 tests pass**, CUDA uninitialized. Tests use actual128³ and256³ CPU tensors for shape, constant prior, state serialization and synthetic32+32 versus64 Adam/sampler restoration. Both resolutions resume bitwise exactly on CPU. Empty rays retain escape1 and contribute no class objective. Invalid512, wrong shape/dtype/bounds/frame, TWO-UNKNOWN comparison, changed packet/data identity, changed prior/seed/LR and a false GPU release fail closed. GPU atomic trajectory agreement remains a separate measured technical question.

Every new checkpoint and completion records actual resolution, initialization, optimizer settings, FOUR-UNKNOWN count, experiment family and source/capture/packet hashes. Resume checks these fields, the actual Adam LR and tensor shape/floor bounds before strict state restoration. The root release binds worker, helper, source inputs, qualification, matched128 spec and each execution spec.

| Independent FP32 grid | Nodes | Logits | Logits + gradient + two Adam moments |
|---|---:|---:|---:|
|128³|2,097,152|8 MiB|32 MiB|
|256³|16,777,216|64 MiB|256 MiB|

The256 minimum adds224MiB over128, while the worker retains the existing8GiB allocator limit. Cache transfers, interpolation/autograd activations, optimizer temporaries and allocator reservation are additional; actual memory fit was subsequently measured by the bounded technical and fresh runs below. Source cache arrays stay on CPU and only one camera packet is transferred per update. No512 option is prepared.

The [worker diff](assets/marshawn_LC_semantic_resolution_20261107/head_worker_vs_sealed128.diff) is separate from the original worker. The original [physical constructor proposal](assets/marshawn_LC_semantic_resolution_20261107/physical_constructor_PROPOSAL_ONLY.diff) adds an explicit expected-resolution guard and substitutes that resolution at construction. After the actual head completed, it was applied only in a separate false-release physical namespace, described below. It accepts the historical128 checkpoint schema only with explicit128 completion/spec and matching shape;256 requires the new worker identity and FOUR-UNKNOWN metadata. All existing physical source/label/spec checks remain necessary.

## Actual GPU technical gate and fresh2,064 update result

Root released the technical gate under exact release SHA `251e8f4855e6e160f258052c874deef40d48b734467f90089eef6e182c8da65f`. The three actual GPU1 jobs finished normally:32 updates, an exact-start resumed32, and an independent64. [CPU continuation comparison](assets/marshawn_LC_semantic_resolution_20261107/actual256_technical_comparison.json) and [Adam/RNG audit](assets/marshawn_LC_semantic_resolution_20261107/actual256_technical_Adam_RNG.json) confirm exact resume-start head, Adam, RNG and sampler restoration. Continued64 versus uninterrupted64 max logit difference is1.90735e-6; max probability difference is3.53903e-8. Adam first/second moment max differences are7.27596e-11/3.10862e-15, and final Python/NumPy/Torch CPU/Torch CUDA/sampler RNG states are bit-identical. These are measured atomic trajectory differences, not a claim of deterministic CUDA backward operations.

The technical jobs observed282,690,560 allocated and419,430,400 reserved GPU bytes at compact progress boundaries, with first-job NVML usage1,052MiB. The fresh job observed436,207,616 reserved bytes. These are observed samples, not unsampled absolute peaks. All workers finished normally, allocator limits were respected, and guardians reported no OOM. All225 bound source files rehashed unchanged after the runs. Native field/RGB/occupancy were never instantiated or optimized by head fitting.

After root accepted this numeric/resource gate, release SHA `7a3e76ca310a409ee6e6597c2fda650232beadfc268cf5592ebb10267c4f29a2` enabled a fresh2,064-update256 head. Actual saved head SHA is `befa50aa2b178938fc2e918b8d78d3e59c7c39699d63ccc45663fb5851ce9d21`. [Matched128/256 CPU comparison](assets/marshawn_LC_semantic_resolution_20261107/actual256_matched128_comparison.json) confirms zero initial Adam steps,2,064 actual updates, the same FOUR-UNKNOWN qualification, exactly the same camera packet history and final RNG/sampler state as128 head `7937ab3d…81beb`. Frozen LC pipeline/capture remain `c83bc7db…d46636` / `fb8b875d…d99073d`.

| Fixed diagnostic | FOUR-UNKNOWN128 | FOUR-UNKNOWN256 |
|---|---:|---:|
|Actual DC retained bag-contact sample22 class probability|0.8265304|0.8193661|
|Minimum probability on109 sparse actor observations|0.5110674|0.2619596|
|Sparse observations retained at threshold0.1|109/109|109/109|
|Sparse observations retained at threshold0.5|109/109|106/109|

The three0.5 crossings are the same physical black sole witness6862 observed from JA/JB/KB. They are one unique point, not three independent geometry failures. The result flags a shoe-preservation risk at0.5. All109 remain above0.1, but that sparse set contains only41 distinct positions and does not certify complete hair/feet or physical empty space. The bag sample remains above both fixed thresholds; finer resolution alone has not removed that contact attachment. Full physical render/geometry review is still required before any quality promotion.

The [CPU physical materializer](../scripts/prepare_marshawn_LC_resolution256_physical_cpu.py) then prepared `/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_resolution256_physical_plan_v1`. [Actual CPU validation](assets/marshawn_LC_semantic_resolution_20261107/actual256_physical_CPU_validation.json) verifies358 bound inputs, three worker/controller dry-runs and the strict constructor against actual saved256 tensor, completion and spec metadata. The two runtime changes are guarded head construction in rendering and TRAIN-distribution audit; the existing tested helper supplies resolution validation. Source field, colour, density, occupancy, interval sampling, native weights, independent transmittance reference, thresholds and physical selectors are unchanged. The three-job DAG is physical render → fixed V1 metrics → TRAIN audit. Its SHA is `a9c322c581cc2be47b4b975fb048abd3bd71b004ee74b41eab0e7462adc5af00`; false release template SHA is `33f21bcb36b4cf8fd832603c40626eec4612b783046bfb9fd1831136f098289c`. This CPU preparation starts no GPU and grants no physical release. Matched128 physical outputs will be compared after both candidates complete.

## Insights and required gates

The prepared DAG is256 technical32 → resumed32 → independent64, followed by root review of exact resume-start restoration, reported CUDA trajectory drift, actual peak memory/OOM evidence and immutable source/cache checks. At preparation, all technical and fresh2064 authorizations were false; subsequent root releases and measured results are recorded below. The fresh2064 controller is separate and requires another explicit root release after the technical gate; no physical or production release is created.

After the FOUR-UNKNOWN128 result and the256 technical gate, a candidate comparison must use matched fresh2,064 histories and identical qualification bytes, then the same physical selectors and reviewed views. Any baseline label benefit must be separated from resolution benefit. Evaluation and visual review should explicitly retain hair, shoes and body while inspecting floor, bag, crew, fringe and interleg attachments.

A finer independent low-prior grid can localize selection but also suppress geometry outside observed supports. The packed points are integration samples from ray-level supervision, not independent3D labels; finer cells do not resolve occlusion ambiguity or missing evidence. Soft class probabilities can still leave high-sigma floor opaque. No regularizer, new scientific prior, source label change, quality improvement or preference over the hierarchical option is claimed.
