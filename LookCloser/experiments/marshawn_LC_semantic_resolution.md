# Independent semantic node-resolution preparation

## What was tested

CPU-only preparation of a separate [128/256 worker](../scripts/probe_marshawn_LC_semantic_resolution_head.py), copied from the sealed head worker SHA `6f6bca9b69c97f0375d943a37f208a17407fcb64f046e920556e101382ed149d`. The existing worker, `FrozenActorGrid`, original capture, model, labels, boxes and renderer are unchanged. The class function remains sigmoid of trilinear XYZ logits, with Z/Y/X sampling coordinates, `align_corners=True`, inclusive accepted bounds and the identity floor transform. Native visibility weights and the existing conditional class objective are unchanged.

The scientific comparator is **FOUR-UNKNOWN128 versus FOUR-UNKNOWN256**, both independent fresh heads with p=.01, seed42, Adam LR=.03 and the same 2,064 camera packets. The four reviewed UNKNOWN records are IA40470/40622 and LA54411/54594; positive targets remain1 and confidence is0. The new worker reuses the exact root-qualified label file SHA `44e7734ed93a2c10742de6dc55886978d7c2097ee62779915dada93feda94df4` and matched128 spec SHA `ccfdd894d29d06ef17b8dffd8fa66b7db06b4a8383d7691246632d8fff7fba93`. The original TWO-UNKNOWN128 baseline is immutable and is a separate label ablation, not the resolution comparator.

The local [paper](../Paper%20LookCloser.md), §4.2, motivates matching feature resolution to observed detail; it does not propose or validate this independent semantic grid. Finer semantic support may separate floor/fringe/interleg/shoe attachments, but its ability to preserve unobserved body regions requires an experiment. The earlier [hierarchical residual helper](marshawn_semantic_residual_head.md) remains a separate CPU option.

## Results

The [materializer](../scripts/prepare_marshawn_LC_semantic_resolution_cpu.py) created the local false-release namespace:

`/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_resolution256_fourUNK_plan_v1`

Its specs reference the corresponding `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/` execution namespace. Inherited source bindings were verified against local bytes; the old control head uses an explicit same-SHA local archive evidence alias. Actual native source step126316, source SHA `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`, sealed capture SHA `fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d`, all 69,888 queries and 12,993,330 packed points remain unchanged.

The [CPU stager](../scripts/stage_marshawn_LC_semantic_resolution_silver.py) subsequently verified235 remote files, reusing216 existing files and uploading only19 new package files (152,099 bytes). [Staging receipt](assets/marshawn_LC_semantic_resolution_20261107/stage_receipt.json) and [remote CPU proof](assets/marshawn_LC_semantic_resolution_20261107/remote_CPU_proof.json) bind all225 execution inputs, source checkpoint/aliases/history, the four UNKNOWN qualification references and their IA/LA leaves, label bytes and controller SHA `2e7a1be53794882ffc3c0a6a82fff3cc24070b36fbf7593e65771772e0dadf39`. All nine sibling modules and the controller import on silver; all four worker dry-runs pass with CUDA uninitialized. No256 actual release, run/queue/output or bytecode cache was created. No controller/GPU was launched, no source head was fitted, rendered or exported, and the active FOUR-UNKNOWN128 run was untouched.

The final false-template SHA is `b53ce6871a01a41f03c6b1a294724ae494dbb8ae5a9d64d810004faea15a19cb`; worker SHA is `8e9e62d86b5cde6bfb7b08839ee0fe6dcec71530e1283f91670efe1535c85102`. Exact technical/fresh DAG and all four spec hashes are in the staging receipt. Technical256 is intended for GPU1 only after the healthy128 run finishes and root grants a separate release.

[CPU proof](assets/marshawn_LC_semantic_resolution_20261107/CPU_proof.json) covers 19 new tests plus 17 existing head/LC contracts: **36 tests pass**, CUDA uninitialized. Tests use actual128³ and256³ CPU tensors for shape, constant prior, state serialization and synthetic32+32 versus64 Adam/sampler restoration. Both resolutions resume bitwise exactly on CPU. Empty rays retain escape1 and contribute no class objective. Invalid512, wrong shape/dtype/bounds/frame, TWO-UNKNOWN comparison, changed packet/data identity, changed prior/seed/LR and a false GPU release fail closed. GPU atomic trajectory agreement remains a separate measured technical question.

Every new checkpoint and completion records actual resolution, initialization, optimizer settings, FOUR-UNKNOWN count, experiment family and source/capture/packet hashes. Resume checks these fields, the actual Adam LR and tensor shape/floor bounds before strict state restoration. The root release binds worker, helper, source inputs, qualification, matched128 spec and each execution spec.

| Independent FP32 grid | Nodes | Logits | Logits + gradient + two Adam moments |
|---|---:|---:|---:|
|128³|2,097,152|8 MiB|32 MiB|
|256³|16,777,216|64 MiB|256 MiB|

The256 minimum adds224MiB over128, while the worker retains the existing8GiB allocator limit. Cache transfers, interpolation/autograd activations, optimizer temporaries and allocator reservation are additional; actual GPU memory fit is **not measured yet**. Source cache arrays stay on CPU and only one camera packet is transferred per update. No512 option is prepared.

The [worker diff](assets/marshawn_LC_semantic_resolution_20261107/head_worker_vs_sealed128.diff) is separate from the original worker. The [physical constructor proposal](assets/marshawn_LC_semantic_resolution_20261107/physical_constructor_PROPOSAL_ONLY.diff) adds an explicit expected-resolution guard and substitutes that resolution at construction. It has not been applied or integrated. It accepts the historical128 checkpoint schema only with explicit128 completion/spec and matching shape;256 requires the new worker identity and FOUR-UNKNOWN metadata. All existing physical source/label/spec checks remain necessary.

## Insights and required gates

The prepared DAG is256 technical32 → resumed32 → independent64, followed by root review of exact resume-start restoration, reported CUDA trajectory drift, actual peak memory/OOM evidence and immutable source/cache checks. All technical and fresh2064 authorizations are false. The fresh2064 controller is separate and requires another explicit root release after the technical gate; no physical or production release is created.

After the FOUR-UNKNOWN128 result and the256 technical gate, a candidate comparison must use matched fresh2,064 histories and identical qualification bytes, then the same physical selectors and reviewed views. Any baseline label benefit must be separated from resolution benefit. Evaluation and visual review should explicitly retain hair, shoes and body while inspecting floor, bag, crew, fringe and interleg attachments.

A finer independent low-prior grid can localize selection but also suppress geometry outside observed supports. The packed points are integration samples from ray-level supervision, not independent3D labels; finer cells do not resolve occlusion ambiguity or missing evidence. Soft class probabilities can still leave high-sigma floor opaque. No regularizer, new scientific prior, source label change, quality improvement or preference over the hierarchical option is claimed.
