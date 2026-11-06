# Training-time semantic admission candidate

## What was tested

One isolated change to the frozen native-exp actor/body80/R0 parent: fixed, conservative TRAIN semantic admission during density evaluation and traversal. SH correction, exp(h+1), frequency projection, LR schedule, sampling, AABB, seed42 and mandatory106316→plateau remain unchanged. No geometry prefix or positive-band objective is added.

The source is the independently reviewed occlusion_v2/W64 pad1 grid from63TRAIN cameras. Whole-cell mapping retains any source overlap, boundary contact and semantic UNKNOWN. It preserves103 witness XYZ/pixel-depth points and both qualified shoe anchors; source whole-TRAIN-ray coverage and source/frame hashes are sealed. DC/HC/KA never contribute admission labels. The original learned occupancy EMA and core are saved separately: active is native noncumulative dilation (or native all-active binary warmup) intersected with fixed admission.

A separate inference-reloadable model subclass gates the native common position selector. Therefore fixed256-sample warmup, adaptive ray marching, `get_density`, `query_points`, inherited `density_fn` and evaluation use the same support. Native encoding, activation and admitted-point derivatives remain intact. The primary model defaults and running snapshots are unchanged. Snapshot schema2 stores actual learned core, EMA, admission and active separately; skipped cells are not certified physically empty.

## Technical results

Generation1 was preserved after a zero-update diagnostic failed on an empty native parameter-gradient tensor. Generation2 changes only that diagnostic and its shape/dtype/empty regression checks; production model/worker files are byte-identical. Ten CPU tests pass, including brute-force whole-cell overlap, XYZ/contact/outside UNKNOWN, density and gradient gating, all-ones identity, source/bounds tampering, core/halo separation and empty-gradient mismatches.

| Actual native GPU check | Result |
|---|---|
| Full4096-ray all-ones warmup RGB/alpha difference |0 /0|
| Full4096-ray all-ones ARM RGB/alpha maximum difference |2.38e-7 /4.17e-7|
| Largest parameter-gradient difference warmup/ARM |1.19e-6 /2.74e-6|
| Forbidden-point density_fn/query_points |Exactly0 in both paths|
| Warmup successful Adam updates before→after resume |16→32|
| ARM successful Adam updates before→after resume |15→30|
| RNG restore |Exact recorded state digest in both phases|
| Technical rendered DC/HC/KA |Finite complete rasters in both phases|
| Peak monitored candidate process memory |7,948MiB|

ARM smoke starts cold at step4096 only to exercise the postwarmup code path; it is not a learned4096-step checkpoint. GradScaler legitimately skipped one update in each16-attempt ARM segment (32768→16384 across resume); successful Adam states and finite moments are verified on both sides. There is no quality claim from these32-attempt runs.

The root-approved technical overlap lasted189.7s. The independent production worker remained alive and progressed82639→88301 without OOM. Its global GPU lock and files were untouched. The candidate used a separate overlap lock,10GiB allocator cap and12GiB process-memory ceiling; that lock was confirmed free afterward. These concurrent timings are not performance benchmarks.

[Complete technical receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/technical_complete.json), [GPU identity/path proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/native_probe/probe.json), [warmup resume](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/verification_warmup.json), [ARM resume](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/verification_ARM.json), and [visually inspected three-view technical montage](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/technical32_overview.jpg).

## Insights and launch state

Technical gates pass; initial renders contain broad actor silhouettes and retained foreground/mat regions. This is consistent with conservative semantic UNKNOWN and does not establish good geometry. Production remains separately guarded by an exact root release, binding19 artifacts and1495 frozen source files. The concrete single-job plan queues a cold actor+admission arm on the first idle GPU under the common global lock; production overlap is prohibited. Compare matched geometry rays, DC failures, both actors, hair/feet and per-view PSNR/SSIM/LPIPS before any promotion.

Frozen root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106`; remote counterpart under silver-sable `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`. Source manifest SHA `f428fdc2527d189cec319001aced76f9de7b7a6caae5b5a1e8702106b5ab434c`; support manifest SHA `fce7fc24735cd62d5242ce719964c5d6d61291feef1cc1c536c12f2ce194b15e`.

Root production review approved the exact19-artifact release (`c97566fb2ab559aa1c22839e81373637f4c1ba9015a2c9ed07611f84d75d42ab`). Controller676019 is durably queued; independent watcher691770 retains the first15188 checkpoint/grid/renders before replaying the sealed103TRAIN and12DC diagnostics. Its semantic-aware adapter uses schema2, keeps fixed density admission in every profile and computes native halo from the saved learned core before intersection. Six per-view/actor metric records and DC replay/PNG parity are preserved; DC diagnostic pixels never become TRAIN labels.

A separate R1 control is prepared, not launched, at `semantic_actor_r1_prepared_20261106`. Actual resolved config differs only in native occupancy radius0→1 and output/support path relocation; source pad1 bytes are identical. Its only source-code change is exact-spec validation of radius1. Preparation and config-diff receipts require R0's first15188 visual/geometry gate plus separate release before launch. Source SHA `804b6858155537565b38080c17304037e19bbb64e3094b2d35ff2281c4981f5a`; spec SHA `08de9623c26c78d49fec04b33b21b399be574aa7c7cb5b01f6348b6fae59f298`.
