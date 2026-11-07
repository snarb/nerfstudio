# Marshawn cold hash capacity and corrected allocator — frozen campaign

**Final status: USER_CANCELLED.** The user explicitly stopped all experiments and runs immediately. All three workers, continuation retainers, gallery watchers, controllers and guardians are terminal; all eight Silver GPUs report0MiB and no compute applications. Checkpoints and results are preserved. No run reached its final plateau and no candidate is promoted. No restart or further optimization is authorized.

What was tested: three cold seed42 arms on the frozen frame1660 actor dataset. Control uses hash23 and the legacy allocator; capacity changes only hash23→24; allocator changes only `corrected_arm_allocator=False→True`. No combined hash24+corrected arm was run. All retain floor AABB `[-.84,-.21,-.07]..[.15,.42,.63]`, frozen Log3G10/RWG gain4 color and shared Reinhard, 63 TRAIN/3 EVAL split, frequency maps, semantic support, quarter coarse step .0015625, distortion coefficient .1, W64/body80/R0 and native ARM cap1024. No masks, query labels, calibration, bounds or external weights changed.

The audit found the actual 126316 detail source (`340e794e…d641`, pipeline `c83bc7db…6636`) already uses the larger hash budget from the [local paper](../Paper%20LookCloser.md): 16 levels, two features per level, min16 and hash23. Its explicit max resolution is8192; `max_res_base=2048` does not override that value. Field coordinates normalize per AABB axis. The frozen frequency maps and feature reweighting retain their existing8192/16-level contract.

| Encoding | Encoding parameters | Parameter, gradient and two FP32 Adam buffers |
|---|---:|---:|
| Source/control hash23 |171,739,264 |2.748GB |
| Capacity hash24 |322,734,208 |5.164GB |

The TCNN storage calculation exactly matches the trained source tensor. Source1495 already contained the corrected allocator branch; model, sampler and math files remain byte-identical. The common factory sets the case fields and operational guards; probe metadata records the actual case. Actual Silver typed-config comparison proves control equals the archived source config after output normalization, and each variant differs by exactly one model field. Both1495-file trees, 165 actual project imports, all66 RGB/mask records and frozen frequency/semantic/bounds inputs were checked. All63 TRAIN RGB headers are1920×1012. This is separate from the actual4K-trained `semantic_4k_production_v1` source.

The source run was cold: actual step0 logs, no initial checkpoint load and own-output-only resume establish its lineage. It reached126316 in4,860.16 training seconds (81.0min), with3.733GB peak allocated memory. Its automatic selector chose60752 (PSNR31.81075, SSIM.83793061, LPIPS.01947274); the separately designated minimum-LPIPS source126316 remains31.63973/.83126406/.01877312. Both selectors remain recorded. The old [hash-capacity report](marshawn_hash_capacity.md) was an untrained hash21→23 loose/full-scene stage, not this matched experiment. An inventory of320 campaign configs found no higher hash/resolution within its recorded search scope.

Results: the original three-case ZERO phase remains **failed**. Hash24 exceeded the unchanged `atol=3e-6, rtol=1e-3` gradient check at one of322,734,208 encoding entries: absolute difference3.4570694e-6 at index4412. The old log does not establish the step or signed values. Control23 and corrected23 passed independently.

A separately released hash24 diagnostic repeated backward twice on each retained graph without optimizer updates. Same-graph encoding differences reached2.20537e-6. This establishes backward repeat variation and is compatible with reduction noise; it does not prove the original failure's cause. Parameter/source/RNG hashes were preserved. That diagnostic omitted later support queries and supplied no full admission qualification. A subsequent separately released, **byte-identical original full native probe** passed both phases, including the actual support callbacks and density/query checks. Its gradient differences were1.31130e-6/2.50340e-6. No tolerance changed; all three scopes remain separate in the [qualification receipts](assets/marshawn_cold_hash_capacity_20261107/qualification_receipts.json).

Every case then passed actual warmup32→resume32 and ARM32→resume32 qualification. Adam counters were32→64 for warmup and30→62 for ARM, including two initial AMP skips. All persistent state families, source-cache/code hashes, RNG restoration, learned-core persistence and three native validation renders were verified. Hash23 technical peaks were3.727GB/3.591GB; hash24 peaks6.628GB/6.588GB. These are qualification results, not quality improvements.

| FixedV1 aggregate at20k | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Retained source126316 |31.639727 |.83126406 |.01877312 |
| Cold hash23 legacy |31.893421 |.84994805 |.02191202 |
| Cold hash23 corrected |31.878567 |.85003969 |.02168882 |

[Root’s exact20k visual receipts](assets/marshawn_cold_hash_capacity_20261107/root_twenty_k_visual_receipt.json) bind continuation permission. Root personally reviewed the [three actor panels](assets/marshawn_cold_hash_capacity_20261107/all3_actors.jpg) and [nine fixed detail panels](assets/marshawn_cold_hash_capacity_20261107/all9_details.jpg), then authorized both continuations. Recognizable bodies, hair and shoes remain; large floor and boundary halos remain too. Corrected20k reduces LPIPS by.0002232 versus matched legacy while reducing PSNR by.01485dB. Neither is promoted. [Source hashes and fixed ROI provenance](assets/marshawn_cold_hash_capacity_20261107/summary.json) bind the raw FHD RGB/alpha and original metrics. Exterior-opacity counts remain diagnostics, not nonactor, alpha0 or free-space truth.

Both hash23 runs restored their own20k checkpoints, unchanged source/input/spec hashes and RNG before advancing to20001. Before cancellation, detached guardians and retainers preserved the original plateau protocol without a22k cap. Hash24 was stopped before its20k review; its retainer and gallery watcher were terminated first to prevent relaunch. Every checkpoint and raw render is retained remotely. Shared guards enforce a10GiB allocator ceiling,32GiB per-process GPU/RSS limits, physical GPU UUID leases, exact process identity and independent heartbeat supervision. These byte limits are distinct from total `nvidia-smi` memory.

| Last preserved result | Saved checkpoint | PID, now terminal | Last evaluated step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Control23 / GPU0 |52,000 |645723 |50,000 |31.663084 |.83926668 |.02025108 |
| Corrected23 / GPU2 |52,180 |646389 |50,000 |31.668245 |.83967152 |.02006853 |
| Hash24 / GPU1 |17,652 |733082 |15,188 |32.012046 |.85193190 |.02321292 |

These are interrupted results at different evaluated steps, not a final capacity ranking. [The final cancellation receipt](assets/marshawn_cold_hash_capacity_20261107/user_cancelled_final.json) records exact checkpoint hashes, process identities and stop signals. Retainers2216273/2218613 and gallery watcher2218733 were stopped before verified workers645723/646389/733082 received SIGTERM. All workers exited within the bounded grace period without SIGKILL or OOM; remaining controllers and guardians exited automatically. Their nonzero cancellation/abort status is **USER_CANCELLED**, not scientific failure. Root independently confirmed no campaign processes or GPU work remained.

The earlier [11:48 freeze snapshot](assets/marshawn_cold_hash_capacity_20261107/freeze_status.json) remains historical evidence. Full artifacts remain under `/fsx/tmp/marshawn_001660_hd/tuning_20261106/hash24_cold_matched_candidate_v1` and its Silver mirror; checkpoint blobs are retained remotely.

Insights: the completed qualification and20k review establish valid execution, not a new best recipe. The user cancellation prevents a final plateau comparison. Preserve the designated comparison source and the global recipe freeze; do not infer a final hash-capacity or corrected-allocator advantage from these interrupted runs. All remaining work is documentation only. No further experiments, probes, initializations, watchers or optimization will run.
