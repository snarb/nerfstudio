# Marshawn cold hash capacity and corrected allocator — frozen campaign

**Status at 2026-11-07 11:48 UTC:** the user stopped new experiments. Only the three already running cold arms may finish their original plateau protocol. Final quality comparisons remain pending; the original source leader is unchanged. Hash24 still requires its existing 20k visual review before continuing the same run. No additional ablations, initializations or probes are planned.

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

Both hash23 runs restored their own20k checkpoints, unchanged source/input/spec hashes and RNG before advancing to20001. Detached per-case guardians and retainers continue the original plateau protocol with no22k or overall iteration cap. Hash24 is also running from cold; its existing single-case retainer waits for root's20k visual decision. Every checkpoint and raw render is retained remotely. Shared guards enforce a10GiB allocator ceiling,32GiB per-process GPU/RSS limits, physical GPU UUID leases, exact process identity and independent heartbeat supervision. These byte limits are distinct from total `nvidia-smi` memory.

| Snapshot11:48 UTC | Actual step | PID | Latest evaluated step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Control23 / GPU0 |38,310 |645723 |35,000 |31.718333 |.84339942 |.02070181 |
| Corrected23 / GPU2 |39,230 |646389 |35,000 |31.722755 |.84494442 |.02028371 |
| Hash24 / GPU1 |6,975 |733082 |5,000 |31.788713 |.84788740 |.02747147 |

The different evaluated steps are not a capacity comparison. [The exact freeze snapshot](assets/marshawn_cold_hash_capacity_20261107/freeze_status.json) records checkpoint hashes and process memory. Full artifacts remain under `/fsx/tmp/marshawn_001660_hd/tuning_20261106/hash24_cold_matched_candidate_v1` and its Silver mirror. Existing running-job supervision continues after this report commit.

Insights: preserve the original source recipe and leader until these existing runs finish and receive their final fixedV1 metrics and visual review. No capacity bottleneck, final allocator benefit or new best model is established. The user freeze prohibits extending the campaign with new experiments; remaining work is supervision, the already planned hash24 visual decision and final reporting.
