# Quarter integration step × distortion regularization

## What was tested

An explicit two-factor interaction: coarse step .0015625 versus original .00625, and final distortion coefficient .1 versus original .01. The quarter+.1 cold arm retains semantic R0/W64, main body80 sampling, original global2048 frequency updates/maps, legacy allocator with cap1024, original LR/FR schedule and full106316→plateau trajectory. No focused-frequency hook, bag correction, source-point gate or 4K bundle.

The frozen generation is `C/semantic_quarter_distortion_candidate_20261107` (silver `SC` equivalent). Source inventory SHA `ad32a034c492314162fac8d966be2215c834393a1bb984beb05f1eed8f9c43b3`, spec SHA `54e55e212bb7c1d27c12c9428cef8cd6b99ccdefa9ca3fbef776ba01d828aaa8`. Successful warmup/ARM Adam-resume checks preceded the cold run. Minimum trajectory and historical selector remain unchanged.

## Results

At the matched15188 gate, quarter+.1 LPIPS .022190927 was1.37% worse than quarter .01 (.02189069), with black actor3.88% worse and grey1.56% better. Native band locality improved: old18/new66/old8reserve/new9reserve82→86.57 /84.35→95.27 /86.47→89.68 /91.71→99.32%. Native cap hits0, cloth/DC maxima369/158. This did not justify early promotion or early stopping.

At the evaluated intermediate106316 gate, the combined arm has PSNR31.664709, SSIM.832548, LPIPS.018903045. It is still training; the checkpoint is not labelled terminal. The original quarter arm completed and its raw minimum is111316, so the comparison below has explicitly unequal checkpoint ages.

| Group | Quarter .01 raw111316 LPIPS | Quarter .1 at106316 LPIPS | Relative change |
|---|---:|---:|---:|
| All actors |.019231721|.018903045|−1.71%|
| Black actor |.020379021|.020238167|−.69%|
| Grey actor |.018084421|.017567923|−2.86%|

All three views improve LPIPS: DC−1.66%, HC−3.17%, KA−.42%; PSNR and SSIM also improve in each view and actor. Exact saved metrics and file hashes are in `C/independent_occupancy/quarter_distortion_106316_exterior/metrics_comparison.json`. No metric was recomputed or silently reranked.

Personally inspected DC black, HC grey and KA black GT/R0/quarter .01/quarter .1 contacts. The combined arm narrows the exterior halo while preserving gross hair/shoes and body detail. Floor/bag remnants and head halo remain obvious. Pixels with alpha>.1 beyond16HD pixels outside the fixed mask change from190316/111420/76436 for quarter .01 to118213/79911/55067 for quarter .1 (DC/HC/KA). This mask-exterior diagnostic is not physical empty-space truth.

## Insights and pending final qualification

Early detail regression reversed after full training; the actor and per-view checks rule out an aggregate-only win at106316. The combined arm is a plausible structural base for later controlled frequency-map experiments, but no clean-asset or ideal-grid claim is justified.

A durable supervisor waits for natural completion, preserves established selector, raw LPIPS minimum and terminal separately, and runs raw-min first then distinct selector through the same103 TRAIN clothing and12 EVAL-only DC diagnostics, native/dense convergence and actual cap/tail checks. It does not fit EVAL rays or modify the producer. Final geometry remains pending at this intermediate report.

`C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`; `SC=/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`. Immutable106316 checkpoint, actual schema2 grid, config/spec/source/release and renders are in `independent_occupancy/quarter_distortion_106316_snapshot`. Final audit is `independent_occupancy/semantic_quarter_distortion_final_geometry`.
