# Cold HD frequency-map trials

## What was tested

Prepared two cold NeRF trials using the validated63-TRAIN map inventories: SSIM.95 with per-level LR decay and SSIM.93 with constant fitter LR. Both start from seed42 and the same quarter-step.0015625, distortion.1 actor recipe, original W64/body80 confidence, R0, mapped semantic admission, legacy projection and original learning-rate/FR schedules. No warm checkpoint or pre-existing frequency grid is loaded. The frozen baseline uses SSIM.95/constant.01 maps and remains a separate archived comparison.

Only map inputs change. Their effects propagate through the existing FAS ray proposal and frequency-field updates; those two consumers are not changed independently. The original1495 scientific source files are byte-identical. A separate factory accepts the exact sealed.93 provenance value by delegating every other validation to the original.95 validator, then passes the actual unchanged spec into the original config builder and training loop. It does not relabel the maps or install model/sampler hooks.

## Prepared results

Current package: `C/frequency_HD63_trials_v3`, mirrored under the silver campaign root. `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`. Policy SHA256 is `08af9a3cceebe7c5a977e62417643b283575b1cf6cceaf8c1f054e3fe081897e`. All2305 remote source/data/script bindings pass. The two isolated datasets and six source/map reviews are documented in [the map activation report](marshawn_frequency_HD63_activation.md). Earlier generations remain preserved.

CPU dry-runs compare the actual inherited and proposed runtime configurations after explicitly normalizing only four path fields: output directory, datamanager data, dataparser data and model semantic data. The first generation's comparison failed because string substitution did not normalize PyYAML's component-list representation of pathlib paths. Its logs and package remain in `frequency_HD63_trials_v1`. Generation2 fixes only this CPU comparison; no scientific differences were tolerated and no GPU update occurred.

Generation2 passed configuration-only CPU checks and was root-released/launched. Both workers passed physical UUID/lease validation but stopped during model construction, before the first training iteration: the inherited admission receipt binds the entire old ready.json, so its strict dataset identity rejected the new map inventory. No progress, latest checkpoint or Adam update was produced; no OOM occurred. `frequency_HD63_trials_v2/startup_failure.json` and original logs/releases preserve this failure. Configuration equality alone did not cover the loader contract.

Generation3 copies each qualified admission source directory. Its semantic_grid.npz and all prior qualified artifacts remain byte-identical except grid_receipt.json, whose only changed field is source_ready_sha256. The derived manifest updates that receipt hash and binds an explicit lineage record with old/new ready hashes, original manifest, map assembly and unchanged nonfrequency inputs. No geometry, camera, mask, support or mapping is recomputed.

The actual frozen `load_support` now passes for the parent and both derived datasets; all2097152 mapped cells compare exactly equal. Source-grid SHA256 remains `e804d736dd766e8adf11df07e346a6317af87e0b106a94dfc524ab32f1690d71`; mapped boolean SHA256 remains `d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f`. Configuration comparison additionally normalizes only the proved-equivalent semantic manifest path/hash. Both normalized configs have SHA256 `17723ecbd12275b1f790d978aaf514803d5421db7b2ca35270b7a26b97e1a2d1`.

Current `CPU_complete.json` SHA256 is `7d7a58c7ecabfdc61910bf959c401ea1c47bef133ad0dbdafc2e306e01dd3af9`; factory SHA256 is `9bd5e734cb952147cb71ee7774362c3e0702aec6e3796f7f8646b822f13ab424`. All1495 science files remain unchanged. Root independently rechecked2305 bindings on each host and the actual loader/config proofs, then released and launched generation3. Root review SHA256 is `e17ed54f772a1a49f3cd48c71f653b5278da6b3148c2653a741592d3767c6cdf`; controller2148145, decay worker2782320/GPU4 and.93 worker2782549/GPU7. Both passed physical UUID checks and reached the training loop. `manual_supervision.jsonl` records continuing health checks; no restart or sampler reset was performed.

The unchanged loop has mandatory end106316 followed by its existing automatic plateau rule, no fixed iteration ceiling, and the original FR switch75941→.3. It saves the LPIPS.001-tie/PSNR frontier and resume state under the established checkpoint policy. A map change starts cold, including frequency-grid warmup; technical and older trained checkpoints are not ancestors.

The existing local-origin transport controller dispatches to silver GPUs4/7 using global leases. The factory verifies the actual supervising PID/start/boot and physical UUID before importing Torch, then checks the single visible logical CUDA0 and applies a10GiB allocator cap. Parent code and data hashes are checked before execution; original per-arm release verification remains active. Training is not authorized by dataset preparation or CPU proof alone.

After exact per-arm root releases, the existing launch entry is:

```bash
python3 /fsx/tmp/marshawn_001660_hd/tuning_20261106/transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py launch --root /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/controller --manifest /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/controller_plan.json
```

Root-release templates live separately under each arm. Their four inherited mechanism-proof flags are explicitly attributed to the unchanged reviewed quarter.1 parent, not new GPU tests. Root release must additionally bind the new policy, factory, exact spec and CPU proof. Hourly process/progress/GPU/OOM supervision and each required three-view visual gate remain mandatory after launch.

## First matched15188 gate

Both arms passed cold warmup4096 and all four Adam counters reached1998 in the saved attempt2000 checkpoint. Peak Torch allocation was approximately3.48GiB; no OOM or source/runtime changes occurred. Two durable CPU watchers preserve evaluated checkpoints remotely and fetch actual grids, metrics and renders to `C/independent_occupancy/frequency_HD63_{lr_decay,threshold093}_progress`.

| Map recipe | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Frozen baseline .95/constant .01 | 32.09120 | .854067 | .02219093 |
| .95/per-level LR decay | 31.98976 | .854178 | .02291078 |
| .93/constant .01 | 32.04205 | .851926 | .02239882 |

At this early matched gate neither new map improves actor LPIPS. All three source/actor panels and nine hair/hand/feet panels were visually reviewed: no gross new actor cuts, but substantial head/body halos and floor/mat/bag fragments remain in every arm. This is not a complete sole-opacity certificate. [Actor comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/reviews/step-000015188/all_actors.jpg) and [details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/reviews/step-000015188/all_details.jpg); visual receipt SHA256 `1656a661ad5b788c090490155139c46b672c27d9275c1fdbe38398386f35bb10`. Both trials continue under the unchanged stopping protocol.

## Matched30376 gate

| Map recipe | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Frozen baseline | 32.01860 | .847094 | .02053743 |
| LR decay | 31.89553 | .846629 | .02052348 |
| SSIM.93 | 31.92200 | .845573 | .02067298 |

All3 actor and9 detail rows were reviewed again. Garment structure and shoe shapes remain broadly comparable; there is no gross newly missing actor region at contact resolution, while strong halo and floor/mat/bag remnants persist. Decay differs from baseline by only−.00001395 LPIPS and has lower PSNR; this does not establish an improvement. [Actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/reviews/step-000030376/all_actors.jpg), [details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/reviews/step-000030376/all_details.jpg), visual receipt SHA `ee4f5c11201882393668fee9c6fc961462267780a83bf09066f463d25670ff6a`. Both workers remain healthy and continue. Saved alpha16 diagnostics outside a16HD-pixel Euclidean mask margin show decay has+4.65%/+4.29%/+2.49% alpha>.1 pixels in DC/HC/KA; .93 changes−1.05%/+.11%/+1.79%. Thus no consistent exterior cleanup is measured. These fixed image-mask margins are not physical empty-space labels; full0/3/8/16/32-pixel diagnostics and input hashes are in the gate exterior/receipt.json.

## Insights and limits

The source/map review supports a controlled experiment, not a predicted improvement. SSIM.93 coarsens actor map levels, while decay varies across materials and views. The legacy baseline is not a newly repeated fitter seed/runtime control. The first matched gate supplies early NeRF measurements, not a terminal map selection or geometry/cleanup promotion.

Reusable [assembler](../scripts/assemble_marshawn_HD_frequency_variants.py), [trial preparer](../scripts/prepare_marshawn_HD_frequency_trials.py), and [external exact-spec factory](../scripts/train_marshawn_HD_frequency_variant.py).
