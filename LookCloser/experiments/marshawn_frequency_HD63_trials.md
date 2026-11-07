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

Current `CPU_complete.json` SHA256 is `7d7a58c7ecabfdc61910bf959c401ea1c47bef133ad0dbdafc2e306e01dd3af9`; factory SHA256 is `9bd5e734cb952147cb71ee7774362c3e0702aec6e3796f7f8646b822f13ab424`. All1495 science files remain unchanged. No generation3 GPU worker has launched.

The unchanged loop has mandatory end106316 followed by its existing automatic plateau rule, no fixed iteration ceiling, and the original FR switch75941→.3. It saves the LPIPS.001-tie/PSNR frontier and resume state under the established checkpoint policy. A map change starts cold, including frequency-grid warmup; technical and older trained checkpoints are not ancestors.

The existing local-origin transport controller dispatches to silver GPUs4/7 using global leases. The factory verifies the actual supervising PID/start/boot and physical UUID before importing Torch, then checks the single visible logical CUDA0 and applies a10GiB allocator cap. Parent code and data hashes are checked before execution; original per-arm release verification remains active. Training is not authorized by dataset preparation or CPU proof alone.

After exact per-arm root releases, the existing launch entry is:

```bash
python3 /fsx/tmp/marshawn_001660_hd/tuning_20261106/transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py launch --root /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/controller --manifest /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v3/controller_plan.json
```

Root-release templates live separately under each arm. Their four inherited mechanism-proof flags are explicitly attributed to the unchanged reviewed quarter.1 parent, not new GPU tests. Root release must additionally bind the new policy, factory, exact spec and CPU proof. Hourly process/progress/GPU/OOM supervision and each required three-view visual gate remain mandatory after launch.

## Insights and limits

The source/map review supports a controlled experiment, not a predicted improvement. SSIM.93 coarsens actor map levels, while decay varies across materials and views. The legacy baseline is not a newly repeated fitter seed/runtime control. No new NeRF quality, geometry, footprint cleanup or convergence claim is made before the trials.

Reusable [assembler](../scripts/assemble_marshawn_HD_frequency_variants.py), [trial preparer](../scripts/prepare_marshawn_HD_frequency_trials.py), and [external exact-spec factory](../scripts/train_marshawn_HD_frequency_variant.py).
