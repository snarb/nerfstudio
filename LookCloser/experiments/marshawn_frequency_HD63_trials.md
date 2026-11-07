# Cold HD frequency-map trials

## What was tested

Prepared two cold NeRF trials using the validated63-TRAIN map inventories: SSIM.95 with per-level LR decay and SSIM.93 with constant fitter LR. Both start from seed42 and the same quarter-step.0015625, distortion.1 actor recipe, original W64/body80 confidence, R0, mapped semantic admission, legacy projection and original learning-rate/FR schedules. No warm checkpoint or pre-existing frequency grid is loaded. The frozen baseline uses SSIM.95/constant.01 maps and remains a separate archived comparison.

Only map inputs change. Their effects propagate through the existing FAS ray proposal and frequency-field updates; those two consumers are not changed independently. The original1495 scientific source files are byte-identical. A separate factory accepts the exact sealed.93 provenance value by delegating every other validation to the original.95 validator, then passes the actual unchanged spec into the original config builder and training loop. It does not relabel the maps or install model/sampler hooks.

## Prepared results

Package: `C/frequency_HD63_trials_v2`, mirrored under the silver campaign root. `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`. Policy SHA256 is `21dced3de3b420abb68cb27471597a1d4f0afc3a5534a9f66d70be407af256d1`. All2285 remote source/data/script bindings pass. The two isolated datasets and six source/map reviews are documented in [the map activation report](marshawn_frequency_HD63_activation.md).

CPU dry-runs compare the actual inherited and proposed runtime configurations after explicitly normalizing only four path fields: output directory, datamanager data, dataparser data and model semantic data. The first generation's comparison failed because string substitution did not normalize PyYAML's component-list representation of pathlib paths. Its logs and package remain in `frequency_HD63_trials_v1`. Generation2 fixes only this CPU comparison; no scientific differences were tolerated and no GPU update occurred.

Both actual CPU dry-runs now pass. Their normalized configuration SHA256 is identical: `2b53babd3590de563d5ad581313892177baca66ca26b519712648768c0d31ec6`. `CPU_complete.json` SHA256 `3998499ea2a3a495f2187af754612fbceb5445d3e087ccb30584056c09e19568` binds the two proofs, policy, source-map root review and all2285 staged bindings. The factory hash is `b81944ad40c854e2670741f2c6d18d2e4787a57fe1531845b46ee70f87661d0c`. Both confirm no CUDA initialization, no loaded checkpoint and seed42; no NeRF training has launched.

The unchanged loop has mandatory end106316 followed by its existing automatic plateau rule, no fixed iteration ceiling, and the original FR switch75941→.3. It saves the LPIPS.001-tie/PSNR frontier and resume state under the established checkpoint policy. A map change starts cold, including frequency-grid warmup; technical and older trained checkpoints are not ancestors.

The existing local-origin transport controller dispatches to silver GPUs4/7 using global leases. The factory verifies the actual supervising PID/start/boot and physical UUID before importing Torch, then checks the single visible logical CUDA0 and applies a10GiB allocator cap. Parent code and data hashes are checked before execution; original per-arm release verification remains active. Training is not authorized by dataset preparation or CPU proof alone.

After exact per-arm root releases, the existing launch entry is:

```bash
python3 /fsx/tmp/marshawn_001660_hd/tuning_20261106/transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py launch --root /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v2/controller --manifest /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_trials_v2/controller_plan.json
```

Root-release templates live separately under each arm. Their four inherited mechanism-proof flags are explicitly attributed to the unchanged reviewed quarter.1 parent, not new GPU tests. Root release must additionally bind the new policy, factory, exact spec and CPU proof. Hourly process/progress/GPU/OOM supervision and each required three-view visual gate remain mandatory after launch.

## Insights and limits

The source/map review supports a controlled experiment, not a predicted improvement. SSIM.93 coarsens actor map levels, while decay varies across materials and views. The legacy baseline is not a newly repeated fitter seed/runtime control. No new NeRF quality, geometry, footprint cleanup or convergence claim is made before the trials.

Reusable [assembler](../scripts/assemble_marshawn_HD_frequency_variants.py), [trial preparer](../scripts/prepare_marshawn_HD_frequency_trials.py), and [external exact-spec factory](../scripts/train_marshawn_HD_frequency_variant.py).
