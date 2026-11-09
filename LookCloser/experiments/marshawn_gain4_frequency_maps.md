# Marshawn1660 corrected-color frequency maps

## What was tested

Fresh train-only frequency maps are fitted for the immutable corrected gain4 SDR PNG dataset. All63 source hashes differ from the original gain1 PNGs, so the previous maps are excluded. After the scalar exposure candidate won the matched22k common-target visual/metric gate, a separate fresh63-map queue was authorized for scalar-normalized gain4 SDR. Its images also all differ from gain4; no map is reused across representations.

Both queues use the tested fitter SHA256 `6ef90cbbbd6f277886c3aaa2112d4b151f08c325c4ae5fb7b0b63da3d3b75bd7` and identical preprocessing source on local, DEV3, and clever. Parameters are explicit: SSIM0.95,1000 optimizer steps per level,16 levels, resolution16..8192, two features, patch8/stride8, evaluation patch batch1024. Only63 training images are fitted; the3 eval images are excluded.

## Results

| Source | Status | Strict source/recipe QA | Manual review |
|---|---|---|---|
| Gain4 SDR control |63/63 complete |63/63 passed |Every overlay on all3 pages inspected; passed |
| Scalar-normalized gain4 SDR |63/63 complete |63/63 passed |Every overlay on all3 pages inspected; passed |

[Gain4 completion receipt](/fsx/tmp/marshawn_001660_hd/frequency_gain4/complete.json), [source/recipe/statistics](/fsx/tmp/marshawn_001660_hd/frequency_gain4/review/strict_review.json), [manual review](/fsx/tmp/marshawn_001660_hd/frequency_gain4/review/manual_review.json), [all source/overlay pages](/fsx/tmp/marshawn_001660_hd/frequency_gain4/review/index.html).

Gain4 fits came from local24,DEV3 10,clever29 cameras. The average maximum-level fraction is34.38%; the largest fraction in any single level for any map is50.32%. All maps are finite126×240 scalar-resolution lattices, correctly aligned to1920×1012 sources. Uniform white softboxes generally occupy lower levels, while detailed people, rig and backdrop occupy higher levels. Alignment and recipe QA do not establish an absolute ground-truth spatial frequency.

Local fitting was paused with SIGSTOP to prioritize the primary physical-PQ8k visual gate, preserving the current EB fit and lock. Other workers use nonblocking per-image locks and continued. EB resumed with SIGCONT and completed the same16×1000 optimizer steps. Its wall time includes the pause. An initial DEV3 directory-permission failure occurred before fitting; only the newly created output directory was made writable, and its clean retry exited0. Both local workers and both clever relays exited0. These recoveries are documented rather than omitted from the provenance.

The scalar queue completed using one local worker, one DEV3 worker, and two clever workers. Each clever worker has a separate remote ownership file and holds the shared central per-image advisory lock through fit, validation, and atomic publication. Foreign GPU jobs are left untouched. Per-map source/fitter hashes, exact recipe metadata, and30-second process/GPU/OOM records are retained.

[Scalar input contract](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/input_contract.json), [authorization and matched-quality binding](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/promotion_authorization_receipt.json), [completed supervision](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/supervision_current.json).

## Insights

Color, exposure normalization and linear prefiltering change the fitting target; a matching filename or unchanged pose does not qualify an old frequency map. The strict wrapper requires every new source hash and exact SSIM/step/chunk recipe, with no historical-map exceptions. Existing gain1 maps and wrong-threshold archives remain separate.

All images leave four bottom rows outside the full-patch lattice. This means lack of FAS patch support on those rows; the whole-image uniform sampling branch still reaches them. The maps do not certify occupancy or correct hidden geometry.

Scalar fits finished without OOM or failed queue/relay controllers: local22,DEV3 11,clever30 cameras. The average maximum-level fraction is34.51%; the largest single-level fraction for any map is51.20%. All63 actual scalar input hashes, fitter source, exact threshold/steps/chunk, tensor lattice and finite values passed. Every source+overlay on all3 final pages was manually inspected; no alignment or borderstretch defect was identified. Both source variants are now complete, with no frequency GPU workers remaining.

[Scalar completion receipt](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/complete.json), [strict gate](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/review/strict_review.json), [manual review](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/review/manual_review.json), [all source/overlay pages](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/review/index.html), [download63 maps+sidecars+QA](/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar/download/marshawn_1660_scalar_SDR_frequency63.zip). The archive SHA256 is `d6ca3d047f228470cd73010837f08b3e9bd72cb78ef59421428b36c9461943d5`.

The scalar map gate applies exclusively to `gain4_scalar_trainfit_poseeval/sdr_png`; it does not qualify HDR, raw gain1, or the gain4 control source.
