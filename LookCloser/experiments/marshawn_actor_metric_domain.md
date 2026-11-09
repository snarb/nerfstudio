# Two-actor evaluation domain

What was tested: foreground metrics must exclude other people and background,
including indirect contamination through perceptual receptive fields. The helper
`scripts/marshawn_actor_metrics.py` takes a frozen boolean semantic mask, display
RGB, and a cached eval-mode AlexNet LPIPS model. It neither changes RGB nor selects
the mask using predictions.

Results: three CPU integration tests passed in 4.42 seconds. Independently changing
both prediction and target RGB outside the mask left PSNR, SSIM and LPIPS unchanged;
damaging admitted actor pixels worsened all three. Full-image support matched
standard TorchMetrics AlexNet LPIPS. A thin mask correctly returned unavailable
LPIPS when deeper feature footprints lacked support.

Insights: PSNR covers all admitted pixels, including hair/limb boundaries. SSIM
and LPIPS describe complete interior footprints and report coverage explicitly.
They cannot certify omitted boundaries, and thin or small HD actors may require
higher-resolution evaluation to obtain a complete LPIPS score. These are actor-domain
metrics, not replacements for previously reported whole-image results. Actor masks,
training integration and actual actor-only model quality remain pending.

[Test log](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/actor_metric_tests_root.log).

DEV3 independently passed the same background-invariance, full-LPIPS parity and
thin-support checks on Python3.8.20/Torch2.4.1. Outside-RGB score deltas were exactly
zero; full-domain LPIPS differed from its native reference by8.73e-11. Its environment
has no pytest, so this was a standalone assertion runner; the initial missing-pytest
log is retained separately. [DEV3 proof](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/actor_metric_DEV3_standalone_root.log).

The provisional GE SAM3 pair has1,214,583 pixels in its native3542×4221 crop;
all five AlexNet levels have valid interior support (deepest1570 features).
An approximate4.27× preview reduction has no valid features at levels2–4.
This is support analysis only, before native boundary acceptance or the actual
RAW-to-undistorted mask remap; it is not a quality score. It motivates evaluating
native actor rays instead of silently weakening LPIPS. [Support counts](/fsx/tmp/marshawn_001660_hd/two_actor_isolation/actor_metric_GE_provisional_support_root.json).
