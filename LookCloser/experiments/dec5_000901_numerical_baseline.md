# DEC5: numerical baseline and checkpoint-phase corrections

## What was tested

The packed distortion regularizer used global FP32 cumulative sums, then
subtracted the prefix preceding each ray. Thin surfaces have a small legitimate
within-ray contribution; subtracting a large preceding opaque batch loses that
signal. An analytic two-sample ray changes when4095 unrelated opaque rays precede
it: the nonnegative regularizer becomes negative and a weight-gradient sign
reverses. This is a reproducible implementation error, not a new hair prior.

The fix computes packed prefixes, per-sample terms and reductions in FP64, then
returns the original weight dtype. Dense fixed-warmup math and all hyperparameters
remain unchanged. Independent pairwise-oracle tests check values and gradients,
unrelated-ray invariance, ragged/empty rays and dtype preservation.

A separate reload audit found that the manually controlled FR1→FR.3 phase was
not represented by a tensor in old sequence checkpoints. Their YAML stayed at
the initial phase. The trainer now records runtime strength and declares the
schedule in saved config; public pipeline loading restores the correct phase.
The sequence export helper supports older sequence checkpoints using their
explicit `fr_switch_after` request, without changing legacy fixed-phase runs.

## Results

Eighteen focused precision/runtime/RNG tests pass, including actual tensor-state
pipeline reloads before and after the FR switch. A matched5000-update precision
continuation from the same15188 full-state parent has completed:

| Arm at20188 | Hair PSNR | Hair SSIM | Hair LPIPS | Boundary LPIPS | Face LPIPS |
|---|---:|---:|---:|---:|---:|
| Old packed math | 23.9461 | .77461 | .238265 | .109563 | .127544 |
| Corrected packed math | 23.9149 | .77433 | .236715 | .108750 | .127242 |

Hair LPIPS improves0.65%, boundary0.74%, face0.24%. This is a small continuation
effect, not a demonstrated substantial hair fix. Correctness invariants justify
the numerical repair; scene-quality benefit still needs the full recipe.
[Corrected continuation](assets/dec5_silver_sable/stable_distortion_continuation_hd.json).

A separate **cold full106317-update HD recipe** is running on
clever-shadow with the numerical fix and validated8192-ray evaluation batches.
These runs are not native50-frame production completions.

No hair-quality benefit is claimed from synthetic invariants. Their purpose is
to establish correct baseline math before evaluating new techniques.

## Insights and next steps

- Preserve the old full HD pilot as an explicitly labeled pre-fix control.
- Check whether the correction changes real hair/background quality, then
  validate the complete corrected recipe before native training begins.
- Verify a real late-FR checkpoint against its sealed saved render. Correct
  runtime restoration is necessary even when parameter tensors are identical.
  This gate now passes on the real91128 checkpoint: correct phase max RGB
  difference1.31e-6, versus.2876 with initial FR1. The old1055 floor benchmark
  uses a pre-switch5000-step field and is unaffected by this phase issue.
  [Real roundtrip](assets/dec5_silver_sable/late_fr_checkpoint_roundtrip.json).
- Use `dec5_sequence_checkpoint.py` for serving the field with its hash-matched
  external background and correct FR phase. Bare field loading omits that
  background; the adapter avoids GPU optimizer restoration.
- Productionv6 now includes the correctness fixes. It preserves the old sampler's
  exact default pixel indices across warmup, ramp, mature and late steps; actual
  fixed/adaptive synthetic updates and full-state restoration also pass.
- New matched5000-update ablations start from the corrected cold15188 parent:
  control, hair-focused core/FAS mixture, and a separate relaxed frequency-update
  opacity gate. The hair mixture changes ray allocation11.3%→37.9% while keeping
  whole-image draws20%; this is not an accepted quality result yet.
- Numerical diagnostics do not establish millimetric hair depth or physical
  temporal stability. Final quality reporting uses PSNR, SSIM and LPIPS.

Artifacts on clever-shadow:
`/home/ubuntu/lookcloser_transfer/recipe_pilot_20261028/frequency_depth_stable_distortion`
and `stable_full_v6`; immutable corrected code is in `code_stable_v6`.
