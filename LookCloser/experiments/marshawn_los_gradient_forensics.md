# Read-only LOS gradient forensics

What was tested: the actual preserved canonical HDR6k field and the frozen
128×64 observed-train free-space samples at step6000. There were no callbacks,
optimizer updates, RGB fitting, field replacements or occupancy changes. The
existing field evaluates density as `exp(h0+1)` but its truncated-exponential
backward uses `exp(clamp(log_sigma,-15,15))`.

Results: all8192 actual free samples are finite. Raw h0 spans−184.125…−5.8633
with median−19.71875;88.208% of samples have log-density below−15. Those clear
samples contribute0.712% of the forward integral, but7.289% of the native
point-gradient L1. The true exponential derivative assigns0.712%, consistent
with the forward contribution. Independent finite differences at log-density
−20/−40/−60 match the true derivative; native STE exceeds it by148×,
7.2×10¹⁰ and3.49×10¹⁹ respectively.

| Actual mixed-precision gradient measurement | Scale32768 | Scale65536 |
|---|---:|---:|
|Native low-log-density component NN norm / total native norm|0.186%|11.600%|
|Parameters with nonzero native low-component gradient|2,907|248,564|
|Parameters with nonzero true low-component gradient|0|10,832|

At scale65536 the low-component native norm is about47 times the corresponding
true norm. Total native LOS gradient norm is about7.5% above true. A scale1
probe gives zero NN gradients for all compared terms because the gradients
underflow while returning through FP16 TCNN outputs; such an unscaled test
would miss the actual mixed-precision behaviour.

The uninstalled [prototype](../scripts/marshawn_true_log_los_prototype.py)
exposes the existing differentiable raw geometry output with the reviewed
temporary density proxy. `exact_tau` preserves the integral's forward value
and uses its true derivative. Explicit `robust_log1p_tau` evaluates
`softplus(logsumexp(log_sigma)-log(S)+log(length))`; its log-density derivative
is bounded and vanishes in clear space. The robust mode deliberately changes
the penalty for large optical thickness and is not presented as an identical
objective. Both produce finite actual6k gradients; their current small-tau
forward values are almost identical. Four independent CPU integral,
finite-difference, sample-count and extreme-value tests pass.

Insights: the free-space auxiliary has a measured surrogate-gradient mismatch,
including nonzero gradients on many already clear parameters. This does **not**
prove that it caused the later training failure: the failed7k state and batch
were not saved. No corrected objective was installed in a production model.
Any future trial still requires exception-time model/batch/RNG/activation dumps.

[Actual report](/fsx/tmp/marshawn_001660_hd/canonical_LOS_gradient_forensics_STAGED/report.json) ·
[Exact sample/value arrays](/fsx/tmp/marshawn_001660_hd/canonical_LOS_gradient_forensics_STAGED/actual_free128x64.npz) ·
[CPU checks](/fsx/tmp/marshawn_001660_hd/canonical_LOS_gradient_forensics_STAGED/CPU_tests.log).
