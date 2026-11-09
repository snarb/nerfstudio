# Exact-forward LOS derivative and failure evidence

What was tested: the frozen LOS objective evaluates optical thickness correctly,
but its density activation uses the truncated-exponential surrogate derivative.
At raw log-density below −15 that derivative retains a floor while the real
exponential derivative vanishes. Actual canonical6k samples confirm the mismatch;
this does not establish the cause of the later lost-state numerical failure.

`scripts/marshawn_true_los.py` is a new opt-in plain adapter around the existing
validated geometry objective. It reuses its private step sampler, field query,
native density bytes, FP32 integral and metrics. A zero-valued differentiable
correction substitutes the true exponential derivative. Native cached density
logits and initialization scale are supported; the reviewed temporary raw-density
proxy supports the bounded frozen LookCloser field. Neither field implementation
nor the original objective is changed. No robust log1p penalty is installed.

Results: four CPU checks pass, including exact forward equality, finite-difference
gradients at logit−20/−40/−60, unchanged Torch RNG and schema, and injected failure
capture. The separate MVS candidate's actual64/adapted1M zero-update probe also
passed full-state restoration and finite combined AMP checks. This is preparation,
not authorization to train or a rendered-quality result.

Before the separately authorized launch, root corrected diagnostic serialization
to preserve dictionary key types and tuples. This keeps integer Adam state keys
and permits actual Python/NumPy RNG restoration. Four CPU checks passed again,
including `random.setstate` and `np.random.set_state` on the saved payload.
The active helper SHA256 is
`7ee4d98fa24bfec73de57a609481072bb75f1f714ee7d35d2c47b9e3b477104c`;
the earlier `true_LOS_capture_checks/receipt.json` records the preceding version.
The active-version addendum is
`true_LOS_capture_checks/receipt_post_root_type_hardening.json` in the MVS stage,
bound to `root_capture_type_hardening.json` and the actual source bytes.

`scripts/marshawn_crash_capture.py` installs explicit iteration-scoped hooks and
releases detached packet references after each attempt. On a training-iteration
exception it records the exact failed attempt, declared ancestry count, pre/post
Python/NumPy/Torch/CUDA RNG, actual sampled batch/rays, first main field inputs,
raw geometry/color outputs, field density/RGB and packed depth-renderer inputs.
It also saves pipeline, Adam, scheduler and scaler state, then re-raises the
original exception. The files live in `failure_diagnostics`, are explicitly
**not valid resume checkpoints**, and are excluded from checkpoint selection.
CUDA/context failures may prevent individual copies; the receipt lists omissions.
Eval/callback failures outside `train_iteration` are outside this hook's scope.

The new MVS stage uses parent14k full state, start14001/end18830 exclusive,
with conservative prior17,170 + at most4,829 new attempts =21,999. Existing frozen
stages and accepted downloads remain unchanged. No training was launched by this
implementation task.

Root subsequently reviewed and launched one MVS continuation controller1979063,
with its first visual gate at16k. `actual_launch_root.json` and
`independent_review_root.json` record that separate authorization. This records
an experimental launch, not a quality acceptance; no additional process was
launched during the receipt verification.
