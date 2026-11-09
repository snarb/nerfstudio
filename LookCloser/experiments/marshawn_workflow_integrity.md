# Marshawn supervision and artifact integrity

## What was tested

Cold corrected-PNG R0/R1 runs, external review gates and exact-checkpoint
read-only renderers. These checks concern workflow integrity, not clean-image
acceptance or physical occupancy certification.

## Results

- A naive `8000`→`12000` source substitution generated the invalid filename
  `step-0000012000`, missing both 12k review gates. R0 subsequently stopped by
  its default plateau rule; its conservative attempt upper bound is12261.
  R1 was paused after CSV12070, conservatively12081 attempts at that point.
  Both remained below22000. Existing checkpoints and renders were valid.
- `scripts/marshawn_external_checkpoint_gate.py` now formats integer steps with
  nine digits, validates PID birth and checkpoint/grid hashes, pauses the worker,
  and seals a separate read-only checkpoint copy. Checks include2k,8k,12k and21999.
  The corrected16k/20k R1 gates succeeded.
- A saved checkpoint number is not proof of the live optimizer boundary.
  Sparse500-step AMP traces must be considered alongside synchronously flushed
  10-step CSV progress. Continuing one existing worker does not replay its
  earlier overrun; resuming an older checkpoint would replay attempts and needs
  a reduced remaining budget.
- Native pre/post-eval saves can replace a checkpoint container at the same
  step. At8k, all model tensors, core and EMA matched exactly while the container
  SHA changed. Historical render hashes were retained; tensor equivalence and
  a new immutable source copy were recorded separately. No container equality
  or optimizer/RNG equality was inferred from model equality.
- Completed background agents do not start work when merely sent a message.
  This delayed the first bounded-HDR reader. New tasks must use `followup_task`;
  verify a live agent/process handle instead of treating a queued message as work.

## Insights and next steps

Prepare the first visual-gate renderer before launching a new training worker.
Archive its exact checkpoint before any CONT. For physical EXR models, native
RGB is linear: export PNG through Reinhard+sRGB exactly once, retaining raw HDR
buffers. An identity/clamp preview is invalid. Keep training-attempt accounting,
formal checkpoint selection and visual cleanliness as separate recorded claims.

R1 finished normally at its20k plateau, with conservative total upper20211;
the selected20k immutable model is tensor-identical to the reviewed20k gate.
Residual floor/mat mottling remained, so neither terminal success nor these
integrity checks establish completion of the image-quality goal.
