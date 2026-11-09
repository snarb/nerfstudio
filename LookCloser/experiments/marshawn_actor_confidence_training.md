# Actor-only confidence preparation

Training has not started. The user rejected the previous actor AABBs; those bounds and their training approvals are frozen. Final native hair/occlusion data, revised bounds, actor-only geometry and actual zero-update proof remain required.

What is implemented:

- `scripts/marshawn_actor_training_pipeline.py`: unchanged RGB3 targets, common GLOBAL-uniform W64 pixel proposal, fixed TRAIN-wide mean confidence, and one confidence factor in RGB and packed linear distortion. Hard/W32/W64 differ only in their confidence arrays. No alpha labels or target premultiplication.
- Native actor evaluator uses actual RAW-derived 8192×4320 RGB/K and renders only a separately frozen semantic actor domain. SSIM/LPIPS use complete foreground receptive fields; unavailable levels fail closed. Excluded people/background do not contribute.
- `scripts/run_marshawn_actor_cold.py`: seed42, attempt indices0–21999 including AMP skips, default all-three-actor geometric-MSE stopping, PSNR selection with LPIPS tie within0.07dB, checkpoint before evaluation, immutable2k snapshot and same-worker SIGSTOP before2001, health and crash capture.
- `scripts/stage_marshawn_actor_training.py`: preparation only, unique frozen helper copies and three gated configs. Requires an explicit revised foot-inclusive bounds source; never falls back to rejected subjectLOOSE. All production flags remain false.

Common proposed baseline is native NoApp0, hash23, maxres8192, fixed black background, core0 and world step0.0005. Step0.001 is a zero-update cost/coverage comparison only before ONE common setting is selected. Random prediction-only background would add expected residual `(1-alpha)^2/12` for RGB3 and therefore force opacity even in the uncertain weighted band. Fixed black avoids that particular constraint but retains RGB/opacity ambiguity.

Foreground-only RGB leaves other volume unconstrained. Geometry owns separately vetted actor SfM (at least3 actual TRAIN semantic observations) and a conservative semantic-absence candidate beyond256 native pixels, with other-person/UNKNOWN/invalid vetoes and body/hair conflict checks. Those are bounded assumptions, not scanner depth or automatic empty-cell truth. Historical fullscene MVS52/floor priors are excluded. Geometry auxiliaries will be identical across confidence policies; no production run may silently omit the reviewed common geometry policy.

Validation so far:10 CPU tests pass (confidence+pipeline contracts), public native-class YAML roundtrip passes, no actor model/optimizer/GPU forward constructed. Independent66-camera selected-identity review found and resolved FD grey-fragment selection and FD/ED target-leg veto errors. Its final receipt is `/fsx/tmp/marshawn_001660_hd/two_actor_isolation/independent_identity66_recipes/independent_manual_review_final.json`; this certifies identity/major visible body at contact resolution, not pixel-perfect hair, AABB adequacy or training release.
