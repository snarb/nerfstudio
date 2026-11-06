# Marshawn: matched late distortion continuation feasibility

## What was checked

The Luster accepted sequence started its density compactness intervention from
checkpoint106316. Its actual `train_face_core.py` immediately changed the
existing distortion coefficient0.01→1, then ran4000 training attempts to110316
with Adam, scheduler, AMP scaler and RNG resumed. It did **not** ramp that
coefficient over4000 steps. FR remained0.3. Later face ROI sampling and occupancy
halo were separate interventions; they are not part of this proposed pair.

Evidence: [historical recipe](luster_000540_face_fix_occupancy.md), lines17–22 and
30; actual script `/fsx/tmp/luster_000540_leader_mask_ablation/scripts/train_face_core.py`,
SHA256 `3cbcf9074e81a1092ea82f6d9113c28c50d8fb4f4ac4f704feb27ba741a6ff1d`.
The earlier Marshawn cold strong-distortion arm does not test this late
continuation hypothesis.

## Actual source and compatibility audit

The completed structural exp(h+1), corrected-SH, full/body80/R0 run selected
checkpoint60752. Its fixed v1 metrics are PSNR26.87955, SSIM0.792158 and
masked actor LPIPS0.0540711. The checkpoint exists on silver and has SHA256
`edb8a0a7d88bc932b6bbf944e97dda98bd29b6f64d0a37e41c1a121f375e5109`.

The CPU checkpoint audit confirms pipeline weights/buffers, Adam60723,
scheduler state with LR0.0024704154798885205, AMP scale16384, Python/NumPy/Torch
CPU/CUDA RNG, body80 sampling state and actual occupancy capture. Sampler
sample_count is60753; the discrepancy from Adam is native skipped AMP updates,
which must remain explicit. FR is still1 because the absolute switch is75941.
The proposed60753–64752 continuation therefore preserves FR1; it is not a
literal reproduction of Luster's FR0.3 context.

The existing frozen worker only resumes its own matching spec/code/data hashes.
It has no external-fork entrypoint. A child must use an explicit isolated fork
bootstrap with source lineage, while leaving ordinary resume checks intact.
Framework checkpoint loading already supports full optimizer/scheduler/scaler
and pipeline restoration; its RNG must be restored after sampler setup.

Private image-cache IDs are not serialized in the source checkpoint. The
complete63-camera cache is recreated with Python `random.sample` during seeded
setup; body80 uses the global NumPy RNG and immutable source pools. Before
launch, the fork must verify identical cache camera ordering/source association,
sampler counter restoration and bucket reconstruction. This audit has **not**
yet run a reconstructed trainer or proved cache parity. It is a required gate,
not an assumed consequence of loading model weights.

## Concrete proposed pair

Both arms start from exactly the same selected60752 checkpoint, data, config and
RNG; run4000 attempts60753–64752; and evaluate identical gates61252,61752,62752,
64752. The control keeps distortion0.01; treatment sets1 immediately. Everything
else, including LR, density activation, SH mapping, body80, FR, grids, AABB and
the v1 actor evaluation protocol, remains identical. No LR reset, occupancy
reset, dilation, new prior or source overwrite is included.

Record actual Adam counts and initial/matched ray-index digests. Compare parent,
control and treatment renders, actor PSNR/SSIM/LPIPS, fixed-ray depth mass and
native occupancy diagnostics. This4000-attempt endpoint is a diagnostic visual
gate, not a claim that quality optimization has completed. Only a reviewed
improvement would justify matched continuation toward plateau; an intermediate
coefficient0.1 would be a separate later experiment.

Artifacts are in
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/late_distortion_feasibility/`:
`source_checkpoint_audit.json`, exact historical script, `audit_source.py`, and
`proposed_plan.json`. **No continuation worker or training has been launched.**

Distortion regularizes sampled termination mass. It cannot by itself certify
empty space or repair density hidden by false-empty occupancy cells; dense-grid
and native-grid tracing remain separate checks.

## Isolated fork prepared for review

The candidate now exists at `tuning_20261106/late_distortion` with a separate
`late_distortion_code` snapshot. All1485 original files remain byte-identical;
only the explicit fork worker and technical sequence runner are added. The
selected2.094GB parent checkpoint is copied and SHA-verified on silver, leaving
the source run untouched. Six source-bound specs and both controller manifests
are staged; all releases remain absent and no GPU fork test has run.

Eleven CPU tests passed, including real Adam/scheduler moment continuation,
individual pipeline/optimizer/scheduler/scaler/RNG mismatch rejection, cache
ordering, refusal before importing training code without a release, and paired
sampling comparison. The normalized config difference is exactly
`pipeline.model.distortion_loss_mult:0.01→1`.

The technical plan first creates two zero-update state proofs. It compares all
serialized training state and reconstructs the current seeded63-camera cache,
checking every cached RGB/validity row against its dataset source. **Historical
cache IDs were not observed in checkpoint60752:** this proves mutual/current
setup consistency, not independently witnessed historical bitwise sampling.
The proofs explicitly retain `historical_cache_order_observed:false`.

Both zero proofs must agree before either technical continuation executes.
Each arm then runs32 attempts, saves, exits, resumes the full checkpoint, and
runs another32. Actual Adam progress is required in both segments. Initial
pre-update sampled rays must match between arms; later stream digests are
recorded but legitimate adaptive/RNG divergence is reported rather than
artificially prevented. No sampling policy is changed to force agreement.

The separate4000-attempt diagnostic plan remains unreleased. Its endpoint is a
visual/depth review gate with the original v1 metrics, not automatic promotion
or a declaration that quality has saturated. Ordinary worker/resume code is
unchanged; child resumes enforce their own source/spec/code lineage.

After root releases the technical specs and allocates a suitable slot:

```bash
C=/fsx/tmp/marshawn_001660_hd/tuning_20261106
"$C/python" "$C/transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py" launch --root "$C/late_distortion_technical_controller"
```

After both technical arms finish, the CPU pair auditor is
`run_marshawn_late_distortion_technical.py --audit-pair CONTROL_SEQUENCE.json TREATMENT_SEQUENCE.json`.
This command does not release or launch the4000-attempt pair.
