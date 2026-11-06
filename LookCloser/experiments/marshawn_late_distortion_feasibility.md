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

Paired technical overlap is prepared separately at
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/late_distortion_overlap`.
The frozen fork code and all six specs remain unchanged. The supervisor requires
an exact policy release and four technical-only child releases before GPU work.
It serializes two zero-state proofs and each arm's 32+32-attempt Adam/resume check
on silver GPU2, preserving the producer's global lock. Limits are 900 seconds,
10 GiB Torch allocator and 12 GiB candidate process memory; producer identity,
progress age and OOM evidence are checked every five seconds. Failure terminates
only its own process group. Nine CPU scope/release/state-gate tests pass. No GPU
technical or 4000-attempt diagnostic launch is implied by this preparation.

The root-released technical gate passed in 220.73 seconds on silver GPU2.
Peak candidate process memory was 6,990 MiB, within the 12 GiB cap. Both zero
proofs matched restored pipeline, optimizer, scheduler, scaler, RNG and paired
reconstructed cache. Each coefficient had actual Adam steps
`60723 → 60755 → 60787` across the first 32 and resumed 32 attempts; final global
step was 60816. Initial rays and both later sampled-stream digests matched;
no sampling policy was changed to enforce this. Historical cache order remains
unobserved, so this is controlled paired-fork evidence, not a claim of observed
historical stream continuity. Producer PID487868 advanced 105377→110387 with no
OOM, and all technical candidate GPU processes exited. Receipts and small
state proofs are in `late_distortion_overlap/evidence` under the campaign root.
The 4000-attempt diagnostic remains unreleased.

After this successful technical review, root separately authorized the two
4000-attempt diagnostic specs. Controller2076985 queues both on exclusively
idle silver GPU locks, fresh from source60752. It does not reuse the technical
60816 checkpoints. CPU watcher2077216 prepares source-hashed GT/parent/control/
treatment comparisons for DC/HC/KA and both actor crops at61252 and64752, with
unchanged V1 per-view/per-actor metrics. The diagnostic remains a visual review
gate, with no automatic geometry approval or optimization-complete claim.

## Matched4000-attempt result

Both diagnostic arms completed without OOM. Control had3999 actual Adam
updates and treatment3998 across4000 attempts; native AMP was preserved.
Initial pre-update rays and the complete sampled4000-attempt stream matched
between arms. This does not change the historical cache-order caveat.

| Attempts | Control PSNR | Control SSIM | Control LPIPS | Treatment PSNR | Treatment SSIM | Treatment LPIPS |
|---:|---:|---:|---:|---:|---:|---:|
|500|27.0045|.793477|.054457|30.7605|.830997|.035832|
|1000|26.9082|.792273|.054750|30.8301|.832285|.036386|
|2000|26.8969|.792599|.054545|30.7421|.832623|.036193|
|4000|26.8650|.792562|.054859|30.6608|.833477|.037054|

The global improvement is supported by the matched control, but it is not
uniform by view: at4000, KA LPIPS worsens .024687→.034549 and PSNR31.520→30.500,
while DC LPIPS improves .072454→.043710 and HC .067437→.032903. The treatment's
raw LPIPS minimum is500; applying the existing .001-window/PSNR selector among
these four diagnostic gates would select1000. Neither is automatically promoted.

All3matched500/4000 view/actor contacts and source hashes are saved under
`late_distortion_visuals/step-000061252` and `step-000064752`. Visual inspection
confirms clearer clothing/hair/silhouettes in DC/HC; background haze becomes
sharper floating clumps and floor/boundary artifacts remain. This is not an
ideal-grid result. The independent103TRAIN+12DC geometry audit was assigned
before any geometry claim. The paired final checkpoint hashes and actual-Adam
receipts are in `late_distortion/diagnostic_completion.json`.

These results apply to the full-objective exponential parent60752. A late
factor on semantic actor training requires a new exact source-bound fork;
this causal conclusion must not be transferred to that ancestry implicitly.
