# Marshawn independent-frame video campaign

## What was tested

The target is 72 independently initialized models at source frames 1660–2015,
step 5: 24 fps for the interval [0,3) seconds at source 120 fps. Global source
frame IDs remain the directory keys so a future 60 fps campaign can add step-2
frames without renaming or borrowing neighboring models.

All 66 physical cameras train. DC/HC/KA are overlapping **in-sample monitoring**
views; their PSNR/SSIM/LPIPS are not held-out generalization measurements.

The canonical model implementation and recipe are copied from
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_quarter_distortion_candidate_20261107`.
The factory verifies its 1495-file source manifest before copying. It retains
the actor objective, body80 sampling, W64 confidence/support, radius 0 native
occupancy, coarse step 0.0015625, distortion final 0.1, seed 42, SSIM frequency
threshold 0.95, corrected SH, trunc_exp density, legacy frequency projection,
200000-step LR decay, full historical ancestry and native FR switch. There is
no new iteration ceiling. The existing plateau worker controls stopping.

Adaptations are operational: all66 dataset guards; explicitly overlapping
monitor membership; frame-local conservative image-derived admission;
campaign release verification; read-only cold initialization assertions;
explicit visibility handling; and retention of distinct output selection
checkpoints. Factory AST checks prove the scientific config matches the
canonical worker and the train/evaluate functions match after reversing their
enumerated operational/visibility edits. Present-actor crop/metric math remains
unchanged; the complete six-pair monitoring case calls the original aggregator.

Only reviewed known absence permits an empty actor mask and null crop. UNKNOWN
identity/visibility pauses preparation. An absent pair is excluded explicitly;
each fighter must remain visible in at least one monitoring camera. For partial
coverage, visible views average within each actor, then the two actors average
equally. Receipts record visible pair count/camera sets, and metrics from frames
with differing visibility are explicitly not comparable. DC/HC/KA stay the
default; alternative TRAIN monitoring cameras require a reviewed evidence
reference in the prepared input. CPU tests cover absence/UNKNOWN and exact
complete-pair equivalence.

Each frame builds independent frequency maps from that frame's RGB. Reusing
source-verified image inputs at the same source frame is allowed. Neighboring
weights, Adam/scaler/scheduler state, learned occupancy/EMA, sampler/RNG state,
or frequency-fit optimizer state are not inputs. A crash can resume only from
the same frame/output, with exact input/code/spec/checkpoint hash bindings.

## Results

CPU verification passed on 2026-11-07:

| Check | Result |
|---|---|
| Canonical source inventory and factory AST parity | Passed |
| Foreign-frame checkpoint / `..` traversal rejection | Passed |
| Retry forbidden without fresh remote death evidence | Passed |
| Retry forbidden while worker survives supervisor | Passed |
| SSH UNKNOWN does not authorize retry | Passed |
| Confirmed-dead retry creates a new attempt on same frame/host | Passed |
| Controller restart adopts the existing attempt | Passed |
| Raw actor LPIPS differs correctly from historical .001 tie selector | Passed |
| Python compile | Passed |

CPU-only factory evidence is at
`/fsx/tmp/marshawn_video_factory_cpu_check_20261107/factory_proof.json`.
This is a factory check artifact, not the final campaign release: helper files
were still being implemented when it was created. Final staging must freeze
the complete source package again. No training quality is inferred from CPU
checks. Actual cold/resume GPU qualification and frame input qualification
remain required before a production release is written.

Training renders/PSNR/SSIM/LPIPS will be linked from each frame's
`frame_completion.json`; this document does not claim completed models.

## Insights and operation

`marshawn_video_campaign.py` wraps the existing fleet controller rather than
introducing a second GPU lease protocol. It uses the same per-GPU global locks,
SSH probes, PID/start-time/boot identities, detached supervisors and per-job
progress/GPU/OOM receipts. A separate detached watchdog restarts a dead queue
dispatcher and adopts live workers. A failed attempt can retry only after a
fresh host probe proves both supervisor and optimizer are dead. The retry gets
a new immutable attempt name but the same frame directory and host, enabling
verified same-frame resume. Unreachable hosts remain UNKNOWN.

Preparation runs chronologically on olive GPU0; qualified RGB/semantic inputs
are then pushed to silver-sable. The staging adapter dereferences RGB symlinks,
verifies every immutable input remotely, and commits a central dispatch marker
only after verification succeeds. Frames without that marker claim no training
GPU. Each admitted silver-sable frame runs source-bound frequency fitting →
final input inventory → fresh admission grid → cold training; there is no
barrier waiting for frequency preparation of every other video frame. The precise commands
are immutable argv arrays in campaign `pipeline.json`, so there is no shell
interpolation. Stage validators run again on restart; file existence alone is
insufficient. Controller and supervisors record progress, GPU/OOM evidence
and free disk space every 30 seconds by default, satisfying the mandatory
hourly supervision cadence while the detached campaign runs.

The queue plan sets `prepared_dispatch_marker` to the central
`frames/{frame}/dispatch_ready.json` pattern; `{frame}` expands to a six-digit
global source ID. Marker host identity must match the frame's assigned host.
`marshawn_video_stage.py push` publishes inputs and `verify` rechecks them.
`marshawn_video_frequency_frame.py` creates one immutable spec per physical
camera and runs the canonical per-image frequency worker in separate processes,
then seals `frequency_ready.json` only after all 66 maps validate. Its locks are
per filesystem; using a second host to fit maps on private NVMe requires an
additional remote claim-and-copy protocol, not an assumed shared local flock.

The default render checkpoint is the **raw minimum actor LPIPS** with PSNR
then earlier step as exact ties. The historical .001-LPIPS/PSNR selector is
retained separately. A PSNR selector with a 0.07 dB LPIPS tie window is also
retained separately. The pruner protects all three output choices, latest
resume, the FR boundary and the historical moving frontier.

The shared renderer loader is
`marshawn_video_train.load_pipeline(frame_root, checkpoint_kind='actor_lpips', device='cuda:0')`.
It validates the sealed code/checkpoint and returns `(pipeline, binding)` with
checkpoint hash. Completion receipts carry the exact config, checkpoint path,
metrics and loader arguments. An optional `after_training_command` lets the
durable campaign continue into rendering after every model is complete.

CPU checks can be repeated with:

```bash
python scripts/marshawn_video_selftest.py
python scripts/marshawn_video_freeze.py --root /path/to/new/campaign
```

The queue commands are `prepare`, `launch`, and `status`; launch uses a sealed
`video_plan.json` with hosts/GPUs/frames. Production training additionally
requires campaign `release.json` with `production_enabled: true`,
`all66_training: true`, and the exact `code_seal_sha256`, written after concrete
cold/resume qualification. This gate is technical qualification, not a request
for another user authorization.

## Production launch, 2026-11-07

The 72-frame controller/watchdog was launched at 12:47 UTC. Frame1660 entered
full cold training on silver GPU0 at 12:49 UTC, loading all66 cameras/maps.
The recorded cold audit has zero occupied cells, zero nonzero EMA cells, empty
Adam state and start step0. This launch is not a completed quality result.

Both silver-sable and dev3 passed actual cold/pause/resume GPU qualification:
warm0–8 →9–31 and cold ARM4096–4104 →4105–4127. Expected pause exit143 and
successful resume exit0 were observed in both phases on both hosts. These are
runtime checks, not comparisons of training quality. CPU checks cover lease
inheritance after both parent processes exit, same-frame ownership, live-worker
and SSH-UNKNOWN retry fences, and all66 frequency source/provenance checks.

Campaign: `/fsx/tmp/marshawn_video_1660_2015_24fps_20261107`.
Its `release.json`, `code_seal.json`, `factory_proof.json`, `video_plan.json`,
`pipeline.json`, `qualification_silver.json` and `qualification_dev3.json` bind
the release. The immutable production package has1530 source files; its seal
SHA256 is `54cbb1723a2fe41149f829c3a51dbfaa8721080eea4948fab86f42a602de5b6c`.

The conservative common3second box is min[-.91,-.26,-.08],
max[.15,.50,.63], in the unchanged floor frame. See
[the saved box](../configs/aabb_review/marshawn_1660_2020_3s_floor.json) and
`tail_native/review_v3_sixview_floor_cleanup/` within the campaign for source,
box and depth-proxy reviews at1660/1900/1960/2020. The grey actor lies on the mat
at the end, so mat intersection is necessary for actor containment. This is not
a certificate of perfect physical occupancy or of the minimum possible box.

Operational preparation and dev3 frequency assistance live under campaign
`operations/`, outside the sealed model code. The producer received a documented
startup-only variable-shadow/path-adapter correction before processing frames;
its original source/photometry/rig checks remain intact. Dev3 computes six
reserved same-frame maps before verified silver delivery; silver then computes
the other60. No two hosts concurrently fit the same reserved map. All72 final
models remain on silver, preserving native checkpoint/config paths for rendering.

At launch no new frame model or final video had completed. The supplied TIFFs
end at2020; actual GT2025–2255 for the requested extra2seconds remains missing.
The export pipeline can publish two3second neural videos while waiting for
those actual photographs. No synthetic continuation has been authorized.

A subsequent CPU packaging check found that the manual ray-audit helper compiles
`marshawn_semantic_ray_support.cpp` via its adjacent pathname, which Python
import-closure inspection alone missed. The repository factory now includes and
hashes that required source asset for future freezes. The running production
package/model seal was left unchanged; this fix concerns the optional CPU audit
package, not field training or checkpoint math.

The first new-frame native review also found a partial crew merge in MC: the
grey actor's global person instance included the bald helper's head. Correct
identity selection and small contact sheets were insufficient to certify clean
mattes. Frame1665 field dispatch was held before any training; fresh native1008
interactive head/visible-shoulder masks with actor-negative prompts supply a
reviewed current-frame crew veto. The original masks and correction receipts
remain available. Add-only hair refinement cannot remove this false foreground.
All current person candidates and persistent identity/UNKNOWN guards are being
qualified before the rest of the input pipeline is released. Existing frame1660
training uses the prior reviewed masks and continues independently.


### Supervised continuation (2026-11-07 15:11 UTC)

Frame1660 and1665 are training independently from cold state on silver. Both
include ALL66 train cameras; DC/HC/KA scores are in-sample monitoring only.
Frame1665 was released only after current-image crew correction, all66 fresh
MAT/BAG visibility layers, final frequency inventory and the original admission
loader passed. Its four shoe proxy cells and921 sampled positive boundary/core
rays survived the conservative admission grid (zero failing sampled rays).
This sampled evidence does not certify every physical surface cell.

For frame1670, current-image all-person proposals recovered two omitted black
actor lower-leg components in JE. Add-only repair preserved existing actor,
crew-veto and UNKNOWN arrays. A95% current/prior same-actor upper-body anchor
qualified these fragments and rejected the earlier MC crew merge. Independent
geometry UNKNOWN and ambiguous people remain conservative; previous-frame
identity proposals never initialize model weights or learned grids.

The full native producer now runs under a durable watchdog. The CPU source
service owns decoding1675–2015; the producer adopts its exact source receipts.
Dev3 computes six reserved same-frame frequency maps, excluding those cameras
from silver fitting. The previous late-push frequency helper was retired before
activating the source-bound admission-safe dispatcher. Future dispatch must wait
for the final identical LOCAL/silver ready hash, current MAT/BAG visibility,
frozen admission loader and positive-source witnesses. No input mutation is
permitted after dispatch.

The optional CPU audit package fix passed306 independent synthetic ray/DDA
crosschecks using the compiled source in the new factory snapshot; proof:
`/fsx/tmp/marshawn_video_factory_native_asset_check_20261107/cpu_ray_test/verification.json`.
The active1530-file production model snapshot remains unchanged.
