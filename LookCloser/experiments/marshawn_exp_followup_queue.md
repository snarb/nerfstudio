# Marshawn corrected-SH exponential follow-up queue

## What was tested

Prepared five cold one-factor arms relative to the existing corrected-SH,
`exp(h+1)`, full/body80/R0, seed42 recipe. Preparation does not authorize training.
The parent is `structural_exp_hplus1`; it must finish and receive depth-audit
review before this queue can launch.

| Arm | Only scientific change |
|---|---|
| actor | Full objective → existing actor confidence weighting, width64 |
| lr50k | Exponential LR decay duration200000 →50000 |
| step_half | Coarse step .00625 →.003125 |
| jacobian | Legacy → existing normalized_jacobian mode |
| r1 | Occupancy dilation radius0 →1 during cold training |

All arms preserve mandatory106316 steps followed by the existing plateau rule,
fixed v1 actor evaluation, source RGB, masks, AABB and frequency maps. No frequency
threshold sweep is included. The strict validator rejects combinations and
additional knobs. R1 must train with the expanded grid; post-training dilation
can expose density in cells that were skipped during training.

## Results

Nine CPU tests passed. Every arm has exactly one normalized runtime-spec change;
the four non-objective arms also have exactly one TrainerConfig change. Actor
weighting is installed by the existing runtime sampling adapter, so its
TrainerConfig remains identical. The control config matches the parent exactly.

The actual native scheduler yields LR.001 at25000, .0001 at50000 and retains that
floor at106316,300000 and1e9, including after scheduler-state restoration. No
new scheduler implementation was introduced. Original decay yields.001 at100000.

Silver staging verified1485 source files,11 specs and the sealed data receipt.
No production or GPU technical worker was launched during preparation. These CPU
results establish configuration/scheduling correctness, not scene-quality gains.

Artifacts under `/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_followup` include
`preparation.json`, `normalized_config_diffs.json`, `source.diff`, source/spec
inventories, tests and `staging_verification.json`. The isolated snapshot is
`exp_followup_code`; the prepared durable queue is `exp_followup_controller`.

## Insights and release

`launch_reviewed.py` requires explicit authorization bound to plan/source/spec/
test hashes, the completed exponential parent's exact result, and a reviewed
depth-audit artifact. The supplied release template is intentionally unauthorized.
The launcher rechecks local and remote source/spec bytes before starting the
detached dispatcher. Shared GPU locks prevent collision with other campaigns.

Cross-campaign reporting remains separate: the completed numerical winner,
per-actor3% guard and manually reviewed geometry candidate are distinct. Running
metrics are provisional and must not replace the completed-parent release gate.

After the root reviewer supplies a concrete authorized release receipt:

```bash
C=/fsx/tmp/marshawn_001660_hd/tuning_20261106
"$C/python" "$C/exp_followup/launch_reviewed.py" --release "$C/exp_followup/root_release.json"
"$C/python" "$C/cross_campaign_report_exp_followup/report_marshawn_cross_campaign.py" launch --root "$C/cross_campaign_report_exp_followup"
```

The second command starts a separately prepared seven-source reporting generation,
including the new queue. Its registry and frozen reporter are already saved; it
does not mutate or replace the existing six-source report. Both commands remain
unexecuted during this preparation. The release template must not be mistaken
for a signed authorization.

## Conditional launch authorized after matched-depth review

Root reviewed the exact source/config differences, CPU checks, matched15188 RGB
crops and native/halo/dense depth plots. A separate durable watcher
`watch_marshawn_exp_followup_release.py` now waits for the unchanged exponential
parent to finish106316 ancestry plus plateau. The five experiments may launch
only if completed v1 LPIPS is≤.06 and neither actor regresses>3% relative to the
fixed corrected-SH15188 reference. Parent configuration and reviewed artifacts
are hash/config bound. No model/grid promotion is authorized by this policy.

Policy: `<campaign>/exp_followup/conditional_launch_policy.json`.
Watcher PID2058869; checks and launch receipts are in the same directory. The
watcher supplies the exact final parent hash to the existing independent release
verifier, then launches the prepared queue and separate seven-source reporter.
A failed scientific condition writes a blocked receipt instead of relaxing it;
transient SSH reads retry every60seconds. Five focused eligibility checks passed.
This makes the reviewed next experiments durable without claiming their results
in advance. The root continues supervising the active campaign through the
user's minimum20-hour deadline.


## Matched15188 visual gate

The first three follow-up arms use identical sources, seed and evaluation protocol. These are provisional equal-step results, not final selected checkpoints.

| Arm | Actor PSNR | Actor SSIM | Actor LPIPS | DC PSNR | DC LPIPS |
|---|---:|---:|---:|---:|---:|
| `structural_exp_hplus1` | 27.1029 | 0.798674 | 0.056944 | 25.3546 | 0.075550 |
| `exp_followup_actor` | 27.5159 | 0.809772 | 0.053507 | 20.5371 | 0.099261 |
| `exp_followup_lr50k` | 26.3000 | 0.791343 | 0.061211 | 22.6785 | 0.088271 |
| `exp_followup_step_half` | 27.0478 | 0.811806 | 0.050777 | 24.2982 | 0.074221 |

Root inspected the [three-view comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_followup_review_15188/three_view_comparison.jpg) and the [DC metric-support diagnostic](/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_followup_review_15188/DC_actor_metric_support.jpg). The actor objective improves the aggregate while producing large yellow holes through DC clothing. The frozen masks include those holes: independent PNG masked PSNR reproduces DC black17.5784dB and grey23.4947dB, within quantization error of the original float metrics. Better HC/KA scores offset this damage in the average. This is a view-specific reconstruction failure, not evidence of a scoring implementation error.

The half-step arm improves detail and average LPIPS but still has foreground/background fog. None is promoted. The primary score remains fixed; per-view/per-actor tables and visual/depth gates expose regressions hidden by its average. Protected15188 checkpoints permit independent103-ray tracing, with separate DC diagnostic queries kept out of TRAIN supervision. Evidence hashes are in `/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_followup_review_15188/root_review.json`.
