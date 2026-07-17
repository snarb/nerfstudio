# Static leader end-to-end speed optimization

Status: active. The reproduction baseline is frozen in
`experiments/static_leader_reproduction_recipe.md`; this campaign changes only named speed fields
and retains the same numeric, artifact and detail reporting.

## What was tested

The first ablation is fixed-batch point normalization on the accepted stable-FP16 historical
recipe. Increasing rays per optimizer update is compensated in update space so the warmup,
occupancy update, frequency-grid update, depth-loss window, exponential LR horizon, checkpoint
cadence and FR transition occur at the same nominal ray/point exposure:

| Batch | Scale | Warmups | Occ/grid interval | Depth steps | Scheduler | FR switch | Accepted boundary |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1× | 4096 | 16 / 1024 | 5000 | 200000 | 75940 | 91128 |
| 8192 | 2× | 2048 | 8 / 512 | 2500 | 100000 | 37970 | 45564 |
| 16384 | 4× | 1024 | 4 / 256 | 1250 | 50000 | 18985 | 22782 |

The fixed warmup remains exactly 4,294,967,296 point samples in every row. Adaptive ARM, FAS from
step zero, FR `1.0→0.3`, hash23, losses, Adam endpoints, stable occupancy and the historical FP16
TCNN binding remain frozen. Batch compression is still algorithmic: it reduces Adam update count
and changes gradient statistics even at matched point exposure.

The speed worktree is `/home/brans/repos/nerfstudio_leader_speed` on the named branch
`nerfstudio_leader_speed`, with historical base `85818149`. Its complete current patch set is
committed, and the controller requires the exact branch, committed HEAD and clean status. Its first
telemetry-only generation added an explicit scheduler-horizon runner field and a checkpointed
cumulative-point counter/CSV column and had fingerprint `0913233e…9e6da87`. Later fingerprinted
generations add the named default-off schedule hooks and hot-path patches documented below; the
current committed-source fingerprint is `6cf7eb95…6e4c258`. A dry-run proves that optional hooks do not appear in the
fixed-batch default command. `scripts/run_static_leader_speed_e2e.py` exposes one safe
`--batch-scale {1,2,4}` instead of independent low-level knobs and asserts all derived values.

## Mature solo throughput profile

Each profile loaded accepted S1@91128 model weights, reset optimizer/scheduler for timing only,
discarded 50 of 500 steps, disabled eval/renders and ran alone on the GPU. The same speed worktree,
historical FP16 TCNN overlay and compensated maintenance cadence were used. The exact point counter
was verified monotonic and persisted in each final checkpoint.

| Batch | Median points/step | Median step | Median points/s | Peak VRAM | Median GPU | Peak power |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2.818 M | 64.71 ms | 43.43 M/s | 16.52 GiB | 81% | 311 W |
| 8192 | 5.629 M | 107.47 ms | 52.35 M/s | 33.28 GiB | 86% | 340 W |
| 16384 | 11.264 M | 207.92 ms | 54.28 M/s | 58.55 GiB | 86% | 359 W |
| 6144 | 4.217 M | 85.58 ms | 49.30 M/s | 21.86 GiB | 86% | 337 W |
| 7168 | 4.934 M | 95.53 ms | 51.62 M/s | 24.76 GiB | 86% | 339 W |

Raw profiles:

- `/home/brans/lookcloser_leader_speed_profiles/profile500_b4096_20260715/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/profile500_b8192_20260715/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/profile500_b16384_20260715/profile.json`.
- `/home/brans/lookcloser_leader_speed_profiles/profile500_b6144_20260715_v1/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/profile500_b7168_20260715_v1/profile.json`.

The intermediate profiles were added after B8192 phase controls exposed an update-count rather
than point-exposure limitation. B6144 is 4.5% slower than B7168 in mature points/s but supplies
16.7% more Adam updates at equal point exposure, so B6144 is the first quality screen. Its cadence
uses the 1.5x point-normalized matrix: warmup 2731, occupancy interval 11, frequency-grid interval
683, depth window 3333 and scheduler horizon 133333. The first seed-42 boundary is step 10000 and
is retained as a common parent for later LR/scheduler forks.

## Batch-8192 end-to-end quality control

The first point-normalized quality run used seed 42 and changed only the derived 2x batch matrix.
Its campaign manifest is
`/home/brans/lookcloser_leader_speed_runs/campaigns/speed_b8192_seed42_v1/campaign.json`.

| Matched boundary | Exact points | PSNR | SSIM | LPIPS | Delta vs S1 PSNR / SSIM / LPIPS |
|---:|---:|---:|---:|---:|---:|
| 15188 ↔ 30376 | 54.65 B | 29.3327 | 0.662522 | 0.307934 | -0.1219 / +0.001856 / +0.012800 |
| 22782 ↔ 45564 | 92.63 B | 29.6005 | 0.666282 | 0.274333 | -0.0922 / -0.001061 / +0.010750 |
| 30376 ↔ 60752 | 132.69 B | 29.7057 | 0.667782 | 0.256849 | -0.0738 / -0.001608 / +0.011477 |
| 37970 ↔ 75940 | 174.019 B | 29.7060 | 0.670153 | 0.244725 | -0.1395 / -0.001464 / +0.012064 |
| 45564 ↔ 91128 | 216.162 B | 29.7362 | 0.669351 | 0.233536 | -0.1039 / +0.000148 / +0.014057 |

The final scheduled checkpoint passed PSNR and SSIM but missed the LPIPS gate by 0.002401. The
controller therefore correctly stopped before renders/artifact gates with a quality exit code 2,
not an infrastructure failure. Stage A took 3636.1 seconds, Stage B 871.0 seconds, and the complete
controller took 4511.1 seconds (75.2 minutes). This is not a <=60 minute candidate.

Nominal update scaling slightly underexposed the final model: 216.162 B exact samples versus the
accepted S1 estimate of 218.383 B. A continuation from the same checkpoint to step 45963 reached
218.395 B, a <0.01% exposure match, in 90.1 seconds including one full eval. It produced
`29.7452 / 0.671725 / 0.235453`: PSNR and SSIM improved, while LPIPS regressed and missed its gate
by 0.004318. The checkpoint is
`/home/brans/lookcloser_leader_speed_runs/speed_b8192_seed42_exactpoints_ext/lookcloser/20260715_100310/nerfstudio_models/step-000045963.ckpt`.

This rules out sample underexposure as the main cause. At matched point exposure, halving the
number of Adam updates while retaining the same instantaneous LR reaches a different and worse
perceptual trajectory. The next diagnostic is a common-checkpoint
`LR x {1,4} x {loaded Adam, reset Adam}` fork; a winning behavior must then be tested from scratch,
because a late fork is diagnostic rather than an end-to-end speed result.

## Common-checkpoint LR and Adam diagnostic

Six seed-42 forks started from the same batch-8192 step-45564 model and sampling-stream reset. Each
ran exactly 1000 further updates with FR 0.3 and one full three-view eval. LR multipliers were
applied consistently to the loaded optimizer and exponential scheduler. Reset variants removed
Adam moments but retained model, scheduler position and AMP scaler.

| LR multiplier | Adam state | PSNR | SSIM | LPIPS | Numeric gate |
|---:|---|---:|---:|---:|---|
| 1x | loaded | 29.7390 | 0.669868 | 0.231192 | fail: LPIPS +0.000057 |
| 1x | reset | 29.7448 | 0.669573 | 0.231018 | pass |
| 2x | loaded | 29.7270 | 0.668306 | 0.230935 | fail: SSIM -0.000144 |
| 2x | reset | 29.7295 | 0.669264 | 0.230759 | pass |
| 4x | loaded | 29.6425 | 0.668945 | 0.234514 | fail: LPIPS |
| 4x | reset | 29.6578 | 0.666544 | 0.233723 | fail: SSIM and LPIPS |

The structured record is `experiments/artifacts/b8192_lr_optimizer_forks.json`. A 4x late LR kick
is decisively harmful with either optimizer policy, so this is not evidence for a deep local
minimum that needs a large escape step. Adam reset gives a small benefit at 1x and lets 2x pass all
aggregate numeric gates. The 2x/reset fresh evaluator produced
`29.729460 / 0.669264 / 0.230732`, with significant artifacts 0/3 and serious ROI artifacts 0/10.
The 1x/reset fresh result was `29.744841 / 0.669573 / 0.230989`, also automatic-clean.

Neither late-fork numeric pass meets the frozen detail target. Both pass tangled cable holes, but
thin pipe and fingers regress against the archive reference in LPIPS; stand and label also fail the
strict all-metric comparator. Therefore neither is promoted as a quality winner. The useful policy
signal is moderate LR scaling, not the late checkpoint itself. The next experiment applies LR
scaling from scratch so micro-detail can form along the whole trajectory; it must beat the same
detail comparator before any wall-clock claim.

## Early and piecewise LR screen

From-scratch batch-8192 seed-42 screens kept every recipe field fixed and changed both LR endpoints
together. At the first step-7594 boundary:

| LR scale | Exact points | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 1.0 | 21.002 B | 28.6327 | 0.649828 | 0.384590 |
| 1.25 | 21.310 B | 28.3125 | 0.650863 | 0.378613 |
| 1.5 | 20.236 B | 26.2334 | 0.678944 | 0.399374 |
| 2.0 | 22.201 B | 28.2398 | 0.653871 | 0.377754 |

At the second boundary, diagnostic resumes of the two Pareto moderate-LR runs produced
`28.9047 / 0.664100 / 0.296579` for 1.25x (55.206 B) and
`28.9038 / 0.670292 / 0.305029` for 2x (57.343 B). Increasing LR from step zero accelerates
SSIM/LPIPS at the cost of about 0.43 dB PSNR; 1.5x is anomalously worse in PSNR and LPIPS and is
rejected. No from-scratch LR scale dominates the historical 1x trajectory.

The next screen therefore kept the exact accepted batch-8192 step-7594 parent and changed only the
loaded LR/scheduler state. Because historical checkpoints lack RNG state, all boost forks and an
explicit 1x resume control used the same seed-reset. Their step-15188 point counts agree within
0.12%, making this a point-matched LR comparison:

| Schedule after step 7594 | Exact points | PSNR | SSIM | LPIPS | Delta vs resume control |
|---|---:|---:|---:|---:|---:|
| 1x resume control | 54.759 B | 29.2773 | 0.671745 | 0.309458 | — |
| 1.25x boost | 54.735 B | 29.2572 | 0.670939 | 0.307602 | -0.0201 / -0.000806 / -0.001856 |
| 2x boost | 54.796 B | 29.2292 | 0.666836 | 0.303940 | -0.0481 / -0.004909 / -0.005518 |

The boost effect is real rather than point exposure or resume noise: more LR improves LPIPS but
trades away PSNR and SSIM. The 1.25x boost is the conservative Pareto candidate; 2x consumes too
much SSIM/detail margin. The 1x and 1.25x trajectories are continued to one more paired boundary
before any piecewise scheduler is promoted to the end-to-end controller.

At step 22782, the 1.25x trajectory fully dominates its matched 1x control at equal point exposure:

| Trajectory | Exact points | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| 1x resume control | 92.880 B | 29.3535 | 0.658020 | 0.251710 |
| 1.25x boost | 92.795 B | 29.3957 | 0.658627 | 0.248527 |

The boost delta is `+0.0422 dB / +0.000607 / -0.003183`, so the initial trade-off eventually pays
back on all three metrics. However, both diagnostic trajectories restarted the sampling stream at
steps 7594 and 15188. The nominal 1x resume control differs from the uninterrupted 1x run at the
same step by `-0.2470 dB / -0.008262 / -0.022623 LPIPS`. This is far larger than the accepted
same-seed noise and identifies deterministic sampling-stream restart as a major basin selector,
not GPU nondeterminism.

Continuing 1.25x for another restarted phase reaches
`29.3751 / 0.656137 / 0.221550` at step 30376: LPIPS clears its final gate by 0.009585, while PSNR
and SSIM miss by 0.243 dB and 0.012313. Reverting the same parent to 1x improves PSNR but not SSIM,
yielding `29.4192 / 0.655017 / 0.221708`. Repeated restarts are therefore useful for perceptual
rate but over-trade fidelity; they are not silently promoted as a speed recipe.

The decisive control used exactly one deliberate restart at step 7594, applied 1.25x LR, then
trained continuously through step 37970. Intermediate model checkpoints were saved without full
eval or sampling restarts. Its run directory is
`/home/brans/lookcloser_leader_speed_runs/speed_b8192_single_restart_lr125_at7594_seed42/lookcloser/20260715_145000`.
It reached `29.7668 / 0.670603 / 0.241176` at 173.722 B exact point samples. PSNR and SSIM pass,
but LPIPS misses by 0.010041. Reconstructing the initial step-0-to-7594 wall from the parent run and
adding the continuous segment gives about 58.7--59.1 minutes of training to this boundary. This is
not a milestone result because it fails quality and is not yet a single controller-to-gates wall.

## Early FR and restart separation

Three forks used the continuous 1.25x checkpoint at step 30376 as a common parent. The historical
FR transition was moved forward from step 37970 to 30376 in the first fork. The second retained FR
1.0 and therefore measures restart/RNG-stream reset alone. The third extended the early-FR fork by
1000 updates.

| Fork from step 30376 | Final step | PSNR | SSIM | LPIPS | Segment wall |
|---|---:|---:|---:|---:|---:|
| early FR `1.0->0.3` + restart | 37970 | 29.6505 | 0.665845 | 0.231454 | 840.9 s |
| restart only, keep FR `1.0` | 37970 | 29.6833 | 0.665418 | 0.231287 | 825.9 s |
| early FR `0.3`, extend 1000 | 38970 | 29.7270 | 0.663544 | 0.231024 | 946.1 s |

The restart-only result misses the final gates by 0.003032 SSIM and 0.000152 LPIPS; PSNR passes.
Changing FR to 0.3 at the same restart costs 0.0328 dB PSNR and 0.000167 LPIPS while gaining only
0.000427 SSIM. Thus almost the entire late LPIPS jump comes from the deterministic sampling-stream
restart/basin change, not the early FR transition. Training the FR-0.3 branch 1000 updates longer
clears PSNR and LPIPS but moves SSIM farther from its gate, so simply waiting longer is not the
joint solution.

The next matched fork keeps the useful single restart at step 7594 and the continuous 1.25x middle
phase, but at step 30376 returns the loaded optimizer and scheduler to their historical 1.0x LR
using a provenance-recorded 0.8 multiplier. FR remains 1.0. This tests whether a conservative late
LR can retain more structural fidelity while the one late restart supplies the missing perceptual
rate. It reached `29.6814 / 0.666219 / 0.232924` at step 37970. Relative to the matched restart-only
1.25x control, SSIM improves by only 0.000801 while LPIPS worsens by 0.001637 and PSNR is unchanged
within 0.002 dB. It therefore fails both SSIM and LPIPS and is rejected.

The next diagnostic records dense model checkpoints along the matched 1.25x restart-only
continuation without intermediate evaluation. Those checkpoints are evaluated only after training
so evaluation cannot perturb the live sampling stream. This determines whether a joint gate pass
existed before the final step or whether the restart-induced PSNR/SSIM-versus-LPIPS trade-off itself
must change.

| Dense checkpoint | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 34000 | 29.698139 | 0.667852 | 0.239165 |
| 35000 | 29.719851 | 0.665122 | 0.235645 |
| 36000 | 29.754173 | 0.667196 | 0.234897 |
| 37000 | 29.738438 | 0.666225 | 0.232789 |
| 38000 | 29.674103 | 0.664071 | 0.229575 |
| 38970 | 29.713924 | 0.663769 | 0.230438 |

No checkpoint crosses all gates. At 34000 SSIM is only 0.000598 below its gate but LPIPS is
0.008030 too high; by 38000 LPIPS passes by 0.001560 while SSIM has fallen 0.004379 below its gate.
This rules out checkpoint cadence as the explanation. The dense diagnostic is at
`/home/brans/lookcloser_leader_speed_runs/speed_b8192_lr125_restart30376_dense_seed42/lookcloser/20260715_184500`.

The next update/point ablation uses batch 8192 through step 30376, then batch 4096. At the switch,
the scheduler epoch is remapped `30363 -> 60726` while current LR stays exactly 0.003087806; its
horizon changes `100000 -> 200000`. Occupancy and frequency-grid cadence return to `16/1024`.
This keeps LR and maintenance continuous in point-time while adding late Adam updates with a
smaller gradient batch. Target step 43000 is expected to keep training plus one eval near 60
minutes when reconstructed from the common initial segment.

| Hybrid checkpoint | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 40000 | 29.760521 | 0.667776 | 0.246442 |
| 42000 | 29.746092 | 0.668282 | 0.241127 |
| 43000 | 29.752714 | 0.671094 | 0.240429 |

The smaller late batch recovers SSIM: step 43000 passes PSNR/SSIM with substantial margin. LPIPS,
however, remains 0.009294 above the gate. This branch began its B4096 phase before B8192 had formed
the low-LPIPS state. The next fork therefore uses dense B8192 step 36000
(`29.754173 / 0.667196 / 0.234897`) and switches to B4096 for only 3000 updates. The observed final
3000-update slope of the first hybrid is approximately `+0.0033 SSIM / -0.0060 LPIPS`, making this
a targeted structure-recovery phase rather than a blind extension. The resumed fork instead
produced `29.7235 / 0.666528 / 0.239354`: restart-induced sampling changed the basin and invalidated
the slope extrapolation.

The speed worktree now exposes an optional live ray-batch boundary in the LookCloser pipeline. It
calls `set_num_rays_per_batch` before the scheduled update in the same trainer process, logs the
actual batch, and is disabled by default. The next same-parent diagnostic runs B8192 from step
30376 through 35999 and B4096 from step 36000 through 39000 without a checkpoint/RNG restart at
the batch transition. LR horizon and occupancy/grid cadence deliberately remain at the B8192
values in this first isolated live-batch test; those point-cadence changes are reported rather than
silently called equivalent. It produced `29.7306 / 0.668646 / 0.238634`: the in-process smaller
batch restores SSIM without a restart, but LPIPS misses by 0.007499. Thus checkpoint reset was not
the only issue; reducing the late ray batch itself gives up too much perceptual sampling rate.

The next in-process control keeps B8192 and decays FAS from 1.0 at global step 38000 toward 0.35 at
step 41000. The historical sampler expresses decay in sampler calls, so the resumed step-30376
diagnostic uses `decay_start=7623`, `decay_steps=4615`; no RNG, model, optimizer or scheduler state
is reset. This tests whether the already low-LPIPS state can be retained while a growing uniform
pixel fraction recovers SSIM. It remains diagnostic: no schedule is promoted until a from-scratch
one-command run passes fresh numeric, artifact and strict-detail gates.

The FAS-decay final reached `29.8133 / 0.673315 / 0.252250`: it recovers structure decisively but
destroys the LPIPS margin. Offline checkpoints show the same incompatible transition:

| FAS trajectory point | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| step 38000, strength ~1.0 | 29.671528 | 0.664647 | 0.229813 |
| step 40000, strength ~0.57 | 29.863342 | 0.670866 | 0.243134 |
| step 41000, strength ~0.35 | 29.8133 | 0.673315 | 0.252250 |

Field-only weight interpolation between steps 38000 and 40000 also lacks a hidden joint pass. The
model-only checkpoints retain the step-38000 buffers and interpolate only TCNN encoding/geometry/
color parameters with SHA-recorded provenance:

| Right-weight alpha | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 0.25 | 29.731104 | 0.666897 | 0.232589 |
| 0.50 | 29.774155 | 0.668603 | 0.235737 |
| 0.75 | 29.800083 | 0.669861 | 0.239186 |
| 1.00 | 29.809223 | 0.670666 | 0.243018 |

The next diagnostic isolates the historical leader's actual late change, FR `1.0->0.3`, without
the historical checkpoint/RNG reset. B8192, FAS, optimizer and scheduler remain live and unchanged;
the pipeline mutates the field's FR strength at global step 38000 and continues to step 41000.
This distinguishes FR from the basin-selection effect that confounded the earlier phase forks. It
reached `29.7171 / 0.666127 / 0.228581`: FR 0.3 improves LPIPS further but does not recover SSIM.
The historical leader therefore retained its SSIM because its B4096 Stage A supplied more Adam
updates, not because FR 0.3 repairs structure.

The last short cycle diagnostic resumes the high-SSIM FAS-decay step-40000 checkpoint
(`29.863342 / 0.670866 / 0.243134`) with full FAS for 2000 updates. Although the resume resets
sampling streams, it cheaply tests whether biased sampling can return LPIPS while retaining the
new structure. A positive direction would be reimplemented in-process; a failure rejects the
uniform-to-FAS cycle before another long run. It reached
`29.6983 / 0.663913 / 0.225562`: LPIPS returns strongly, but the recovered SSIM is lost. The cycle
is rejected.

Together the matched controls show that B8192 is not merely point-underexposed. At fixed or greater
point exposure it can move rapidly between a high-SSIM state and a low-LPIPS state through restart,
FAS or FR changes, but none retains both, and the detail gate already rejected late aggregate-only
passes. The remaining causal difference from the accepted B4096 leader is Adam update count and
gradient batch statistics. The next speed screen therefore profiles intermediate ray batches rather
than adding more B8192 phase knobs.

## Intermediate batch screen

B6144 seed-42 Stage-0 used the 1.5x point-normalized cadence and historical LR 1.0 through step
10000. It reached 19.826 B exact point samples in 560.7 seconds and evaluated at
`28.1532 / 0.649015 / 0.385292`. This is 5.6% less exposure than B8192 step 7594 (21.002 B), while
LPIPS is nearly identical. The immutable step-10000 checkpoint is the common parent for a 1.25x
loaded-Adam/scheduler fork targeting step 21600, approximately the same 54--55 B cumulative region
as the earlier B8192 LR fork.

The B6144 fork reached `28.970001 / 0.667645 / 0.299887` at step 20000 and 52.576 B exact points;
at step 21600 and 58.288 B it reached `28.9402 / 0.671171 / 0.295462`. Against the B8192 1.25x
fork at 54.735 B, B6144 has a 0.0077--0.0121 LPIPS advantage but remains roughly 0.29--0.32 dB
behind in PSNR. It is continued to the approximately 90 B region to test whether fidelity catches
up while preserving the perceptual advantage. That continuation is an explicit second resume/
sampling-stream boundary at step 21600. It reached `28.8914 / 0.657089 / 0.240900` at step 30000:
LPIPS improved, but PSNR/SSIM did not recover. The multi-restart 1.25x path is rejected.

A paired 1.0x LR control now resumes the same immutable step-10000 parent with identical seed,
batch, cadence and target step 21600. This separates the B6144 update-budget effect from the
aggressive LR multiplier before deciding whether to reject the intermediate batch itself. The
control reached `29.0032 / 0.665087 / 0.295638`, versus
`28.9402 / 0.671171 / 0.295462` for 1.25x. Higher LR trades 0.063 dB PSNR for 0.006084 SSIM while
LPIPS changes by only 0.000176. Neither dominates the matched B8192 trajectory or closes the final
joint gap. B6144 is not promoted.

Fixed-ray batch changes have now exposed the limiting trade-off on both sides: B8192 lacks Adam
updates for joint fidelity/perceptual convergence, while B6144 gives up throughput and early PSNR
without enough compensating LPIPS gain. The next speed axis should preserve optimizer-update count
and cap point samples per update (corrected point-budget ARM/dynamic rays), rather than adding more
fixed-batch phase knobs.

## Update-preserving corrected-ARM point budget

The speed worktree now exposes a default-off dynamic point controller and a default-off corrected
ARM allocator. The frozen reproduction remains `target_num_samples_per_batch=0` with the legacy
allocator. The explicit speed ablation uses an EMA of observed samples/ray to change the next
pixel-sampler ray batch, checkpoints the EMA/current rays and cumulative points, and logs current
versus next rays separately. The corrected allocator uses minimum-one plus largest remainder and
deterministically merges closest intervals when interval count exceeds the cap, so it cannot drop
an internal interval or the far tail. ARM/occupancy property tests and the CUDA train/resume smoke
passed (`22` focused tests; the smoke restored cumulative points and dynamic ray state).

The first faithful-leader p20 candidate kept the historical seed42, 75940/91128 optimizer-update
boundaries, LR horizon, FR/FAS trajectory, hash23, Charbonnier loss, stable occupancy, shared RNG
policy and historical FP16 TCNN. Only the corrected allocator and target `2^20=1,048,576` field
points/update were enabled. The canonical solo manifest is:

`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_pointcap_p20_seed42_v1/campaign.json`.

| Scheduled step | PSNR | SSIM | LPIPS | Exact cumulative points |
|---:|---:|---:|---:|---:|
| 15188 | 28.2720 | 0.638498 | 0.386040 | about 15.9 B |
| 30376 | 28.7544 | 0.654121 | 0.330273 | about 31.9 B |
| 45564 | 29.1422 | 0.660887 | 0.299633 | about 47.8 B |
| 60752 | 29.2887 | 0.668111 | 0.286501 | about 63.7 B |
| 75940, FR 1.0 | 29.2970 | 0.667353 | 0.274570 | 79.637 B |
| 91128, FR 0.3 | 29.4864 | 0.670329 | 0.267281 | 95.555 B |

Mature updates held approximately 1.04--1.07 M points with 1.5--1.7k rays and about 36--38 ms
step time. Controller wall was `3567.9 s` (`59:27.9`), including provenance, both stages and all
scheduled evals. This is not a <=60-minute milestone: the numeric selector was empty, so renders
and artifact/detail gates correctly did not run. The final checkpoint passes SSIM but misses PSNR
by 0.1316 dB and LPIPS by 0.03615. Thus preserving optimizer-update count is insufficient at p20;
field-point exposure remains causally important, especially for perceptual detail. A p19 run would
move in the rejected direction. The next cheap diagnostic uses the saved p20 step75940 parent for
paired late LR/Adam forks; only a joint pass is eligible to be encoded into a from-scratch
one-command schedule. Otherwise the point bracket moves upward toward p21.

Paired step75940 forks retained the same Adam/scaler/scheduler epoch, FR0.3, seed and point budget,
but multiplied current/base/scheduler LR by 2x or 4x. Because both diagnostics ran concurrently,
their segment wall is deliberately not used for a speed claim. Fresh exact-checkpoint evals gave:

| Late segment from p20 step75940 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| historical LR 1x | 29.4864 | 0.670329 | 0.267281 |
| loaded Adam, LR 2x | 29.3328 | 0.668598 | 0.266099 |
| loaded Adam, LR 4x | 29.1403 | 0.666810 | 0.273058 |

LR 2x gains only 0.00118 LPIPS while losing 0.154 dB; LR 4x is worse than baseline on every final
metric. A simple high-LR escape does not substitute for the missing p20 point exposure. During
this diagnostic an evaluator bug was found and fixed: resumed YAML configs carried an explicit
parent `load_checkpoint`, so changing only `load_step` could silently evaluate the parent. The
helper now parses the config object, clears `load_dir/load_step`, and sets the exact candidate file.
Both reported fork metrics are fresh evals whose JSON records the final step91128 checkpoint.

A 500-update mature p21 profile from the p20 step75940 model held 2.08--2.12 M points/update with
about 3.1k rays and 51--53 ms/update. Full p21 is therefore an approximately 79--82 minute quality
bracket, not a <=60 candidate by itself. Its intermediate checkpoints are needed to determine how
far update count/LR horizon can be shortened while retaining the additional point exposure.

The canonical solo p21 control subsequently completed under
`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_pointcap_p21_seed42_v1/campaign.json`.
It used the same seed42, corrected allocator, historical 75940/91128 update boundaries and exact
FR1.0 -> FR0.3 continuation as p20; only the point target changed to `2^21=2,097,152`.

| Scheduled step | PSNR | SSIM | LPIPS | Exact cumulative points |
|---:|---:|---:|---:|---:|
| 15188 | 27.8045 | 0.661090 | 0.356394 | about 31.9 B |
| 30376 | 28.0675 | 0.668505 | 0.304750 | about 63.7 B |
| 45564 | 28.2977 | 0.672046 | 0.276377 | about 95.6 B |
| 60752 | 28.1610 | 0.678092 | 0.259601 | about 127.4 B |
| 75940, FR 1.0 | 28.2619 | 0.677854 | 0.247696 | 159.274 B |
| 91128, FR 0.3 | 28.1898 | 0.681731 | 0.243141 | 191.109 B |

Controller wall was `5116.5 s` (`85:16.5`): Stage A took `4239.7 s` and Stage A_fw03 took
`871.3 s`. The numeric selector was empty, so final renders and artifact/detail gates correctly did
not run. The final checkpoint passes SSIM by 0.01328 but misses the PSNR gate by 1.4282 dB and the
LPIPS gate by 0.01201. FR0.3 improved LPIPS by only 0.00456 while losing another 0.072 dB PSNR.

This is not ordinary undertraining: at step75940, p21 train PSNR was 33.22 versus 33.44 for the
accepted fixed-ray reproduction, yet full-view PSNR was 28.26 versus 29.85. Doubling the p20 point
batch improved SSIM/LPIPS but moved farther into a low-eval-PSNR basin. The next bounded diagnostic
therefore forks the saved p21 step15188 checkpoint with loaded Adam and scheduler, raises LR by 2x,
and evaluates at step30376. It tests whether the larger effective batch needs an early LR/noise-scale
correction before spending another full p21 campaign.

The completed p21 manifest also exposed a bookkeeping issue: for dynamic runs the top-level printed
total used the legacy CSV estimate (`182.519 B`) even though the checkpointed exact cumulative count
was `191.109 B`. The controller now records `total_point_samples` with
`point_sample_accounting=exact_checkpointed_cumulative` in dynamic mode and keeps the legacy estimate
as a separate diagnostic field.

The first early-LR diagnostic forked the immutable p21 step15188 checkpoint, retained loaded Adam,
scaler, scheduler epoch and dynamic-point state, and multiplied optimizer/current/base scheduler LR
by 2x. Its provenance is
`/home/brans/lookcloser_leader_speed_runs/diagnostic_checkpoints/p21_step15188_lr2.ckpt.fork.json`;
the evaluated child checkpoint SHA-256 is
`7d0be7f513c02e427e4ca0746388c840c2d6adc745c12b76c2f9cfb8c14eaa0d`.

| p21 step30376 path | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| continuous historical LR | 28.0675 | 0.668505 | 0.304750 |
| resumed loaded-Adam LR 2x | 28.1498 | 0.662840 | 0.304330 |

The apparent change is `+0.0823 dB / -0.005665 / -0.000420`: a small PSNR/LPIPS gain traded for
SSIM, not a joint escape. Historical checkpoints do not contain RNG state, so the continuous-versus-
resumed comparison also contains restart RNG effects. A matched LR1 resume from the exact same parent
and seed is required before assigning this delta to LR or continuing the fork.

The matched LR1 resume produced `28.1613 / 0.668081 / 0.302997`. Therefore the causal LR2-minus-LR1
delta is `-0.0115 dB / -0.005241 / +0.001333 LPIPS`: LR2 is worse on every metric and is rejected.
The LR1 resume differs from the continuous LR1 control by
`+0.0938 dB / -0.000424 / -0.001753 LPIPS`. Its PSNR delta exceeds the accepted same-seed bound of
0.06 dB and quantifies the previously uncontrolled checkpoint-restart variance.

The speed worktree now checkpoints and restores Python, NumPy, Torch CPU and all CUDA RNG states,
with restore deferred until immediately before the first resumed iteration. CPU/CUDA RNG round-trip
tests pass. A 30-step control and two identical step20 -> step29 resumes verified that both resumes
end with the same Torch CPU RNG state. They still differed by 173 cumulative points out of 10.94 M
and in TCNN/occupancy tensors, despite identical random streams. This isolates the remaining floor to
CUDA/TCNN/occupancy numerical nondeterminism: atomic reductions perturb density and then traversal
counts. Future forks use RNG-complete checkpoints and judge the residual at full metric scale. The
fingerprinted speed source for this behavior is
`74595b202c01b8d51754027ab48cf69450664c41b05f52758f03db3a8a088b79`.

## Corrected-versus-legacy allocator attribution

The first dynamic point campaigns changed both ray-batch control and the allocator correction. A
seed42 p21 screen therefore kept target points, optimizer updates, LR, FAS, FR1.0, occupancy and
frequency-grid schedules fixed and changed only `corrected_arm_allocator=true -> false`. It ran solo
from scratch through step30376. The legacy mode is diagnostic only: it can omit a far-tail interval
under cap pressure and is not eligible for promotion as the corrected implementation.

| Step | Allocator | Exact points | PSNR | SSIM | LPIPS |
|---:|---|---:|---:|---:|---:|
| 15188 | corrected | about 31.9 B | 27.8045 | 0.661090 | 0.356394 |
| 15188 | legacy | about 31.9 B | 28.0435 | 0.656372 | 0.358801 |
| 30376 | corrected | about 63.7 B | 28.0675 | 0.668505 | 0.304750 |
| 30376 | legacy | about 63.7 B | 28.4899 | 0.666031 | 0.298829 |
| 30376 | accepted fixed-ray legacy | about 76 B | 29.4546 | 0.660666 | 0.295134 |

The legacy screen took `1697.6 s` (`28:17.6`) including both scheduled full evals. At step15188 it
moves `+0.2390 dB / -0.004718 / +0.002407 LPIPS` relative to corrected p21: initially a fidelity-
versus-perceptual trade. By step30376 it moves `+0.4224 dB / -0.002474 / -0.005921 LPIPS`, improving
both PSNR and LPIPS while giving up a small amount of SSIM. Thus corrected full-tail allocation is a
real basin selector and explains a substantial part of p21's failure. It does not explain all of it:
legacy dynamic p21 still trails the fixed-ray leader at the same update by 0.9647 dB PSNR and
0.003695 LPIPS, despite having 0.005365 higher SSIM. The remaining difference is the dynamic
ray-batch/point exposure and its gradient statistics. Corrected p21 is not promoted, and the buggy
legacy diagnostic is not mislabeled as a correctness-preserving speed result.

The exposure comparison revealed a second confound. The accepted leader has only about 56.5 B
points through step30376, versus about 63.7 B for dynamic p21, so missing total samples cannot
explain the dynamic gap. The old p21 controller was already active during the fixed warmup: it
rapidly raised rays from 4096 to 8192 while every ray still used 256 fixed samples. The accepted
leader kept exactly 4096 rays for all 4096 warmup updates. The speed controller now exposes
`dynamic_rays_start_step` and the canonical dynamic command sets it to 4096. A 4110-step CUDA smoke
verified current/next rays remained 4096/4096 with zero EMA through step4095, then changed under the
1.25x limit only after ARM activation. Its checkpoint contains cumulative/EMA/ray state plus the
complete RNG snapshot. The new source fingerprint is
`4bd72fe628508f1f7669cc346dcff090c8e8032962fa41b568ac8472c9095366`.

The next default-off speed hook adds an in-process dynamic point-target boundary. It logs the
active target, uses the new target only to calculate the ray batch for the update after the named
step, and does not reload model/Adam/scaler/scheduler/RNG/occupancy/EMA state. The reviewed source
fingerprint is `c8b9fd2b581963fc4f57d91a72a72ca62352d1861bdb2540b7b056fcd3047c9d`.

## Delayed corrected-p21 trajectory

The corrected p21 control was repeated with dynamic rays disabled through the exact historical
4096-update fixed-march/occupancy warmup. All other fields remained fixed: seed42, corrected ARM,
target `2^21` points/update after warmup, hash23, stable occupancy, FAS/FR1.0, Adam and the original
LR horizon. The first segment is
`/home/brans/lookcloser_leader_speed_runs/p21_corrected_delayed4096_seed42_screen30376/lookcloser/20260715_221100`;
the RNG-complete continuation is
`/home/brans/lookcloser_leader_speed_runs/p21_corrected_delayed4096_seed42_continue60752/lookcloser/20260715_224000`.

| Step | Exact cumulative points | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 15188 | about 27.56 B | 28.7765 | 0.647632 | 0.346686 |
| 30376 | 59.433 B | 29.3803 | 0.662264 | 0.286686 |
| 45564 | 91.286 B | 29.5782 | 0.667334 | 0.263411 |
| 60752, FR 1.0 | 123.134 B | 29.6229 | 0.668059 | 0.244658 |

At step15188 this trajectory beats the accepted fixed-ray leader by
`+0.0285 dB / +0.001732 SSIM / -0.014912 LPIPS`; it also gains 0.972 dB over the original
corrected-p21 controller. This makes warmup batch history, not the allocator correction or total
point count alone, the dominant cause of the earlier low-PSNR basin. At step60752 PSNR already
passes the archived-leader gate by 0.0049 dB. SSIM misses by only 0.000391, while LPIPS remains
0.013523 above target.

The two diagnostic segments took 1652.6 s and 1712.7 s including their scheduled evals, or
`56:05.3` combined. This is not a milestone claim: it contains a manual resume, omits automatic
candidate renders/detail gates, and was not launched by the one-command controller. It is a
quality-screen upper bound for encoding a final from-scratch schedule.

A paired continuation from the same RNG-complete step45564 checkpoint changed only FR
`1.0 -> 0.3`. At step60752 it reached `29.6893 / 0.668966 / 0.246000`, versus
`29.6229 / 0.668059 / 0.244658` for FR1.0. Early FR0.3 therefore improves PSNR/SSIM but worsens
LPIPS by 0.001342 and is rejected as the missing perceptual accelerator. The remaining controlled
diagnostic resets only Torch CPU sampling RNG at step45564 while preserving the CUDA RNG,
model/Adam/scheduler/scaler and FR1.0; its purpose is to attribute the residual LPIPS gap to the
pixel/FAS/frequency sampling stream without conflating occupancy randomness. Its immutable fork
provenance is
`/home/brans/lookcloser_leader_speed_runs/diagnostic_checkpoints/p21delayed_step45564_cpu_rng42.ckpt.fork.json`.

The selective branch completed at `29.6811 / 0.668561 / 0.246327`. Relative to the uninterrupted
FR1.0 control it changes `+0.0582 dB / +0.000502 SSIM / +0.001669 LPIPS`: fidelity improves, but
the perceptual metric moves in the wrong direction and remains 0.015192 above its gate. The
checkpoint retains the same 123.134 B point exposure. A late CPU sampling-stream reset is therefore
rejected; the earlier B8192 restart effect cannot be attributed to CPU pixel/FAS/frequency sampling
alone and may involve the CUDA occupancy/traversal stream or an earlier basin boundary.

## Live p21-to-p20 point schedule

A 500-update mature p22 profile measured 4.197 M points/update, 85.57 ms/update, about
49.1 M points/s and roughly 7147 rays/batch. A p22 step42000 candidate would expose about 163 B
points but consume an estimated 56--57 minutes of training before finalization, while retaining
only 42k Adam updates. It leaves too little <=60-minute gate headroom and moves toward the already
problematic large-gradient-batch B7168/B8192 regime, so it is retained as a profile rather than the
next quality run. The profile is
`/home/brans/lookcloser_leader_speed_profiles/profile_p22_from_delayed_p21_step30376/lookcloser/20260716_000100`.

The selected candidate instead keeps corrected delayed p21 through step30376, then changes only
the live point target from `2^21` to `2^20` for the ray batch chosen after that update. It trains
continuously through step75940 with seed42, FR1.0, full FAS1.0, the historical LR horizon and
unchanged Adam/scaler/RNG/occupancy/frequency-grid state. It has one full eval at the final step;
model checkpoints remain available every 15188 updates. Its run directory is
`/home/brans/lookcloser_leader_speed_runs/p21_to_p20_delayed4096_live30376_seed42_screen75940/lookcloser/20260716_002000`.

The hypothesis is quantitative. Delayed p21 reaches 59.433 B points and
`29.3803 / 0.662264 / 0.286686` at the switch. Applying only the measured p20
step30376-to-75940 deltas predicts approximately `29.9229 / 0.6755 / 0.2310`, a narrow joint
numeric pass. Point exposure should finish near 107.2 B while retaining about 75.9k Adam updates;
estimated training plus a single finalization is 56--59 minutes. This is preferable to p22 because
early p21 establishes perceptual detail and late p20 buys optimizer updates and wall speed rather
than enlarging the gradient batch.

The live target hook is default-off and fail-closed. It logs the active target in compact CSV and
uses the post-switch target only for the next ray batch; a CUDA smoke showed p21 at step30380 and
p20 with rays reduced from about 3573 to 1784 by step30390. An RNG-complete resume retained p20,
EMA, current rays and cumulative point count. Thirty-one focused ARM/controller/RNG/occupancy tests
pass. The frozen reproduction dry-run remains fingerprint `69d4f36c...8857b252`; the current speed
training fingerprint for this run is `c8b9fd2b...3047c9d`.

After training, the first offline intermediate-eval attempt exposed a speed-worktree helper
regression: it wrote an exact `load_checkpoint`, but historical `eval_utils` ignores that field and
selected latest because `load_step` was `None`. Both attempted 45564/60752 JSON files therefore
identified step75940 and are invalid as intermediate evidence. The helper now binds both the exact
path and exact `load_step`; a dedicated regression test plus the prior focused suite gives 32
passing tests. This semantics-preserving evaluator fix does not change the trained weights. That
generation's reviewed speed fingerprint was `021ff4df...478a6f6`.

The live p21-to-p20 run completed in 3337.7 s (`55:37.7`) including its one scheduled full eval.
It therefore met the diagnostic training-time envelope but not quality:

| Exact checkpoint | Approx. cumulative points | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 15188 | 27.561 B | 28.7999 | 0.648639 | 0.350635 |
| 30376, p21 boundary | 59.433 B | 29.3197 | 0.660242 | 0.292010 |
| 45564, p20 | 75.356 B | 29.4695 | 0.660196 | 0.279376 |
| 60752, p20 | 91.285 B | 29.5985 | 0.664352 | 0.262455 |
| 75940, p20 | 107.213 B | 29.5359 | 0.662392 | 0.255285 |

The final misses the leader gates by `0.0821 dB / 0.006058 SSIM / 0.024150 LPIPS` and is rejected
before artifact/detail promotion. The actual post-switch slope contradicts the p20 extrapolation:
PSNR and SSIM peak by step60752 and then regress, while LPIPS continues to fall too slowly. This is
a different point-schedule basin, not a candidate that merely needs a short extension. Numeric
proximity from unrelated p20 states is therefore no longer accepted as evidence for another mixed
target schedule.

The repeated p21 prefix also refines the variance estimate. Against the prior delayed-p21 run,
step15188 changes by `+0.0234 dB / +0.001007 / +0.003949 LPIPS`, inside the accepted
`0.06 / 0.01 / 0.005` bounds. At step30376 the exact same current evaluator gives
`-0.06067 dB / -0.002022 / +0.005358 LPIPS`, exceeding the PSNR and LPIPS bounds only by
0.00067 dB and 0.000358 respectively. This is marginal but formally outside the accepted repeat
gate; future speed predictions must retain explicit variance margin rather than target a threshold
within a few ten-thousandths.

## Semantics-preserving continuous-p21 acceleration

Because p21→p20 changed the optimization basin, the next branch keeps the corrected delayed-p21
target, LR, update count, FR/FAS trajectory, occupancy and checkpoint boundaries unchanged. A
PyTorch trace at step30381 found the mature step was split between about 26 ms forward/loss and
30 ms backward/Adam. The dominant avoidable CPU symptom was 7188 `aten::item` calls: the FAS
metadata path read image IDs from CUDA twice per ray. The following changes are limited to tensor
plumbing and cached evaluations:

- cache the exact Eq. 6 feature-reweighting weights for the 16 discrete frequency-grid levels;
- cache dense per-image height/width LUTs on the sampling device instead of Python dictionary
  lookup plus two CUDA scalar reads per FAS ray;
- process only overflowing ARM rays, replace iterative closest-gap merging with the equivalent
  stable sort/group reduction, pass known output sizes to `repeat_interleave`, and reuse sampler
  `packed_info`;
- keep the dynamic EMA as a Python mirror while persisting its tensor buffer, derive point count
  from packed tensor shape, and persist the FAS schedule counter across resumes;
- share packed black-background accumulation across RGB/accumulation/depth during training.

All profiles resume the same seed-42 p21 step30376 checkpoint, discard the early setup region and
report steps30500--30870:

| Patch set | Mean step | Median step | Throughput | Mean-step speedup |
|---|---:|---:|---:|---:|
| pre-patch control | 54.200 ms | 54.151 ms | 38.760 Mpoints/s | — |
| ARM/controller sync | 52.338 ms | 52.231 ms | 40.195 Mpoints/s | 3.44% |
| + discrete FR LUT | 49.324 ms | 49.244 ms | 42.486 Mpoints/s | 9.00% |
| + shared packed render | 48.932 ms | 48.784 ms | 42.924 Mpoints/s | 9.72% |
| + vector gap merge | 48.678 ms | 48.596 ms | 43.159 Mpoints/s | 10.19% |
| + device FAS shape LUT | **45.086 ms** | **44.995 ms** | **46.509 Mpoints/s** | **16.82%** |

Raw runs are under `/home/brans/lookcloser_leader_speed_profiles/` with experiment names
`profile_p21_syncfix_from30376`, `profile_p21_syncfix_frlut_from30376`,
`profile_p21_syncfix_frlut_render_from30376`, `profile_p21_all4_vector_merge_from30376` and
`profile_p21_all5_faslut_from30376`.

Parity is checked at three levels. Fifty-eight focused CPU/CUDA property tests cover every discrete
FR level at strengths 1.0/0.3, mixed/no metadata image IDs, invalid IDs, randomized ARM allocation
and merging, controller/resume state, and packed-render forward/gradient equivalence. On the frozen
p21 step30376 checkpoint, fast-vs-analytical packed weights, ray indices, spacing and sample counts
are bit exact. CUDA `nerfacc` accumulation itself is atomic: reference-vs-reference RGB/depth differ
by up to `2.7e-6 / 9.8e-6`; fast-vs-reference differs by `4.3e-6 / 7.8e-6`. Gradient differences
(`<=9.54e-7` in the fast comparison) remain below the measured reference-repeat TCNN floor
(`6.56e-7` hash-grid and `3.81e-6` color MLP). Therefore CUDA RGB/depth are bounded by the measured
native repeat floor rather than incorrectly described as bit exact.

The paired 500-update optimizer smoke from the same p21 step30376 checkpoint gives old/fast
step30876 metrics `29.4371 / 0.661593 / 0.287415` versus
`29.4582 / 0.660480 / 0.286276`. Deltas `+0.0211 dB / -0.001113 / -0.001140` are inside the approved
repeat limits `0.06 / 0.01 / 0.005`.

The reviewed speed-source fingerprint for this patch set is `0a623fc1...059049d`; the frozen accepted
recipe remains `69d4f36c...8857b252`. The speed dry-run rejects extra files and records every source
hash. A solo end-to-end continuous-p21 quality run is the next acceptance test; profile speed alone
is not a milestone result.

### Solo continuous-p21 end-to-end result

The fingerprinted seed-42 controller run is
`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_p21_semanticfast_seed42_v1/campaign.json`.
Dataset provenance matched `dev3`; the run used one uninterrupted p21 Stage A followed by an exact
checkpoint/Adam/scheduler/RNG/FAS-counter continuation at FR0.3. The trajectory was:

| Step | Exact cumulative points | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 15188 | ~27.56 B | 28.8019 | 0.653279 | 0.354059 |
| 30376 | ~59.43 B | 29.3676 | 0.664559 | 0.299220 |
| 45564 | ~91.28 B | 29.5516 | 0.670193 | 0.267532 |
| 60752 | ~123.13 B | 29.7163 | 0.674312 | 0.249791 |
| 75940, FR1 | 154.991 B | 29.6993 | 0.672950 | 0.240324 |
| 91128, FR0.3 | 186.826 B | **29.8000** | **0.672415** | **0.232865** |

Stage A including five full evals took 3518.9 s (`58:38.9`). It therefore reached the ancestry
boundary inside 60 minutes, but its LPIPS did not pass. The full controller through step91128 took
4244.9 s (`70:44.9`) and missed only the LPIPS gate by `0.001730`; PSNR and SSIM passed. The
fail-closed controller recorded `complete_no_accepted_candidate` and did not run artifact/detail
gates. The next diagnostic is an exact continuation to the historical full step106316 boundary:
if it passes, insufficient p21 cumulative exposure rather than collapse/local minimum is the leading
explanation, and that longer pass becomes the speed-optimization baseline rather than a ≤60 claim.

The exact continuation reached step106316 at 218.661 B cumulative points in another 736.0 s. Its
fresh full-precision result is `29.810398 / 0.671881 / 0.227649`: all three aggregate numeric gates
pass, with LPIPS improving by 0.005216 from step91128. The complete automatic protocol also passes
with significant artifacts `0/3` and serious ROI artifacts `0/10`. This supports the
insufficient-exposure hypothesis and provides no evidence that a large LR kick is needed on this
trajectory.

It is not promoted as the speed winner because the frozen priority detail gate fails. Relative to
the archive, eval1 thin pipe passes, while tangled cable holes regress by `+0.007817 LPIPS` and
fingers by `+0.001470 LPIPS` despite better PSNR/SSIM. The structured result is
`/home/brans/lookcloser_leader_speed_runs/leader_p21_semanticfast_seed42_v1_A_fw03_ext106316/lookcloser/20260716_113600/candidate_evaluation_step-000106316.json`;
its contact sheet and five ROI metrics are in the sibling `detail_candidate_step-000106316/`
directory. This diagnostic extension was launched after the controller and therefore is not an
end-to-end milestone time.

A solo mature profile then applied the same hot-path patch set to the safest fixed-ray leader path:
4096 rays, historical allocator, FR0.3, and the accepted S1 checkpoint. Median update time fell from
the earlier 64.71 ms control to 53.721 ms (`−16.98%`), at 2.819 M points/update and 52.50 Mpoints/s.
The profile is
`/home/brans/lookcloser_leader_speed_profiles/b4096_semanticfast_all5_20260716/profile.json`.
Even before scheduled eval/finalization overhead, `91128 × 53.721 ms ≈ 81.6 min`; therefore a full
fixed-ray semantics-fast rerun cannot meet the current ≤60-minute milestone. It remains the safest
quality validation path, but it does not dominate the next algorithmic screen on wall-clock.

The planned TCNN JIT check was attempted next on the two small MLPs only. The binding reported JIT
support, but both RTC compilations failed before training because the isolated historical overlay's
single RTC include root does not contain `cuda_fp16.h`; TCNN then disabled JIT automatically. The
resulting 53.901 ms median is therefore a non-JIT fallback, not evidence about JIT speed. Its log is
`/home/brans/lookcloser_leader_speed_profiles/profile_b4096_semanticfast_jitmlp_20260716/lookcloser/20260716_120325/train_stdout.log`.
The default-off source experiment was removed, restoring reviewed fingerprint
`0a623fc1820596070cc7d676e08f92f593932fb36bee6724dde91fb1f059049d`.
JIT remains blocked until a separately fingerprinted RTC-header overlay is built; the canonical
TCNN overlay is not mutated in place.

## Count-aware FAS diagnostic

The next single-change diagnostic tested whether the remaining cable-hole regression came from
under-sampling the frequency levels that actually dominate this dataset. Across the 66 training
frequency maps, levels 12--15 contain 77.442% of classified patches, but the frozen linear
`1->3` FAS distribution samples those levels only 35.000% of the time. Setting only
`fas_level_count_alpha=0.5` changes the sampling probability to
`linear_ramp(level) * bucket_count(level)^0.5` and raises their expected share to 64.809%.
The exact counts and both 16-level probability vectors are frozen in
`experiments/artifacts/fas_alpha05_distribution.json`.

The RNG-complete seed-42 fork loaded continuous-p21 step45564 at exactly 91.286 B cumulative point
samples and retained model, Adam, scheduler, scaler, CUDA/Python/NumPy RNG, occupancy state, FAS
sample counter, FR1.0, LR and the p21 controller. It changed only the count exponent and trained to
step72000. The segment took 1242.2 s and ended at 146.728 B points. This is a diagnostic checkpoint
fork, not an end-to-end wall-clock result.

| Step72000 result | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs |
|---|---:|---:|---:|---:|---:|
| fresh full-precision eval | 29.926001 | 0.681952 | 0.274660 | 0/3 | 0/10 |

PSNR and SSIM pass comfortably, but LPIPS misses the gate by 0.043525. The failure is not confined
to the aggregate metric: all five frozen crops improve in PSNR and SSIM versus the archive while
regressing in LPIPS.

| Crop | Delta PSNR | Delta SSIM | Delta LPIPS | Priority gate |
|---|---:|---:|---:|---|
| stand | +0.108292 | +0.017278 | +0.029451 | diagnostic fail |
| thin pipe | +0.340261 | +0.016611 | +0.020085 | fail |
| stand label | +0.321737 | +0.013144 | +0.023486 | diagnostic fail |
| tangled cable holes | +0.260574 | +0.013216 | +0.032072 | fail |
| fingers | +0.547081 | +0.017208 | +0.015854 | fail |

The full structured result is
`/home/brans/lookcloser_leader_speed_runs/p21_semanticfast_fasalpha05_from45564_to72000_seed42/lookcloser/20260716_120930/candidate_evaluation_step-000072000.json`;
the visually inspected blind sheet is in the sibling
`detail_candidate_step-000072000/blind_contact_sheet.png`. The candidate checkpoint SHA-256 is
`1518e2bf1793eab527f4cc2a70b235937f1e3e2a32a6839c6210bcb60efed30d`.

This rejects count-aware `alpha=0.5`: increasing high-frequency bucket exposure does not recover
the archive's perceptual detail. It selects a higher-PSNR/SSIM but substantially smoother LPIPS
solution across every measured micro region. It is therefore not added to the one-command
controller, and no projected `<=60` minute claim is made from this fork.

## Historical all-stream restart diagnostic

The accepted historical leader began its second process from a checkpoint that did not persist
Python, NumPy, Torch CPU or Torch CUDA RNG. The process was seeded with 42, performed normal setup,
then entered its first resumed update with the resulting post-setup streams. The semantic-fast p21
path instead restores all four captured streams exactly. To test this remaining ancestry difference,
the diagnostic fork removed only `rng_state` from the RNG-complete p21 step60752 checkpoint. Model,
Adam, scheduler, scaler, occupancy, dynamic p21 controller/EMA, FAS counter/distribution, FR1.0 and
all config fields remained unchanged. The immutable fork provenance is:

`/home/brans/lookcloser_leader_speed_runs/diagnostic_checkpoints/p21_semanticfast_step60752_rngrestart_seed42.ckpt.fork.json`.

Its source SHA-256 is `07ca1169010b634a93b2093dffebcfa4d900a6a32f8c043061e21971ccab65f4`;
the RNG-dropped fork SHA-256 is
`4dfec3f3fb3dddeb0e94f97b8bc8a5f19edd1a75ebe2da20af763e7f221e6cbf`. A new
`--drop-rng-state` mode in `scripts/fork_static_checkpoint_optimizer.py` records this operation and
rejects combination with the existing CPU-only reset. Two focused tests verify that weights,
Adam/scheduler and trainer step are unchanged while only the RNG snapshot is removed.

The valid seed-42 branch trained from step60752 to 72000 in 540.9 s and ended at 146.728 B exact
point samples. Its fresh full protocol result is:

| Step72000 result | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs |
|---|---:|---:|---:|---:|---:|
| historical-style all-stream restart | 29.745617 | 0.673307 | 0.242379 | 0/3 | 0/10 |

PSNR and SSIM pass, but LPIPS misses by 0.011244. All three priority crops fail archive parity:
thin-pipe LPIPS is `+0.001597`, cable-hole LPIPS is `+0.016358` with PSNR also `-0.044207 dB`,
and fingers LPIPS is `+0.005779`. The fresh result is
`/home/brans/lookcloser_leader_speed_runs/p21_semanticfast_rngrestart_from60752_to72000_seed42_v2/lookcloser/20260716_124100/candidate_evaluation_step-000072000.json`;
the visually inspected blind sheet is in the sibling `detail_candidate_step-000072000/` directory.
The candidate checkpoint SHA-256 is
`188f95f56c7b30b659cba37844959817888891da919309e81bcdc3be6848ee35`.

The candidate improves LPIPS relative to its earlier step60752 parent, but that comparison includes
11248 additional optimizer updates and is not a matched causal estimate of RNG reset. A matched
exact-RNG step72000 control would be required to quantify the restart delta. It is not run because
this candidate already fails aggregate LPIPS and every priority detail gate by margins well above
the accepted repeat floor. The all-stream restart axis is rejected and is not added to the speed
controller.

An initial preflight command omitted the explicit `--max-res 8192`, generated 2048 in the config,
and was stopped after 168 updates. It has no evaluation and is excluded. The valid `_v2` command
and config bind 8192 for both the FAS sampler and model; its config SHA-256 is
`308ad67e61350fc1b9ce78d57d2f2e76b329b84c054a1a8e72a5efa6dd9eccc3`.

## Fused Adam and compressed B6144 screen

A detailed PyTorch trace of the mature fixed-4096 path measured about 44.1 ms/update in GPU kernels.
TCNN hash-grid backward alone used about 10.2 ms; the remaining large optimizer-side kernels came
from foreach Adam and AMP bookkeeping. The speed worktree therefore adds an explicit
`AdamOptimizerConfig.fused` field and `--fused-adam` runner flag. The field defaults to `None`, so
the frozen reproduction remains historical non-fused Adam. Loading an old foreach checkpoint into
an explicit fused run retains weights, moments and step count while moving scalar step tensors to
the parameter device as required by fused CUDA Adam.

The controlled mature profiles were:

| Recipe | Median step | Median points/update | Throughput |
|---|---:|---:|---:|
| fixed 4096, semantic-fast, historical Adam | 53.721 ms | 2.819 M | 52.50 Mpoints/s |
| fixed 4096, semantic-fast, fused Adam | **48.767 ms** | 2.819 M | **57.94 Mpoints/s** |
| B6144, semantic-fast, fused Adam | 64.972 ms | 4.218 M | **65.02 Mpoints/s** |

Thus fused Adam reduces fixed-4096 update wall by 9.22% and raises point throughput by 10.16%.
The raw fused profiles are
`/home/brans/lookcloser_leader_speed_profiles/profile_fixed4096_fusedadam500_20260716/lookcloser/20260716_130200`
and
`/home/brans/lookcloser_leader_speed_profiles/profile_b6144_semanticfast_fusedadam500_20260716/lookcloser/20260716_130400`.

Optimizer semantics were checked with a scaled CUDA one-update comparison, including parameters
and moments; maximum parameter difference is bounded by `1.2e-7`. A paired 500-update smoke loaded
the same accepted step-91128 parent and reset optimizer/scheduler identically. Historical versus
fused fresh metrics at step91628 were respectively
`29.532640 / 0.664061 / 0.231220` and
`29.536882 / 0.664703 / 0.232146`. The fused-minus-historical delta
`+0.004242 dB / +0.000642 / +0.000926` is inside the accepted repeat limits; both artifact protocols
were clean. The focused speed suite, including the new optimizer migration/parity properties, is
60/60 passing. That fused-only generation's reviewed speed source fingerprint was
`10001f353e243671a2bb33801cef711ef0d15c44576577a9e7eec6f233459d89`.

The previously blocked TCNN network JIT was then repaired without mutating the canonical binding.
The separate overlay `/home/brans/deps/tcnn_2e757_py310_jit_rtc` copies the five CUDA headers that
the upstream package setup selects for runtime compilation; its compiled binding remains bit
identical to the historical overlay. Both geometry and color networks compile forward/backward JIT
kernels and remain JIT-enabled. With fused Adam, mature fixed-4096 median falls further from
48.767 to **45.026 ms** (62.62 Mpoints/s), and B6144 falls from 64.972 to **60.759 ms**
(69.39 Mpoints/s). Raw profiles are
`/home/brans/lookcloser_leader_speed_profiles/profile_fixed4096_fusedadam_jit500_20260716/lookcloser/20260716_143500`
and
`/home/brans/lookcloser_leader_speed_profiles/profile_b6144_fusedadam_jit500_20260716/lookcloser/20260716_144500`.

A same-parent 500-update fused-versus-fused+JIT smoke at exact step91628 gives
`29.536882 / 0.664703 / 0.232146` versus
`29.533745 / 0.665318 / 0.231646`. The JIT delta
`-0.003137 dB / +0.000615 / -0.000500` is inside the accepted repeat bounds and both automatic
artifact protocols are clean. JIT is still default-off; the reviewed speed source fingerprint for
the hook is `5383f012988e0a8804b0ad79cd99aa33f31a43173420088cd5fdb029ddde5d4f`.
The focused suite is 61/61 passing.

The next seed-42 screen combined this speed primitive with a 6144-ray compressed schedule. It kept
hash23, historical ARM, full FAS, stable occupancy and LR endpoints; warmups were 2731 updates,
occupancy/grid intervals `11/683`, depth window 3333, and scheduler horizon 114129. Stage A used
FR1 through step43355, then the exact checkpoint/Adam/scheduler/scaler/RNG state continued with
FR0.3. This was intentionally a quality screen, not a controller milestone run.

| Boundary | Exact points | PSNR | SSIM | LPIPS | Artifacts | Priority detail |
|---:|---:|---:|---:|---:|---:|---|
| 52000 | 175.900 B | 29.464399 | 0.678306 | 0.245791 | 0/3; ROI 0/10 | fail all 3 |
| 59000 extension | 204.646 B | 29.515232 | 0.679500 | 0.240072 | 0/3; ROI 0/10 | fingers pass; pipe/cable fail |

Stage A took 2404.374 s and the first continuation took 600.709 s. The single predeclared
exposure extension from 52k to 59k took another 510.520 s, for 3515.603 s (`58:35.6`) of summed
training processes including their scheduled online evals. This is not a canonical end-to-end wall
because the branch was manually extended and separately finalized. It is enough to reject the
schedule: at 59k the aggregate still misses by `0.102732 dB PSNR` and `0.008937 LPIPS`. Relative to
the archive, thin-pipe and cable-hole LPIPS regress by `0.009742` and `0.005322`; fingers improves
by `0.001303`.

The 52k and 59k structured results are respectively
`/home/brans/lookcloser_leader_speed_runs/b6144_fused_compressed52k_seed42_v1_A_fw03/lookcloser/20260716_135100/candidate_evaluation_step-000052000.json`
and
`/home/brans/lookcloser_leader_speed_runs/b6144_fused_compressed59k_seed42_v1_ext/lookcloser/20260716_140900/candidate_evaluation_step-000059000.json`.
The final checkpoint/config SHA-256 values are
`7e9501dc38fe5304eb8e818e5cbac875588876571f00973c8140198be3798610` and
`0e114806dfc9575a37adce1e0ce53d36153133c712c47d5c12ec8c6c69e6b08a`.
Visual inspection agrees with the metrics: the candidate retains structures but smooths fine cable
and pipe texture. The early point-normalized FR transition therefore enters a worse perceptual
trajectory; simply waiting to the last plausible <=60-minute exposure does not recover it. B6144
compressed scheduling is not encoded in the controller. Fused Adam remains a valid default-off
speed primitive for the next schedule.

## B6144-to-B4096 point-time hybrid

The next predeclared seed-42 diagnostic tested whether B6144 could be used only for the early
point-throughput advantage and then hand optimization back to the accepted B4096 gradient
statistics. It used fused Adam plus repaired TCNN JIT throughout. Stage A ran B6144/FR1 through
step43355. Its immutable checkpoint was forked by scaling the *actual* scheduler epoch by 1.5 while
preserving its instantaneous LR; weights, Adam moments/steps, scaler, all RNG streams, cumulative
points and FAS count were structurally equal before and after the fork. B4096/FR1 then ran through
step54263, very close to the accepted FR boundary in point exposure, and B4096/FR0.3 ran to the hard
candidate boundary step65000. No extension or alternative checkpoint was selected.

| Boundary | Rays / FR | Process wall | Exact points | Adam updates | Scheduler epoch | Fresh PSNR / SSIM / LPIPS |
|---:|---|---:|---:|---:|---:|---|
| 43355 | 6144 / 1.0 | 2243.292 s | 141.943 B | 43337 | 43337 | 29.252958 / 0.677563 / 0.247614 |
| 54263 | 4096 / 1.0 | 505.905 s | 171.443 B | 54240 | 75909 | 29.299082 / 0.677818 / 0.239037 |
| 65000 | 4096 / 0.3 | 500.906 s | 200.902 B | 64973 | 86642 | **29.336306 / 0.678491 / 0.231781** |

The three training processes total 3250.103 s (`54:10.1`). Final evaluation took 22.293 s and the
automatic protocol remained clean at significant artifacts 0/3 and serious ROIs 0/10. Aggregate
quality nevertheless fails by 0.281658 dB PSNR and 0.000646 LPIPS. Only fingers passes the priority
detail gate. Thin pipe regresses by `-0.229784 dB / +0.002622 SSIM / +0.004906 LPIPS` relative to
the archive; cable holes regress by `+0.189674 / +0.002959 / +0.007820`.

The intermediate fresh evaluations quantify the trajectory: each additional roughly 29.5 B points
improves aggregate LPIPS by 0.00858 and then 0.00726, but PSNR only by 0.0461 and 0.0372 dB. Thus
another 17.5 B points could plausibly close the tiny aggregate LPIPS gap but cannot explain or close
the 0.282 dB PSNR gap. The early large batch formed a different optimization basin rather than
merely leaving the candidate under-trained.

At nearly the same FR point boundary, hybrid has 54240 Adam updates versus 75905 for the accepted
fixed-B4096 path, a 28.5% deficit. Scheduler point-time is nevertheless matched to four epochs
(`75909` versus `75905`), and the fork's first resumed LR is continuous. The mechanical switch,
state restore and scheduler remap are therefore exonerated; the supported cause is the early B6144
batch-gradient statistics and missing small-batch updates. Fused/JIT paired differences are two
orders of magnitude smaller than this failure. The final checkpoint/config SHA-256 values are
`7484a0a6162f8598935a9d2c12713418483849ed9fa87128a82e2620ac8f036d` and
`cb8ddf9f718a97527d81383c7d9d47c5b5989088098d9776f03237ab8205d8c8`;
the structured result is
`/home/brans/lookcloser_leader_speed_runs/hybrid_b6144_b4096_jit_seed42_v1_C_fr03/lookcloser/20260716_152800/candidate_evaluation_step-000065000.json`.

## Fixed-B4096 fused+JIT feasibility control

The last predeclared <=60-minute screen restored the proven fixed-B4096 ancestry from update zero
and changed only the parity-tested fused Adam and repaired TCNN JIT speed primitives. A single
process kept FR1 through step75940 and changed the live field to FR0.3 before step75941; it saved the
phase checkpoint but did not evaluate or select it during training. The only candidate was hard
step80000, with no extension or alternate seed/checkpoint.

Training completed in 3318.678 s (`55:18.7`) at 187.309 B exact cumulative points. The observed
runner-launch-to-complete-summary wall was 3396.084 s (`56:36.1`), including an explicit checkpoint
state/hash inspection between training and finalization; adding the immediately preceding 2.415 s
read-only dev3 provenance dry-run still gives `56:38.5`. Thus the speed path demonstrates sufficient
wall feasibility, although it is not promoted because quality is mandatory.

| Candidate | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Priority detail |
|---|---:|---:|---:|---:|---:|---|
| fixed4096 fused+JIT step80000 | **29.748022** | **0.668694** | **0.235192** | 0/3 | 0/10 | fail all 3 |

PSNR and SSIM pass, but LPIPS misses by 0.004057. All three blind priority crops visibly retain
structure but smooth perceptual detail. Relative to the archive, thin-pipe metrics are
`+0.033823 dB / +0.004900 SSIM / +0.010229 LPIPS`; cable holes are
`-0.081736 / +0.001317 / +0.005791`; fingers are
`+0.327911 / +0.007072 / +0.003899`. Numeric pass plus automatic-clean therefore remains a quality
failure, not a speed milestone.

The saved FR boundary was fresh-evaluated only after the predeclared candidate had failed. At
step75940 it gives `29.771954 / 0.672312 / 0.235748`, whereas the historical accepted S1 online full
evaluation at the same step was `29.8455 / 0.671617 / 0.232661`. The long-run delta
`-0.0735 dB / +0.000695 / +0.003087` was already present before FR0.3 and exceeds the relaxed
same-recipe PSNR repeat bound by 0.0135 dB. The 4059-update FR0.3 tail then changes the current run by
`-0.023932 dB / -0.003618 / -0.000556`; it is too short to reproduce the historical perceptual drop.
The 500-update fused/JIT smoke tests were necessary but not sufficient to predict this chaotic
long-horizon divergence.

The final checkpoint/config SHA-256 values are
`a0c06e8280a37e96359022f4d898b1ef765522ce6101e6b9cfb645b8951e2e0e` and
`4f86b070928729b260f48000d793f6f886230fc889e5a1da4bc10ebc42adf075`.
The structured final result is
`/home/brans/lookcloser_leader_speed_runs/fixed4096_fused_jit_s80000_seed42_v1/lookcloser/20260716_154600/candidate_evaluation_step-000080000.json`;
the post-hoc boundary diagnostic is the sibling `candidate_evaluation_step-000075940.json`.

## Insights and next step

Batch 8192 raises mature point throughput by 20.5% over 4096. Batch 16384 gains only another 3.7%
while using 25.3 GiB more peak VRAM and halving the remaining optimizer updates again. The first
quality candidate was therefore batch 8192, seed 42, at the matched step-45564 boundary. It failed
LPIPS both at its scheduled boundary and after exact point matching. Batch 16384 is not promoted:
its throughput gain over 8192 is only 3.7%, while another halving of Adam updates moves in the
direction already shown to hurt perceptual quality. LR/optimizer diagnostics take priority.

Both large-batch solo profiles already use about 86% GPU. Two 8192 runs fit in memory but would be
compute-contended; 8192+16384 would leave little finalization headroom and two 16384 runs cannot fit.
Quality runs are therefore serialized, and only a winning recipe is repeated with a recorded random
seed. Parallel quality screening is not used when it would distort canonical wall and offer little
aggregate throughput. Final milestone claims always use a solo controller-to-gates run. The final
fixed-B4096 fused+JIT screen passed the wall budget but failed LPIPS and every priority-detail crop;
the <=60-minute campaign is therefore closed without promoting a speed recipe or changing the
frozen no-argument reproduction defaults. Reopening it requires a new, separately justified
long-horizon numerical-stability hypothesis rather than another batch/LR/FR/RNG retune.

## Predeclared eval-cadence/shared-RNG causal screen

Forensic review found one previously uncontrolled trajectory difference. The accepted S1 run
performed train-time batch, image and all-image evaluation every 15188 steps, whereas the first
fixed-B4096 fused+JIT feasibility control disabled those evaluations. Training and evaluation pixel
sampling share the global Torch RNG in this frozen implementation. A direct sampler fixture confirms
that inserting one evaluation sample changes the next training sample. Consequently, after step
15188 the two runs did not see the same training-ray/FAS stream despite sharing seed 42.

The predeclared screen is
`fixed4096_fused_jit_evalcadence_s80000_seed42_v1`. It is a solo, uninterrupted run from scratch and
changes exactly one setting relative to `fixed4096_fused_jit_s80000_seed42_v1`: it restores
`steps_per_eval_batch = steps_per_eval_image = steps_per_eval_all_images = 15188`. It retains eval
chunk 16384, fixed 4096 rays, historical ARM, hash23, stable occupancy, fused Adam, TCNN network JIT,
seed 42, FR1 followed by the in-process FR0.3 switch before update 75941, and the sole hard candidate
at step80000. There is no extension, alternate checkpoint, seed substitution or failed-run repeat.

Frozen provenance before launch:

- speed-source fingerprint:
  `5383f012988e0a8804b0ad79cd99aa33f31a43173420088cd5fdb029ddde5d4f`;
- TCNN commit: `2e757bbe781db59c4980d389d7dccbf5edc09669`;
- compiled binding SHA-256:
  `f2163346afd103c27e78b9f56f8d82b6eeb3317c1ce11caf57d45f0216aece36`;
- RTC overlay provenance SHA-256:
  `e5d67f9750465112e3996b13c74e43175a142986590dea90503116dd8aa29606`;
- CUDA 12.6 RTC header hashes are recorded in
  `/home/brans/deps/tcnn_2e757_py310_jit_rtc/rtc_overlay_provenance.json`;
- the immediately preceding local-versus-read-only-dev3 dry-run matched all 202 files, 69 images,
  66 frequency maps and 66 metadata files; transforms SHA-256 remains
  `022f8748a1a039861a754e68ab3ef830beeb3e5dd94ccb00457a630d28f64aa1`.

The step75940 scheduled evaluation is a causal diagnostic against accepted S1 under the relaxed
repeat limits `0.06 dB / 0.01 / 0.005`; it is not an eligible candidate. Step80000 alone must pass
PSNR >=29.617964, SSIM >=0.668450, LPIPS <=0.231135, significant artifacts 0/3, serious ROIs 0/10
and all three priority-detail gates within 3600 seconds controller-to-gates. Any miss permanently
rejects this recipe. A pass would only authorize one separately predeclared identical confirmation;
both runs must pass before promotion.

### Result

The immediately preceding same-checkpoint exact-RNG tail control was itself not fully identifiable.
Reloading the immutable no-eval step75940 checkpoint and continuing to step80000 with its saved RNG
state produced `29.808233 / 0.666252 / 0.233172`, versus
`29.748022 / 0.668694 / 0.235192` in the original uninterrupted run. The delta was
`+0.060211 dB / -0.002442 / -0.002020`: PSNR exceeded the relaxed repeat limit by 0.000211 dB.
The historical-RNG-restart treatment was therefore not run and neither tail result is eligible for
promotion. This bounds checkpoint-resume nondeterminism but does not explain the already present
pre-boundary divergence.

The eval-cadence screen completed its one uninterrupted hard trajectory. Its scheduled full-eval
comparison against accepted S1 was:

| Step | Current PSNR / SSIM / LPIPS | S1 PSNR / SSIM / LPIPS | Current - S1 |
|---:|---|---|---|
| 15188 | 28.5956 / 0.650295 / 0.364300 | 28.7480 / 0.645900 / 0.361598 | -0.1524 / +0.004395 / +0.002702 |
| 30376 | 29.2384 / 0.672993 / 0.299446 | 29.4546 / 0.660666 / 0.295134 | -0.2162 / +0.012327 / +0.004312 |
| 45564 | 29.4902 / 0.671648 / 0.268324 | 29.6927 / 0.667343 / 0.263583 | -0.2025 / +0.004305 / +0.004741 |
| 60752 | 29.4727 / 0.673530 / 0.250598 | 29.7795 / 0.669390 / 0.245372 | -0.3068 / +0.004140 / +0.005226 |
| 75940 | 29.3260 / 0.669644 / 0.239800 | 29.8455 / 0.671617 / 0.232661 | -0.5195 / -0.001973 / +0.007139 |

The first row occurs before evaluation can alter a later training sample, so fused/JIT already had
a non-bit-exact long-horizon offset before the cadence intervention. Later scheduled evaluations do
change the shared training RNG exactly as expected, but the trajectory moves farther from S1 rather
than converging to it. At step75940 the current checkpoint has approximately 175.913 B cumulative
points, essentially the accepted exposure, so under-training is not the explanation. Restoring
cadence is a real semantic correction for historical trajectory matching, but it is not sufficient
when fused Adam and TCNN JIT are enabled together.

The sole hard step80000 candidate is a complete failure:

| Candidate | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Priority detail |
|---|---:|---:|---:|---:|---:|---|
| eval-cadence fused+JIT step80000 | **29.374962** | **0.669688** | **0.239359** | **1/3** | 0/10 | pipe fail, cable fail, fingers pass |

It misses PSNR by 0.243002 dB and LPIPS by 0.008224 and has one serious full-view artifact in
eval0. Relative to the archive, thin pipe changes by
`-0.366999 dB / +0.003440 SSIM / +0.007650 LPIPS`, cable holes by
`-0.010168 / +0.002201 / +0.010934`, and fingers by
`+0.264519 / +0.004994 / -0.000334`. Numeric, full-view artifact and priority-detail gates all
reject the recipe.

Training took 3461.593 s (`57:41.6`). The candidate summary was written 3509.930 s (`58:29.9`)
after the training config was created; allowing the measured roughly six-second runner setup gives
approximately `58:36` controller-to-gates. Thus the cadence screen fits the <=60-minute wall but
does not earn a milestone because quality is mandatory. Checkpoint/config SHA-256 values are
`156066c06f6d8cf7bb1f9a77859cb1b500385f5f4f9987897e29af291efc4421` and
`81525c04ab5de661dc68434c8c4e398caefc41f322205393a2a23d4818ddce88`.
The structured result is
`/home/brans/lookcloser_leader_speed_runs/fixed4096_fused_jit_evalcadence_s80000_seed42_v1/lookcloser/20260716_170200/candidate_evaluation_step-000080000.json`.

This recipe is permanently rejected: no repeat, extension, alternate checkpoint or seed may erase
the result. The next long screen must isolate fused-only from JIT-only under the historical cadence;
changing LR, batch or FR simultaneously would confound the now-demonstrated numerical-basin issue.

A follow-up 500-update mature profile measured the previously missing JIT-only cost with historical
Adam (`fused=None`, `tcnn_network_jit=True`). It gives 49.839 ms/update at median 2.818 M points,
or 56.45 Mpoints/s: 7.23% faster than the 53.721 ms historical-Adam path, but 10.69% slower than
fused+JIT. Scaling the measured full run predicts roughly 64 minutes end-to-end through step80000
once historical cadence and final gates are included. JIT-only can isolate long-horizon quality but
cannot by itself satisfy the <=60-minute milestone at this stopping point. The raw profile is
`/home/brans/lookcloser_leader_speed_profiles/profile_fixed4096_jitonly500_20260716/lookcloser/20260716_180421`.

## Predeclared early fused x JIT factorial

Full fused-only and JIT-only step80000 runs are not launched: the measured mature costs project to
roughly 63--65 minutes controller-to-gates and therefore cannot be <=60-minute candidates. Instead,
one bounded causal campaign freezes current speed source, seed42, fixed B4096, historical ARM/hash23,
stable occupancy, full FAS/FR1, LR schedule, shared RNG and eval/save cadence15188, and stops every
new arm at hard step15188:

- `H`: historical Adam, JIT off;
- `F`: fused Adam, JIT off;
- `J`: historical Adam, JIT on;
- `FJ`: already measured by the eval-cadence run at
  `28.5956 / 0.650295 / 0.364300`.

Accepted S1 at the same scheduled boundary is `28.7480 / 0.645900 / 0.361598`. The three new arms
are serialized so GPU contention cannot change their timing or execution environment. Each has one
predeclared step15188 scheduled full eval and checkpoint; no extension, alternate checkpoint, seed
or artifact promotion is allowed. All arms use the same RTC overlay, whose compiled binding is
bit-identical when JIT is disabled.

First, `H` must match S1 within `0.06 dB / 0.01 / 0.005`. If it does not, fused/JIT attribution is
invalid and the cause lies in the common semantic-fast source or its long-horizon numerical
interaction. If `H` passes, compare `F-H` and `J-H` with the same limits. A single failing component
identifies that primitive; two individually passing components with failing `FJ` identify an
interaction. Exposure must agree within 0.1%, with matched Adam/scheduler counts and no anomalous AMP
skip burst. These are diagnostic arms, not speed winners, and no cell can be promoted from this
short screen.

### Factorial result

The first `H` process used eval chunk16384 and completed all training updates, but TCNN failed a
large final-render allocation with CUDA OOM before any eval row or checkpoint was written. It is an
infrastructure-only run with no quality result. The complete factorial therefore uses chunk2048,
matching accepted S1. The excluded run is
`factorial15188_H_histadam_nojit_seed42_v1`; its `_v2_chunk2048` replacement and both component arms
were rerun from scratch, serialized, with no other change. The existing `FJ` checkpoint was also
fresh-evaluated at chunk2048 and reproduces its online numbers.

| Cell | Adam / JIT | PSNR | SSIM | LPIPS | Delta from H | Relaxed repeat |
|---|---|---:|---:|---:|---|---|
| S1 | historical / off | 28.7480 | 0.645900 | 0.361598 | reference | -- |
| H | historical / off, speed source | 28.7406 | 0.648612 | 0.359473 | reference | pass vs S1 |
| F | fused / off | 28.6758 | 0.650305 | 0.357639 | -0.0648 / +0.001693 / -0.001834 | **fail PSNR** |
| J | historical / on | 28.6093 | 0.647185 | 0.365135 | -0.1313 / -0.001427 / +0.005662 | **fail PSNR, LPIPS** |
| FJ | fused / on | 28.595589 | 0.650295 | 0.364262 | -0.145011 / +0.001683 / +0.004789 | **fail PSNR** |

`H-S1` is only `-0.0074 dB / +0.002712 / -0.002125`, so the common semantic-fast source passes and
is not the cause of the early offset. Fused Adam alone exceeds the PSNR repeat limit by 0.0048 dB;
JIT alone exceeds PSNR by 0.0713 dB and LPIPS by 0.000662. Both primitives are therefore unsafe for
from-scratch faithful training even though their same-parent 500-update late smoke tests pass.

The point-exposure audit refines, but does not weaken, that conclusion. All four speed cells are
bit-identical through the fixed warmup at step4090 (`4.28972 B` cumulative points). Once adaptive
occupancy starts, their closed loops diverge. At step15180, H/F/J/FJ have respectively
`22.4922 / 22.5379 / 22.3638 / 22.3396 B` points. This exceeds the predeclared 0.1% direct-attribution
bound, so the metrics cannot be described as a pure optimizer or MLP arithmetic effect at exactly
matched exposure. Instead, the primitive's first numerical differences alter density/occupancy,
which then alter traversal, training point exposure and the later basin. That feedback is itself a
disqualifying semantics change for the frozen recipe.

Checkpoint state is otherwise coherent. S1 and H both have 15179 Adam/scheduler updates, LR
0.00705033901 and AMP scale8192. F and J have 15180 updates and LR0.00705017667; F's scaler reached
16384 while J remained8192. This is a one-update numerical overflow-path difference, not an
anomalous skip burst, and it is downstream of the primitive choice. Checkpoint SHA-256 values for
H/F/J are respectively
`d4522c86c5e5bb4c4cd8db1451f018db2bbb25a9ebd200e2c5e98430c9e34461`,
`482a0c852755c5778a24ae8cab68638d2435fee08f457e099ae50e4ed9c32d32` and
`12dbc981d9b09319380c35a3fe779cc45e943c2fbee55fe379cbd5c8733b655a`.

The component campaign is closed without a full fused-only or JIT-only run. The supported next
direction is delayed activation from a proven historical-Adam/no-JIT checkpoint, with a paired
historical-resume control. This tests whether the accepted basin becomes robust to the fast kernels
after an early stable phase; enabling either primitive at initialization is no longer admissible.

## Predeclared delayed-activation pair at step15188

The next bounded experiment tests whether the fast kernels are safe after a reproducible historical
basin has formed. Both arms load the immutable current-source H checkpoint
`factorial15188_H_histadam_nojit_seed42_v2_chunk2048/.../step-000015188.ckpt`, SHA-256
`d4522c86c5e5bb4c4cd8db1451f018db2bbb25a9ebd200e2c5e98430c9e34461`. It contains the matched
historical Adam/scheduler state (15179 updates, LR 0.00705033901), AMP scale8192, cumulative point
counter and Python/NumPy/Torch CPU/CUDA RNG snapshot captured after the scheduled step15188 eval.
Runtime restores the RNG only after model/data/callback/writer setup, immediately before the first
resumed iteration.

The serialized arms continue through the next hard cadence boundary step30376 with FR1, B4096,
historical ARM/hash23, full FAS, stable occupancy, scheduler horizon200000, eval/save cadence15188
and eval chunk2048:

- `H-resume`: historical Adam, JIT off;
- `FJ-delayed`: migrate the same Adam moments/steps to fused Adam and construct the same field
  weights with TCNN JIT enabled.

No LR/scheduler/scaler/RNG reset, extension, alternate checkpoint or seed is allowed. Step30376 is
the only diagnostic boundary. First, H-resume must match uninterrupted accepted S1 within the
relaxed `0.06 dB / 0.01 / 0.005`; otherwise the added restart is not a faithful baseline. If the
control passes, FJ-delayed must match H-resume inside the same limits with coherent Adam/scheduler
counts and no anomalous AMP skip burst. Adaptive point exposure is reported as a downstream
closed-loop effect rather than forced equal by changing steps or LR. A passing pair authorizes a
longer predeclared delayed-activation candidate; neither short arm is independently promotable.

### Delayed-pair result

Both arms completed from the common H step15188 parent. The historical `H-resume` control reached
step30376 at `29.3432 / 0.662563 / 0.296550`; relative to uninterrupted S1 at the same boundary
(`29.4546 / 0.660666 / 0.295134`), its delta is
`-0.1114 dB / +0.001897 SSIM / +0.001416 LPIPS`. SSIM and LPIPS remain inside the relaxed repeat
bounds, but PSNR exceeds the `0.06 dB` limit. The added process restart is therefore not a faithful
substitute for the uninterrupted historical trajectory, even with the saved post-eval RNG state.

The paired `FJ-delayed` arm reached `29.3246 / 0.664055 / 0.297543`. Its delta from `H-resume` is
`-0.0186 dB / +0.001492 SSIM / +0.000993 LPIPS`, passing all three relaxed repeat limits. Training
wall for the resumed segment was `761.132 s` for H and `670.955 s` for FJ. Cumulative point exposure
at step30370 was `56.8173 B` versus `56.8314 B`, only `0.0248%` apart. Both checkpoints retain
coherent step30376 trainer state, 30360 Adam/scheduler updates, LR `0.00497049911` and AMP scale8192.

Artifacts:

- H run:
  `/home/brans/lookcloser_leader_speed_runs/delayed15188_Hresume_to30376_seed42_v1/lookcloser/20260716_185400`;
  step30376 checkpoint SHA-256
  `c6326d7e4dff56eacd252f35621fe5b3e066a2d5bdceaa4480ebe521ad428f2c`;
- FJ run:
  `/home/brans/lookcloser_leader_speed_runs/delayed15188_FJ_to30376_seed42_v1/lookcloser/20260716_190700`;
  step30376 checkpoint SHA-256
  `6d4db0407e40bdfaf6bc400eef27b65655394a4e3ebbfb8cc3c6517add0bf460`.

This result supports a narrow conclusion: once the historical prefix has formed, fused Adam plus
TCNN JIT is safe relative to its paired resumed control over steps15189--30376. It does not
authorize a promoted speed recipe because the process restart itself moved PSNR outside the
reproducibility bound. The next test must activate both primitives inside the original process.

## Predeclared uninterrupted live activation at step15189

The next causal test is one seed-42 run from scratch with no checkpoint reload. It uses historical
Adam and TCNN JIT off for updates 0--15188, including the scheduled step15188 evaluation, then
switches both primitives before the forward/optimizer update at step15189 and continues in the
same process through the sole diagnostic boundary step30376. B4096, historical ARM/hash23, full
FAS, FR1, stable occupancy, scheduler horizon200000, eval/save cadence15188 and eval chunk2048 stay
fixed. Its step30376 metrics must match uninterrupted S1 within
`0.06 dB / 0.01 SSIM / 0.005 LPIPS`; no extension, alternate checkpoint or seed is eligible.

The live controls are opt-in and default to `None`, so neither the frozen reproduction command nor
ordinary speed commands change. The core rejects initial plus scheduled double-enable for either
primitive. The TCNN switch mutates the existing geometry/color modules' JIT flags without
reconstructing their parameters; its callback is prepended so it runs before occupancy callbacks
and the training forward at the boundary. The Adam switch rejects foreach or differentiable
optimizers, synchronizes `_step_supports_amp_scaling`, moves scalar `step` state to the parameter
device and sets fused execution while preserving parameter groups, weights, moments and optimizer
step counts. A post-checkpoint-load resync reapplies those invariants. Scheduler, AMP scaler, RNG
streams and cumulative-point telemetry remain continuous. The switch is performed before
update15189 and its active state is logged; it must not create a checkpoint boundary or reseed any
subsystem.

Three excluded attempts refined only the launch and switch implementation. `v1` used a fresh
`TORCH_EXTENSIONS_DIR`; `v2` selected the canonical cache but omitted `CUDA_HOME` and
`TORCH_CUDA_ARCH_LIST`. Both reached approximately step4096 and then tried an unsupported
`compute_120` extension build, so both are infrastructure-only with no diagnostic metric. `v3` used
the full canonical CUDA/cache environment and was intentionally stopped before the switch after
review found that live fused Adam had not synchronized `_step_supports_amp_scaling`. The corrected
implementation passed 66 focused tests plus real CUDA integration and checkpoint-resume smokes
before `v4` was launched.

### Live-activation result

The uninterrupted corrected run is
`/home/brans/lookcloser_leader_speed_runs/inprocess15189_FJ_to30376_seed42_v4_ampfix/lookcloser/20260716_211500`.
Before activation, its scheduled step15188 full evaluation was
`28.7861 / 0.647185 / 0.360978`. The train log then records both exact boundary events at step15189:
the `fields` optimizer switched to fused Adam and TCNN network JIT was enabled. Fresh evaluation of
the sole step30376 checkpoint gives `29.437275 / 0.665461 / 0.298865`. Against uninterrupted S1 at
the same boundary, the delta is `-0.017325 dB / +0.004795 SSIM / +0.003731 LPIPS`, passing the
predeclared `0.06 / 0.01 / 0.005` reproducibility limits.

Training took `1277.004 s`; controller total through the fresh evaluation was `1304.995 s`.
Checkpoint state records `57,053,993,419` cumulative points, 30360 Adam and scheduler updates, LR
`0.004970499105762349`, AMP scale8192 and fused execution active. Provenance hashes are:

- step30376 checkpoint:
  `6d65b7ecba4677b6fdc96cb3c76d5dcbab43320ffaf3205549349f691e945682`;
- `config.yml`:
  `965d6f40c5c3d169a7d706cd596520c903c163d4e9a5763187795e536bde31ac`;
- fresh `eval_latest_step-000030376.json`:
  `8090d7d38c5e5f85b40f8c7092daf9a878d3b6b97a224e970358402a8dbeba8c`.

Activation at step15189 is therefore accepted for a longer predeclared diagnostic relative to S1
at step30376. This early-boundary pass is not a final quality result and is not yet a complete
`<=60`-minute end-to-end candidate; the frozen reproduction defaults remain unchanged.

## Predeclared <=60-minute hard candidate: live activation plus eval-trajectory replay

The next and only eligible candidate in this experiment is a solo seed-42 end-to-end controller
run through hard checkpoint step80000. It keeps B4096, historical ARM/hash23, stable occupancy,
full FAS, LR `0.01 -> 0.0001` over 200000 scheduler updates and FR1 through Stage A. Historical Adam
and TCNN JIT remain active exactly as in the reproduction through update15188; both fast kernels
activate in-process before update15189. The exact historical Stage-A process boundary is retained
at step75940, after which Adam/scheduler/scaler/checkpoint state is restored and only FR changes to
0.3 for updates75941--80000.

Scheduled evaluation still advances every stateful sampler and RNG consumer in the original order,
but at intermediate boundaries it uses `replay_eval_trajectory=true`: eval batch sampling, random
eval-image selection and all three fixed-image dataloader iterations execute without model forward,
metric reduction or rendering. This removes only intermediate evaluation compute. It does not skip
the stochastic trajectory that historically changes later training rays and FAS samples. The final
step80000 checkpoint receives the ordinary fresh full evaluation, renders, automatic artifact
gates and frozen detail/contact-sheet protocol.

A real CUDA preflight loaded the same step30376 checkpoint twice, restored its saved RNG state and
compared one ordinary batch/image/all-image evaluation with the replay path. Post-action Python,
NumPy, Torch CPU and every CUDA RNG state matched exactly; eval counters and sampler counts matched;
the following training image, pixel indices, ray origins/directions, pixel areas and camera IDs
were byte-identical. Relevant speed-worktree tests pass `66/66`, including replay ordering and
runner/controller plumbing.

The candidate is fixed before launch:

- campaign: `leader_live15189_evalreplay_hard80000_seed42_v1`;
- sole quality checkpoint: step80000; no extension, alternate checkpoint, seed substitution or
  failed-run repeat;
- numeric and visual/detail gates are unchanged from the frozen reproduction recipe;
- controller wall-clock from provenance start through final gates must be `<=3600 s`;
- a quality pass after 3600 seconds is recorded separately as a wall-milestone failure.

The frozen no-argument reproduction defaults remain historical Adam, JIT off and full evaluation;
live activation, trajectory replay and hard step80000 are explicit opt-in speed controls only.

### Hard-candidate result

The solo controller campaign completed its entire protocol in `3511.836 s` (`58:31.8`), so the
`<=3600 s` wall gate passes with `88.164 s` of headroom. Stage A took `3263.016 s`, the restored
FR0.3 segment took `195.283 s`, and final fresh evaluation plus all gates completed inside the same
controller clock. Exact checkpoint telemetry reports `188.656 B` cumulative point samples, trainer
step80000, 79963 Adam/scheduler updates, LR `0.00158624403` and AMP scale16384.

The sole hard checkpoint passes all aggregate numeric and automatic artifact gates but is rejected
by the priority-detail gate:

| Candidate | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Priority detail |
|---|---:|---:|---:|---:|---:|---|
| live15189 + eval replay, step80000 | **29.741262** | **0.676903** | **0.228587** | **0/3** | **0/10** | pipe fail, cable fail, fingers pass |

Relative to the frozen archive detail reference, thin pipe is
`-0.046883 dB / +0.001678 SSIM / +0.000785 LPIPS`; cable holes are
`+0.020723 / +0.002226 / +0.003640`; fingers are
`+0.407448 / +0.005859 / -0.001722`. The gate requires every metric of each priority crop to equal
or beat the reference within `1e-4`, so the positive cable PSNR/SSIM cannot compensate its LPIPS
regression. Full views and the ten structural ROIs are clean; this is a localized perceptual-detail
failure rather than an occupancy-hole failure.

A post-hoc fresh evaluation of the immutable Stage-A step75940 checkpoint is diagnostic only and
cannot replace the predeclared hard candidate. It gives `29.780493 / 0.673463 / 0.231288`, clean
`0/3` and `0/10`. The FR0.3 tail to step80000 improves aggregate LPIPS by `0.002701` while changing
PSNR by `-0.039231` and SSIM by `+0.003440`. Thin-pipe ROI improves by
`+0.091089 / +0.000529 / -0.001924`, but cable ROI changes by
`+0.022038 / +0.000237 / +0.000848`: its LPIPS moves farther from the archive. Therefore simply
extending this failed checkpoint is neither eligible nor supported by the observed cable slope.

Immutable result artifacts:

- campaign manifest:
  `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_live15189_evalreplay_hard80000_seed42_v1/campaign.json`;
- step80000 checkpoint SHA-256:
  `0c3492e40cf67037f5c4acae7e1cc26aeb021850f25e31b0909628b9678e6751`;
- config SHA-256:
  `9a7d90e3ac5082f8ee5079083dbe50b06da6d3b3ae081557acd17eee37ea8362`;
- candidate summary SHA-256:
  `b55d0ba2d81b3538565756c89f71ce206942cbc448d7574845fe080ca7e9a8f0`.

The `<=60` compute milestone is feasible but not yet quality-accepted. This campaign is not repeated
and no later checkpoint from it is eligible. The next recipe must target the demonstrated
long-horizon detail-basin effect while retaining the measured wall feasibility.

## Forensic Stage-B RNG correction and next hard candidate

Post-run checkpoint audit found one material mismatch with the frozen leader restart. The new speed
worktree persists Python/NumPy/Torch CPU/all-CUDA RNG state, so the failed hard run restored its
step75940 sampling stream before the first FR0.3 update. The archived leader checkpoint contains no
RNG snapshot: its new Stage-B process retained the deterministic seed-42 post-setup streams. Model,
Adam, scheduler and scaler continuation were correct, but the failed run did not reproduce the
historical Stage-B ray/FAS stream.

The controller now exposes the fail-closed
`--historical-stage-boundary-rng-reset` hard-candidate control. After Stage A it automatically uses
`fork_static_checkpoint_optimizer.py --drop-rng-state`, hashes the source and output, and validates
that trainer step, model payload, Adam moments/steps, scheduler, LR and scaler are unchanged. A real
fork of the failed run's 2-GB step75940 checkpoint took `3.35 s`; before/after telemetry is trainer
75940, Adam/scheduler75904, LR `0.001741646454`, scaler8192, with only `rng_state_present` changing
from true to false. Stage B then obtains the same documented new-process seed semantics as the
historical leader.

Final-eval chunk size was profiled independently on the immutable failed step80000 checkpoint:

| Eval chunk | Fresh eval seconds | PSNR | SSIM | LPIPS | Automatic/detail outcome |
|---:|---:|---:|---:|---:|---|
| 2048 | 27.337 | 29.741262 | 0.676903 | 0.228587344 | clean; pipe/cable fail |
| 4096 | 23.675 | 29.741262 | 0.676903 | 0.228587240 | identical outcome |
| 8192 | **22.061** | 29.741262 | 0.676903 | 0.228587613 | identical outcome |

All renders, `0/3` full-view artifacts, `0/10` serious ROIs and detail decisions agree. Chunk8192
is selected; chunk16384 remains excluded because an earlier hash23 final render OOMed. This is a
validated finalization-only optimization and does not alter training or checkpoint selection.

The next sole E2E candidate is predeclared as
`leader_live15189_evalreplay_rngreset_hard80000_seed42_v1`. It changes only the historical Stage-B
RNG-reset semantics relative to the failed training recipe, plus the independently parity-checked
eval chunk8192. Fused Adam and JIT still activate together before update15189, intermediate eval
trajectory replay remains exact, FR changes only after step75940, and step80000 is the only quality
checkpoint. No extension, alternate checkpoint, seed substitution or failed-run repeat is allowed.
The measured fork cost and eval saving approximately cancel, so expected controller wall remains
about `3510--3520 s`, inside the same `3600 s` gate.

If this faithful-reset candidate fails, the next causal training change is to keep fused Adam at
15189 but delay only TCNN JIT to the validated cadence boundary30377. JIT is the larger early LPIPS
perturbation; that staggered recipe is not mixed into the RNG-reset control.

### Faithful-reset hard-candidate result

The candidate completed the full controller protocol in `3495.525 s` (`58:15.5`), including the
`3.412 s` verified RNG-reset fork. Stage A took `3247.870 s`, Stage B took `195.258 s`, and final
step80000 contains `188.306 B` exact cumulative points. Aggregate metrics improve to
`29.765554 / 0.669484 / 0.225159`, comfortably passing all three numeric gates.

Quality still fails. The full-view detector finds one significant eval1 component, score `0.172`
and area419 px; all ten ROI gates remain non-serious. Historical reset makes thin pipe pass at
`+0.139521 dB / +0.001195 / -0.003538 LPIPS` relative to the archive, but cable and fingers fail
only LPIPS at `+0.005673` and `+0.002719` respectively while beating the reference in PSNR/SSIM.
Thus the reset is a real aggregate/pipe improvement, not a complete leader reproduction.

The next and only eligible recipe is predeclared as
`leader_fused15189_jit30377_evalreplay_rngreset_hard80000_seed42_v1`. It changes exactly one
training factor: fused Adam remains active from15189, while TCNN JIT moves from15189 to30377, just
after the second cadence replay. Historical Stage-B RNG reset, FR boundary, hard step80000,
chunk8192 and every gate stay fixed. Mature profiles predict about `56.8 s` extra training wall,
or roughly `3552 s` total with about `48 s` headroom. No extension, alternate checkpoint, seed
substitution or failed-run repeat is allowed.

### Staggered fused/JIT hard-candidate result

The predeclared staggered campaign completed the full controller protocol in `3510.713 s`
(`58:30.7`), passing the `<=3600 s` wall gate by `89.287 s`. Stage A took `3262.732 s`, the verified
historical RNG-reset fork took `3.376 s`, Stage B took `195.282 s`, and finalization took `44.452 s`.
The exact boundary log confirms fused Adam before update15189, replay at step30376, and TCNN JIT
before update30377. Total estimated point exposure was `188.420 B`; checkpoint telemetry records
`188.403 B` exact cumulative points.

The sole hard checkpoint passes aggregate and automatic artifact gates but again fails priority
detail:

| Candidate | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Priority detail |
|---|---:|---:|---:|---:|---:|---|
| fused15189 / JIT30377, step80000 | **29.758568** | **0.678173** | **0.224903** | **0/3** | **0/10** | pipe pass, cable fail, fingers fail |

Relative to the frozen archive detail reference, thin pipe is
`+0.192583 dB / +0.004795 SSIM / -0.000789 LPIPS` and passes. Cable holes are
`+0.186100 / +0.003119 / +0.001164`; fingers are
`+0.224449 / +0.004958 / +0.006197`. Both failed crops therefore beat the reference in PSNR and
SSIM and miss only local LPIPS. Unlike the joint-switch faithful-reset run, the staggered run has
no full-view artifact. Delaying JIT through the second cadence reduced the cable LPIPS miss from
`+0.005673` to `+0.001164`, but increased the fingers miss from `+0.002719` to `+0.006197`. This is
a measured local-perceptual trade rather than a monotonic all-crop recovery, so the candidate is
rejected and no checkpoint/seed substitution or extension is eligible.

Immutable result artifacts:

- campaign manifest:
  `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_fused15189_jit30377_evalreplay_rngreset_hard80000_seed42_v1/campaign.json`;
- step80000 checkpoint SHA-256:
  `0ef7ea53cfcdcc68e67c1dc8811584115d6263ae513370072c6bdcb8dda5f8ed`;
- config SHA-256:
  `da6caa7c00ee873d1a9d5560ce2664b458faed577d8e916cfed554ffb5a6b5ea`;
- candidate summary SHA-256:
  `54601c4365e5be450edf763906e2281a10d7f5bfb1541a7b821a878b0a5b0391`;
- detail JSON SHA-256:
  `8c6a9460eeaff203dbe37708f3d6f9808b1d002a5c9cbbdb3d34e8eb94b085ec`.

This result is wall-feasible but not promoted. Frozen reproduction defaults remain unchanged while
the next experiment isolates the remaining JIT-versus-tail cause from these saved checkpoints.

## Predeclared parallel Stage-B LR/Adam diagnostic after the staggered run

Before another from-scratch wall candidate, a diagnostic reuses the immutable staggered Stage-A
step75940 checkpoint and changes only late optimizer policy. Every branch uses seed42, the verified
historical process-boundary RNG reset, FR0.3, fused Adam and already-active JIT, and stops at the
same step80000. The baseline `1x / loaded Adam` branch is the rejected hard candidate above; four
new branches form `{2x,4x late LR} x {loaded Adam, reset Adam}`. LR multiplication scales the loaded
optimizer LR, initial LR and scheduler base/last LR together; scheduler epoch and scaler remain
unchanged. Reset branches clear only Adam moments/step after the same LR transformation.

The four short branches may run concurrently because each uses about 10 GiB on the 98-GB GPU.
Their timings are explicitly contended diagnostics and cannot support the `<=60` milestone. Each
receives a fresh step80000 evaluation and the same numeric, `0/3`, `0/10` and frozen priority-detail
gates. Collapse or a detail miss rejects that branch. A passing or clearly Pareto-dominant policy
must then be encoded fail-closed in the controller and rerun solo from scratch before promotion.

### Parallel Stage-B LR/Adam result

Dataset provenance was byte-identical immediately before launch. Four branches ran concurrently,
using about 59 GiB total, and each reached step80000 without NaN/OOM. Their approximately `780.8 s`
per-branch wall is deliberately contended and is not a speed result.

| Late policy | PSNR | SSIM | LPIPS | Artifacts / serious ROI | Pipe LPIPS delta | Cable LPIPS delta | Fingers LPIPS delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x, loaded Adam baseline | 29.758568 | 0.678173 | 0.224903 | 0/3 / 0/10 | -0.000789 | +0.001164 | +0.006197 |
| 2x, loaded Adam | 29.706211 | 0.675034 | 0.225387 | 0/3 / 0/10 | -0.001101 | +0.002608 | +0.006199 |
| 2x, reset Adam | 29.702723 | 0.676112 | 0.225520 | 0/3 / 0/10 | -0.000470 | +0.002983 | +0.006268 |
| 4x, loaded Adam | 29.595901 | 0.668852 | 0.231945 | 0/3 / 0/10 | +0.003065 | +0.008229 | +0.007888 |
| 4x, reset Adam | 29.644838 | 0.672442 | 0.230962 | 0/3 / 0/10 | +0.002905 | +0.009359 | +0.008349 |

The 2x branches preserve aggregate gates but worsen cable and do not move fingers; resetting Adam
changes neither conclusion. At 4x the loaded branch fails aggregate PSNR/LPIPS, and both policies
fail all three priority crops. A late LR impulse therefore does not compress the missing perceptual
tail and is rejected. The supported next single-factor candidate instead moves the already
historical FR `1.0 -> 0.3` transition in-process to update64813. This gives exactly 15188 low-FR
updates through hard step80000 while preserving LR, Adam, rays, JIT, RNG-reset semantics and wall
work. It must still be implemented as an exact fail-closed controller schedule and rerun solo.

Diagnostic root:
`/home/brans/lookcloser_leader_speed_runs/diagnostics/staggered_stageb_lr_adam_seed42_v1`.

## Predeclared early-FR <=60-minute hard candidate

The next sole E2E campaign is
`leader_fused15189_jit30377_fr64813_evalreplay_rngreset_hard80000_seed42_v1`. It retains the
staggered backend schedule, seed42, B4096, historical ARM/hash23/FAS, LR/Adam/scheduler, exact eval
replay, verified Stage-B RNG reset, chunk8192 and hard step80000. Its only training change from the
rejected staggered candidate is an in-process FR switch from1.0 to0.3 before update64813. FR0.3 then
runs for exactly15188 updates through step80000, matching the accepted leader's full low-FR tail
length without adding optimizer updates.

The controller exposes this only as exact `(64813,0.3)` and emits it only into Stage A; arbitrary
FR boundaries/strengths fail closed. Focused controller tests pass `23/23`, and the protocol
fingerprint is `fef10e50e641ada1f7c3387529f4e28b70354b474ba4183dbf6c14a8a540a1b4`. The sole checkpoint must
pass every numeric, artifact, ROI and priority-detail gate within controller wall `<=3600 s`; no
extension, alternate checkpoint, seed substitution or failed-run repeat is eligible.

### Early-FR hard-candidate result

The solo campaign completed the full protocol in `3556.836 s` (`59:16.8`), passing the wall gate by
`43.164 s`. Stage A took `3292.942 s`, the verified RNG-only fork about `3.4 s`, Stage B
`210.280 s`, and finalization `44.445 s`. The FR switch log occurs exactly once, at step64813; Stage
B starts directly at FR0.3. Exact cumulative points were `178.839 B` at step75940 and `190.223 B`
at step80000 (`190.237 B` legacy total estimate).

The sole hard checkpoint is rejected:

| Candidate | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Priority detail |
|---|---:|---:|---:|---:|---:|---|
| fused15189 / JIT30377 / FR64813, step80000 | **29.757217** | **0.669999** | **0.231657** | **0/3** | **0/10** | pipe fail, cable fail, fingers fail |

Aggregate LPIPS misses its gate by `0.000522`. Relative to the archive, thin pipe misses LPIPS by
`+0.000495`, cable by `+0.004532`, and fingers by `+0.007049`; all three otherwise beat the
reference in PSNR/SSIM. Relative to the matched staggered hard candidate, early FR changes aggregate
quality by `-0.001350 dB / -0.008174 SSIM / +0.006754 LPIPS` and changes all local LPIPS metrics in
the rejected direction. Thus additional low-FR exposure is not a compressed substitute for the
accepted late trajectory; this schedule is closed with no repeat, extension or alternate checkpoint.

Immutable artifacts:

- campaign manifest:
  `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_fused15189_jit30377_fr64813_evalreplay_rngreset_hard80000_seed42_v1/campaign.json`;
- step80000 checkpoint SHA-256:
  `0bf69dce13292b2c4672ad47e7538b9cf40b6d93ce13a18b6aa1b4f3d5227ea9`;
- config SHA-256:
  `e566836e6c23e7ba22f8445c17081957da1084f01c76172986b988326098c19b`;
- candidate summary SHA-256:
  `ea52b93adcb0148d738b1c48707e602e1feae656bdf26d3fd5b39ea88773ba31`;
- detail JSON SHA-256:
  `e9b4d07945d8ac77b81caf057cc77475da26741bb8b7bbe61dcd3534e1250cb0`.

A post-hoc fresh evaluation of the matched staggered FR1 step75940 checkpoint further localizes the
failure. Aggregate `29.858776 / 0.676556 / 0.228865`, artifacts `0/3` and serious ROIs `0/10` all
pass, but pipe/cable/fingers miss only LPIPS by `+0.001261 / +0.001808 / +0.006177`. The ordinary
4059-update late FR0.3 tail therefore improves pipe by `0.002050` and cable by `0.000644`, while
fingers changes by only `+0.000020`. The dominant fingers gap is already present before Stage B;
late LR and FR timing cannot be its primary cause. The next treatment must preserve the earlier
training basin more faithfully or recover no-JIT quality with additional measured throughput.

One final bounded same-parent tail control keeps FR1.0 from staggered step75940 through step80000,
using the already verified historical RNG-reset fork and unchanged LR/Adam/JIT. It is diagnostic,
not an eligible replacement checkpoint. This tests whether the extra 4059 updates alone recover
the pre-existing crop gaps, independently of the FR0.3 transition.

The FR1 tail reaches `29.807636 / 0.676702 / 0.225012`, clean `0/3` and `0/10`; pipe passes, but
cable and fingers still miss LPIPS by `+0.001966` and `+0.005174`. Relative to the ordinary FR0.3
tail, FR1 improves fingers by `0.001024` but worsens cable by `0.000802`; both endpoints remain far
from a simultaneous crop pass. Late FR strength therefore redistributes the localized error rather
than removing the early-basin gap. This closes tail-only tuning at hard step80000.

## Exact static-ray cache and post-step80000 exposure screens

The next semantics-preserving optimization caches the immutable training-camera ray fields on the
GPU. It is opt-in and fail-closed for the canonical one-dimensional, uniform perspective-camera
case. The cache stores direction, pixel area and detached direction norm for every train pixel;
origins remain per-camera. It is derived state (`persistent=False`), consumes no RNG and is absent
from checkpoints. On the canonical 66-camera `1920x1080` dataset it builds in `2.308 s`, occupies
`2,737,152,792` bytes (`2.55 GiB`), and peaks at only `3.11 GiB` during an isolated build.

Parity checks used 100,000 random rays with repeated indices and nonzero distortion. Origins,
directions, pixel areas, camera indices and `directions_norm` were byte-identical; CPU and CUDA RNG
states were unchanged. A same-parent mature 500-update profile measured:

| Arm | Median train iteration | Mean train iteration | Final cumulative points |
|---|---:|---:|---:|
| control 1 | 48.6345 ms | 48.4835 ms | 1.38567 B |
| control repeat | 48.7991 ms | 49.0629 ms | 1.38608 B |
| static-ray cache | **43.9656 ms** | **44.0122 ms** | 1.38568 B |

Thus the measured steady saving is `4.67--4.83 ms/update`. Final checkpoint differences between
cache and control have the same 14 tensor entries and the same scale as control-versus-control;
their saved RNG states are identical, and cache cumulative exposure is closer to control1 than the
two controls are to one another. This establishes parity within the existing CUDA/TCNN repeat
floor, not literal final-weight identity. An attempted consolidated FAS transfer is inapplicable to
the canonical path because images and sampling are on CPU; it produced no measured benefit and is
not promoted.

Three saved-checkpoint continuations then tested whether the failed step80000 speed basins merely
needed more point exposure. These are diagnostics, not eligible E2E replacements:

| Ancestry / step | PSNR | SSIM | LPIPS | Artifacts / serious ROI | Pipe | Cable | Fingers |
|---|---:|---:|---:|---:|---|---|---|
| staggered JIT, 91128 | 29.725739 | 0.676195 | 0.216940 | 1/3 / 0 | pass | pass | LPIPS `+0.004040` |
| joint JIT, 91128 | 29.801340 | 0.670472 | 0.214579 | **0/3 / 0** | pass | LPIPS `+0.001066` | pass |
| joint JIT, off for 80001--91128 | 29.809662 | **0.667385** | 0.214962 | 0/3 / 0 | pass | LPIPS `+0.000796` | pass |
| joint JIT, 92152 | 29.767286 | **0.666290** | 0.214473 | 0/3 / 0 | PSNR/SSIM fail | pass | pass |

The first two rows show that extra exposure improves aggregate LPIPS substantially but retains the
same cable/fingers kernel trade. Turning JIT off only for the final 11,128 updates improves cable by
just `0.000271` and fails aggregate SSIM. Another 1,024 joint-JIT updates make cable and fingers
pass, but aggregate SSIM and thin-pipe PSNR/SSIM fail. Therefore neither a longer tail nor a late
full-JIT disable yields one passing checkpoint. Frozen reproduction defaults remain unchanged.

Artifacts for these screens are under
`/home/brans/lookcloser_leader_speed_runs/diagnostics/`; the exact run directories begin with
`staggered_cache_continue80000_to91128`, `joint_cache_continue80000_to91128`,
`joint_cache_jitoff80001_continue91128`, and `joint_cache_continue91128_to92152`.

## Staged color/geometry JIT E2E quality pass at step91128

The next solo seed42 campaign combined only reviewed speed mechanisms: the exact static-ray cache;
historical kernels through step15188; fused Adam plus color-network JIT before update15189;
geometry-network JIT before update30377; exact eval-trajectory replay; the recovered RNG-only
Stage-A boundary; historical FR `1.0 -> 0.3` at step75940; and one hard checkpoint at step91128.
The point schedule, B4096, LR/scheduler, hash23, ARM, occupancy, FAS and losses were unchanged.

This is the first speed-worktree checkpoint to pass every quality gate simultaneously:

| PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs | Pipe | Cable holes | Fingers |
|---:|---:|---:|---:|---:|---|---|---|
| **29.802864** | **0.675499** | **0.222623** | **0/3** | **0/10** | pass | pass | pass |

The priority-crop deltas versus the archive are respectively: pipe
`+0.163433 dB / +0.003346 SSIM / -0.001791 LPIPS`, cable holes
`+0.130796 / +0.000338 / -0.002396`, and fingers
`+0.472807 / +0.005132 / -0.000871`. Stand and label remain visible strict-all-five misses, but are
not priority promotion failures. Total exposure is `220.237 B` points.

The controller wall is `3617.973 s` (`60:17.97`): Stage A `2917.588 s`, verified RNG-only fork
`3.4 s`, Stage B `645.004 s`, and finalization `47.149 s`. The checkpoint therefore establishes the
training recipe and basin but misses the first wall milestone by `17.973 s`. It is recorded as
`complete_quality_pass_wall_fail` and is not promoted to the canonical no-argument defaults. The
next optimization may remove only orchestration or unnecessary intermediate-checkpoint I/O; it
must preserve this exact training trajectory and rerun the same hard checkpoint and gates.

Immutable artifacts:

- campaign: `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_hard91128_seed42_v1/campaign.json`;
- checkpoint SHA-256: `2b1c046ea5481ff2772877c13e8135a810d906f177547bd09345cc8a8a837324`;
- config SHA-256: `748661b0cd414b9121ff551ef3b7d541c563b6e58ce7fbd09d121a9c5ed6d865`;
- candidate summary SHA-256: `f6490198be2f567a64456f3cfabfadeedeaf0db7a066f6df0fc7dd5210db7d89`;
- detail JSON SHA-256: `c9f2a227fd0d652cb9ec5a4e2ecbe4e8500e950e12da6ae35e6a00151b848446`.

## Checkpoint/finalization pruning and second solo same-seed E2E run

The first staged-JIT/cache result left only `17.973 s` between its measured wall and the first
milestone. Before changing optimizer updates, samples or weights, the controller removed work that
cannot affect training semantics:

- a hard single-candidate campaign keeps the `15188` eval/RNG cadence but sets the save interval to
  `final_step + 1`; `_after_train()` still writes the exact Stage-A step75940 and Stage-B step91128
  checkpoints. This reduces six persisted checkpoints to two and removes the four intermediate
  Stage-A writes plus duplicate scheduled writes of both final checkpoints;
- candidate eval disables construction of the unused 2.55-GiB training-ray cache;
- the three full-view artifact detectors run concurrently, while the ROI scorer omits diagnostic
  image writes in gate-only mode. Full eval renders, detector diagnostics, detail crops and the
  contact sheet remain present;
- the controller records a stable checkpoint identity (`device`, `inode`, size, mtime and ctime)
  around its SHA-256 pass. The candidate recorder reuses that digest only for the exact successful
  stage/step while the complete identity is unchanged; ambiguous, partial or changed records fail
  closed, and non-final intermediate checkpoints retain a full hash.

The relevant controller tests pass `39/39`; the complete focused speed suite passes `119/119`.
The frozen speed source fingerprint remains
`69695ee038399b87f0017e307420153efcf5b97a702d3c9dc6ce4075ac185e9d`; the selection/finalization
protocol fingerprint is
`d4a3aae737d3cc319314ea859753dc2fa5bfdafef31571f09084e2d32e85dde7`.

The second solo seed42 run used the identical staged training recipe, hard step91128 and chunk8192,
with only the changes above and runner polling reduced from1.0 to0.1 seconds. It completed in
`3704.831 s` (`61:44.8`), so it missed the milestone by `104.831 s`:

| Component | First run | Pruned repeat | Delta |
|---|---:|---:|---:|
| Stage A | 2917.588 s | 3006.216 s | +88.628 s |
| Stage B | 645.004 s | 655.050 s | +10.046 s |
| Finalization | 47.149 s | 35.301 s | **-11.848 s** |
| Controller total | 3617.973 s | 3704.831 s | +86.858 s |

Finalization therefore improved as designed, but ordinary run-to-run training speed dominated the
saved overhead. At the matched mature Stage-A window 45k--51k, mean iteration time changed from
`39.940 ms` to `41.232 ms` (`+3.2%`); RNG fork plus residual controller overhead outside the two
stages/finalization stayed essentially unchanged (`8.232` versus `8.264 s`). Point exposure changed only
from `220.237 B` to `220.687 B` (`+0.204%`), so sample mix explains little of the wall gap. This is
evidence for a runtime/clock plus CUDA execution fluctuation, not for an orchestration regression;
the exact contribution remains unresolved because clocks, thermals and OS scheduling were not
experimentally controlled.

The phase decomposition localizes the slowdown: mean logged iteration time changed by `+1.15%`
before the first JIT switch, `+1.92%` in the color-only phase, `+4.34%` with both networks JITed,
and `+2.04%` in Stage B. Extra point exposure explains only about `7.3 s` under a linear-cost model,
far less than the `98.674 s` combined Stage-A/B increase. The two configs differ only in paths,
timestamp, save interval and polling; no training-hot-path field changed.

Quality divergence is also observed before any scheduled save: train losses differ by step10 even
though the recorded Stage-A CPU/CUDA RNG hashes are identical. Cumulative points remain identical
through step4090 and first diverge after adaptive occupancy starts at step4100. At step75940 the
second run has one fewer successful Adam/scheduler update because AMP skipped one additional
overflowing update; its scaler is `8192` rather than `16384` and its LR differs by one scheduler
tick. The verified sequence is therefore nondeterministic CUDA/mixed-precision arithmetic first,
then occupancy/sample feedback and GradScaler history. The exact first responsible kernel remains
unresolved.

The fresh metric vector was `29.832731 / 0.676254 / 0.218988`, with significant artifacts `0/3`
and serious ROIs `0/10`. Relative to the first same-seed staged run it changes by
`+0.029867 dB / +0.000755 SSIM / -0.003635 LPIPS`, inside the user-approved reproducibility bounds
`0.06 / 0.01 / 0.005`. The frozen strict micro-detail comparator nevertheless rejects the candidate:

| Priority crop | PSNR delta vs archive | SSIM delta | LPIPS delta | Strict gate |
|---|---:|---:|---:|---|
| thin pipe | +0.182280 | +0.001479 | -0.003681 | pass |
| tangled cable holes | -0.009834 | +0.000344 | +0.000898 | fail |
| fingers | +0.391367 | +0.005919 | +0.001004 | fail |

The failed crop deltas are much smaller than the global repeat tolerances, but those tolerances do
not relax the separately frozen archive-detail gate. The campaign correctly remains
`complete_no_accepted_candidate`; the earlier strict quality pass is not substituted and the new
checkpoint is not promoted. A robust `<=60` recipe now needs at least about two minutes of training
margin, not another sub-ten-second orchestration shave. The next bounded diagnostic should locate
the earliest clean checkpoint in the existing Stage-B tail before changing LR/batch/warmup again.

Immutable repeat artifacts:

- campaign:
  `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_hard91128_seed42_ioprune_v2/campaign.json`;
- step91128 checkpoint SHA-256:
  `af74946b976b01723debcb1b4b290bf80ec64ecbe2d630218b0c694b17b30655`;
- config SHA-256: `bc5b11f2e2bc5f2ba93e30b36a18b1786fd6c0ce99f3006bc96869dc118c9792`;
- candidate summary SHA-256:
  `af25386b49aadc6844df5a539a0c5e440da822cdfc89300bc77691e329242f0a`;
- detail JSON SHA-256:
  `93c8d3a871d01053decb2de66385a7705022af81080c8b8858d4b38d77321f13`.

## Dense Stage-B cutoff screen and selected prefetch axis

A saved-parent diagnostic replayed only the exact historical-RNG-reset Stage B from step75940 to
91128. It kept FR0.3, LR/Adam/scheduler/scaler, static rays and the staged JIT state unchanged, and
added model checkpoints every2000 updates. The run took `664.884 s`, but its wall is not comparable
to a milestone because of the deliberately dense 2-GB checkpoint writes. Four candidate
checkpoints were then evaluated concurrently; their evaluation times are likewise contended and
are used only for quality screening.

| Step | PSNR | SSIM | LPIPS | Artifacts / serious ROI | Pipe | Cable | Fingers |
|---:|---:|---:|---:|---:|---|---|---|
| 84000 | 29.816307 | 0.674623 | 0.226995 | 0/3 / 0/10 | fail | fail | fail |
| 86000 | 29.810772 | 0.673264 | 0.223936 | 0/3 / 0/10 | pass | fail | fail |
| 88000 | 29.810404 | 0.672041 | 0.221720 | 0/3 / 0/10 | pass | fail | fail |
| 90000 | 29.795351 | 0.672571 | 0.220529 | 0/3 / 0/10 | pass | pass | fail |
| 91128 repeat | 29.817698 | 0.675974 | 0.218900 | 0/3 / 0/10 | pass | fail | fail |

At step90000 the only remaining priority miss is fingers LPIPS `+0.001171` above the archive;
the replayed final step still misses cable/fingers by `+0.000261/+0.000842`. Thus no shorter hard
cutoff passes the frozen gate, and even the full tail retains the already measured strict-crop
repeat floor. A shorter tail is rejected rather than being combined with an LPIPS tolerance or
best-checkpoint selection. Diagnostic root:
`/home/brans/lookcloser_leader_speed_runs/diagnostics/staged_tail_dense_seed42_v2/lookcloser/20260717_051621`.

The next semantics-preserving speed axis is therefore an opt-in CPU prefetch of exactly one future
FAS pixel batch while the current CUDA training update runs. A mature profile attributes about
`4.066 ms/update` to pixel sampling, of which `4.038 ms` is base sample/collate; the static ray cache
does not overlap this CPU work. Across 91129 updates the theoretical overlap is about370 seconds.
The go/no-go lower bound is `2.8 ms/update`: that predicts about255 seconds saved and keeps even the
slow v2 timing plus another3% fluctuation under3600 seconds. Saving only the nominal104.8-second
miss is insufficiently robust.

The reviewed design is deliberately narrow: one thread, queue depth one, CPU indices/RGB only,
no RayGenerator or CUDA in the worker, logical FAS `sample_count` commit only on dequeue, and a
lazy start after checkpoint RNG restoration. The queue is derived state and must be empty at
frequency-grid CPU-RNG boundaries, eval/replay, save/final/phase boundaries and before any live ray
batch or point-target switch. Cancellation rolls back the queued Torch-CPU RNG transition before a
checkpoint. Unknown CPU-RNG callbacks disable the optimization fail-closed. Multiprocessing and a
deeper queue are excluded because they duplicate the cached dataset and cannot preserve the shared
RNG order.

Before a long run, focused tests must prove byte-exact pixel batches and cached rays, exact
Python/NumPy/Torch-CPU/all-CUDA RNG states, FAS count, Adam/scheduler/scaler and cumulative points
against sync controls across every barrier. A 500-update real sync/sync/prefetch smoke must bound
model drift by the existing sync-repeat CUDA/occupancy floor, and a solo ABA profile must measure at
least `2.8 ms/update`. The feature remains default-off until all of those checks pass.

## Deterministic CPU FAS prefetch: correctness and mature ABA

The selected implementation is an opt-in, queue-depth-one CPU FAS prefetch. Its worker owns a
private CPU `torch.Generator` and immutable sampler snapshot; it never touches CUDA, the live
sampler, the ray generator, or process-global RNG. Dequeue commits the private post-RNG state and
logical FAS counter only when trainer step, sample count, configuration, tensor identity/storage,
tensor mutation versions, and the live global CPU RNG still match. Any mismatch is discarded and
replayed synchronously. Explicit barriers discard derived work before frequency-grid RNG,
scheduled eval/replay, checkpoint save, and terminal save/shutdown. The supported surface is
fail-closed to one process, the static ray cache, fixed B4096, fully cached homogeneous
`image/image_idx` input, enabled FAS, and no dynamic ray/point schedule.

The canonical runner and E2E controller expose `--cpu-fas-prefetch`, both default-off. The E2E
controller accepts it only as an extension of the complete reviewed staged recipe. The frozen
no-argument reproduction path is unchanged. The relevant speed suite passes `134/134`, the
controller suite passes `41/41`, and both worktrees pass `git diff --check`. The frozen speed
committed-source fingerprint is
`6cf7eb9560403ed05da27b2eb7ce732585e930b2d13a0ccfbfb9dd1766e4c258`; the controller protocol
fingerprint is
`156a73bf475771e357af73afe298f88421502387f8fcda6b24d689c8d50550ad`.

Dataset provenance was rechecked immediately before profiling and matched dev3: 202 files, 69
images, 66 maps plus 66 metadata files, no local/remote differences, and transforms SHA-256
`022f8748...f64aa1`. All mature arms resumed the immutable v2 step91128 checkpoint
`af74946b...0655`, loaded model, Adam, scheduler, scaler and RNG, ran updates 91129--91628 solo,
and discarded the first 50 updates. The two synchronous controls establish the local timing
floor:

| arm | median ms/update | p95 ms/update | median Mpoints/s |
|---|---:|---:|---:|
| sync A | 42.8264 | 44.5324 | 66.109 |
| sync B | 42.3648 | 43.4548 | 66.818 |
| first transactional prefetch | 40.8079 | 41.0522 | 69.485 |
| identity-only live signature | 40.7362 | 40.8888 | 69.685 |
| dense-LUT worker snapshot | **40.4975** | **40.6685** | **70.097** |

The first transactional prefetch preserves exact Torch CPU and CUDA RNG SHA-256, trainer step,
Adam update count, scheduler epoch/LR and AMP scaler across both sync controls and prefetch. Its
model drift is not larger in mean than the sync-vs-sync CUDA/occupancy floor. A direct sampler
diagnostic corrected the earlier cost attribution: the live synchronous sampler is about
`0.981 ms`, while the initial worker snapshot took `5.976 ms` because it accidentally restored a
Python per-ray image-shape lookup. Reusing the canonical dense height/width LUT reduced the worker
snapshot to `0.861 ms` while the byte-exact batch/RNG tests remained green.

Against the mean synchronous median (`42.5956 ms`), the final prefetch saves `2.0981 ms/update`, or
about 191 seconds over 91,129 updates. This is below the predeclared conservative
`2.8 ms/update` robustness screen, so it is not promoted and defaults remain unchanged. However,
that old screen was based on the now-refuted `~4.1 ms` live-sampler attribution. The directly
measured saving projects the slow v2 controller wall from `3704.831 s` to approximately `3514 s`
(`58:34`). One solo diagnostic full E2E seed42 run is therefore allowed to measure the actual
controller wall and full frozen quality/detail gates. It is not accepted from projection: only an
observed `<=3600 s` run with numeric pass, `0/3`, `0/10`, and all three strict priority-detail
passes can change the speed default.

Raw profiles:

- `/home/brans/lookcloser_leader_speed_profiles/prefetch_aba_sync_a_20260717/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/prefetch_aba_sync_b_20260717/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/prefetch_aba_prefetch_lut_v3_20260717/profile.json`;
- `/home/brans/lookcloser_leader_speed_profiles/prefetch_aba_checkpoint_state_20260717.json`.

## CPU-prefetch solo E2E result: wall pass, quality miss

The predeclared solo seed42 diagnostic used the complete reviewed staged recipe plus only
`--cpu-fas-prefetch`. Dataset provenance matched dev3 immediately before launch. It trained from
scratch through the historical step75940 boundary, applied the verified RNG-only fork, continued
with FR0.3 to the single hard step91128 candidate, and ran the complete numeric, render, automatic
artifact, ROI and priority-detail finalization.

The observed controller wall was **3501.901 s** (`58:21.9`), so this is the first complete E2E run
of this speed recipe to pass the `<=3600 s` wall milestone with about 98.1 seconds of margin:

| Component | Wall |
|---|---:|
| Stage A, steps 0--75940 | 2827.003 s |
| Historical RNG-only fork | 3.373 s |
| Stage B, steps 75941--91128 | 630.920 s |
| Full finalization | 35.538 s |
| Controller total | **3501.901 s** |

The final candidate had 220.174 B point samples and the following aggregate result:

| PSNR | SSIM | LPIPS | Significant artifacts | Serious ROIs |
|---:|---:|---:|---:|---:|
| **29.848965** | **0.667368** | **0.219900** | **0/3** | **0/10** |

PSNR and LPIPS pass the archive leader gates, but SSIM misses the absolute `0.668450` gate by
`0.001082`. Relative to the accepted stable reproduction S1, the aggregate delta is approximately
`+0.00882 dB / -0.001835 / +0.000445`, well inside the user-approved repeat limits
`0.06 / 0.01 / 0.005`; those repeat limits do not replace the per-run absolute quality gates.

Of the three priority crops, thin pipe and fingers pass. Cable holes fails only its PSNR comparison
(`-0.088606 dB` versus the archive); its SSIM is `+0.000502` and LPIPS is `-0.000484`, both better
than the reference. Stand and label are also strict-all-five misses, but they are not priority
promotion gates. Automatic artifact detection is completely clean.

The campaign therefore correctly records `complete_no_accepted_candidate`: it proves the wall
milestone is feasible, but does not yet prove a quality-accepted speed recipe. CPU prefetch remains
default-off and the frozen reproduction defaults are unchanged. The next experiment must use the
saved common Stage-B boundary for a predeclared LR/scheduler/Adam margin diagnostic, then rerun any
selected policy solo from scratch; no alternate seed or best-checkpoint substitution is eligible.

Immutable artifacts:

- campaign: `/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_prefetch_lut_seed42_v3/campaign.json`;
- checkpoint SHA-256: `d7f65b80ddd982fab5f0348584c76e0f8ffab22884d5598d3e70e8f1f11c0c7f`;
- config SHA-256: `c346da4f0f1430dc39e1da9709f2a3c36ba87687b32386a4b66d687b21703e8e`;
- candidate summary SHA-256: `5d50249a1a1d7a6f88dba4620277d5ee734f2314779a06afef65eb3674d83f64`;
- detail JSON SHA-256: `069c1c98efb83f4cb6b390f7d518dabf35f7494eba4cc17a5204b650162b6000`.

## Predeclared v3 plateau fork at step91128

The next diagnostic follows the previously frozen plateau protocol rather than starting another
from-scratch candidate blindly. Its common parent is exactly the failed-but-wall-feasible v3
step91128 checkpoint `d7f65b80...f11c0c7f`. The loaded state is trainer91128, Adam/scheduler91086,
LR `0.00122783497`, GradScaler8192 and the persisted post-training RNG snapshot.

An already-existing, independently generated v2 timing continuation supplies useful prior
evidence: three same-parent 500-update arms all moved cable holes to pass and aggregate SSIM to
`0.67364--0.67399`, but fingers remained just outside its LPIPS reference by
`0.000330--0.000777`. That result is not eligible for selection because it belongs to the older v2
parent; it only fixes the diagnostic horizon at 500 updates.

Four v3 branches form the exact factorial `{1x, 0.25x LR} x {loaded Adam, reset Adam}` and run only
updates91129--91628. LR multiplication changes optimizer LR plus scheduler base/last LR while
preserving scheduler time; reset branches remove only Adam moments/step. Model, scheduler epoch,
GradScaler, RNG, FAS/FR, occupancy, B4096, fused/JIT state and CPU prefetch remain otherwise
unchanged. Every branch has exactly one scheduled checkpoint, step91628, at approximately221.6 B
cumulative point samples.

The branches may train concurrently because each uses about10 GiB on the 98-GB GPU; their contended
wall is diagnostic only. Each receives a fresh chunk8192 full eval plus `0/3`, `0/10` and the same
three priority-detail gates. An arm is eligible only if all gates pass. If several pass, choose the
policy with the largest worst normalized priority-crop margin; aggregate LPIPS breaks an exact
margin tie. No seed or checkpoint substitution is allowed. A selected policy must be encoded in
the controller and rerun solo from scratch before it can change defaults.

At the measured `40.50 ms/update`, the 500-update tail adds about20.3 seconds. The observed v3 E2E
projection is therefore about3522 seconds, while the deliberately conservative slow-v2-plus-
prefetch projection is about3534 seconds; both remain under3600 before run-to-run clock variance.

### Plateau-fork result

All four branches completed step91628 and the full evaluator. Concurrent training used about65 GiB
and had no NaN/OOM; its roughly131--135 s per arm is deliberately contended and is not a speed
measurement.

| Late 500-update policy | PSNR | SSIM | LPIPS | Numeric | Pipe | Cable | Fingers |
|---|---:|---:|---:|---|---|---|---|
| 1x, loaded Adam | 29.814407 | 0.668547 | 0.220796 | pass | fail | pass | pass |
| 0.25x, loaded Adam | 29.844330 | 0.668321 | 0.221442 | fail | pass | fail | pass |
| 1x, reset Adam | 29.804007 | 0.668076 | 0.221481 | fail | fail | pass | pass |
| 0.25x, reset Adam | 29.834856 | 0.668245 | 0.221422 | fail | fail | pass | pass |

Every arm remained automatic-artifact clean. The 1x loaded continuation cleared the absolute SSIM
gate and cable/fingers, but moved pipe PSNR to `-0.148029 dB`. The closest arm was 0.25x loaded:
pipe/fingers pass, cable misses only PSNR by `0.003670 dB`, and aggregate SSIM misses by only
`0.000129`. Resetting Adam did not rescue either LR. Thus a late 500-update impulse exposes a stable
pipe-versus-cable trade rather than a clean plateau escape. None is eligible; there is no extra
extension, alternate step or branch repeat.

Diagnostic root:
`/home/brans/lookcloser_leader_speed_runs/diagnostics/prefetch_v3_plateau500_seed42_v1`.

## Predeclared common-Stage-A quality-margin screen

The remaining causal screen moves the optimizer intervention to the start of the complete FR0.3
tail. Its sole parent is the v3 historical-RNG-reset step75940 checkpoint
`22707cffd746d00e4af077e8f1b3b520b72511a9528d4cf855a0934b557119f3`. It is trainer75940,
Adam/scheduler75905, LR `0.001741606352`, scaler8192, with RNG absent so every seed42 branch gets
the same historical new-process stream. The existing v3 step91128 remains the immutable 1x/loaded
reference and is not rerun as an eligible arm.

Three new one-factor branches run the full 15188-update Stage B and expose only step91128:

| Fixed priority | Arm | Checkpoint/config mutation |
|---:|---|---|
| 1 | `S100-loaded` | scheduler horizon `200k -> 100k`; remap epoch `75905 -> 37952` so starting LR is continuous; loaded Adam |
| 2 | `L080-loaded` | scale the entire LR trajectory `0.01 -> 0.008`, final `0.0001 -> 0.00008`; loaded Adam |
| 3 | `L100-reset` | historical LR/scheduler, reset only Adam moments/step at step75940 |

For `S100-loaded`, the expected final scheduler epoch is53133 and LR approximately0.000866. For
`L080-loaded`, the expected starting/final-tail LRs are approximately0.001393/0.000982. All other
state and data are fixed: seed42, FR0.3, B4096, historical ARM/hash23/FAS/occupancy, cache, CPU
prefetch, fused Adam, both already scheduled JIT scopes, replay cadence, GradScaler and gates.

Each arm has one endpoint and is eligible only if all numeric, `0/3`, `0/10`, pipe, cable and
fingers gates pass. If multiple pass, take the first in the fixed table order, not the numerically
best arm. If none passes, reject this LR/scheduler/Adam family without another endpoint, extension,
seed or repeat. A diagnostic pass still requires one solo from-scratch controller run within3600 s
before promotion. The three arms may run concurrently for quality screening; their contended time
does not support any wall claim.

### Common-Stage-A screen result

All three branches reached the only declared endpoint. Their roughly1983 s concurrent wall is not
a speed result. Checkpoint state confirms the intended policies: S100 finished at scheduler
epoch53133/LR0.000865645, L080 at epoch91086/LR0.000982268, and reset Adam made15182 updates while
the historical scheduler reached epoch91087. Every automatic artifact and serious ROI gate was
clean.

| Full Stage-B policy | PSNR | SSIM | LPIPS | Numeric | Pipe | Cable | Fingers |
|---|---:|---:|---:|---|---|---|---|
| S100, loaded Adam | 29.855030 | 0.668673 | 0.221144 | pass | fail | fail | pass |
| L080, loaded Adam | 29.857464 | 0.668945 | 0.220877 | pass | fail | fail | pass |
| L100, reset Adam | 29.851961 | 0.668232 | 0.219229 | fail | pass | fail | pass |

S100 misses pipe/cable only in PSNR by `0.028574/0.047304 dB`; L080 misses them by
`0.119658/0.048407 dB`. Both beat the reference for SSIM and LPIPS on those crops. Reset Adam
preserves pipe/fingers but makes cable materially worse (`-0.270855 dB / -0.000631 SSIM /
+0.000440 LPIPS`) and misses aggregate SSIM by0.000218. No arm passes, so the predeclared family is
closed without endpoint, seed or repeat substitution.

Diagnostic root:
`/home/brans/lookcloser_leader_speed_runs/diagnostics/prefetch_v3_stageb_margin_seed42_v1`.

The state audit exposed a stronger next variance axis. At step91128, both the accepted canonical
reproduction and the first all-quality speed run have exactly91087 Adam/scheduler updates, gap41,
and GradScaler16384. Both failed-quality v2/v3 speed runs have91086 updates, gap42, and
GradScaler8192. This is correlation, not proof of one-update causality: canonical accepted S1 and
v1 reached the same final count through different Stage-A counts, and the model/occupancy states
already differ. It nevertheless identifies the adaptive scaler's overflow threshold as a concrete
feedback path: CUDA-level numerical drift changes whether an update is skipped, which changes the
scheduler and model, then occupancy and future samples.

The next bounded diagnostic should therefore make only GradScaler growth deterministic at a
conservative fixed scale while retaining FP16 autocast. Before another full E2E candidate, two solo
same-seed prefixes must show whether this reduces model/cumulative-point divergence versus the
existing sync-repeat floor without NaN/overflow or a throughput regression. Historical scaler
behavior must remain the no-argument default.

## Predeclared fixed-growth GradScaler variance diagnostic

The speed worktree now exposes two top-level TrainerConfig controls:
`grad_scaler_init_scale` and `grad_scaler_growth_interval`. Their defaults are exactly the PyTorch
historical values `65536/2000`; the quiet runner emits no CLI override unless explicitly requested.
The experimental setting is `8192/1000000`: FP16 autocast and finite-gradient checking stay active,
but the scale cannot grow during this run. A real inf/NaN still causes the safe normal behavior —
the optimizer update is skipped and the scale backs off — so this is not an unsafe forced step.

Validation before GPU use is `141 passed`, `py_compile` clean and `git diff --check` clean. The
defaults are covered independently from explicit forwarding and invalid-value rejection. This is a
variance-control experiment, not a semantics-preserving speed patch; it remains opt-in.

The first screen is a sequential solo `A/B/B` prefix, all seed42 and all stopping at step15188:

| Arm | GradScaler | Purpose |
|---|---|---|
| `A-default` | `65536/2000` by omitted flags | same-source historical control |
| `B-fixed-1` | `8192/1000000` | fixed-growth candidate |
| `B-fixed-2` | `8192/1000000` | exact same-seed repeat |

Every other field is the v3 from-scratch speed prefix: cache, CPU FAS prefetch, B4096,
FR1.0, historical ARM/hash23/FAS/occupancy, LR0.01→0.0001 and exact eval-trajectory replay. The
fused/color-JIT switch remains declared at15189 but is not reached, so the endpoint isolates
pre-switch AMP→occupancy feedback. Runs are deliberately sequential: parallel GPU contention would
defeat the variance and throughput comparison even though memory permits it. Each arm has one
step15188 checkpoint followed by a fresh chunk8192 eval; no earlier endpoint is eligible.

The candidate is worth a longer run only if both B arms finish at scale8192 with zero skipped Adam
updates, differ from each other by at most `0.01 dB / 0.001 / 0.002`, reduce the relative
cumulative-point spread below `0.05%` (historical v1→v3 is about0.289% at step15180), and have no
more than1% solo median throughput regression versus A after occupancy warmup. A large common shift
versus A is reported separately and must remain inside `0.06 / 0.01 / 0.005`; passing repeatability
alone is not permission to accept a biased basin. Failure closes this exact scale rather than
silently lowering it or extending the prefix.

### Fixed-growth A/B/B result

All three solo prefixes completed at the declared step15188 endpoint. Fresh chunk8192 evaluation
used the same evaluator and frozen three-view render protocol for every arm:

| Arm | PSNR | SSIM | LPIPS | Adam counter | Final scale / interval | Points @15180 | Median update, steps 5000--15000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A-default` | 28.645100 | 0.646545 | 0.360836 | 15179 | 8192 / 2000 | 22.4022 B | 35.1361 ms |
| `B-fixed-1` | 28.421326 | 0.647698 | 0.363886 | 15188 | 4096 / 1000000 | 22.4207 B | 35.2372 ms |
| `B-fixed-2` | 28.521709 | 0.648833 | 0.367198 | 15188 | 4096 / 1000000 | 22.4545 B | 35.2126 ms |

Both fixed arms made the same early safe scaler backoff from8192 to4096 and ended with the same
growth tracker15105. Their Adam/scheduler counters and LR are identical, but the scale rollback
means the predeclared zero-backoff condition already fails. The two B evaluations differ by
`0.100383 dB / 0.001135 / 0.003312`, exceeding every repeat bound
`0.01 / 0.001 / 0.002`. Their step15180 cumulative-point range is `33.8 M`, or `0.150640%`, three
times the `0.05%` limit. Median update time regresses by only `0.253%`, so throughput is not the
reason for rejection.

Fixed growth also biases the prefix rather than merely reducing its noise. B1 shifts from A by
`-0.223774 / +0.001153 / +0.003050`; B2 shifts by
`-0.123391 / +0.002288 / +0.006362`. Both exceed the allowed common PSNR shift, and B2 also exceeds
the LPIPS shift. A chunked checkpoint comparison measures B1-versus-B2 symmetric relative L2 of
`0.841501` for all field tensors and `0.508255` for the duplicated occupancy `occs` tensors. Thus
matching scaler/Adam state does not suppress the underlying TCNN/occupancy numerical divergence.

The exact `8192/1000000` fixed-growth policy is closed. It is not extended, repeated at another
endpoint, lowered silently, enabled in the E2E controller or promoted to defaults. The controls
remain available only as explicit diagnostic plumbing. The next isolated variance axis is the
already reviewed independent RNG-stream separation, which prevents an occupancy branch/count
difference from advancing pixel/FAS and frequency-grid streams; it must receive its own
predeclared same-seed screen before any full E2E run.

Evidence:

- root: `/home/brans/lookcloser_leader_speed_runs/diagnostics/fixed_scaler_prefix_aba_seed42_v1`;
- checkpoint state: `checkpoint_state.json`;
- B1-reference drift: `checkpoint_drift_B1_reference.json`;
- per-arm evaluation: `*/lookcloser/20260717_083000/prefix15188_evaluation_step-000015188.json`.

## Predeclared independent-RNG prefix screen

The next axis isolates the pixel/FAS, occupancy and frequency-grid random streams by a stable
`(campaign seed, subsystem, trainer step)` seed. It remains an algorithmic variance ablation:
TCNN/grid atomics, occupancy threshold branching and historical GradScaler behavior are unchanged.
The speed implementation adds a private-generator path for CPU FAS prefetch so the queued step
`n+1` is generated with the pixel seed for `n+1`, is byte-exact to synchronous seeded FAS, and
never reads or mutates process-global RNG. The omitted-flag runner argv and compact summary remain
byte-identical to the pre-port default. The focused suite passes161 tests before GPU use, including
the actual pixel/frequency pipeline boundaries and stable occupancy boundary/no-op paths.

The screen is a new same-source sequential solo `A/B/B`, not a comparison only against the older
fixed-scaler A. All arms use seed42, cache plus CPU FAS prefetch, B4096, stable occupancy, FR1,
historical ARM/hash23/FAS, replayed eval cadence, LR `0.01→0.0001`, default GradScaler and the sole
step15188 endpoint. Fused Adam and color JIT remain declared for step15189 and are therefore not
reached. `A_rng_control` omits the new flag; `B_rng_1` and `B_rng_2` add only
`--independent-rng-streams`. Both `--grad-scaler-*` flags are omitted. Parallel execution is
forbidden because it would contaminate the repeat and timing measurements.

Before launch, dataset provenance must match dev3 and tests must prove: stable/distinct
step-addressed subsystem seeds; restoration of enclosing CPU/CUDA RNG; matching pipeline/model
seed and enable flags; correct pixel and frequency-grid stream names/steps at their actual pipeline
boundaries; occupancy boundary seeding; exact synchronous↔prefetch samples for the same seed, step
and logical FAS count; correct stale-successor recomputation; signature fail-closed; and one logical
FAS-count increment per step.

The exact step15188 decision rule is:

- fresh B1/B2 metric range at most `0.01 dB / 0.001 / 0.002`;
- one-sided non-inferiority of mean(B) versus A: PSNR no lower by more than`0.06 dB`, SSIM no lower
  by more than`0.01`, and LPIPS no higher by more than`0.005`; improvements are not capped;
- exact checkpoint point-count B1/B2 range divided by their mean at most`0.05%`, and mean(B) point
  shift versus A at most`0.5%`; each FAS count exactly15189;
- exact B1/B2 equality of trainer/Adam/scheduler/LR, skipped-update count, GradScaler
  scale/tracker/interval, and Python/NumPy/Torch CPU/all-CUDA RNG hashes; all tensors finite;
- B1/B2 symmetric relative L2 at most`0.841501461` for aggregate field state,
  `0.381191158` for occupancy `occs` (25% below the rejected fixed-scaler pair), and
  `0.304655892` for occupancy binaries; duplicate model/sampler occupancy state exact internally;
- each B median update time over steps5000--15000 at most`1.01×` A;
- every fresh three-view evaluation has zero significant artifacts and zero serious ROIs. The
  immature prefix's absolute archived-detail result is reported but is not a promotion gate.

All conditions must pass to authorize exactly one independent-RNG full E2E seed42 run. A failure
closes this exact mode without another endpoint, seed or checkpoint. A prefix pass alone does not
change defaults; the full run must independently pass controller wall`≤3600 s`, numeric gates,
automatic artifacts and all three priority detail crops.

### Independent-RNG A/B/B result

Dataset provenance matched dev3 immediately before launch:202 files,69 images,66 maps and66
metadata files, with transforms SHA-256 `022f8748…f64aa1` and no differences. All three arms then
completed sequentially at the sole step15188 endpoint. Configs record seed42, stable occupancy,
CPU prefetch and historical GradScaler `65536/2000`; only B1/B2 enable both independent-stream
flags.

| Arm | PSNR | SSIM | LPIPS | Significant / serious ROI | Exact points | Skips / scale / tracker | Median update |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_rng_control` | 28.664501 | 0.648839 | 0.357084 | 0/3 / 0/10 | 22,585,538,378 | 9 / 8192 / 1814 | 35.1759 ms |
| `B_rng_1` | 28.731922 | 0.652861 | 0.362993 | 0/3 / 0/10 | 22,653,819,902 | 9 / 8192 / 612 | 35.2960 ms |
| `B_rng_2` | 28.724264 | 0.650604 | 0.366760 | **3/3** / 0/10 | 22,730,317,299 | 9 / 8192 / 815 | 35.4442 ms |

B1/B2 pass only the PSNR repeat limit: their range is
`0.007658 dB / 0.002257 / 0.003767`, failing SSIM and LPIPS. Mean(B) improves A by
`+0.063592 dB / +0.002894 SSIM` but regresses LPIPS by`+0.007793`, failing the common
non-inferiority bound. B2 also fails the required automatic full-view gate in all three views.

All arms have trainer step15188, Adam/scheduler counter15180, nine skipped updates, LR
`0.00705017667`, scale8192 and FAS count15189. B1/B2 Python, NumPy, Torch CPU and CUDA RNG hashes
are exact, proving the step-addressed outer streams are reproducible. Their GradScaler growth
trackers nevertheless differ (`612` versus`815`), so the timing of their last overflow differs
despite the same final skip count. Exact cumulative-point spread is`0.337111%`, versus the`0.05%`
limit; mean(B) shifts`0.471674%` from A and narrowly passes its separate`0.5%` bound.

Checkpoint drift also rejects the axis. B1/B2 symmetric relative L2 is`0.856276` for aggregate
field state, `0.516133` for occupancy `occs`, and`0.301771` for binaries. Only binaries pass;
field is worse than the rejected fixed-scaler pair and `occs` becomes slightly worse rather than
25% better. Duplicate model/sampler occupancy buffers remain exact inside each checkpoint.
Throughput passes: B1/B2 median-update regressions are only`0.341%` and`0.763%` versus A.

The exact independent-RNG speed mode is therefore closed without a different endpoint, seed or
full E2E attempt. It successfully isolates and restores the named RNG states, but does not control
TCNN atomic arithmetic, occupancy threshold feedback or GradScaler-overflow timing. The flag and
tests remain default-off diagnostic infrastructure.

Evidence root:
`/home/brans/lookcloser_leader_speed_runs/diagnostics/independent_rng_prefix_abb_seed42_v1`.
Machine-readable state and drift are `checkpoint_state.json` and
`checkpoint_drift_B1_reference.json`; fresh summaries are
`*/lookcloser/20260717_091500/rngprefix15188_evaluation_step-000015188.json`.

## Predeclared Stage-B-only CPU-prefetch quality-preservation screen

The first staged-JIT/cache E2E campaign already provides a clean all-gate checkpoint but misses
one hour by17.973 seconds. Its Stage A trained without CPU prefetch and selected the desired basin;
its separate historical-RNG-reset Stage B took645.004 seconds. Mature common-parent ABA measured
CPU prefetch as an exact sampling/RNG/state optimization with about2.098 ms/update saving. Enabling
it only in Stage B should therefore preserve the known Stage-A basin while saving about31.9
seconds, unlike the rejected full-trajectory prefetch run whose Stage A itself selected another
basin.

This diagnostic has one immutable parent and one candidate. Parent:

`/home/brans/lookcloser_leader_speed_runs/campaigns/leader_cache_color15189_geometry30377_hard91128_seed42_v1/stage_a_step-000075940_historical_rng_reset.ckpt`

Its SHA-256 is `0534f7d0…704b838`, trainer step75940, Adam/scheduler75906, LR
`0.00174156625`, scale16384, and deliberately absent historical RNG snapshot. The candidate copies
the recorded v1 Stage-B command exactly—seed42, B4096, cache, stable occupancy, FR0.3, fused Adam,
color-JIT from15189, geometry-JIT from30377, replayed cadence, historical LR/Adam/scaler and hard
step91128—and adds only `--cpu-fas-prefetch`. It does not enable independent streams or fixed
GradScaler. No intermediate/alternate checkpoint, extension, seed or repeat is eligible.

The branch authorizes one from-scratch staged-prefetch E2E campaign only if its fresh step91128
checkpoint simultaneously:

- passes PSNR≥29.617964, SSIM≥0.668450 and LPIPS≤0.231135;
- has zero significant artifacts in all three views and zero serious results in all ten ROIs;
- passes thin pipe, cable holes and fingers against the frozen archive on every metric;
- stays within `0.06 / 0.01 / 0.005` of the original v1 quality checkpoint
  `29.802864 / 0.675499 / 0.222623`;
- has exact FAS sample-count progression and cumulative point exposure within0.5% of v1's
  approximately220.223 B step91128 exposure;
- completes the saved-parent training branch in at most620 seconds, at least25 seconds below the
  original645.004-second Stage B. Diagnostic timing is not a milestone claim.

Passing all conditions permits one controller-integrated recipe with prefetch disabled in Stage A
and enabled only in Stage B. That full run must itself observe wall≤3600 seconds and every frozen
quality/detail gate; projected time or this saved-parent result cannot promote defaults.

### Stage-B-only prefetch result and terminal-save follow-up

The branch completed at step91128 with the exact accepted v1 optimizer exposure: Adam/scheduler
91087, LR`0.0012278067`, 42 skipped updates and scale16384. FAS count is91129 and cumulative point
exposure is220.245 B, only about0.010% from v1. Fresh evaluation is a complete quality pass:

| PSNR | SSIM | LPIPS | Significant / serious ROI | Pipe | Cable | Fingers |
|---:|---:|---:|---:|---|---|---|
| **29.795826** | **0.676245** | **0.222871** | **0/3 / 0/10** | pass | pass | pass |

Relative to the original v1 quality checkpoint the change is
`-0.007038 / +0.000746 / +0.000248`, comfortably inside the repeat limits. This establishes that
CPU prefetch can be confined to Stage B without losing the v1 quality basin.

Training took628.492 seconds, saving16.512 seconds from v1 but missing the conservative620-second
branch threshold by8.492 seconds. Therefore this exact command does not authorize a full E2E run.
Forensic inspection found that it deliberately copied v1's `save_interval=15188`; step91128 is a
scheduled save boundary and `_after_train()` immediately writes the same2-GB checkpoint again.
The later accepted controller pruning already removes this duplicate and uses0.1-second polling.

One final, distinct orchestration screen is therefore predeclared from the same immutable parent.
It keeps every training/sampling/optimizer field and `--cpu-fas-prefetch` exact, but sets the save
interval beyond the endpoint (`1000000`) so only `_after_train()` writes step91128, and changes
runner polling `1.0→0.1` seconds. Both changes occur outside optimizer updates; terminal prefetch
close/barrier and the one retained checkpoint are unchanged. No dense save, alternate endpoint,
seed, extension or repeat is eligible.

This production-orchestration branch must itself finish in≤620 seconds and pass the same numeric,
three-view/ten-ROI, priority-detail, repeat-delta, FAS-count and0.5%-point-exposure gates above.
Only then may the controller encode Stage-A prefetch off / Stage-B prefetch on plus the already
reviewed I/O pruning for one from-scratch E2E attempt.

The production-orchestration branch again passed every quality gate at
`29.805180 / 0.675693 / 0.223097`, with`0/3`, `0/10` and all three priority crops passing. Its
optimizer state is again91087 applied updates, LR`0.0012278067`, scale16384; exposure is220.252 B.
However, wall was627.755 seconds—only0.737 seconds faster and still7.755 seconds above the hard
screen. Directory inspection shows that both branches contain only one step91128 checkpoint; the
assumed duplicate terminal file was not present for this resumed interval. The I/O hypothesis is
therefore falsified rather than carried into a full run. This exact orchestration mode is closed.

The quality result is nevertheless important: two independent saved-parent continuations show
that Stage-B-only CPU prefetch preserves the accepted v1 basin. The remaining branch gap is about
0.51 ms/update. The next semantics-preserving candidate removes only occupancy telemetry that is
currently recomputed every16 updates: max/ratio/flipped-cell reductions, a full2M-cell binary clone
and emission of diagnostic-only metrics. Occupancy thresholding, binaries, warmup, dilation,
adaptive-sampler state and every optimizer input must remain exact. This axis requires explicit
default-on plumbing, parity tests and a new predeclared branch before GPU use.
