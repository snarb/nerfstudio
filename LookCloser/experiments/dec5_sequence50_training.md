# DEC5: 50 native frames, durable LookCloser training

## What was tested

Started 2026-10-28. User redirected **silver-sable to preparation/training**;
new exploratory runs use dev3 and clever-shadow. The old silver campaign is
closed (`research_retired.json`); its final 96-view review completed within the
requested one-hour wrap-up window.

Frames **000901–000950**, contiguous, were selected after inspecting the action
timeline. The subject applies lipstick in this interval. The static 65-camera
rig is validated over 899–1258. [Prepared input preview](assets/dec5_silver_sable/sequence901_input_preview.jpg).

- S3 source: `s3:aina-lustre-west-2/projects/Dec5Shoots/workspace/DEC5_5A_3/pq16_extract/fullres/`.
- Native source: 6144×3072 uint16 **Log3G10 / REDWideGamut**, not PQ. Decoding was
  checked against the existing frozen PQ/AP1 converter: subsampled frame925
  AP1 mean absolute linear difference 0.00000544, median relative difference
  0.0000505 (denominator `abs(reference)+.001`).
- Native training images retain the complete 6144×3072 field of view. The
  established fixed gain/Reinhard/sRGB conversion produces 8-bit training RGB;
  this is not HDR training. Original TIFFs remain available on S3.
- **64 train + one held-out camera**, `D004_A005_12103C`. Held-out RGB is excluded
  from frequency fits, semantics, background fits and training objectives.
- Genuine train-only frequency fits: 1000 steps/level, 16 levels, max8192,
  patch8, SSIM .95. Fresh native maps for every camera/frame. No stretched HD maps.
- Separate HD semantic evidence (1920×960) is mapped into native ray coordinates.
  Semantic labels are sampling localizers, not output masks or depth ground truth.

Full leader schedule: 4096 rays, hash23/max8192, safe truncated exponential,
corrected SH, stable occupancy and fixed256 warmup through4096. Frequency-grid
projection uses normalized image coordinates. FR1 through75940; FR.3 through
106316. Adam/scheduler/scaler/RNG persist within each frame; frames train
independently. Native 6K held-out renders accompany the seven evaluation gates.
Best checkpoint maximizes all-eval PSNR, with LPIPS tie-breaking within .07dB.

Scene adaptations: separate frozen ray-background fit, trusted-background
optical-thickness weight .05, conservative common normalized scene cube ±.15,
geometry-scaled marching, no stale-frame sparse-depth targets.

The requested green-dress recipe was recovered from the later **Luster000540**
campaign on FSx, not the earlier000470 report. Production revision**v4** uses
its **body80/FAS50** recipe: 80% sampling support (core plus protective outer
band),20% **whole image**, with FAS selecting half of core draws. This is not
a strict80/20 foreground/background pixel split. Mature allocation is roughly
one-third core-uniform, one-third body-FAS,13.3% protective band,20% whole image.
FAS ramps from4096 to8192. The upper-body band is constructed in portrait
coordinates before mapping to the sideways native images. Source frequency
classes are chosen with1→3 weights over nonempty levels, then uniformly within
the selected class. Camera IDs, native/semantic raster scaling and actual sampled
classes/field queries are audited.

The same archived recipe supplies normalized **Jacobian** frequency projection,
foreground-supported high-opacity frequency-grid updates, one-cell noncumulative
occupancy dilation, zero-initialized frequency grid and distortion ramp.01→1
over steps4096–12288. New standalone tests check Jacobian finite differences,
scene/image scaling, camera identity and native coordinates. A synthetic
end-to-end check exercised one fixed and one adaptive optimizer update and then
verified native restoration of Adam, scheduler, scaler, RNG and pipeline buffers.
Synthetic renders are **not** scene-quality results. [Recovered leader selection](assets/dec5_silver_sable/luster_body80_leader_selection.json),
[historical actual sampling](assets/dec5_silver_sable/luster_body80_actual_sampling.csv).

Revision**v5** preserves that training recipe and changes only evaluation batch
1024→8192. On1,179,648 rays spanning the native6144×3072 camera raster,
median render time13.44→3.11seconds (4.33×), peak3.01GiB. RGB/opacity differ by
only a few millionths; supported depth maximum difference4.95e-6, with zero
crossings of opacity thresholds.001/.01/.1/.9. Depth of effectively empty rays
is undefined and can differ; no such ray becomes supported. Running preparation
workers were preserved; future native fits usev5. This is a throughput change,
not a hair-quality claim. [Benchmark](assets/dec5_silver_sable/eval_chunk_native_rays_benchmark.json),
[deployed recipe](assets/dec5_silver_sable/sequence50_recipe_v5.json).

At21:00UTC revision**v6** supersedesv5 with two baseline correctness fixes:
stable packed-distortion accumulation and reproducible FR-phase/background
restoration. Eighteen focused tests, two real synthetic fixed/adaptive updates,
full-state resume and five-step default-sampler bit-equivalence checks passed.
A real91128 checkpoint now reproduces its sealed RGB to1.31e-6 max error;
loading its initial FR1 phase instead gives.2876 max error. The full corrected
HD recipe is running on clever-shadow. All eight active preparation processes
were preserved; native training has not started. Experimental hair-core mixture
and relaxed frequency-update gates remain **disabled** in production.
[Numerical baseline report](dec5_000901_numerical_baseline.md),
[deployedv6 recipe](assets/dec5_silver_sable/sequence50_recipe_v6.json).

## Results

Preparation is running; **no new 50-frame training quality results yet**.
At01:05UTC on2026-10-29 all3250 source TIFFs remain copied. Forty frames have
complete native frequency sets; eight more are fitting maps (44–46/64 cameras),
with two queued. All eight worker
processes are alive; no OOM evidence. Training waits until all50 frequency sets
are ready. The complete new recipe is also being exercised in a separate HD
preflight on clever-shadow, not counted as native production training.

Persistent campaign on silver:
`/home/ubuntu/lookcloser/campaigns/dec5_sequence50_20261028`.
Volatile inputs: `/cache/lookcloser/sequence50/raw` (recoverable from S3).
Code snapshots, image/map receipts, metrics, logs and checkpoints use the boot disk.
The `lookcloser-sequence50.service` user service is enabled with lingering and
restart-on-failure. A live restart test preserved **all eight worker PIDs**.
PID starttime prevents adopting a recycled PID. Completed inputs/maps/checkpoints
resume; failed frames back off and do not block remaining frames. Health snapshots
include GPU memory and worker state every30minutes. Root supervision remains active.

The independent enabled `lookcloser-sequence50-postflight.service` waits for all
sealed frame completions. It verifies the50 selected checkpoint checksums,
aggregates PSNR/SSIM/LPIPS, and writes `review50/gt_vs_best.mp4`, a contact sheet
and `campaign_complete.json`. The24FPS preview is a display choice, not verified
capture timing. This CPU-only finalizer does not change checkpoint selection or
claim the temporal-hair problem is solved.
It also builds `review50/final106316/gt_vs_final106316.mp4` and its metric manifest:
one common final FR.3 gate versus individually selected checkpoints. This checks
whether differing selected training stages contribute to temporal variation;
the mandated official per-frame PSNR best remains intact.

One status call returns every frame and all eight GPUs:

```bash
ssh ubuntu@silver-sable 'python3 /home/ubuntu/lookcloser/campaigns/dec5_sequence50_20261028/revisions/v6/code/LookCloser/scripts/run_dec5_sequence.py status --root /home/ubuntu/lookcloser/campaigns/dec5_sequence50_20261028'
```

## Insights and next steps

- Old 96-view fixed-source path: FG/BG q10 versus HD q10 reduced visibility-toggle
  proxy168068→155213 (7.65%), but present-hair proxy fell18795260→16530030 (12.05%).
  Depth-exposure dropouts changed861929→855637 (0.73%). This **does not establish a
  successful hair-popping fix**. [Metrics](assets/dec5_silver_sable/full96_final_metrics.json),
  [problem patch](assets/dec5_silver_sable/full96_problem_patch_0.jpg),
  [HD video](assets/dec5_silver_sable/full96_hd_video.mp4),
  [separated-field video](assets/dec5_silver_sable/full96_fg_video.mp4).
- dev3 now tests opacity floors .001/.01/.03/.1/.2, plus source-only / target-only
  .01 changes against .1. Frozen checkpoint, source order and RGB stay fixed.
  Both the export-opacity gate and quantile-mass gate are parameterized; FP64
  packed mass is retained so thresholding does not substitute a rounded opacity.
  Fourteen focused depth/geometry tests passed before the background sweep.
  The sweep and independent three-view UHD gate are now complete:
  [opacity results](dec5_001055_opacity_floor.md). Both-.01 is promising for
  holes/contour quality; the completed96-view comparison reduced visibility
  toggles22.98%, but transported-RGB change improved only2.80% and background
  fragments remain. It is a component candidate, not a complete hair fix.
- A separate clever-shadow series compares the same parent's uniform hair
  allocation, increased boundary share, frequency-weighted hair, and their
  combination. All three variants failed to improve held-out hair LPIPS; none
  was promoted. A consistency objective remains a candidate, not an accepted
  improvement. A separate real-HD preflight of the full new64+1 recipe runs on
  clever-shadow, reusing only its genuinely completed HD frequency fits.

### Supervision update, 2026-10-28 22:25 UTC

Silver: all3250 native TIFFs are copied;24/50 frequency sets are sealed,8 are
being fitted (48–50 of64 camera fits completed),18 are queued. All8 workers and
the controller are alive, GPU allocation about2977MiB each, persistent space
4675.7GiB free. Native field training waits for all50 frequency sets by design.
Production remainsv6. The corrected hair-sampling pilot improves short-run NeRF
hair LPIPS14.66%, but its matched4K GT-transfer gain is only.39%, with worse
missing-source coverage. No sampling promotion is justified yet; the full
schedule comparison is still running on clever-shadow. See
[geometry and appearance gates](dec5_000901_hair_sampling_after_fixes.md).

CPU-only audit of all64 native frequency maps for frame901 verifies uncropped
6144×3072 inputs. In confident train semantic regions, mean level is14.84 for
hair,11.50 for face,9.58 for trusted background; top-four-level fractions are
99.28%,71.72%,61.22%. Hair is91.65% at the maximum level. The inspected map
retains hair detail but also assigns high levels to noisy wall patches; a
frequency map alone is therefore insufficient as a subject sampling policy.
Keep the recovered person-support quota and reliable-foreground update guard.
This diagnostic is not evidence of improved depth or temporal consistency.
[Region histograms](assets/dec5_silver_sable/native_frequency_audit901/metrics.json),
[native RGB / frequency preview](assets/dec5_silver_sable/native_frequency_audit901/native_rgb_frequency.jpg).

Postflightv3 also compares motion-compensated RGB-error changes for the official
best and common-final sequences. GT-only bidirectional DIS flow is evaluated at
960×480 with a1.5-analysis-pixel forward/backward gate. The metric subtracts
the real GT appearance change, rather than penalizing physical movement or
lighting change directly. Full-frame scores and a propagated, fallible hair
proxy are saved with49 per-pair records and first/middle/last mask previews.
The hair reference comes only from the excluded frame901 evaluation image;
it lives outside train assets and is read only after all predictions are sealed.
This is a temporal RGB proxy, not hair-topology or depth ground truth. Two tests
verify translation/real-appearance invariance and injected-flicker detection,
including rejection of inconsistent flow. CPU dependency smoke passed remotely;
the waiting postflight service is active. Production fitting code remainsv6.

Root supervision now uses one command for all8 Silver workers and both Clever
full pilots, with parallel SSH, bounded timeouts, controller/worker identities,
checkpoint/gate progress, GPU allocation and OOM evidence:
`python3 scripts/check_dec5_campaigns.py --record`.
It appends the full checks to each campaign's supervision log; SSH failure is
reported as interrupted supervision, not a failed training job. Last recorded
check22:37 UTC: Silver24 ready/8 fitting/18 queued, active fits58–60/64;
Clever corrected control67600, hair44800; all expected processes alive, no OOM.
Next mandatory root check is no later than23:37 UTC, or immediately at a finished
training/evaluation gate.


### Current recipe and supervision (2026-10-29 01:10UTC)

**v7_seed42** is the active immutable recipe. It preserves the full corrected
v6 leader and adds explicit seed42 to the frozen background fit. Source archive
SHA256 `3b42a8acfadee99d02988c49e4f3c9ead5109a7dd4eab293e93180027566ff5c`.
No active frequency worker was restarted; all workers remain alive and no OOM
was observed. The deployed background fitting function was compared to the
locally tested function text, not cross-version AST serialization. Two real fits
with distinct incoming RNG states and explicit42 give identical weights.
[Recipe](assets/dec5_silver_sable/sequence50_recipe_v7_seed42.json),
[equivalence](assets/dec5_silver_sable/neural_background/seed42_deployment_function_equivalence.json).

The native cold15k separate-background trial did not pass its quality gate;
strong hair-core sampling also failed completed full-schedule face/contour
checks. Neither is enabled in production. New contour-budget trials preserve
original body/FAS draws. The CPU-only postflight v3 performs both official-best
and common-final videos plus bidirectional-flow RGB-stability diagnostics; it
must finish before campaign completion. No native50 training-quality or temporal
improvement result is claimed before those renders exist.


Postflight **v4** additionally writes two constant-crop head videos, comparing
GT with official best and with final106316 at **native pixel scale**, without
spatial resampling or prediction stabilization. The evaluation-only901 hair/face
reference defines one crop with128native-pixel padding. This complements, rather
than replaces, full-frame reviews; later motion may reach the crop edge. Encoded
native crop pairs are3848×2618 for this reference, not a standard4K frame.
A real native901 single-frame encode/decode smoke passed locally for both mode
labels (same15k prediction, not a completed50-frame result); the decoded preview
was inspected. Silver CPU imports plus an independent two-frame H264 encode/decode
also pass. Only the waiting CPU postflight service restarted; GPU workers did not.
The all50 completion receipt now also requires these native head artifacts.
[Smoke receipt](assets/dec5_silver_sable/native_head_review_smoke.json).

For the completed HD control, official-best45564→final106316 hair LPIPS improves
.229339→.209757 (8.54%) and face .108621→.102883 (5.28%); contour is essentially
unchanged. Background-dominated whole-frame PSNR selection can therefore hide
late foreground improvement. The mandated selection stays unchanged, with a
separate common-final sequence comparison to assess this tradeoff and timing.
