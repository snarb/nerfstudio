# Native EXR, HDR reconstruction, and automatic frequency maps

Status: complete; implementation, full campaign, native evaluation and visual review passed.

## What was tested

The implementation follows the local `Paper LookCloser.md` recovery principle—choose the minimum
frequency that reconstructs a patch—but removes its predefined SSIM threshold. Training inputs,
targets, compositing and saved masters remain scene-linear EXR. No color grade or sigmoid RGB is
used in this path.

The staged comparison is frozen before measurement:

1. HDR reconstruction screen: linear L1, RawNeRF-style weighted L2, Linear-PQ, PQ-L1, and
   EAG-PT-inspired PQ plus DSSIM.
2. Frequency-map screen: scene-calibrated empirical crossing, relative multi-crossing ensemble,
   and threshold-free knee, all derived from one recovery cube per image.
3. Three-point tuning on the selected reconstruction/map pair.
4. One final run capped at the leader Stage-A budget of75941 updates.

Selection uses maximum PQ-domain PSNR, with lower LPIPS breaking ties inside0.07dB. SSIM is reported
and all three metrics must be accompanied by EXR masters and fixed-exposure review sheets. Internal
training/evaluation loss values are not reported.

## Data-quality results

The EXR dataset at
`/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740` parses as66 training and3 evaluation
views at1920×1080. The observed images are float32 RGB, include legitimate negative values and
values above1, and contain no non-finite channels in the audit. Across all69 frames the observed
channel envelope was approximately `[-0.716, 9.84]`.

Train-split robust calibration (stride-8 sample, 2,138,400 pixels):

| Field | Value |
|---|---:|
| Linear output scale (luminance q99.9) | 0.11768185 |
| Initial radiance | 0.00558929 |
| Scene-to-nits scale | 4654.2274 |
| Log-mean luminance | 0.00429717 |
| Luminance q99.99 | 0.46080104 |
| Negative channel fraction | 0.00519999 |
| Non-finite channel fraction | 0 |

These statistics are evidence that an 8-bit/sigmoid input path would discard material signal and
that one fixed threshold calibrated in a display-referred color space is not portable to this data.

## Implementation validation

The combined focused EXR/HDR/frequency/campaign and runner/data regression selection passes64
tests. Covered behavior includes EXR path/byte decoding and round-trip writing, preservation
of negative and super-white values, dataset cache safeguards, ST2084 reference/round-trip/gradient
behavior, all HDR output/loss variants, renderer clamp compatibility, adaptive recovery methods,
bootstrap selection, optimizer compatibility and existing LookCloser sampling/eval plumbing.

The standalone native evaluator was smoke-tested on a synthetic EXR prediction/GT pair. It produced
PSNR `63.69928`, SSIM `0.999976`, LPIPS `0.000005`, zero non-finite pixels and a review contact sheet.
This is a plumbing test, not scene-quality evidence.

## Training results

The full66-view recovery fit completed for every camera. The span between the first and last saved
recovery cube was1413 seconds; the cached PQ-proxy and selection pass took25.4 seconds. Each method
contains66 finite positive maps of shape135×240 (8×8 source patches), and every map hash matches its
sidecar. The selected maps cover the full16–8192 scalar-resolution schedule.

| Automatic family | Mean rank agreement | High-detail overlap | Spatial coherence | Effective bins | Entropy | Top-two-bin share | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated empirical crossing | 0.7714 | 0.5466 | 0.5925 | 9.25 | 0.8010 | 0.4736 | 0.1147 |
| Relative three-crossing ensemble | -0.3092 | 0.1493 | 0.4669 | 9.01 | 0.7914 | 0.3793 | 0.0000 |
| Threshold-free knee | -0.1173 | 0.1885 | 0.4058 | 12.22 | 0.9005 | 0.3395 | 0.0000 |

The pretraining proxy winner is calibrated median scene crossing
`calibrated_q50_t0.92149889_guided3`. It won all200 bootstrap resamples within its family. Knee and
relative maps have attractive bin diversity but fail the preregistered structural gates (mean rank
agreement at least0.1 and high-detail overlap at least0.25); this prevents entropy alone from
promoting maps whose ordering is visually reversed. Side-by-side previews confirm the decision:
the calibrated map forms coherent regions around brick, cables, bodies and thin structures while
leaving the smooth floor low; the relative and knee maps are visibly noisier or assign high bins to
smooth walls.

The five-way HDR reconstruction screen completed at two evaluation boundaries (15188 and30376).
The table reports each candidate's best checkpoint, never its internal training loss.

| Reconstruction path | Best step | PQ PSNR | PQ SSIM | PQ LPIPS | Structural artifact score |
|---|---:|---:|---:|---:|---:|
| Linear L1, softplus head | 15188 | 23.6444 | 0.7737 | 0.5777 | 146.018 |
| RawNeRF weighted L2, softplus head | 30376 | 30.2492 | 0.8714 | 0.3070 | 8.408 |
| Linear-PQ, softplus head | 30376 | 33.3601 | 0.8884 | 0.2573 | 5.944 |
| PQ-L1, PQ head decoded before compositing | 15188 | 29.6407 | 0.8247 | 0.4519 | 44.746 |
| EAG-inspired PQ-L1 plus patch DSSIM | 30376 | **33.6954** | **0.8902** | **0.2435** | **3.663** |

Linear L1 produced severe floor/foreground floaters. The PQ-head candidate desaturated the scene,
showed strong residual structure and collapsed after its first checkpoint. RawNeRF was stable but
visibly softer and hazier. Linear-PQ was clean, while patch DSSIM retained that behavior and gave
the sharpest/coherent result of the screen. EAG-inspired PQ plus DSSIM therefore advances to the
map-family screen. It uses contiguous11×11 training patches; the lightweight unstructured eval-loss
batch falls back to PQ-L1, while the reported full-image SSIM remains the real PQ image metric.

The map-family screen reused the identical calibrated/EAG run and trained the two alternatives.
Relative multi-crossing was stopped after the first boundary because it missed the predeclared
0.5dB PSNR trajectory envelope. Knee stayed inside the envelope and completed both boundaries.

| Automatic map used in training | Best step | PQ PSNR | PQ SSIM | PQ LPIPS | Structural artifact score | Decision |
|---|---:|---:|---:|---:|---:|---|
| Calibrated empirical crossing | 30376 | 33.6954 | 0.8902 | **0.2435** | 3.663 | Proxy winner |
| Relative three-crossing ensemble | 15188 | 32.5486 | 0.8691 | 0.2984 | 7.680 | Early reject |
| Threshold-free knee | 30376 | **33.8492** | **0.8942** | 0.2549 | **1.843** | Downstream winner |

The downstream result overturns the proxy-only choice. Although knee failed the structural proxy
gate, it is0.154dB ahead of calibrated at the second boundary, outside the0.07dB LPIPS tie window,
and its artifact score is50% lower. Fixed-exposure review shows a cleaner floor marker and thin
structures. The selected automatic EXR map is therefore the threshold-free knee, not the
scene-calibrated crossing and not the legacy hard-coded threshold.

Three-point DSSIM tuning kept every candidate inside the two-boundary trajectory envelope:

| DSSIM weight | PQ PSNR | PQ SSIM | PQ LPIPS | Structural artifact score | Decision |
|---:|---:|---:|---:|---:|---|
| 0.1 | 33.7752 | 0.8932 | 0.2590 | 3.740 | Reject |
| 0.2 | **33.8492** | **0.8942** | 0.2549 | 1.843 | PSNR maximum |
| 0.3 | 33.8260 | 0.8940 | **0.2530** | **0.804** | Selected |

Weights0.2 and0.3 are inside the0.07dB PSNR tie window, so LPIPS selects0.3. The same variant also
has the cleanest artifact audit and fixed-exposure review. The final Stage-A-length run uses knee
maps, EAG-inspired PQ plus DSSIM, and DSSIM weight0.3.

The capped final run completed all five evaluation boundaries:

| Step | PQ PSNR | PQ SSIM | PQ LPIPS |
|---:|---:|---:|---:|
| 15188 | 33.1601 | 0.8833 | 0.2895 |
| 30376 | 33.5702 | 0.8934 | 0.2581 |
| 45564 | 33.8063 | 0.8970 | 0.2391 |
| 60752 | **33.8597** | 0.8981 | 0.2286 |
| 75940 | 33.8176 | **0.8984** | **0.2218** |

Step75940 is0.0421dB below the maximum and therefore inside the0.07dB tie window; its lower LPIPS
selects it over step60752. Independent EXR evaluation gives `33.8176 / 0.8984 / 0.2218`, exactly
matching the model evaluator within rounding. Predictions contain no non-finite channels, negative
channels or values above the calibrated PQ peak.

All three selected-view `-2/0/+2 EV` sheets were visually reviewed. Geometry, floor markers, cables,
stands and people remain coherent without the floaters or desaturation seen in rejected losses.
The full-frame artifact detector flags a bottom-border component, but all nine declared detail ROIs
have zero serious components and the stand-connector score is zero; inspection confirms the border
flag is not a scene-structure break.

Final artifacts:

- Checkpoint: `/mnt/data/lookcloser_exr_hdr_runs/exr_hdr_auto_frequency_v1/lookcloser/final_eag_pq_dssim_knee_s42/nerfstudio_models/step-000075940.ckpt`
  (`sha256:7ab468756a8ff1141f1dec30b3d1d96b00b41c14d16f38d13849f12ac5796e13`).
- Paired native EXR renders: `/mnt/data/lookcloser_exr_hdr_runs/exr_hdr_auto_frequency_v1/lookcloser/final_eag_pq_dssim_knee_s42/renders_best_step-000075940/`.
- Fixed-exposure review sheets and per-view metrics: `/mnt/data/lookcloser_exr_hdr_runs/exr_hdr_auto_frequency_v1/lookcloser/final_eag_pq_dssim_knee_s42/hdr_review_renders_best_step-000075940/`.
- Complete campaign manifest: `/mnt/data/lookcloser_exr_hdr_runs/campaigns/exr_hdr_auto_frequency_v1/campaign.json`.
- Map provenance: `/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740/lookcloser_frequencies_exr_auto/provenance.json`.

## Insights and next steps

The one-image one-step diagnostic originally selected a relative candidate, while the adequately
trained full-scene proxy selected calibrated crossing and downstream reconstruction selected knee.
This is direct evidence that tiny preprocessing diagnostics and proxy scores alone are too noisy
for scene policy. The recovery cube and proxy gates remain useful for generating candidates and
early rejection, but the final policy must be chosen by downstream metrics and visual review.

The task's acceptance condition is met: EXR stays native scene-linear through input, training,
compositing and output; the legacy scene-specific threshold is unused; the selected map is
threshold-free and scene-adaptive; the selected HDR objective was tuned; and the full-budget run
improves monotonically through the useful trajectory before the expected late PSNR/LPIPS tradeoff.
