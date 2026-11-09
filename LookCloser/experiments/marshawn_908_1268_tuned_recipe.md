# Marshawn 908–1268 — native hyperparameter campaign

Status: tuning in progress. The recipe below is the control, not a selected winner.
The six-hour wall-clock window began 2026-11-09 11:08:12 PST and ends 17:08:12 PST.

**Confidence-port correction, discovered12:53 PST:** early runs used a linear
outside taper. The historical recipe uses exponent `log(.1)/log(.5)` (~3.32193),
so weight is0.1 at half width, not0.5. These early runs are retained as explicit
linear alternatives, not exact old-recipe reproductions. New reference controls
recompute the sealed historical helper from qualified actor masks, which also
removes the confidence halo around rejected crew. RGB/evaluation actor labels
and static admission geometry remain unchanged. The six-hour clock is unchanged.

## What is tested

Frame908,69TRAIN/3heldout cameras: **CB, GC, NB**. These minimize120-degree azimuth
gap error within the middle60% of camera heights. All four visible performers
receive equal masked AlexLPIPS weight within each view, then equal view weight.
PSNR and SSIM are supplementary. No training-image resizing: calibrated native
actor crops with128px context and integer principal-point shifts.

The starting recipe is hash25/SAM15/core.01/body80,8192rays,uniform16,
Charbonnier with actual proposal-confidence normalization,Adam(.9,.99),
LR.01→.0001 over200k,warmup4096,FR1→.3 at75941 with its seed-reset event,
corrected ARM allocator/R0 occupancy. Every trajectory has a150000-attempt ceiling;
recovery replay and conservatively reserved unfinished blocks consume that budget.

Controls compare uniform16 against source-bound progressive learned maps at
SSIM.90/.95,ARM coarse step/minimum/allocator limits,max3 interval estimates,
field resolution16384,earlier FR relaxation,frequency update512,head20%,batch16384,
and the supported scene-linear EXR path against PNG. GPU-resident gather is paired
with CPU/RAM caching; ordinary training already caches its images in RAM.

## Geometry and data findings

The accepted physical V3 box stays unchanged:
`[-.41,-1.20,-.15]→[3.41,5.14,1.77]` in its canonical rotated-floor rig frame.
The canonical JSON is `marshawn_908_1268_aabb/aabb_original_rig.json`.

Fitting the66 common camera centers to the historical1660 prepared poses gives
**model units/rig unit =0.2217203609534452**,RMS4.6864e-8,max1.1333e-7.
This common isotropic conversion is required to carry the old numerical marching
steps into comparable physical units. Model camera centers and AABB coordinates
are scaled; rotations and focal lengths are unchanged. Physical box membership
was checked on10000 points. Raw rig units remain a separate ablation.

Six source views contained worker fragments according to source-image inspection
and projected COLMAP witnesses: BC,CC,CD,KE,MC,NB. The accepted performer/worker
qualification and exact input bytes are retained in the campaign artifacts.
The three heldout views each contain four retained person domains. These are
camera-local pixel domains, not claimed global identity tracks. Head proposals
use current-image SAM head/hair/face prompts and native refinements; no bbox-top
head proxy is substituted. Renders are saved before metric masking.

PNG photometry: source PQ16/AP1-D65 → linear Rec709 /52.13360318135631 nits,
fixed gain1.6,Reinhard,sRGB8. EXR retains the corresponding nonnegative scene-linear
float32 crop; it adds no independent source information. Both are evaluated
against the same PNG references. EXR adoption requires>=2% actor LPIPS improvement,
head-detail improvement in at least two views and no>5% head-view regression.

## Current validation and execution

All1531 historical source files were reconstructed byte-exactly; new runtime and
four-person/72-camera adapters are separate. Technical GB300 training completed
8194steps including adaptive callbacks. Normalized-unit32-step canaries passed
on GB300 and Silver; Silver own-checkpoint continuation32→64 also passed.
GPU/CPU gather equality with shuffled camera IDs and executor-crash survival were
tested. Original mask jobs were already complete; none were killed for tuning.

One initial resolution16384 launch failed a metadata compatibility guard before
optimizer updates. A separate consumer-grid metadata variant keeps uniform scalar16
and RGB bytes unchanged; the failed preparation is retained, not scored as a
scientific negative result.

The primary coordinator and GPU executors run as systemd services. The coordinator
is on Silver at `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109`; NVIDIA
source owners remain WS07/WS08. Cross-host control uses Silver→calm-otter→NVIDIA
and survives the conversational reverse tunnel. Additional host enrollment checks
real technical completion before dispatch. Active workflow supervision continues.

## Final recipe and sequence

Pending measured selection. After freezing this file and the machine-readable
recipe,all361frames908–1268 cold-start with72TRAIN cameras,including former
heldout views. Their monitoring metrics are explicitly in-sample.120fps remains
assumed; frame-index selection does not depend on that assumption.

### Live campaign, 9 November 2026 11:41 PST

Six-hour tuning interval: 11:08:12–17:08:12 PST. Twelve GPUs are now enrolled: WS07/WS08, eight Silver GPUs, one accessible Olive GPU, one Clever GPU. Dev3 is reserved for inference qualification because its 30 GiB RAM cannot safely hold the native training cache.

At matched 16,000 steps, actor LPIPS: uniform GB300 0.19413; uniform Silver 0.19848; GPU-resident RGB Silver 0.19603; learned .95 0.19497; learned .90 0.19489; half coarse step 0.18631; doubled coarse step 0.21027; 2048 ray samples 0.19402; max3 allocator 0.19580. These are intermediate measurements, not a frozen recipe. CUDA architectures/software differ; comparisons across hosts are not claimed bit-identical.

Early raw renders retain contour haze. Source masks, not composited previews, remain the first check for contour/hair artifacts. Intermediate review: https://drive.google.com/file/d/12FJ60o7WpBqlfhAVEV_Xwh3p7GDL_r0b/view .

Controller: silver-sable `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109`; service `marshawn908-controller.service`. Agents and 5-minute process/GPU checks are independent systemd services. Production allocation respects source frame ownership on NVIDIA workstations. Clever exports completed model tensors to Silver before pruning its own staging data. Optimizer checkpoints are only released after a durable completion receipt; exports verify every saved pipeline tensor exactly. Optional inference reload utility avoids training-image caches.

Dev3 L40S inference reload qualification passed on the exact Silver uniform16k tensor export: maximum per-view LPIPS difference1.03e-6, SSIM4.77e-7, PSNR4.89e-6dB, across CB/GC/NB. Source runtime torch2.7.1+cu128; helper torch2.4.1+cu121. No training image cache or optimizer was allocated. The helper now runs a same-weight inference-cap sweep; these diagnostics do not silently alter the frozen training worker.

Recovery validation: missing export-manifest after tensor-file commit is repaired; an existing export from another source checkpoint is rejected. The focused fault-recovery test passed. Frame archive transfer excludes optimizer states and recreatable linear masters while retaining local resume states until remote SHA verification succeeds.

Native mask review saved at WS07 `mask_edge_review/{GC_person3,CB_person1,NB_person2}.png` under the campaign root. Main hat/hair contours are retained in the reviewed views. The production soft alpha has visible tile seams; training/evaluation use its fixed >=128 foreground, not raw alpha as a color target. NB's fallen actor head proposal has a small disconnected false-positive patch on clothing, enlarging the fixed head ROI and diluting its head LPIPS. Main actor LPIPS is unaffected by this head-only artifact; do not present this auxiliary metric as a perfect facial-quality measurement. Keep all live tuning masks fixed for comparability; examine late raw renders before attributing remaining haze to segmentation.

At matched32k: uniform GB3000.177967; uniform Silver0.179103; learned .950.179873; learned .900.177091; half coarse step0.166759; doubled step0.191442; max3 allocator0.178339. Half-step improves actor LPIPS by~6.9% against the same-hardware baseline. It continues toward150k. A quarter-step cold ablation was added on WS07 after this measured result; the fixed deadline is unchanged. `uniform_clever` is a legacy arm label for another cold uniform repetition actually scheduled on Silver; host columns are authoritative.

An additional W32-confidence ablation keeps RGB, actor/head masks, crop rays and static admission geometry byte-identical. It places the existing W32 confidence raster in the sampler's historical W64 slot, with explicit ready metadata; proposal domains are unchanged and their actual confidence expectation is recomputed. It is not alpha supervision or a binary cutout. This tests whether weaker background-color supervision near silhouettes reduces haze. It has its own immutable dataset and run; all earlier masks/runs stay fixed.

The completed same-hardware32k cache benchmark took3378.36s with RAM-cache gathering and3298.20s with packed GPU RGB/auxiliary gathering: ~2.43% faster, below the5% adoption threshold. Both already avoid rereading training images from disk during iterations. At this evidence level, keep the RAM-cache path for production; the GPU path remains an explicitly tested option.

Final scheduling now reserves weighted quotas over the whole cut before respecting NVIDIA source ranges, preventing the later WS08 half from receiving a disproportionate backlog. Coverage/uniqueness and both41-frame NVIDIA quotas passed the focused allocation test. Production retries use12 bounded attempts with exponential backoff capped at15min; the150k optimizer-attempt ledger still persists across all retries. Scientific workers continued across a live agent restart on all12 GPUs, verified by process identity and fresh progress.

Full-FOV raw FHD renders of the half-step32k export were checked on CB/GC/NB (no GT masking, with original crop offsets undone and ray parity verified). All four performers remain represented. There is substantial unsupported opacity around bodies and, in NB, away from them; ROI overviews understated this limitation. No claim of production visual acceptance is made at32k.

Added one isolated `clear_background` ablation, paired with half coarse step. It adds weight0.01 on squared accumulation only where the production soft mask is exactly0, foreground is farther than64native pixels in both image axes, the pixel is valid, no unknown/crew person is present, and it is at least64px from the crop edge. It ramps4096–12288. RGB/confidence/foreground/head masks and R0 admission stay unchanged. This is a conservative semantic empty-ray prior, not calibrated physical alpha GT. Its extra wrapper is explicitly SHA-bound; existing scientific worker/runtime files stay unchanged. Protection of hair/uncertain/other-person/border pixels passed the focused test. It must earn selection through actor/head quality, with sampled conservative-background opacity only breaking a0.5% actor-LPIPS tie.

EXR now uses the measured half-step configuration, with PNG comparisons at the same parameters, steps and inference cap. Adoption requires>=2% actor LPIPS improvement and>=2% head improvement on at least two views without>5% head-view regression. This compares the supported linear-softplus EXR path against the PNG path, including their output parameterizations; EXR is derived from the same PQ16 source and adds no source information.

At matched32k (linear confidence), EXR half-step0.167769 versus PNG half-step0.166759: no actor improvement. Quarter-step PNG0.159693, raw-rig-units control0.159664, and batch16384 control0.166395. These remain explicitly linear-profile results.

Two inferior linear controls were pruned after committed80k comparisons: max2048 samples0.155607 and initial FR.30.159874 versus half-step0.148266. Their results/checkpoints are preserved; new cold quarter-step/power-confidence controls, with/without the conservative background prior, replace them. They do not inherit weights.

Clever's32k16K-resolution result0.182376 is valid, but its original Python runtime disappeared and an unrelated CA4D job now occupies that GPU. It is excluded from new scheduling;11 GPUs remain active. Replacement CPU dependencies were staged without stopping the foreign job; no GPU recovery qualification is claimed yet.

Final iteration policy distinguishes the measured winning checkpoint from the production budget. With no confirmed plateau, allow up to150k attempts and keep each frame's best scored checkpoint; do not imply that the validation winner was already measured at150k. If late plateau is confirmed, use the observed trajectory length. All72 cameras are included only in final training.

At14:12PST manual fleet supervision verified11 live owned workers with fresh agent states; Clever remains disabled. The8GPU Silver fleet is fully in use. Late80k GC native actor render inspection still shows unsupported opacity around bodies, although clothing detail has improved. This is not a segmentation-only diagnosis. Added one fresh historical-power half-step control with conservative background coefficient0.1 (versus0.01), retaining the same protected hair/unknown domains. It replaces the dominated default-step power control after its committed32k score0.178566 versus half-step0.166072. No transferred weights, altered live datasets, or tuning-clock reset.

At matched32k, historical-power half-step plus conservative background prior0.01: actor LPIPS0.164264 versus0.166072 without the prior. Auxiliary trusted-background opacity within queried native actor ROIs drops from0.224954 to0.098397 (~56.3%). This is not a whole-scene geometry metric; the primary actor/head metrics and raw visuals remain decisive. Late linear half-step96k reaches LPIPS0.145254 but still has sampled background opacity0.257994. Late80k raw full-FOV and nearby novel views retain substantial halos; Drive review: https://drive.google.com/file/d/1rOUx2L6r1gnMdQQiZMIUj9xLzql0Lbgx/view .

Selection clarification before the new quarter-power32k and strong-prior16k results: continued finalists are compared at their latest shared committed evaluation gate (capNone/10 at identical steps), rather than rewarding earlier starts with more optimization. The0.5% actor-LPIPS tie tolerance and5% per-view head-regression guard still apply there. Only after choosing parameters/cap is that arm's own best later measured checkpoint selected. All screened/pruned trajectories remain in the evidence table. A focused unequal-duration test confirms that an older128k result cannot override a better matched64k finalist. This is successive screening within the unchanged six-hour window, not a claim that every arm reached its plateau.

Matched32k historical-power quarter-step: LPIPS0.157821; with background prior0.01:0.158535 (0.45% relative difference). Strong half-step background prior0.1:0.167160. At64k, historical-power uniform16 half-step0.149010 versus learned.95 half-step0.152682. These measurements retain fixed masks and are not plateau claims.

Additional pre-production evidence is saved in `marshawn_908_1268_diagnostics/`: all25,992 selected images have successful production mask-area receipts with no zero-area foreground (minimum325,811pixels; this is a completeness check, not contour-accuracy certification). Source actor semantics preserve the full production subject mask, including unassigned subject/prop pixels, subtracting only qualified crew; SAM person proposals do not replace the training actor union. A same-weight80k inference alpha-threshold sweep0/.001/.003 changed LPIPS by<6e-8 and sampled opacity by<6e-9. The sealed sampler intentionally returns no density callback during evaluation, so this knob has no meaningful effect on that path. No threshold filtering is adopted or claimed to solve halos.
