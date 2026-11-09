# Marshawn 908–1268 — native hyperparameter campaign

Status: tuning in progress. The recipe below is the control, not a selected winner.
The six-hour wall-clock window began 2026-11-09 11:08:12 PST and ends 17:08:12 PST.

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
