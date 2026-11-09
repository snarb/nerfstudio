# Marshawn 1660: patch DSSIM proposal, no training authorized

## What was tested

The new bounded-linear distortion control improves the matched common target
from PSNR/SSIM/LPIPS 27.415559/.781585/.358962 to
28.598103/.796725/.319146. Gray mottling remains on HC/KA floor coverings.
The optional image-structure experiment keeps its native MSE, linear distortion
.01, observed LOS .01 and surface-band .01 and adds
`weight * (1 - functional_SSIM)`, with weights .1 and .3 proposed for gradient
measurement. This convention matches existing LookCloser EAG code; it is not
the alternative half-scaled DSSIM definition. The local paper's training loss
does not prescribe this additional image-structure term.

The frozen parent is the new step8000 checkpoint with SHA
`e9af00c75e8cb58ac822143d67dfbd65587e125369cb38fc4e2f1103d8518960`.
No model/field architecture, RGB display transform, geometry or occupancy rule
changes are proposed. Standard native PatchPixelSampler11 changes the ray
distribution explicitly; a future comparison would change both sampling and
the image-structure objective, not isolate only the objective.

## Results

Nine CPU tests pass. They check camera-constant contiguous patch ordering,
empty/incomplete/misordered batches, structural displacement with equal color
histograms, finite gradients, identity, exception cleanup and the actual native
sampler's initial16-to-zero hazard. The explicit dynamic initial estimate
`max_num_samples_per_ray=2048` with unchanged `target_num_samples=262144`
starts from128 requested rays, rounded to121 by native patch sampling. This
setting is only the dynamic batch constructor estimate, not a native marching
cap. `ignore_mask=True` requires actual all-one QA for all66 source masks.

The functional SSIM uses float32 with autocast disabled for its local moments,
Gaussian kernel11/sigma1.5, data_range1, k1=.01, k2=.03. Its helper is a plain
function and adds no parameters, buffers, checkpoint keys or learned metric
network. Training batch instrumentation exists only inside a context and
restores the datamanager method even on failure.

The read-only CUDA probe is staged separately at
`/fsx/tmp/marshawn_001660_hd/native_png_patch_dssim_PROPOSAL`.
It checks an actual cold121-ray batch first, YAML roundtrip, actual sourcePNG
targets, patch camera/XY order, native callbacks, full combined AMP backward,
then four vetted train-floor patches and four native-random train patches on
the new frozen8k field. It creates no optimizer and performs no training or
post-load occupancy callback. Its gradient comparisons are diagnostic and
cannot demonstrate quality improvement.

## Insights and next steps

Geometry's same-field native/all-cells/protected-grid and finer-quadrature
comparison has priority. A renderer integration error could explain residual
mottling without another learning objective. All DSSIM training remains held.

The parent's actual checkpoint contains Adam, scheduler, GradScaler and
Python/NumPy/TorchCPU/CUDA RNG state. Its saved scale is131072. The proposed
full-component warm start would restore these native components, but the new
patch distribution, dynamic constructor estimate and unsaved dataloader cache
or cursor prevent claiming identical continuation of the old ray trajectory.

A hypothetical global trainer end21000 would run12999 attempts from8001,
giving conservative total8610+12999=21609 including the parent's discarded
post8k attempts. A first2000-attempt review would occur at global10000. This is
a budget proposal, not authorization to train. Accepted PNG release files and
ZIP remain immutable. Actual floor crops, same-target PSNR/SSIM/LPIPS and
ray evidence are required; absence of a broad brown cloud does not satisfy
the user's clean-floor requirement.

### Actual zero-update probe

The initial DEV3 L40S/Torch2.4.1 probe completed: all66 masks are all-one,
all actual sampled targets equal independent PNG decoding exactly, YAML and
state keys pass, and field/grid plus Python/NumPy/Torch RNG are unchanged.
The cold first batch contains121 rays and31240 native samples, not zero rays.

| Train floor patch | DSSIM .1 gradient norm / MSE | DSSIM .3 / MSE |
|---|---:|---:|
| AA (550,650) |3.22|9.66|
| GA (1350,500) |1.46|4.36|
| GA (1450,550) |2.66|7.99|
| OA (1450,750) |10.18|30.51|

These are ratios of gradients of local terms, not quality metrics. Both terms
are divided by eight for the actual eight-patch packet mean. Four standard
random patches yield .1 ratios .82–5.56 and .3 ratios2.45–16.67. Gradients
affect both geometry/hash and color parameters; SSIM on low-contrast11px
patches is not automatically a weak correction.

Cold MSE+existing auxiliaries and DSSIM .1 are finite at the configured65536;
.3 is finite only after a diagnostic backoff to32768. On the frozen8k packet,
both full combined variants overflow at saved parent scale131072. Diagnostic
norms become finite at32768 for .1 and16384 for .3. No scaler/config was
changed. The initial report lacked the full same-packet baseline without DSSIM.
The separate v2 zero-update supplement now confirms that baseline is finite at
131072 (gradient norm .00751854), while adding .1 or .3 reproduces the overflow
on the identical packet. Thus the new term causes this observed parent-scale
overflow. Original source/report are preserved. Diagnostic finite combined
norms are .0230713 at32768 for .1 and .0608437 at16384 for .3; these are not
production scaler changes or evidence of quality improvement.

The probe's `passed=true` means structural/read-only checks completed, not a
production AMP or training release. Both proposed weights currently fail the
finite-at-parent-scale gate; no training is authorized or launched.

### Small-weight full-state phase preparation

The follow-up selects weight .01, without increasing density regularization.
Its isolated stage is
`/fsx/tmp/marshawn_001660_hd/native_png_patch_dssim001_PHASE_STAGED`.
The strict same-packet probe tests only .01 and uses no scaler backoff:
baseline full gradient norm .00751851 and baseline+.01 norm .00842029 are
both finite at the parent's131072. Actual floor .01/MSE norm ratios are
.3221/.1454/.2661/1.0172 (AA/GA/GA/OA); the cold121-ray packet is finite
at its unchanged65536. This meets the narrow measured AMP gate and does not
establish trained image quality.

The concrete proposed phase restores the immutable8k model plus Adam,
scheduler, scaler and native checkpoint RNG. It begins at global8001 and
ends before21000, with an in-place pause before10001 after the global10000
checkpoint, all-view evaluation and normal post-eval save/CSV flush. Native
default plateau stopping remains active. Parent8k remains an eligible best
candidate alongside all saved phase checkpoints. Final evaluation binds the
exact selected checkpoint path, including when the external parent wins;
three targeted selection/evaluation tests cover this case and phase winners.

The actual full-state native Trainer.setup proof passed on DEV3: all pipeline
values/keys, Adam state/groups, scheduler state and scaler are exact; parent
Adam count7999 and scaler131072 are preserved. Native last-point RNG restore
was exercised and all four streams match exactly. No optimizer update or
training iteration was called. Independent production review also passed.
Both controller and direct train entry points
reject launch while any release gate is false. Root subsequently authorized
exactly one reviewed phase; the pre/post-spec hashes and gate evidence are
saved in `authorization_and_gate_promotion.json`. The native component restore is not
an identical ray trajectory: patch sampling, initial dynamic estimate, and
unsaved dataloader cache/cursor are explicitly different.

### Actual first phase gate: regression

The one authorized worker restored parent8k and reached global10000, then
SIGSTOPped before10001 after evaluation, normal checkpoint save and CSV flush.
It spent2000 phase attempts and made1996 Adam updates. Four early native AMP
skips reduced scale131072 to8192; it remained stable afterward, with all
recorded scalar values and weights finite. No manual scaler change occurred.
Total campaign attempts spent through this gate are bounded by10610.

| Same native PNG target, all3 HD views | Parent8k | Phase10k + DSSIM.01 |
|---|---:|---:|
| PSNR |28.709667|26.1586|
| SSIM |.796946|.768428|
| LPIPS |.317703|.399952|

Actual matched600×330 floor crops use exactly equal GT arrays, identical
calibration/bounds and no resizing. HC crop PSNR falls31.7151→28.3437;
KA falls30.4240→28.5999. These crop scores are diagnostic and separate from
the all-view table. All3 640px predictions and both native crop comparisons
were manually inspected. Mottling remains; panel/seam and prop detail become
softer and floor response darkens. Recommendation: reject this phase and
retain parent8k; the clean-floor objective remains incomplete.

Evidence: stage `gate10k_matched_floor_contact.png`,
`gate10k_matched_floor_contact.json`, `gate10k_640/`,
`gate10k_manual_review_occupancy.json`, `gate10k_supervision_receipt.json` and
`parent_plus_phase_selection.json`. The parent remains the best eligible
checkpoint under the prescribed all-view PSNR/LPIPS rule. Root rejected the
phase and terminated worker58773 from the paused gate with SIGTERM followed
by SIGCONT; no update10001 ran. The controller also exited. Its final
selection write encountered a receipt ownership error, so it did not run a
new final evaluation; the existing parent evaluation remains authoritative.
No new model replaces the immutable accepted package.

### Smaller learning-rate follow-up

The parent checkpoint stores Adam learning rate .0083178292 at step8000;
the inherited scheduler has a200000-step horizon. Thus the patch phase
changed the sampling distribution while retaining a comparatively large
learning rate. This is a testable explanation for the regression, not a
proven diagnosis. A separate short phase is being prepared from the same
immutable8k parent with actual current learning rate .0005. Adam moments,
scaler and RNG will be restored; the scheduler's LR metadata must be scaled
consistently so its next step cannot restore the old learning rate. The
first required image gate is global9000, before update9001. Previous work
plus a possible new phase ending before19000 is bounded by21609 attempts.
This proposal does not claim that a smaller step removes the floor spots.

### Actual patch/order/SSIM audit after termination

The frozen actual eight-patch packet was audited on CPU with the exact DEV3
Torchmetrics1.5.2 implementation. Camera-constant row-major indices, HWC→NCHW
reshape and all RGB targets match independent preparedPNG/255 exactly.
There is no BGR swap, gamma/PQ transform or second intensity normalization.
The SSIM has explicit data_range1 and float32 Gaussian11/sigma1.5 moments;
it averages all121 spatial locations and three channels, then the eight
patches. An independent convolution formula agrees within3.16e−6 in score
and1.89e−5 relative gradient error. No layout/reduction bug was found.

There is a concrete objective effect: DSSIM.01's pixel-space gradient lowers
local contrast on all four vetted floor patches. On GA(1450,550), native MSE
instead requests greater contrast. Their luminance factors are already
.9990–.9997 while contrast/covariance factors are .844–.915, so the auxiliary
primarily penalizes structure mismatch. Gaussian11 on an11px patch uses
reflected neighbors in120 of121 windows; its gradient cosine against the
single valid center window is .47–.52 on these floor examples. This boundary
objective differs from full-image SSIM with real neighboring pixels, even
though the implementation matches its declared settings.

RGBA/background paths also match native MSE. Source images have three RGB
channels, background is black and no second opacity multiplication occurs.
HC opacity is essentially one in both renders. KA mean opacity changes
.999935→.998445, but conditionalRGB after dividing opacity remains nearly
as dark: mean red .38145→.36182 versus raw .38143→.36129. Therefore black
background exposure through opacity is a small contribution, not the main
observed color shift. DSSIM brightness derivatives are not uniformly darkening
and are small relative to MSE; no direct systematic brightness bug was found.

Evidence is in stage `patch_math_audit/{pixel_audit.json,crop_color_opacity.json}`;
no field or optimizer was loaded for the pixel audit. Pixel derivative signs
do not establish the direction of coupled Adam/field updates. Without a
matched patch-only control, objective, correlated patch sampling and the high
inherited LR cannot be separated as causes of the trained regression.

### Low-LR phase completed: selected9k, provisional guarded output

The separate .0005 phase completed with normal default plateau stopping at
eval10000; latest compact step10300 and no10500 trace bound its work below2500
attempts. The campaign ledger conservatively charges3000. Both processes
exited and final evaluation succeeded, selecting9k with native all-view
PSNR/SSIM/LPIPS **28.930609/.800138/.315825**. Only the initial AMP attempt
was skipped through the recorded2000 attempts; scale65536 stayed stable.

BODY128 on the fixed9k field has no free-prefix mass above10% and126/128 rays
with surface-band mass at least50%, including the independent8192/16384-sample
reference. The authoritative default grid still misses55/9022 known point
cells, including5/495 body cells with high point density. A fresh inference
candidate uses the actual9k core166359 cells plus37 observed seed cells,
followed by exactly one halo:853498 active cells and0/9022,0/495 known misses.
The8k mask was not reused. Hidden geometry remains unverified.

Actual prediction-only nativeHD files, raw buffers and source-bound manifest
are under `/fsx/tmp/marshawn_001660_hd/native_png_patch_dssim001_LR0005_STAGED/current_best_guarded_HD`.
Guarded all-view metrics are **28.874934/.800104/.316818**, improving the8k
guarded result by+.226775dB/+.003520/−.001903LPIPS. All3 actual HD predictions
and BODY12 crops were viewed; central bodies are coherent, while floor
mottling, wrinkle blur and thin rig/face/logo softness remain. This is a
provisional current-best artifact requiring root review, not clean-scene
acceptance. Original NN/EMA/mask hashes were restored after inference; older
stages and the accepted ZIP remain unchanged.
