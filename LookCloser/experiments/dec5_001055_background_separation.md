# DEC5 001055: separate foreground density and ray-space background

## What was tested

Separate high-level track within the existing 16 GPU-hour campaign, prompted by
persistent background-colored volumetric clouds in both HD and native-RGB
LookCloser fields. The hypothesis is that a separate background color function
can stop the bounded foreground field from explaining the backdrop through
spurious density. This is a hypothesis about the observed failure, not proof
that every artifact has this cause.

A small 3-hidden-layer, width-64 ray-space MLP predicts background display RGB
from raw-world camera origin and direction. Origin/direction positional encodings
have 1/5 levels. It fits only 59 train views, using Sapiens background probability
>.98 eroded by 16 HD pixels. There are 4,096 fitting and 256 withheld monitor
pixels per train camera. The fit uses 5,000 fixed Adam steps; no held-out camera
RGB selects its checkpoint. This is a background color function, not a depth
surface, anatomical model, diffusion painter or foreground reconstruction.

The component gate is >=5% background LPIPS improvement and higher PSNR on the
same excluded-camera background ROI. Only after that gate passes is one warm
foreground/background training screen allowed. Final acceptance still needs
better **hair** metrics/stability with face and contour guards.

## Results

| Excluded-camera BG inside fixed head ROI | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Existing native 20k field RGB | 16.17968 | .77506 | .195391 |
| Frozen ray-space background | 18.70261 | .87620 | .165945 |

The component improves PSNR **2.523 dB** and LPIPS **15.1%**; it passes its
component gate. These are background measurements, not improved hair depth.
The fit takes 4.3 seconds of its measured optimization loop; the conservative
supervisor interval is 30.1 seconds. Training-camera withheld-pixel PSNR is
25.94 dB; this monitor is separate from the three excluded-camera results.
[Component metrics](assets/dec5_001055_background/ray_background_component_metrics.json).

The checkpoint's NumPy center was re-exported as a builtin list for older-host
compatibility. All network tensors match exactly and dev3 CPU loading passes;
the original artifact and conversion receipt remain archived on FSx.

### Foreground training screen (completed)

Warm-start from native LookCloser step 19,999, with fresh optimizer at 0.1 base
LR, mature 60% hair / 30% other-subject quotas, no repeated traversal warmup,
and standard full FR/FAS/ARM. Early sparse support is restarted with the fresh
5k trial. The original initialization RGB function is re-evaluated before
selection; incompatible parent metrics are excluded.

Composition is `C_fg + (1-A_fg)*C_background`, with the background network frozen.
An empty-space optical-thickness term of weight .05 applies only to the same
conservative train BG mask. It uses existing packed RGB-ray density samples,
adding **no foreground density queries**. Fine boundary/uncertain pixels receive
no hard alpha labels. No validation/test pixels enter these objectives.

The field attachment must preserve the initial foreground fingerprint. The RGB
checkpoint selection includes the initial attached function and the 1k/3k/5k
checkpoints, using the existing PSNR / .07-dB LPIPS tie-break rule. Background
weights are an immutable separately hashed dependency of foreground checkpoints.

### Frozen common RGB-transfer comparison

All arms use the same native train RGB, nearest-eight visibility policy and
frozen ray-background fallback. Three excluded cameras, UHD head ROI only.
The 5k foreground checkpoint is an **auxiliary geometry candidate**; native RGB
selection retained the initial attached function by the PSNR rule.

| Geometry | Hair PSNR | Hair SSIM | Hair LPIPS | Face LPIPS | Holes | Halo | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HD LookCloser 20k | 21.05682 | .38830 | .198844 | .282691 | .010975 | .001541 | .920120 |
| Separate FG/BG, 5k | 22.72410 | .39584 | .188515 | .280881 | .003546 | .002405 | .983426 |
| FG/BG only in fixed hair domain | 21.87619 | .39429 | .192012 | .282685 | .013358 | .002332 | .964030 |

Full FG/BG improves hair LPIPS **5.2%** and holes **67.7%**, with slightly better
face LPIPS. Halo increases **56%**. The rendered D-camera preview confirms better
hair completion but residual backdrop fragments around the crown. This passes
an initial hair-quality screen, **not final acceptance**. Routing only inside the
fixed hair domain fails the material-gain gate and increases holes.
[Full FG metrics](assets/dec5_001055_background/fg5k_raybg_metrics.json).

A uniform 2,048-sample occupancy-independent head-compartment diagnostic uses
1,536 excluded-camera semantic hair rays and 1,536 background rays, with a box
defined only by train SfM/semantics. Mean background opacity decreases from
.63158 to .15524 (75.4%); hair rays above opacity .1 remain 100%, but hair rays
above .5 decrease from 99.87% to 98.63%. These are semantic proxies, not true
alpha or independent 3D measurements.

A further 2k continuation substitutes observed train background RGB **only in
the known-background training objective**, avoiding conflict with the approximate
background color function. Its frozen diagnostic does not improve separation:
background mean .15848 versus .15524, hair opacity coverage unchanged. Close
this continuation rather than spending more compute on its local tuning.
The first attempt failed before its first update due to a CPU/GPU image mismatch;
the corrected run completed. The diagnostic rerun completed in 60.2 seconds,
without OOM. All checkpoints remain archived.

## Insights

Higher input resolution alone failed the equal-step head test; the independent
background component now justifies a different representation test. The measured full FG/BG gain now justifies a frozen camera-path pilot. Increased
halo remains an explicit rejection risk; do not promote from static metrics alone.
The known-background continuation failed its diagnostic and is closed.
