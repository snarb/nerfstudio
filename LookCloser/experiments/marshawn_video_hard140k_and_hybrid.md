# Marshawn video: fixed 140,000 iterations and native hybrid tuning

## What was tested

The user replaced unbounded plateau stopping with a hard limit of 140,000
iterations per independently trained frame. The new campaign is
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107`; the active ten-GPU fleet
is `/fsx/tmp/marshawn_fleet10_140k_20261107`. The original uncapped campaign is
retired for production. All 66 cameras train, so reported camera metrics are
in-sample.

`scripts/freeze_marshawn_video_140k.py` changes termination and iteration
accounting only: valid steps are 0 through 139999, the terminal evaluation and
checkpoint are forced, and another optimizer iteration is refused before its
gradient. The selected recipe, optimizer, 200000-step LR decay horizon, ARM,
FR transition, actor sampling, frequency/grid updates and source photometry
remain unchanged. The sealed campaign contains 1531 files; its seal SHA256 is
`d873cd61ad5c00f9501f2f6c477cb42f962d4d15591f5785830e0aeae381ec14`.

Frames 1660 and 1665 restarted cold. Frame 1670 resumed its own checkpoint
122000 after a metadata-only migration. All serialized tensor state, Adam,
occupancy, frequency/EMA, scheduler/scaler and RNG state were compared before
and after serialization. No state came from another frame. Qualification
receipts are under the new campaign's `operations/hard_budget_cpu_qualification.json`
and `operations/same_frame_resume/001670.json`.

## Results

Frame 1670 completed exactly 140000 iterations; its final and selected
checkpoint is step 139999, SHA256
`ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015`.

| Frame | Completed iterations | Selected step | PSNR | SSIM | Actor LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1670 | 140000 | 139999 | 46.12168 | 0.991924 | 0.00094574 |

These values come from the selected-checkpoint receipt and are in-sample
monitoring, not evidence of novel-view accuracy.

The user also authorized using olive-lark's GPU for hybrid-rendering tuning
while the other machines continue the video campaign. The first experiment
uses this genuine capped model, calibrated native RAW texture, and a target
camera excluded from all transfer donors. Target RGB is used only for
measurement. Source geometry and the neural model are frozen.

`scripts/tune_marshawn_hybrid_cache.py` replays qualified donor warps to compare
16 small variants: detail sigma 4/8/12, gain .75/1/1.25, 8/16/24 donors, an
optional confidence reduction near ambiguous donor-score ties, and direct /
multiband transfer controls. It retains
single-source texture selection and exact neural fallback. It never deletes
actor support or transfers colors from a different instant. Native crop
coordinates retain their parent target's pixel scale rather than deriving it
from the crop width.

The replay's baseline RGB, source labels and transfer weights matched the
historical algorithm bit-for-bit in six checks spanning pixel scales 1, 2 and
4.26875 and one/three donors. Receipt:
`operations/hybrid_tuning/cached_replay_qualification.json`. This is algorithm
qualification only; real-image quality selection and temporal validation are
still pending.

The actual capped 1670 checkpoint was copied unchanged for a private local
render-only harness. Five data/checkpoint path bindings changed in memory;
the original YAML and checkpoint remained untouched. Loaded pipeline tensor
state matched exactly. Its automatic FR was .3. Native HC output differed
from the selected archived image by at most one 8-bit code, affecting
0.000001887 of channel entries. Setting the bookkeeping step explicitly
produced a maximum float difference of 0.000003159; that extra override was
discarded, preserving original loader behavior. Inference uses adaptive
marching independently of the training warmup condition.

Eight native 768-square contexts were cached with 384-square metric cores:
HC black face, shirt and grey face at pixel scales 2 and 4.2667, and IC black
and grey faces at scale 4.2667. The first scale follows delivery width; the
second follows actual native focal length. Geometry is shared and every
target excludes its own camera from all 24 donors. Actor metric weights are
the current black-or-grey semantic mask looked up at native coordinates;
this HD-derived metric support is not a newly inferred native mask. RAW RGB
is sampled at native resolution without an HD intermediate. Model tensor
state and checkpoint SHA were unchanged after all geometry work. Receipts:
`operations/hybrid_tuning/native_HC_parity.json`, `manifest.json`, and
`geometry_complete.json`. The GPU lease was released after geometry; the
remaining sweep runs on CPU while local GPU work resumes.

Initial one-patch results (HC black face, pixel scale 2; provisional):

| Variant | Actor PSNR | Actor SSIM | Actor spatial LPIPS |
| --- | ---: | ---: | ---: |
| LookCloser | 27.92344 | 0.571092 | 0.551203 |
| Existing detail sigma8 / gain1 / 16 donors | 27.79847 | 0.565546 | 0.508786 |
| Detail sigma4 / gain.75 / 16 donors | 27.87412 | 0.568930 | 0.506886 |

The hybrid improves this patch's LPIPS by about 8% relative to LookCloser;
the tuning improvement over the existing hybrid is only about 0.37%.
These are native close-up patch metrics, not the much smaller in-sample HD
whole-actor monitoring number above. PSNR and SSIM do not improve. Root
visually inspected the actual GT / neural / baseline hybrid / candidate
384-square comparison; detail gains are subtle and the native photograph
is still clearer. No final recipe change is justified by one patch.

On this actual native cache the replay's baseline PNG matched the historical
24-donor hybrid byte-for-byte; see `real_native_cached_replay_parity.json`.
Full eight-patch selection, visual review and temporal checks remain pending.

## Insights

The iteration limit now has direct pre-gradient enforcement and terminal
accounting. Retaining the previous LR schedule preserves the chosen recipe
under the shorter user-selected duration.

Hybrid tuning must compare actual native patches with an independent donor
set. A camera that supplies its own photograph would make a transfer score
misleading. Numerical candidates require visual checks of faces, hair,
clothing and seams, followed by adjacent-camera/adjacent-frame review before
changing the final video renderer.
