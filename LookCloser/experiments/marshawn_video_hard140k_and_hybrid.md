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
The full eight-patch sweep subsequently completed (16 variants each).

| Common variant, mean over eight patch/scale records | PSNR | SSIM | Spatial LPIPS |
| --- | ---: | ---: | ---: |
| LookCloser | 28.97795 | 0.625870 | 0.471636 |
| Existing detail sigma8 / gain1 / 16 donors | 28.55423 | 0.606737 | 0.416033 |
| Detail sigma4 / gain.75 / 16 donors | 28.77568 | 0.616199 | 0.410062 |

The common candidate improves LPIPS by 1.44% over the existing hybrid and
13.05% over LookCloser, while PSNR/SSIM remain worse than LookCloser. These
are equally weighted close-up records; three HC patches appear twice for
the paired pixel-scale test. They are not eight independent cameras or
whole-body metrics. Clothing drives much of the candidate's gain; several
individual face patches prefer gain1 or multiband transfer. Increasing the
donor count beyond eight did not change the measured baseline patches.
Keep 16 donors until full-image/novel-view checks establish a safe reduction.
Focal-based scale admits more transfer and improves the three matched HC
patches, but requires full-image/temporal review before production use.

`scripts/refine_marshawn_hybrid_warps.py` additionally tested bounded
Farneback image alignment using only frozen neural RGB as its reference.
Original admission/score/support remain exact, excessive or unhelpful shifts
retain the original donor, and GT never enters prediction. Synthetic shift,
identity, empty-support and maximum-displacement checks passed. Two search
radii, .75 and 1.5 reference-HD pixels, were evaluated on all eight contexts.
Their best common LPIPS values were 0.410077 and 0.410074, versus 0.410062
without alignment. This alignment experiment did not improve the objective
and is excluded from the production candidate. It neither changes geometry
nor trains the NeRF.

### Smaller detail radii, 2026-11-08

Two further finite cached sweeps tested 16 and nine configurations, including
controls, on the same eight bound patch/scale contexts. No NeRF or new
geometry was trained. The local GPU ran spatial LPIPS; target photographs
remained measurement-only and every target excluded itself from its donors.

| Same-GPU control/candidate, eight-context mean | PSNR | SSIM | Spatial LPIPS |
| --- | ---: | ---: | ---: |
| Selected sigma4 / gain.75 / 16 donors | 28.77568 | 0.616199 | 0.410071 |
| Sigma2 / gain.75 / 16 donors | 28.79278 | 0.616520 | 0.408689 |
| Sigma1.5 / gain.75 / 16 donors | 28.80739 | 0.616916 | 0.408492 |

Sigma1.5 improves the mean LPIPS by 0.385% against the same-GPU control.
Both shirt contexts and all four face contexts at native focal scale improve;
the two face contexts at pixel scale2 worsen by 0.65–1.15%. The eight rows
still represent one model, two train cameras, and repeated physical regions,
not eight independent views. Root inspected native face/shirt crops: visual
differences are small and the photographs retain substantially finer detail.
Production remains sigma4 pending matched novel-view and temporal review.

Full receipts and PNGs are under the capped campaign's
`operations/ROOT_hybrid_refine_20261108` and
`operations/ROOT_hybrid_refine_low_sigma_20261108`.
A separate private CPU render of canonical records0,2,5 uses the unchanged
qualified shader, existing R0 float geometry, original RAW/donors, and a
sigma1.5-only profile. Its native output review is pending under
`operations/ROOT_hybrid_sigma1p5_novel_first3_20261108`; no production profile
or geometry receipt was overwritten.

Root inspected the native three-region HC comparison. Detail changes are
subtle; the real photographs remain clearer. Full 4K V4 camera record2 was
also rendered from the completed 1670 model with unchanged weights and
native rays. Both actors fit the framing, but floor fragments and a shell
around the hair remain visible in the frozen neural result. Hybrid transfer
alone is not a demonstrated remedy for that geometry. No temporal quality
claim or final renderer promotion has been made.

## Insights

### Production decision, 2026-11-07 22:48 UTC

The provisional selection above has now passed a matched full-resolution
review on the independently trained 1660, 1665 and 1670 models. The selected
export uses 16 donors, detail sigma 4, gain .75 and the smooth fixed-anchor
reference-focal scale along IC–JC–KC–LC–MC. Geometry remains each model's
native saved R0. The experimental weak-grid guard and IC direct-pose fade
are excluded. This changes rendering parameters only; training stays HD
1920×1012, and delivery rendering is 3840×2024.

Root visually inspected the full-frame contact and native face/foot crops:
both actors and their feet remain visible, with subtle texture changes.
Floor fragments and the silhouette shell persist and vary between cold
models; the profile is not a demonstrated geometry or temporal-flicker fix.
The matched comparison is recorded in
`operations/video_fleet10/private_profile_v1/matched_first3/finite_profile_matrix_complete.json`
under the capped campaign.

The production profile SHA256 is
`2bd9bca54f43bb5da2920a5f95124677ee2754762bb8ad5551163d9dbf06eb81`.
The current launcher is the [production manifest](/fsx/tmp/marshawn_fleet10_140k_20261107/postprocess_command.json).
At the ROOT-approved DEV3 compatibility update V5 its SHA256 is
`4ca257d6430314085fda414a9e75f605680ee094e8f69e96ee6513aab9bebd71`.
V5 proves the actual Torch CUDA-context UUID on older DEV3 Torch before invoking
the unchanged GPU shader. The explicit-color encoder V4 and rendering profile
remain unchanged; the previous launcher is archived.
Its durable wrapper preserves source/model bindings and recovers GPU and
CPU launch intents after coordinator failure. Actual first-frame neural
and hybrid outputs match the qualified profile pixel-for-pixel. The frozen
2020 GT endpoints at output records 72 and 119 also passed an actual native
crop/shift check. See the
[video export report](marshawn_video_path_1660_24fps.md) for transport,
failure-injection checks and final-video requirements.

An intermediate download is available at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/review/Marshawn1660_hybrid_tuned_140k_4K.png`.
This is one 140,000-iteration HD-trained model rendered at native 4K,
not the completed 72-model video. Full-sequence temporal inspection remains
required after training and assembly.

A read-only sequence check subsequently inspected all 4686 canonical source
receipts across 71 prepared frames and 66 cameras. Every camera retained
the same fixed EV, photometry-fit hash and calibration hash throughout the
sequence; no frame-dependent exposure fit was found. Receipt:
`operations/root_sequence_photometry_consistency.json` under the source
campaign. This verifies configuration consistency, not absence of temporal
artifacts in the independently learned fields.

The iteration limit now has direct pre-gradient enforcement and terminal
accounting. Retaining the previous LR schedule preserves the chosen recipe
under the shorter user-selected duration.

Hybrid tuning must compare actual native patches with an independent donor
set. A camera that supplies its own photograph would make a transfer score
misleading. Numerical candidates require visual checks of faces, hair,
clothing and seams, followed by adjacent-camera/adjacent-frame review before
changing the final video renderer.
