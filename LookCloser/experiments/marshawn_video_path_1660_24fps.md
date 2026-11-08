# Marshawn 1660: independent-model video camera path and native GT continuation

## Current delivery: v4 with a paused native photograph

The user explicitly authorized holding source frame2020 for the last two
playback seconds and moving a native crop over the heads. The final120-frame
request therefore contains72 unchanged v3 neural frames, followed by48
photographic outputs from **the same source instant2020**. This is a paused
photograph pan, not motion from a future two-second continuation.

The tail keeps the exact MC pose and3840×2024 native pixel scale. A quintic pan
changes only integer crop origins, from `[3500,650]` through `[3710,606]` to
`[3920,563]`. There is no resize or pixel magnification. Both heads, hair and
full actors remain inside every crop; the minimum positive actor-envelope
margin is149 native pixels. Independently rendered start/mid/end crops have
**zero RGB-code difference on their overlapping native pixels**. The maximum
24fps step is17 horizontal and4 vertical native pixels. The continuous pan
has zero velocity and acceleration at its ends; integer quantization is explicit.

The campaign owner inspected and approved the
[actual paused-photo start/mid/end contact](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/endpoint_v3/MC2020_paused_tail_start_mid_end.jpg>).
[v4 CPU validation](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K_v4/cpu_validation.json>)
checks the constant2020 source,48 outputs, unchanged72 neural records, exact
t3 splice, native rays, bounded integer crops and future60fps120-frame paused
tail. The former47 missing future photographs are no longer required. No
synthetic actor motion, source-time interpolation or additional training occurs.

Approved v4 path SHA256:
`e76cea4b4499fd7ef38215a7abc8a4cad4912606cd652d92faaf4e79f843a41e`.
`video/export_pipeline_v4.json` and the stable plan alias select it. The durable
user-decision receipt is `video/path_4K_v4/user_paused_source_decision.json`.
The existing source-only v3 RAW transfer continues: every required neural
source and the available MC2020 photograph are unchanged. An explicit v4
inventory-equivalence receipt records this; control-plane suppression prevents
that transfer from publishing an old request. Final staging publishes v4.
The v3 selected-model IC gate remains valid as a diagnostic because its actual
initial camera, rays and source frame are identical.

## Neural trajectory v3, preserved by v4

The approved path now starts at TRAIN camera IC and follows IC→JC→KC→LC→MC.
It ends exactly at t=3.000 on MC with native1:1 crop
`[3500,650,3840,2024]`. This replaces the earlier DC→HC v2 delivery request;
v1/v2 remain diagnostics. No model, source calibration or training recipe changed.

The earlier HC crop cut off the grey actor's head. That missed framing error
was found by inspecting the actual2020 photograph. Nine camera photographs
were reviewed, and six cameras' trusted current native actor masks were mapped
through the exact RAW-to-native-rectified calibration. MC's full-actor union is
`[4069,854,6472,2432]` in8192×4320 pixels. The approved crop contains this union
with left/top/right/bottom margins569/204/868/242 pixels. Native RGB review
confirms both heads, hair and feet are included. The grey actor's face remains
physically turned into the mat; no camera or crop reveals unobserved features.

IC's actual1660 mask bbox is `[1037,231,1328,740]` at1920×1012. Initial zoom1.45
and actor recentering project it to4K bounds
`[1498.05,273.95,2341.95,1750.05]`. The campaign owner inspected the actual IC
photograph framing and the actual MC native crop and approved both.

Simple initial-to-endpoint intrinsics failed to follow the action toward the
mat. v3 therefore fits a natural cubic through seven current mask-support
center/size keys at half-second intervals, composed with the quintic time
clock. Initial IC intrinsics and exact MC endpoint intrinsics are preserved.
All seven projected support envelopes fit, with minimum148.24 pixels of margin;
interior vertical margins are at least200 pixels. These are conservative
multiview mask-support proxies, not measured hidden geometry or NeRF renders.

CPU checks passed: exact start and t3 endpoint, native-crop ray equivalence,
C2 pose/intrinsic endpoints, positive focal lengths over10001 samples,72+48
timing, TRAIN-only controls, camera centers outside the accepted actor box,
and future60fps accounting180/common36/additional144. The production camera
poses already include the historical rigid floor transform; the framing audit
applies it exactly once.

- [Approved IC/JC source-framing comparison](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/endpoint_v3/IC_JC_start_framing.jpg>)
- [Actual MC and shifted-HC crop comparison](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/endpoint_v3/MC_vs_shifted_HC_native_crops.jpg>)
- [Full actual MC native crop](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/endpoint_v3/MC2020_native_crop_3500_650.png>)
- [Seven-time support-proxy framing](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/endpoint_v3/seven_time_support_framing.jpg>)
- [CPU validation](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K_v3/cpu_validation.json>)

Approved v3 path SHA256:
`265a467879316d4742ba8654804ca32c79cf31411b8fcafacfa3732fa3954351`.
`video/export_pipeline_v3.json` and the stable export-plan alias now select it.
The old RAW prefetch process group was retired after quiescence checks; a
versioned source-only prefetch stages IC's first16 cameras before continuing
the72-frame native inventory at4MiB/s. It cannot publish a cached old path.
The selected-model watcher was likewise replaced before any worker started;
it will first prove canonical HC checkpoint parity, then render actual new4K
IC LookCloser and hybrid rays. This actual neural visual gate and all final
video-frame reviews remain required. The47 missing continuation photographs
are still missing; no duplicate or synthetic GT was authorized.

## Earlier v1/v2 preparation: what was tested

CPU planning and renderer preparation for 72 cold-trained models, original frames
1660–2015 at source step5/24fps. Each model supplies one neural output frame;
the 3-second boundary is **frame2020 at t=3.000**, not neural frame2015 at t=71/24.
The approved endpoint is TRAIN camera HC. All66 views are now TRAIN views.

The path uses TRAIN controls DC→EC→FC→GC→HC, a Bezier interpolation of their
centers (always inside the control centers' convex hull), RotationSpline
orientation and a quintic time warp. The approved v2 uses quintic intrinsics
from an initial1.45zoom centered on the actual actor labels to the unchanged
native HC crop at t=3.000. Velocity and
acceleration approach zero at the endpoints. A continuous correction preserves
the original calibration's approximately1e-7 rotation-matrix roundoff, so the
endpoint remains exactly the stored TRAIN matrix.

Approved delivery is3840×2024. During the neural segment, focal length and
principal point smoothly approach a native HC crop. The following48 photographs
keep a fixed3840×2024 raster and permit only integer crop translations. This
provides native face detail without resizing photographs. The source is first
corrected at native8192×4320 using the fixed lens/photometry calibration; the
optical remapping itself uses linear-light interpolation. It is explicitly not
claimed to preserve the bytes of the original distorted TIFF.

## Results

| Check | Result |
|---|---|
| Timeline |120 frames:72 neural +48 actual GT; exactly5 seconds at24fps |
| Neural samples |1660,1665,…,2015; last neural time2.958333…s |
| First actual GT |2020 at exactly3.000s |
| Exact TRAIN endpoint pose and native-crop intrinsics |CPU checks passed |
| Native crop/output pinhole-ray equivalence |CPU checks passed at corners and center |
| Pose and intrinsics endpoint smoothness |CPU checks passed |
| Future60fps indices |180 neural models1660,1662,…,2018, retaining original IDs |
| Native photometry replay |32×32 HC1660 patch vs existing native target: max1/255, mean0.0667/255 code error |
| GT availability |2020 exists;2025–2255 absent in the two supplied RAW roots |
| Neural/hybrid GPU validation |Pending model completion; no GPU launched by this preparation |

The later uniform60fps neural timeline contains180 original source IDs at
stride2,1660–2018. Its intersection with the current72 stride5 IDs contains
36 models (stride10). Therefore upgrading this interval to60fps needs
**144 additional independently trained models**, not108. The36 common frame
models can be reused exactly; the36 odd-indexed current24fps models remain
preserved but are not samples of the uniform60fps timeline. No existing model
needs retraining, no neighboring weights are transferred, and no additional
training jobs are scheduled by this accounting change. The current24fps run
is unchanged.

The1-code photometry replay difference is consistent with direct floating-point
native lens interpolation versus the earlier OpenCV interpolation-table rounding.
Both decode Log3G10/RWG native samples before linear-light interpolation and apply
the same fixed gain4, k4 radial correction, TRAIN scalar EV and Reinhard+sRGB.

Visual review:

- [All66 source-camera contact](</fsx/tmp/marshawn_video_path_20261107/train_cameras_first_frame.jpg>)
- [Actual2020 DC/HC/KA/MD candidates](</fsx/tmp/marshawn_video_path_20261107/endpoint2020_candidates.jpg>)
- [HC2020 crop comparison](</fsx/tmp/marshawn_video_path_20261107/HC_2020_native_crop_review.jpg>): red is native3840×2024 crop `[1000,650,3840,2024]`; cyan is a native1920×1012 face crop.
- [Rendered actual2020 native crop](</fsx/tmp/marshawn_video_path_20261107/GT2020_native_crop_preview.jpg>) is a diagnostic thumbnail; the archived full3840×2024 PNG is at campaign `video/diagnostics/GT2020_v1_cpu_preview/lookcloser.png`. The grey actor's face is physically facing down/occluded in the inspected HC/DC/KA views. The crop cannot reveal an unobserved face.
- [Six native actor closeups](</fsx/tmp/marshawn_video_path_20261107/six_native_head_crops2020.jpg>) additionally inspect FC/MC/NA. [Native512-pixel head crops](</fsx/tmp/marshawn_video_path_20261107/grey_head_native512_2020.jpg>) show that MC/NA expose an ear/back of the head, rather than readable eyes/nose/mouth. [Three further low-camera head crops](</fsx/tmp/marshawn_video_path_20261107/low_native_head_crops2020.jpg>) inspect MD/ND/OD and confirm the physical face-down occlusion. HC remains the proposed endpoint because the black actor's face is clearer there; no claim of two visible faces is made.

The actual third second contains a major action change: the grey actor is lying
on the thick mat. The earlier2-second AABB that excludes this mat cannot be
assumed valid for the third second. This was reported to the campaign owner
before authorizing full training; temporal AABB preparation is a separate gate.

## Implementation and durable execution contract

- `scripts/marshawn_video_path.py` builds a hash-bound120-frame path and explicit
  missing-GT inventory. Source IDs are never renumbered.
- `scripts/marshawn_video_path_cpu_test.py` checks timing, original indices,
  TRAIN endpoint, smoothness and exact native-crop ray equivalence.
- `scripts/marshawn_video_render.py --spec SPEC --path PATH --index N` renders a
  single resumable unit; `--encode` requires all120 hash-verified units and
  produces both deliveries. PNG inputs use exact24fps image2 timing.
- `scripts/marshawn_video_render_controller.py --spec SPEC --path PATH --gpus 0 …`
  is the detached post-training dispatcher. It waits for all72 qualified cold
  checkpoints, uses only idle local GPUs, logs worker/GPU/OOM evidence hourly,
  retries failed units up to three times and waits explicitly for missing actual
  GT. Exclusive campaign/unit locks protect restart and surviving pre-crash
  workers. It encodes both movies after120 completed units, then leaves a visual
  review gate. The controller was prepared, not launched during this CPU work.
- The GPU dispatcher shares fleet locks
  `/tmp/lookcloser_tuning_gpu_UID_INDEX.lock`, holds them across the entire
  worker lifetime and passes the lock file descriptor to the child. It sets
  `CUDA_VISIBLE_DEVICES` to the physical UUID and verifies that UUID with Torch.
  A crashed dispatcher therefore does not expose its surviving worker's lease.
- `scripts/marshawn_video_render_stage.py --plan EXPORT_PIPELINE_JSON` stages
  same-instant native TIFFs and the frozen calibration/color/hybrid references
  from olive to silver, renders on silver beside its original models/configs,
  and gathers only output PNGs, receipts and movies back to olive with image
  hash verification. It never rewrites a serialized model configuration or
  relocates a checkpoint. The native inputs are about16 TRAIN cameras per
  neural frame; substantial RAW transfer is required before hybrid rendering.
- When72 neural frames finish but actual GT remains unavailable, the dispatcher
  exports both3-second neural movies and continues waiting for the48 real
  continuation photographs. This is separate from a completed5-second delivery.
- Neural rendering calls campaign `marshawn_video_train.load_pipeline` for that
  frame's selected actor-LPIPS checkpoint. Model, source geometry and native
  textures are all from the same original frame. Geometry caches bind the exact
  camera and checkpoint hash; model weights or occupancy grids never transfer
  between video frames.
- Hybrid rendering reuses the measured Misha native-pixel transfer algorithm:
  `/fsx/lookcloser_misha50_20261101_pavlo/code/campaign/misha50_hybrid.py`, with
  its `hybrid_reference` helpers. The Misha-specific PQ decoder is replaced by
  Marshawn's frozen Log3G10/RWG decoder. Default detail sigma8, gain1,16 nearest
  TRAIN sources; source-camera visibility, conditional q10/q50/q90 depth tests,
  angular support and unchanged neural fallback remain active.
- Hybrid source images are native TIFF taps, not enlarged HD training pictures.
  Photometry fit, historical hybrid code, frame checkpoint, source transforms,
  native source files and output images are hash-bound in receipts.
- Pixel-transfer source eligibility is explicitly `W64>0 AND valid AND NOT
  source_ineligible AND NOT unknown AND calibrated_RGB_valid`. A camera-border
  valid mask alone is insufficient because crew can occlude a reconstructed
  actor. Missing crew/UNKNOWN maps fail closed. The eligibility bitmap and
  semantic source are hashed, with rejected pixels retaining neural fallback.
  No extra hard erosion of hair is introduced. The CPU test proves crew,
  UNKNOWN and invalid RGB pixels are vetoed, and missing veto maps are rejected.
- Actual GT output after3s is identical in both variants. Missing source frames
  raise an error; they are not silently duplicated or synthesized.

Campaign path:
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K_v2/path.json`.
The original `video/path_4K/path.json` is retained as a v1 diagnostic.
Render configuration:
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/render_spec.json`.
Concrete transport/CLI/after-training plan:
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/export_pipeline.json`.
It targets `ubuntu@silver-sable` at
`/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107`
and the already qualified silver Python wrapper. Its `after_training_command`
must be attached to the production training plan so export survives a Codex
session ending. Native TIFF staging, remote GPU rendering and gather have not
been launched during this CPU preparation.
These are prepared inputs, not a claim that the movies or GPU visual gate have
completed. The final source transforms and common AABB must remain tied to the
qualified campaign data before render dispatch.

An actual GPU export gate has now passed for both LookCloser and hybrid using
the same-frame intermediate checkpoint30376. Native/packed HD camera parity,
full pipeline/checkpoint immutability and gathered PNG hashes passed. The
campaign owner inspected both v2 exports and approved their framing. This is a
technical export check, not a quality-leader claim or completion of video
training. Details are in `marshawn_video_initial_export_gate_20261107.md`.
The existing physical occlusion of the grey actor's face remains.

## Insights and next steps

Choose4K for a native1:1 photographic crop that can include the two actors'
vertical head range at2020. An HD1:1 crop is useful for an individual face and
can pan between faces, but cannot contain this entire vertical head range.
Complete the zoom during the neural segment: changing zoom continuously during
fixed-raster GT would require resizing source pixels, contradicting the user's
crop/shift constraint.

Obtain actual2025–2255 photographs for a dynamic two-second continuation; the
campaign owner has asked the user. After models complete, inspect path start,
intermediate poses, final neural view, exact t=3 transition and both full videos.
Approve camera motion and hybrid visibility from these renders before delivery.

The matched first-three-frame 4K profile comparison is complete under
`C/operations/video_fleet10/private_profile_v1/matched_first3`. The full-frame,
native head and native foot contacts were inspected. Both fighters and their
hair/feet remain framed. The candidate (`sigma4`, gain `.75`, smooth angular
pixel scale and the eligible own-grid weak-ray guard) shows subtle detail
changes; existing floor fragments and silhouette shells remain in both
profiles and vary between independently trained frames. The six image hashes
and own checkpoint bindings are in `finite_profile_matrix_complete.json`.
This finite comparison does not automatically promote a profile or establish
full-video temporal consistency.

The first1660 baseline was also rebuilt by moving only bound floating-point
geometry from the model's native SG host to olive, then applying the existing
hybrid code on CPU against the original FSX TIFFs. Both LookCloser and hybrid
4K PNGs were pixel-identical to the archived SG render (maximum code difference
zero). This supports replacing bulk RAW copying with geometry-only transport;
checkpoints, YAMLs, image resolution and calibrated photometry stay unchanged.

A separate finite CPU anchor test used the existing IC-winning labels and
confidence at the exact IC starting pose. Two direct blends (`.75` and `1`)
changed only461056 already eligible IC pixels; all other donors and fallback
pixels remained bitwise unchanged. The detail replay matched the existing
candidate PNG exactly. Native face comparisons show modest tone/contrast
changes, without a dramatic new detail gain. These remain private tests;
no target-GT metric or default profile change was made.

ROOT subsequently approved the **parameters-only** production profile:
`sigma4`, gain `.75`,16 sources and the smooth IC/JC/KC/LC/MC focal reference.
Geometry remains the native saved R0; the weak-grid guard and IC direct-pose
blend are disabled. The direct-pose branch showed no clear additional benefit
and was stopped. The third private comparison's145-cell guard is not part of
production. Full72-frame temporal inspection is still required.

The final parameter CPU worker reproduced the approved first1660 candidate
at4K with zero differing RGB codes for both LookCloser and hybrid. The baseline
sigma8 reconstruction had likewise matched the original native-SG render
pixel-for-pixel. Sources are archived in
[`scripts/marshawn_video_export_geometry_cpu`](../scripts/marshawn_video_export_geometry_cpu/README.md).
The final immutable profile SHA is
`2bd9bca54f43bb5da2920a5f95124677ee2754762bb8ad5551163d9dbf06eb81`;
ROOT release SHA is
`cd88438ff3cf16a3b7505e6065f62fb8a713f26e3f096f8fb53e179cb064acf7`.

The fleet's final postprocess hook SHA is
`0a98c1cc132aba3a1e7140eb5bd46956ebde24f79bae954705115b6f94c0f992`.
It runs the external geometry/CPU router after all72 capped models complete;
training plan SHA remains `74c92390646504254133402b1f8f6f6649364b0770e167269bb5e99541397050`.
Models/checkpoints/YAMLs stay on their training hosts. Only bound float geometry
is gathered to olive, where original FSX TIFFs supply native texture. This
removes the bulk RAW-copy dependency. The optional copy controller and its
exact three rsync/SSH children were retired;601 completed cached TIFFs were
SHA-identical to originals, two partials were retained, and no data was deleted.
The byte inventory SHA is
`127a78325e866d7ceab40ab8cd36eea87da5cee756696bbaf95a3f4f4c36afb1`.

Final output directory is
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/video/geometry_cpu_delivery_params_v2`.
The movies are not yet complete. They will contain72 neural frames followed
by48 native integer crops of paused source2020, with no GT image resize.

A subsequent OPS-only crash audit fixed two launch windows: the route now
reloads the authoritative `state.json` written before SSH dispatch, and CPU
renders have a prelaunch intent/nonce plus an inherited execution lease. The
child writes PID/start/boot before executing the unchanged shader; controller
restarts adopt that same owner. Busy locks do not consume shader retry budget.
Four injected coordinator crashes, actual inherited locks, capacity limits,
cross-source rejection and a frozen-fleet ACK crash with a CPU receiver passed.
A real completed SG1660 geometry artifact passed the orphan-adoption proof;
wrong camera/frame requests were rejected. No GPU/model experiment was needed.

The actual frozen GT CLI check found a missing metadata key:
`one.py --cpu` also requires `source_visibility_adapter_sha256`. The outer
spec adapter now binds the unchanged57734bb4… helper. Old qualification specs,
receipts and PNG hardlinks were preserved; shader51239b1e…, GPU6603aa06… and
profile2bd9bca5… remain byte-identical. In a separate new fixture, actual4K
GT72/119 ran without a2020 model/DATA/geometry, shared native pixels matched
exactly, and the neural metadata roundtrip matched the approved PNGs exactly.

Final OPS release SHA:
`fb5b920433f6f59a1e2ed16bd6b8ee79435060ec38c401e8b2ca9e734e529c3e`.
The final fleet posthook SHA is now
`390ef99d70780ff6f3a39ed18bd72fc2edf792c47c204c29fdae82a75df5fe4d`;
the earlier parameter-only hook remains archived. The training plan and all
model/source data are unchanged. Qualified sources and CPU tests are in
[`durability`](../scripts/marshawn_video_export_geometry_cpu/durability/README.md).

A read-only storage check measured the completed1660 geometry package at92.77MB
logical (98.65MB allocated), giving6.68GB for72 equal packages. The conservative
uncompressed shape bound is79.13GB for all72, including1% metadata allowance;
compression varies with scene complexity. No eviction/deletion was introduced.
The optional CPU background-support study retained37,817/37,821 original R0
cells (99.9894%) after all66 positive-W64 ray footprints and one clipped halo.
It stopped without GPU work or production changes. Tiny original tail mass
cannot bound changed RGB under the legacy ARM allocator because interval
pruning changes sample allocation. Details and source-bound receipts are in
[`marshawn_export_background_support_feasibility.md`](marshawn_export_background_support_feasibility.md).

## Current launcher and actual six-frame moving preview

Use the live [production launcher manifest](/fsx/tmp/marshawn_fleet10_140k_20261107/postprocess_command.json)
for the current entry rather than a historical hook SHA in the sections above.
The ROOT-approved DEV3 UUID compatibility wrapper V5 is active at SHA
`4ca257d6430314085fda414a9e75f605680ee094e8f69e96ee6513aab9bebd71`.
It proves the actual Torch CUDA-context UUID and inherited supervisor lease,
then calls the unchanged GPU shader. Native HC parity≤1, full model tensor
identity and16-donor geometry passed on the real finished1680 model. CPUshader,
profile2bd9 and encoderV4 are unchanged; no final movie was started early.

Both actual first-six moving-path4K previews are ready in
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/video_partial_first6_v1/`.
Each has6 frames at24fps (0.25s), accompanied by full-frame contacts and native
face/foot crops. Full actors remain framed; existing floor/silhouette fragments
vary across the independent models. See the [qualification and preview report](marshawn_dev3_export_compatibility.md)
for exact artifacts, operational failure/fix evidence and scope limits. The
complete72-frame temporal review remains required.
