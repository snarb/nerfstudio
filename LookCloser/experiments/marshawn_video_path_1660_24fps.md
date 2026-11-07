# Marshawn 1660: independent-model video camera path and native GT continuation

## What was tested

CPU planning and renderer preparation for 72 cold-trained models, original frames
1660–2015 at source step5/24fps. Each model supplies one neural output frame;
the 3-second boundary is **frame2020 at t=3.000**, not neural frame2015 at t=71/24.
The proposed endpoint is TRAIN camera HC. All66 views are now TRAIN views.

The path uses TRAIN controls DC→EC→FC→GC→HC, a Bezier interpolation of their
centers (always inside the control centers' convex hull), RotationSpline
orientation, smooth cubic intrinsics and a quintic time warp. Velocity and
acceleration approach zero at the endpoints. A continuous correction preserves
the original calibration's approximately1e-7 rotation-matrix roundoff, so the
endpoint remains exactly the stored TRAIN matrix.

Proposed delivery is3840×2024. During the neural segment, focal length and
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

The1-code photometry replay difference is consistent with direct floating-point
native lens interpolation versus the earlier OpenCV interpolation-table rounding.
Both decode Log3G10/RWG native samples before linear-light interpolation and apply
the same fixed gain4, k4 radial correction, TRAIN scalar EV and Reinhard+sRGB.

Visual review:

- [All66 source-camera contact](</fsx/tmp/marshawn_video_path_20261107/train_cameras_first_frame.jpg>)
- [Actual2020 DC/HC/KA/MD candidates](</fsx/tmp/marshawn_video_path_20261107/endpoint2020_candidates.jpg>)
- [HC2020 crop comparison](</fsx/tmp/marshawn_video_path_20261107/HC_2020_native_crop_review.jpg>): red is native3840×2024 crop `[1000,650,3840,2024]`; cyan is a native1920×1012 face crop.
- [Rendered actual2020 native crop](</fsx/tmp/marshawn_video_path_20261107/GT2020_native_crop_preview.jpg>) is a diagnostic thumbnail; the full3840×2024 PNG is at campaign `video/delivery_4K/frames/0072/lookcloser.png`. The grey actor's face is physically facing down/occluded in the inspected HC/DC/KA views. The crop cannot reveal an unobserved face.
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
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K/path.json`.
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

The silver transport/controller and neural GPU branches have been syntax
checked, but have not yet run on a completed video model. The native GT branch
has executed successfully for2020, and its full3840×2024 output was manually
inspected. The existing physical occlusion of the grey actor's face remains.

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
