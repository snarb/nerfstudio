# DEC5: slow central-space flight, 150 real instants

**2026-09-14 correction:** the user rejected the camera movement as visually
near-static. Its actual extent was only 1.228 horizontal / 0.650 vertical grid
intervals. Smoothness and containment were insufficient acceptance criteria.
See [camera-grid diagnosis and wider static controls](dec5_camera_grid_diagnosis.md).

## What was tested

Opt-in `scripts/central_space_temporal_flythrough.py` reuses the corrected geometry
inventory of `lookcloser_dec5_5a3_smooth_temporal_150_repaired_v3`, including the
real-depth full-block TSDF repairs of 000971 and 000973. All 150 target views are
newly rendered. Frames remain 000899..001197, stride two, at 30 fps; neither actor
slow-down nor synthetic temporal interpolation is used.

The camera traverses the first 150 samples of a 360-sample, radius-0.65 central
arc. Its centers are strict convex combinations of four central **train** camera
centers: G004/B005, I004/B005, I004/D005, G004/D005. It looks toward a fixed point,
uses fixed intrinsics, and moves continuously in the shared calibration space;
each mesh's normalization is applied only afterward. The five-second clip is
**open, not a seamless loop**. Repeating it in a player would restart the camera
and actor; that restart is not part of the delivered trajectory.

Frozen texturing: 62 real train EXRs, fixed exposure 10.570320292648677, fixed
camera profiles and static subpixel registration, hard surface source labels,
native RGB from one camera per sample, no RGB averaging or diffusion. Held-out
RGB is not read. Four supervised workers share the local GPU; this is rendering,
not four simultaneous PatchMatch jobs.

Workspace: `/mnt/data/lookcloser_dec5_5a3_central_space_flight_150`.
The immutable request records parent geometry, all hashes and all target poses.
The optical target in the path report uses the reference 000973 normalized gauge;
the motion measurements below use **raw calibration coordinates**, not meters.

## Results

| Camera check | Measured result |
|---|---:|
| Angular speed min / median / max | 2.588 / 2.703 / 2.746 degrees/s |
| Linear speed min / median / max | 0.360147 / 0.360155 / 0.360158 calibration units/s |
| Speed max/min | 1.00002882 |
| Largest consecutive velocity-direction change | 1.636 degrees |
| Minimum convex anchor weight | 0.02019182 |
| Adjacent motion samples | 149, no artificial last-to-first jump |

[Rig and central arc](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/camera_path.png).
[Five canary face crops](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/canary_review/face.png),
[lipstick crops](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/canary_review/lipstick_hand.png).
Each chronological four-frame group has native face/lip/ear contact sheets plus
nearby real-train comparisons. A nearby train camera is **not** exact novel-view
GT: no new 150-view PSNR/SSIM/LPIPS claims are made.

### Artifact hypotheses and controls

1. **Fallback-source switching causes the beige hair fringe.** Attribution on
   001047 found 1,929 fallback pixels in the entire image, only 132 inside the
   52,800-pixel right-hair diagnostic rectangle. Actual overlay inspection shows
   most beige plates use the preferred hard surface label. Fallback is not the
   main explanation; removing RGB averaging cannot fix this because there is
   no RGB averaging in this renderer.
2. **Real-train silhouette support can safely remove the fringe.** The separate
   experimental `test_train_silhouette_mesh_trim.py` projects mesh vertices into
   six nearby train silhouettes, requiring four outside votes at every triangle
   vertex. It preserves any triangle whose removal exposes deeper target
   geometry. Monotone restoration runs to a fixed point. Remaining target RGB
   is byte-identical; no generated colors, eval RGB or target-defined masks.

| Experiment | Removed triangles | Removed target pixels |
|---|---:|---:|
| 001047, semantic probability 0.1, dilation 20 | 121 | 480 |
| 001047, probability 0.5 + real-color GrabCut, dilation 8 | 4,785 | 16,123 |
| 000899, same refined rule | 3,317 | 13,322 |
| 001197, same refined rule | 2,386 | 10,746 |

The first control barely changes the fringe. The refined rule trims some outer
plates but leaves most visible fringe and shoulder seams. Native comparisons
were inspected on all three refined test instants; face/ear and visible tube
remain coherent. This is **not a demonstrated repair of the main artifact** and
has no temporal stability validation, so it is not promoted into the video.
Controls and rejected proposals remain in `silhouette_trim_test/`; the production
meshes and predictions are not overwritten.

[001047 hair comparison](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/silhouette_trim_test/001047_d8/comparison_hair.png),
[000899 face/hand comparison](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/silhouette_trim_test/000899_d8/comparison_face_hand.png),
[001197 hair comparison](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/silhouette_trim_test/001197_d8/comparison_hair.png).

## Insights

The camera can cover more of the central camera space without jumping between
physical views: interpolate a continuous calibration-space curve and control
distance traveled, not camera-list indices. Open clips must not be audited as
closed loops. The audit now reconstructs actual centers from their declared
convex weights, rather than trusting a boolean “inside hull” flag.

Residual tan boundary plates are consistent with uncertain peripheral geometry
receiving skin/background samples from otherwise selected train cameras. This
is an inference, not proof that every plate has the same cause. Conservative
silhouettes cannot recover subpixel hair geometry or make depth self-occlusion
independently trustworthy. The existing 000971 large lipstick extrusion repair
survives the new path; small skin seams, under-chin slits and peripheral fringe
remain and must not be described as artifact-free reconstruction.

All **150/150** instants were visually inspected in native face/ear and lip/hand
panels; no catastrophic geometry was found. Small local defects remain, including
the neck triangle at 001045 and chin-edge fragments around 001125. The actual MP4
was decoded into a 15-sample overview and consecutive native crops around the
lipstick repair and later chin edge. These reveal local fringe flicker, not a
camera-pose jump. This was frame-based inspection, not real-time video playback.

Four-worker rendering finished in **1,112.3 seconds** (18.5 minutes, five canary
renders already cached); 31 targeted tests passed. Encoding checks 150 frames,
1080x1920, 30 fps and exactly 5 seconds. Video SHA-256:
`3eb665a6e0030d5fcf6381df5ea2fefa2a7b5f0b08612c9b1a3dbcae9bcca0cd`.

[Decoded overview](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/encoded_temporal_overview.png),
[lipstick transition](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/encoded_transition_032_043/contact.png),
[chin transition](/mnt/data/lookcloser_dec5_5a3_central_space_flight_150/encoded_transition_110_115/contact.png).
The viewing result is accepted **with known artifacts**; the request to eliminate
all surface artifacts is not claimed fully achieved. Final integrity checks and
all reviewed hashes are in `audit.json`, `frames_audit.csv`, and `visual_reviews/`.
