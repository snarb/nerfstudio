# Local train-waypoint camera workaround

## What was tested

User-authorized temporary shot adaptation: keep the dynamic actor and substantial
camera travel, but pass smoothly through an actual train pose where the late jaw
fleck is less visible. This tests a camera workaround, **not geometry repair**.

The current `dec5_incidence2_unwarped_dynamic_150` request supplies all 150 ordered
actor instants, unchanged meshes, intrinsics, fixed exposure/profiles and hard
single-source texture rendering. A compact periodic cosine-to-the-fourth pose
displacement reaches `G004_C005_121037` at sample 147 / actor time `001193`.
Half-width is 35 samples. Position and rotation change smoothly; actor times are
not retimed. The other 81 camera poses remain exactly unchanged. Camera poses
are transformed to common calibration coordinates before deformation and back
to each mesh's normalization afterwards. No crop or stabilization is applied.

Seven actual times were rendered with three disjoint workers on clever-shadow:
`000899, 000929, 001149, 001169, 001189, 001193, 001197`.
This is a sampled diagnostic, **not a newly rendered 150-frame video**.

## Results

**Reject this periodic candidate:** it hides the late small fleck but exposes a
larger under-chin cutout at the beginning. Published video and geometry unchanged.

| Check | Result |
|---|---|
| Camera orientation extent | Original 31.513°, candidate 31.224° |
| Actual actor / mesh inventory | All 150 entries unchanged |
| Exact selected train pose | Verified within saved float32 calibration tolerance |
| Train camera-center convex hull | All 150 candidate poses inside; this is not a two-row clearance certificate |
| Local translation speed | Nonzero sampled steps; max step 3.786× original max; no constant-speed claim |
| Initial `000899` | New/larger black under-chin cutout behind the hand: **fail** |
| Late `001193` | Isolated under-jaw fleck not visible; thin neck boundary and sharp false nose edge remain |
| Other five samples | No obvious local jaw regression; rough hair, crown gaps and source boundaries remain |

The main agent viewed all seven native-scale head comparisons plus the separate
`000899` and `001193` jaw comparisons. Other saved jaw/overview panels are not
claimed as inspected. Per-frame pass labels apply only to the **local jaw
regression gate**, not whole-frame artifact-free quality. The global gate fails.
Different camera views are not compared with pixelwise image-quality metrics;
no full-frame PSNR/SSIM/LPIPS or loss was added. No encoded motion preview was
generated after the initial visual failure, so temporal smoothness is not
visually certified from these snapshots.

- [Early regression, native jaw comparison](/mnt/data/dec5_local_train_waypoint/review/000899_jaw.png)
- [Late local benefit, native jaw comparison](/mnt/data/dec5_local_train_waypoint/review/001193_jaw.png)
- [Full explicit visual verdict](/mnt/data/dec5_local_train_waypoint/visual_review.json)
- [Independent geometry/inventory audit](/mnt/data/dec5_local_train_waypoint/geometry_audit.json)
- [Retained artifact hashes](/mnt/data/dec5_local_train_waypoint/artifact_manifest.json)

The first initialization stopped before writing any render request because an
overstrict `1e-10` pose assertion rejected ~`5.7e-9` stored rotation round-off.
The producer now checks float32-appropriate tolerance and preserves the exact
saved waypoint. Failure and successful logs are retained; no rendered result was
overwritten. Two helper tests pass. All three workers finished normally; terminal
PID/GPU/error/free-space evidence is in `checks.jsonl`.

The copied request's historical `initial_gate` concerns the parent's texture
experiment only; it is **not** acceptance of this camera change. This is recorded
explicitly in the new verdict; `full_video_candidate` and artifact approval are
false.

## Insights

A real train pose is helpful for the late specific fleck, but does not guarantee
an intact mesh or texture at another actor pose. Periodicity matters: a smooth
late correction wraps into the beginning, where the mouth and raised hand expose
a different under-chin surface. The next candidate must respect early **and**
late visibility constraints and retain the elevated early envelope. Dropping
bad frames, silently making the loop discontinuous, or freezing the actor would
not satisfy the requested dynamic flythrough.

Reusable opt-in helper: `scripts/smooth_train_waypoint.py`. Diagnostic producer:
`scripts/study_local_train_waypoint.py`. Existing model and renderer defaults are
unchanged. Geometry-prior work remains separate and unfinished.
