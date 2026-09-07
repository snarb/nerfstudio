# DEC5 150-frame PatchMatch-TSDF temporal fly-through

## What was tested

The first 150 chronological DEC5 5A-3 instants (`000899` through `001197`)
were assigned one fixed-pose COLMAP PatchMatch-to-TSDF mesh and one pose on a
closed camera path. The path follows the calibrated D..L by A..E outer rectangle,
uses 149 interpolation intervals plus the repeated first pose, and stays two to
four camera-grid rows from the H/C center. Target-view RGB is never read.

The geometry recipe is the accepted fixed 62-train-camera two-pass PatchMatch
configuration. The first 50 meshes are adopted only after validating their source
campaign and retained-file manifests; later meshes use the same commands through
TSDF extraction. A geometry-only worker omits the old discarded physical-eval
texture render. Its `000899` canary reproduced the full worker mesh byte-for-byte.
Final color uses hard seam-cut labels with no RGB averaging: rank zero is selected
from the calibration-only angular 16, with up to seven visibility fallbacks from
all 62 train cameras.

Two hosts process atomic dynamic claims, with one GPU stage at a time on each:
clever-shadow (RTX PRO 6000 Blackwell) and dev3 (L40S). Immutable request
`48f259da7a052f861b3a7d24dd27120e7dd2af60bf5f6d48d38c4ca1b9c062f8`
pins all 9,750 EXR hashes, calibration, code, path, geometry, and render settings.
Visual review runs in parallel and treats dark outer contours, small skin seams,
and ragged boundaries as acceptable; only catastrophic appearance or broken
geometry fails a frame.

The original geometry-only worker also required exactly one connected component.
Frames `001035`, `001037`, and `001039` showed that this is not a valid proxy for
catastrophic geometry when a raised arm, sleeve, hair patch, or prop crosses the
normalized crop boundary. Two clean `001035` runs reproduced component sizes
exactly, and independent full-resolution review identified the large secondary
surface as the genuine arm/sleeve. A campaign-wide opt-in amendment therefore
retains secondary components only when their total is at most 40,000 triangles
and 25% of the largest component, publishes the render as pending visual review,
and does not repeat PatchMatch. It changes neither PatchMatch/TSDF commands nor
mesh bytes; every affected render still requires an explicit visual verdict.

## Results

Campaign output:
`/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150`

The initial full-resolution gate passed for adopted `000899` and new-geometry
frames `000999`, `001001`, and `001005`. The first three new meshes each had 62
full-resolution geometric depth maps and one connected component.

| Frame | Host | Mean coverage | Min coverage | Triangles | Visual |
|---|---:|---:|---:|---:|---:|
| 000899 | clever-shadow (adopted) | 0.38550 | 0.26587 | 157,756 | pass |
| 000999 | clever-shadow | 0.38525 | 0.25009 | 153,266 | pass |
| 001001 | dev3 | 0.38531 | 0.25142 | 153,196 | pass |
| 001005 | clever-shadow | 0.38614 | 0.24874 | 153,383 | pass |

The final atomic audit passed with exactly 150 ordered frame directories, 150
results, 150 unique render hashes, 150 visual-review receipts, no pending
review, and no catastrophic failure. All 1,200 files named by the per-frame
retained manifests were independently rehashed with zero missing files or hash
or byte-size mismatches. Every reprojection audit records
`uses_eval_rgb_for_prediction=false`, `eval_rgb_use=not_read`, no masks, and
only `frame_train_*` sources.

| Final inventory | Value |
|---|---:|
| Ordered frames / results | 150 / 150 |
| New meshes: clever-shadow / dev3 | 63 / 37 |
| Reused, hash-verified meshes | 50 |
| Visual pass / fail / pending | 150 / 0 / 0 |
| Retained files rehashed / mismatches | 1,200 / 0 |
| Minimum non-black render fraction | 0.343899 |

| Geometry statistic over 150 frames | Min | Median | Max |
|---|---:|---:|---:|
| Mean per-camera depth coverage | 0.374105 | 0.386928 | 0.397431 |
| Minimum per-camera depth coverage | 0.236680 | 0.251417 | 0.284775 |
| Mesh vertices | 64,835 | 68,425 | 83,627 |
| Mesh triangles | 125,746 | 132,107 | 161,781 |

All 100 newly reconstructed frames produced 62 full-resolution geometric depth
maps. The adopted first 50 use the hash-verified source-campaign meshes and do
not duplicate that discarded dense workspace in this campaign. The final mesh
component-count distribution is 80 one-component, 42 two-component, 19
three-component, 7 four-component, and 2 five-component frames; full-resolution
review accepted every bounded secondary component as attached, coherent actor
geometry.

Both videos contain exactly 150 frames at 1920x1080, 30 fps, and 5.0 seconds.
The lossless FFV1 decode is byte-identical in RGB to the source PNG sequence.

| Video | Bytes | SHA-256 |
|---|---:|---|
| `dec5_patchmatch_tsdf_flythrough_150_lossless_ffv1.mkv` | 165,246,851 | `2032260f27e1ed96a23d969090adf25ef36a756306f6a0572743d85b13aea44d` |
| `dec5_patchmatch_tsdf_flythrough_150_hq_h264.mp4` | 27,283,455 | `e4a292aa289715aec3767dd6446e08cf5cc805f17649a1e00077d009bc921a37` |

Contact sheets: [000--024](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_000_024.png),
[025--049](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_025_049.png),
[050--074](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_050_074.png),
[075--099](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_075_099.png),
[100--124](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_100_124.png), and
[125--149](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/contact_sheets/frames_125_149.png).

Independent review of all contact sheets plus fully decoded samples from both
videos found no detached or duplicated anatomy, catastrophic holes, gross mesh
shear/fold, collapse/explosion, or catastrophic view/batch transition. Accepted
non-catastrophic defects are ragged silhouettes, attached warm/grey hair halos,
small edge gaps, blocky shirt patches, and hard-source tone/texture seams. The
most visible tolerated seam is a rectangular lower-face/lip band around temporal
indices 138--142. The final audit is
[`audit.json`](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/audit.json),
and the independent verdict is
[`final_visual_review.json`](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/final_visual_review.json).

## Insights

The accepted TSDF surface generalizes to distant outer-ring novel views without
the catastrophic ear/lipstick breakup seen in rejected surface-refinement
canaries. Camera motion closes exactly, but subject motion does not, so the final
temporal transition is not expected to loop seamlessly even though the camera
pose does.

Keeping JPEG conversion on clever-shadow and transferring only one 63-camera
staged JPEG dataset to dev3 avoids moving the EXR parent. Removing the discarded
eval render preserved mesh bytes in the canary and reduced the per-frame critical
path without changing PatchMatch or TSDF.

Measured steady-state reconstruction medians were about 13.6 minutes per frame
on clever-shadow and 23.9 minutes on dev3. Concurrent workers therefore produced
one new temporal result every roughly 8.7 minutes, about 1.6x faster than
serial execution on the faster host. A second dev3 dispatcher overlaps local EXR
to JPEG ingest with the active GPU stage. Re-rendering a verified mesh from the
first 50-frame campaign took roughly 42--60 seconds instead of a full geometry
pass. The bounded component amendment additionally avoids a full duplicate pass
on articulated/crop-boundary frames.
