# Dec5 cinematic: true 6K output

## What was tested

User correction requires **3456×6144 portrait output**, not merely 6K textures
sampled into the previous 1080×1920 delivery. This isolated rerender keeps the
exact dynamic `wide_spiral_free` camera path and all 150 actor times. The previous
source-only delivery remains intact at `/mnt/data/dec5_cinematic_wide_spiral_6k_v2`.

- Baseline: `/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free`.
- New root: `/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1`.
- Renderer: `scripts/render_cinematic_6k_output.py`; audits/review/package:
  `scripts/review_cinematic_6k_output.py`.
- 150 distinct source times, 24 fps, 6.25 seconds: 118 pure 3D frames,
  8 explicit dissolves, 24 moving H004_C005_1210SZ train-camera frames.
- No held-out RGB, retraining, calibration solve, mesh edit, synthetic detail,
  temporal interpolation, or HD output/source-ID upsampling.

Output rays are generated on a fresh 6144×3456 calibrated landscape lattice,
then rotated once. All target focal lengths and principal points multiply by
3.2; ray coordinates use pixel + 0.5, so no additional half-pixel principal-point
offset is appropriate. Camera poses and field of view remain fixed.

Original immutable dev3 PQ/AP1 PNGs are 6144×3072. The same crop
`[341,0,5802,3072]` gives 5461×3072; both historical Lanczos resizes are skipped.
Source UV mapping is `(uv_HD + 0.5) * [5461/1920,3072/1080] - 0.5`.
Source depth/visibility is freshly raycast at 5461×3072, using the unchanged
foreground masks mapped by native pixel centers. Only mesh-face graph labels
are reused; visibility, fallback choices, and source-ID pixels are recomputed
for every new target ray. The separate preflight binds all 150 source camera
orders to the baseline and verifies all 32 ending UV rectangles are in bounds.

Color uses the exact existing ST2084 inverse, fixed gain-to-nits
356.95123731403817, 0.005-nit floor, AP1→Rec709 transform, frozen camera gains,
and one global exposure (10.570320292648677). No frame-specific color fit.
Original source SHA256 values and selected-sample counts are retained per frame.

Native source depths occupy about 4 GiB GPU memory. Source raycasts use four
workers; target source selection uses 60,000-pixel batches. Raw PNGs are decoded
read-only on dev3 using four workers and 500,000-sample batches; only selected
linear RGB values are transferred. Successful exact scratch jobs are removed;
original PNGs and previous/failed output roots are preserved.

## Results

Status: **complete, reviewed with known residuals**. All 150 native PNGs and both
video streams verify as 3456×6144, 24 fps, 6.25 seconds. The final
[delivery seal](/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/delivery.json)
binds media, review, frozen request/scripts, source provenance and terminal checks.

| Deliverable | Bytes | SHA256 |
|---|---:|---|
| [HEVC MP4](/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/presentation/video.mp4) | 93,720,314 | `447bf60f513c495ea9af997007f529a9b414d54d27a995511ecacc36b024353e` |
| [H.264 MP4](/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/presentation/video_h264.mp4) | 141,279,363 | `2893f8d3fc4333b6a7fc02e1da4d1a4ea6671ebabda234437cfa43d84910ff5d` |
| [Lossless PNG archive](/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/presentation/frames.zip) | 2,078,164,643 | `dcb462b45c591409fac63645208f3854666fa032dbe4515702e515061d055e97` |

All 150 PNGs are distinct; 120 camera centers are distinct (the ending holds the
camera, not the actor). Every pose and mesh remains bound to the baseline.
Median measured frame seconds: 67.64 pure 3D, 109.12 dissolve, 49.39 pure train.
The run read 7,834 original PNGs / 677,845,992,458 bytes. Minimum supported
mesh-hit fraction was 99.972%; this is coverage, not a geometry-accuracy metric.

| Canary | Output | Measured seconds | Review |
|---|---:|---:|---|
| 001083, dynamic 3D hair close-up | 3456×6144 | 74.7 | Framing coherent; finer hair/skin/lip detail; source seams remain |
| 001197, actual dynamic train ending | 3456×6144 | 42.2 | Framing coherent; actual native source detail |

Native 1:1 comparison crops are in the new root's `review/` folder:
`001083_{hair,ear,lips}_AB.png`, `001197_{hair,ear,lips}_AB.png`.
Only the explicitly labeled diagnostic control enlarges the previous HD output;
the delivered frames are independently rendered at 6K. Overview pairs show the
same framing. Parent independently reviewed both full-resolution canaries and
verified pixel-center ray equivalence to 7.49e−8 at tested positions.

Native temporal strips cover `000899..000909` hair and `000995..001007`
lipstick/rear hair. Motion remains evident. The latter interval retains the
known skin-colored polygon beside the lipstick and rear hair/background leakage
in both controls; 6K makes the polygon boundaries sharper. This interval was
reviewed without changing production geometry or source labels.

An independent bounded review additionally viewed 14 paired six-frame sheets
(83 distinct pure3D times) and four existing native panels. It found no new
catastrophic temporal regression or frozen actor/camera relative to the prior
HD-output control. This supplements, not replaces, final all-frame and decoded
video review. Its exact scope/input hashes are recorded at
`/mnt/data/dec5_cinematic_6k_independent_review/review.json` and copied into
delivery provenance; it makes no artifact-free or continuous-playback claim.

Five focused tests pass with
`../.venv/bin/python -m pytest -o addopts= -q tests/test_cinematic_6k_output.py`.
They cover target ray scaling, anisotropic source crop coordinates, and the
ending lens's finite source footprint, and no-overwrite directory publication.
The symlink-publication test also passes on the actual `/mnt/data` filesystem.
Full camera order, unchanged poses,
and train UV bounds are checked by `preflight.json`.

## Insights and limitations

True 6K output exposes finer source detail and also makes some fine triangular
texture boundaries more visible. The frozen reconstruction still has pale/tan
hair fringe and polygonal neck/chin silhouettes. This is not an artifact-free
geometry claim. The final 1.9× lens uses approximately 2874×1617 original source
pixels, so a 6144-pixel output cannot create optical detail absent from that crop.

Late 3D frame001123 also retains a thin **internal** dark nose-side seam visible
in both HD and 6K controls, more continuous in native6K, and a pink/tan chin-edge
flange. Its cause is not established here; it is not labeled a natural shadow.
Parent independently viewed this overview/lips/nose-seam comparison and the full
001151/001197 train frames; the train views show changed head tilt/gaze and no
obvious synthetic seam or hole in those two inspected images.

Parent's independent native-label attribution at
`/mnt/data/dec5_6k_source_seam_diagnosis/mesh_label_attribution` found the bright
upper-lip line exactly on the frozen H/C↔I/C mesh-face label boundary: no missed
rays and no visibility fallbacks in that 262,144-pixel crop. The hair crop had
no missed rays and only seven fallbacks; narrow triangles mostly retain the
frozen alternative source. This establishes inherited source boundaries, not a
new native-visibility failure. Parent's subsequent single-H/C lip control still
retains a faint glint, also present as genuine gloss in the original native
source: boundary correlation alone does not make every bright lip line an
artifact. No corrective variant enters this delivery.

Final HEVC/H.264 encodes retain native 3456×6144 with no resize filter;
the PNG archive is the lossless output master. Two independent encodes ran
concurrently, eight encoder threads each, and both full frame-count probes pass.
All 150 PNGs and archive entries have verified actual-byte hashes; internal
relative directory links are resolved into actual PNG bytes in the archive.
Rendering supervision is recorded
in `supervision.json` and `checks.jsonl` including controller/remote worker,
frame/stage, GPU, free disk, and error evidence.

Final manual review actually viewed all 150 chronological thumbnails in 15
overview sheets, ten decoded-video thumbnails, late 3D/dissolve native crops,
and all 24 moving train times in four native strips. The actor/camera remain
dynamic and the explicit black-to-real-background transition remains coherent;
known contour and source patches persist. Exact viewed paths/hashes and the
additional parent review are in `visual_review.json`. This is contact-sheet and
crop review, not continuous normal-speed playback or an artifact-free claim.

Matched native grid-8 foreground RGB comparisons against FFmpeg-decoded HEVC
at 001083/001197 show channel mean differences about −1.09 to −1.36 /255,
without a broad cast/range-loss flag. This is a color sanity check, not PSNR or
a held-out quality evaluation. Matrix/transfer/primaries tags are unspecified;
HEVC reports limited range, while H.264 range is also unspecified. Other players
may interpret untagged color differently. Exact checks: `review/color_sanity.json`.

All workers and the checks writer were terminal before sealing. The primary
outer execution returned 143 after the final `requested_frames_finished` record
and all 150 receipts; the termination cause is not established. No render error
or OOM was observed. Completion rests on independently verified retained bytes,
full PNG/ZIP validation and both video probes, not an assumed clean primary exit.
The parallel controller and packaging exited 0. Failed workspaces remain intact.

### Bounded parallel ending work

During the latter half of rendering, a disjoint worker computed pure-ending indices
126..148 with the same frozen renderer/request. It skips initialization and
verifies remote script hashes read-only, avoiding shared progress/code-copy races.
The two-frame pilot took 48.3/47.9 seconds; concurrent main frames remained
69.1–72.0 seconds versus a preceding 70.4-second median. The approved worker then
continued the remaining endings without changing production sampling.

Continuous overlap then exposed dev3's four-core CPU limit: three main frames
took 102.7/90.3/85.7 seconds. With parent authorization, an exact-argv/UID/frame
allowlisted watcher lowers only ending-worker threads to nice10. It leaves main
workers, sampling, request bytes, and unrelated jobs untouched. Exact PID/argv
and old/new priorities are retained in `scheduling_checks.jsonl`; the watcher
ends when its local ending controller exits. Critical-path progress is monitored,
not inferred from the favorable initial pilot alone.

An initial `renameat2(RENAME_NOREPLACE)` publication attempt failed safely with
EINVAL on `/mnt/data`'s fuseblk filesystem; its script/request/log remain saved.
The working publication moves completed verified payloads into the exclusively
owned output-root `parallel_completed/` namespace, then atomically creates
`frames/<id> -> ../parallel_completed/<id>`. Link creation refuses any existing
destination, including an empty directory. Both canonical and payload paths are
hash-checked; the frozen primary renderer was explicitly tested to resume through
the relative directory link without rerendering or rewriting progress. Output
references stay inside the main root, and the archive contains actual PNG bytes.
