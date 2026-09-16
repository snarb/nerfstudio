# Contiguous dynamic validation of face-angular source recovery

## What was tested

The frozen [face-angular/skin-consensus rule](dec5_face_angular_visibility.md)
on two24-frame, normal-rate cinematic intervals:

- `camera_tie`:001061..001107, covering actual I/B→I/C→H/C angular-source changes;
- `nose_seam`:001087..001133, covering the previously persistent nose line.

The intervals overlap by11 frames: **37 distinct actual times**, not48 unique
reconstructions. Four audited pilot times are reused;33 are newly staged,
segmented and rendered with three parallel workers. Each time uses its own
existing mesh, source RGB and moving camera. No fixed actor, duplicated time,
pose alteration, color refit, exposure change, image averaging or target RGB.
The production150-frame6K video is unchanged. These are **HD diagnostic pairs**,
not a replacement6K delivery or new mesh-reconstruction campaign.

The supervisor records process/stage, GPU memory, error scan and free space
every10 seconds. A completed first clip is reviewed while other workers keep
rendering. All33 new workers terminate with exit0; no recorded OOM/CUDA/traceback
errors. The four reused outputs pass the earlier835-binding pilot seal.

## Results

### Render and chronological visual checks

| Interval | RGB changes/frame min / median / max | Newly zero RGB samples | Observation |
|---|---:|---:|---|
|Camera transitions|0 /479 /132304|8|Broad source replacement at a few transitions; no new gross defect in inspected panels|
|Nose line|200 /523.5 /3781|0|Black line consistently removed across all24 times|

The main agent viewed **26 images**: four chronological overview sheets,
twelve strongest-change native sheets, six native nose strips and four exception
panels. All37 unique times are represented. This is actual sequential image
inspection, **not a claim of continuous normal-speed video playback**.

The nose-side artificial line is removed while real nose shading remains. No
obvious extra face blur or detached island appears in the inspected comparisons.
The inherited ragged crown/neck geometry and brown hair-edge contamination remain.
The earlier lipstick-action interval is outside these two clips and is not
validated by this result.

- [Camera-transition comparison,24fps/1s](/mnt/data/dec5_temporal_face_angular_control/review/camera_tie/matched.mp4)
- [Nose-line comparison,24fps/1s](/mnt/data/dec5_temporal_face_angular_control/review/nose_seam/matched.mp4)
- [Transition native examples](/mnt/data/dec5_temporal_face_angular_control/review/camera_tie/largest_change_native_02.png)
- [Four consecutive native nose comparisons](/mnt/data/dec5_temporal_face_angular_control/review/nose_seam/nose_native_04.png)
- [Explicit visual decision](/mnt/data/dec5_temporal_face_angular_control/visual_notes.json)

Both paired MP4s decode to24 frames,2160×1920,24fps. Each side is the original
1080×1920 diagnostic render. Encoding adds no slow motion or interpolated times.

### Temporal diagnostics, not perceptual scores

Quarter-resolution Farneback motion is estimated from **baseline RGB only**.
Both arms use identical cycle-consistent valid samples and a fixed head/actor
context. It is not ground-truth motion, does not measure missing geometry and
must not be interpreted as face PSNR/SSIM/LPIPS or proof of artifact freedom.

| Interval | Common tracked source samples | Baseline→candidate source switches | Weighted mean tracked RGB step,0..255 |
|---|---:|---:|---:|
|Camera transitions|1458942|150806→150798|3.93248→3.93498|
|Nose line|1653476|30504→29538|3.17723→3.17376|

Source-switch and appearance diagnostics use explicitly documented, different
fixed context heights; their sample counts are not interchangeable. The RGB
step is weighted by each transition's common appearance-sample count.

The largest candidate increases in mean tracked RGB step occur at the transitions
ending001073 (+.18477/255) and001083 (+.13350/255). Overall transition-clip change
is only+.00251/255, but that aggregate does not erase local changes. The nose
clip has slightly lower mean steps at every inspected pair. These measurements
do not establish absence of fine flicker or correct view-dependent eye highlights.

### Explicit exceptions and audit

The eight newly black samples occur at001083 (5) and001085 (3), all in the
pupil/highlight region. Positive mesh depth and valid source IDs remain.
Independent source-RGB replay passes for both times (001083 in the sealed pilot;
001085 newly audited here). These are not newly missing geometry or source
holes, nor are they near-zero old samples: an actual reflected highlight changes
with the source. Correct novel-view specular appearance remains unproven.

[Marked001085 pupil crop](/mnt/data/dec5_temporal_face_angular_control/exceptions/001085_new_black.png).

001063 and001077 have exactly identical baseline/candidate RGB and source arrays.
Their strongest-change sheet rows are blank because the zero-valued change map
has no meaningful maximum crop. Explicit complete-frame comparisons preserve
their visibility in the review; neither time is omitted from the clips.

Every frame's output/request hashes, baseline depth reference and unchanged-source
RGB are checked. The renderer itself reraycasts the target and asserts exact
baseline depth equality. Separate independent audits replay all52982 and46129
changed points at001071 and001085: intersections, source UVs, visibility/skin
quorum, angular weights and actual calibrated RGB. This is added stress coverage,
not a claim that every new frame received a second full ray audit.

Six focused tests cover contiguous unique time inventories, output isolation,
raster/consensus constraints and a candidate-only color-step diagnostic. Defaults
and hash-pinned previous helpers are unchanged. No full-frame quality metrics or
new face-metric protocol were introduced for these interpolated views.

## Insights

The fix is no longer supported only by isolated stills: the same frozen rule
removes the line throughout a contiguous24-time sequence, with no new black
samples there. It is selected for the **next opt-in integration test**, not
promoted as a fully artifact-free video pipeline. Camera-transition appearance
still deserves scrutiny, especially at001073 and001083, without treating any
source change as automatically wrong.

Stop generic nose-weight sweeps. The next substantive work is integration with
the separately validated geometry/layer-aware texture changes and correction of
cheek/lipstick geometry. This experiment deliberately changes **no mesh** and
cannot satisfy that separate requirement. The full150-time native6K result,
including lipstick action and inherited hair/neck contour defects, remains open.

Reproduction in fresh isolated roots:

```text
run_temporal_face_angular_control.py init
run_temporal_face_angular_control.py supervise
review_temporal_face_angular_control.py --clip camera_tie
review_temporal_face_angular_control.py --clip nose_seam
inspect_temporal_face_angular_exceptions.py
```

Use the repository venv with two OpenMP/OpenBLAS threads. The controller invokes
the existing isolated MediaPipe interpreter for inference; no model download or
environment upgrade. Root:
`/mnt/data/dec5_temporal_face_angular_control`.
