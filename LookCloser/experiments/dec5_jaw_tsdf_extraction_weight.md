# Cached-depth TSDF extraction weight at the late jaw hole

## What was tested

Frames `001193` and `001195`: reuse all 62 cached native geometric depth maps
and the original fixed calibration. Repeat the original CUDA tensor fusion
with extraction weights `0.5`, `1`, and `2`. Only the output path and extraction
weight change; voxel size, truncation, block activation, component filtering,
depths, camera poses and RGB are unchanged. No new PatchMatch was run.

These are extracted mesh controls, not saved raw TSDF volumes. An extraction
weight is not a verified count of independent agreeing PatchMatch cameras.
No control has replaced production geometry or entered the 6K video.

Artifacts: `/mnt/data/dec5_jaw_tsdf_extraction_weight`.
Producer: `scripts/study_jaw_tsdf_extraction_weight.py`; native geometry review:
`scripts/review_jaw_tsdf_extraction_weight.py`; independent retained-input,
command and ray-count replay: `scripts/audit_jaw_tsdf_extraction_weight.py`.

## Results

Missing pixels use the same pre-existing fixed moving-camera spot rectangle,
not a candidate-selected ROI. They are geometry diagnostics, not RGB metrics.

| Frame | Weight | Vertices | Triangles | Components | Missing spot pixels |
|---|---:|---:|---:|---:|---:|
| 001193 | Published / 2 | 66373 | 128203 | 1 | 44 |
| 001193 | 0.5 | 73733 | 141682 | 2 | 14 |
| 001193 | 1 | 68830 | 132721 | 1 | 14 |
| 001193 | 2 repeat | 66373 | 128203 | 1 | 44 |
| 001195 | Published / 2 | 66279 | 127932 | 1 | 73 |
| 001195 | 0.5 | 73702 | 141843 | 2 | 43 |
| 001195 | 1 | 68788 | 132688 | 1 | 73 |
| 001195 | 2 repeat | 66277 | 127928 | 1 | 73 |

All six controls completed. Independent audit verified 165 input/artifact
bindings, exact matched command arguments, component counts, finite meshes,
and replayed all six moving-camera spot counts. Two tests reject undeclared
parameter changes and duplicate control flags. The `001195` weight-2 repeat
differs by four triangles from the historical mesh: do not claim byte-exact
fusion determinism or explain those four triangles as a weight effect.

Main-agent visual inspection covered all eight retained panels: both moving
spot details and all six moving/F-E/M-B face panels. Lower weight partly
fills the neck edge, but keeps missing surface and adds fringe geometry around
hair and silhouette. Clay inspection alone does not establish texture quality
or multi-view correctness of the additions.

- [001193 moving detail](/mnt/data/dec5_jaw_tsdf_extraction_weight/001193/review/old_moving_spot.png)
- [001195 moving detail](/mnt/data/dec5_jaw_tsdf_extraction_weight/001195/review/old_moving_spot.png)
- [001193 F/E face](/mnt/data/dec5_jaw_tsdf_extraction_weight/001193/review/F004_E005_1210FP_face.png)
- [001195 M/B face](/mnt/data/dec5_jaw_tsdf_extraction_weight/001195/review/M004_B005_12109O_face.png)

## Insights

The extraction threshold suppresses some surface near this hole, but is not
its sole cause. Weight `1` helps only one of the two frames; even `0.5` does
not close either hole and introduces a second retained component. A global
threshold reduction is therefore not an accepted repair.

If pursued further, lower-weight geometry must be treated as a candidate-only
local addition, independently checked against measured train depth, foreground
footprints and free-space evidence. Previously validated local boundary repairs
already improve these particular old-path holes more than this global control;
do not describe this test as the first partial repair or promote it over them.
