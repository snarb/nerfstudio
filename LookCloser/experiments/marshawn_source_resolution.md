# Marshawn source-resolution feasibility

## What was tested

One TRAIN camera (AA), reprocessed directly from original8192×4320 vendor TIFF
at3840×2024. The existing gain4, frozen TRAIN-fitted camera scalar, shared radial
k4 correction and Reinhard+sRGB encoding are inherited exactly, without any
refit. Current1920×1012 source is rebuilt first and must match every byte.
This diagnostic changes no training data, masks, frequency maps or metrics.

## Results

The FHD rebuild is byte-identical. Root visually inspected the same actor region
in an FHD display upscale and the actual3840 source: finer folds, hair edges and
shoe highlights survive, but high-resolution noise is also more visible. This
is source-detail evidence, not an observed training improvement.

[Side-by-side source crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/resolution_feasibility_AA_v2/actors_source_comparison.jpg)
· [Source hashes and fixed photometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/resolution_feasibility_AA_v2/receipt.json)

Downsampling the new image toFHD differs from the frozen source by1.729 mean
8-bit code values and11 at the99thpercentile across RGB channels. Linear RAW
prefiltering, calibrated remapping, radial correction and nonlinear tone mapping
do not commute; this difference must not be hidden by changing the held-out
reference images.

## Insights and next steps

A later training-resolution comparison is feasible after structural factors are
settled. It must preserve originalFHD eval images/masks/crops, camera poses and
accepted AABB; generate trainRGB and source-native masks using the exact scaled
calibrated mapping; and refit source-bound train frequency maps. Higher resolution
also changes pixel footprints, noise and effective sampling coverage, so it is
not a pure encoding-capacity experiment. No such training has been launched.

Reproduce using `scripts/probe_marshawn_source_resolution.py --data <campaign>/data
--output <new-directory> --camera AA`. The helper rejects held-out cameras and
fails if the rebuilt original source is not identical.
