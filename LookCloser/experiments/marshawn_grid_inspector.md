# Offline occupancy slice inspector

## What was tested

Reusable exporter `scripts/export_occupancy_grid_viewer.py` produces one standalone HTML file from named boolean XYZ NPZ layers. It synchronizes XY/XZ/YZ slices, compares two layers, reports exact voxel bounds in the declared coordinate frame, and downloads manual observations with source hashes. It never changes a model or grid. Different bounds, resolutions, frames, and ambiguous multiple cascades are rejected instead of silently overlaid.

Example:

```bash
python scripts/export_occupancy_grid_viewer.py \
  --spec /fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/spec.json \
  --output /fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/index.html
```

The JSON spec declares `coordinate_frame` and a `layers` list, each with `name`, `archive`, `key`, and optionally `bounds_key` (default `cascade_aabbs`). Arrays must be boolean XYZ, optionally with a single leading cascade dimension. All data and scripts are embedded; no server or network connection is needed to open the exported file.

## Results

Initial artifact: [interactive inspector](/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/index.html), [browser screenshot](/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/browser_review.png), [source hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/index.json).

It compares actual actor-only exp step15188 occupancy (123,817 active cells) with the proposed conservative TRAIN support (131,911 admitted cells) on identical 128³ cell boundaries. The actual archive hash is `6c7558a912946833ab2b397595d62960e3354fa1234f0ecfb17320000aa75fca`. This comparison is hypothetical; it does not retroactively alter that checkpoint.

Five CPU tests passed: unequal-axis XYZ bit addressing including final-byte padding, exact boundary mismatch rejection, frame mismatch rejection, multiple-cascade rejection, and landmark mapping including outer boundaries and outside-box detection. Headless Chromium passed actual local-file loading, three synchronized slices, numeric coordinates, positive-Y-up click mapping, layer swapping, comparison toggling, observation JSON download, and landmark navigation, with no page errors. Root visually inspected the screenshots. Test observations are explicitly marked as browser checks, not geometry findings.

The exported scene also contains39 unique triangulated TRAIN feature landmarks from the sealed103 camera-ray records. Selecting one navigates to its cell; provenance and material limitations remain available. Point5568 explicitly retains its braid/boundary ambiguity instead of being labeled verified opaque clothing. These sparse landmarks supplement visual navigation and do not certify the rest of the volume.

## Insights

The inspector makes individual cells and all slices accessible for manual review. It supplements camera renders and ray traces; an occupied or admitted cell is not proof of physical geometry, and an excluded cell is not an independently verified empty-space certificate. The displayed actor-only checkpoint still has scattered false foreground occupancy; the new support is a separate training hypothesis being tested.

A subsequent [matched15188 comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/semantic_15188/index.html) contains the original actor-only active grid (123,817 cells), the actual newly trained semantic-admission active grid (51,894), and its fixed mapped admission (131,911). These are identical native cell boundaries; the raw semantic source has slightly different float64 boundaries and is deliberately not overlaid. Chromium landmark navigation passed and root viewed the comparison. Newly trained geometry differs from a posthoc intersection: some cells appear where the old model lacked appropriate support. Lower cell count alone is not an accuracy metric; remaining halos/floor fragments are documented in the separate ray and render audits.

Download and open locally:

```bash
scp coder@olive-lark:/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/index.html ~/Downloads/marshawn_grid_inspector.html
```

## Complete orthogonal sweep at semantic R0 step60,752

`scripts/export_occupancy_slice_sweep.py` exports every slice from the same validated viewer specification. Example arguments: `--spec SPEC.json --output DIRECTORY --active active_R0_60752 --support fixed_admission`. White denotes active cells within support, blue support alone, and red active cells outside support. Coordinates increase right/up on explicitly labeled axes. The two orientation/color tests and the five existing viewer tests pass.

Root visually scanned all six pages (384 slices, 128 per axis) under `/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/semantic_60752/sweep/`; source/page hashes and findings are saved in `receipt.json` and `root_manual_review.json`. There are zero active cells outside sealed admission. A broad, patchy near-floor component is apparent around Z slices0–25 and agrees with floor fragments in camera renders. Upper-body support changes continuously across adjacent slices, without obvious global axis reversal or remote high-Z blocks.

This is an exhaustive slice-page scan, **not** independent certification of every physical cell. Empty interior cells, imperfect silhouette boundaries, and occluded geometry still need ray evidence. No cells were removed based on component size or this visual scan.
