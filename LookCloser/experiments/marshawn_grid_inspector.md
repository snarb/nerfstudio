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

Four CPU tests passed: unequal-axis XYZ bit addressing including final-byte padding, exact boundary mismatch rejection, frame mismatch rejection, and multiple-cascade rejection. Headless Chromium passed actual local-file loading, three synchronized slices, numeric coordinates, positive-Y-up click mapping, layer swapping, comparison toggling, and observation JSON download, with no page errors. Root visually inspected the screenshot. Test observations are explicitly marked as browser checks, not geometry findings.

## Insights

The inspector makes individual cells and all slices accessible for manual review. It supplements camera renders and ray traces; an occupied or admitted cell is not proof of physical geometry, and an excluded cell is not an independently verified empty-space certificate. The displayed actor-only checkpoint still has scattered false foreground occupancy; the new support is a separate training hypothesis being tested.

Download and open locally:

```bash
scp coder@olive-lark:/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/index.html ~/Downloads/marshawn_grid_inspector.html
```
