# Frequency Map Visual Debug

## What was tested

Generated frequency-map debug artifacts for the requested HD image and then checked a targeted 6K crop with small cables/rods:

- HD source: `/home/ubuntu/repos/look-closer/E004_D014_HD.jpg`
- HD crop: `x=512, y=104, w=512, h=512`
- 6K source: `/home/ubuntu/repos/look-closer/E004_D014_graded.png`
- 6K crop: `x=2560, y=1152, w=512, h=512`
- Config: `patch_size=8`, `ssim_threshold=0.97`, `ssim_window_size=7`, `train_steps_per_level=160`, `max_res=2048`.

## Results

| Run | Patch grid | Non-empty levels | Min-level frac | Max-level frac | Visual read |
|---|---:|---:|---:|---:|---|
| HD crop | 64x64 | 16 | 0.003 | 0.041 | Not collapsed. Wires, hard vertical structures, and small details are generally higher than smoother background regions. |
| 6K target crop | 64x64 | 12 | 0.000 | 0.020 | Not collapsed. Rods/cables and hard object boundaries are high, while many wall regions stay mid-frequency. Brick texture still contributes some high-frequency patches. |

Artifacts:

- [HD frequency map](../lookcloser_debug_outputs/freq_hd/freq_map.pt)
- [HD heatmap](../lookcloser_debug_outputs/freq_hd/freq_heatmap.png)
- [HD overlay](../lookcloser_debug_outputs/freq_hd/freq_overlay.png)
- [HD histogram](../lookcloser_debug_outputs/freq_hd/freq_histogram.png)
- [HD stats](../lookcloser_debug_outputs/freq_hd/freq_stats.json)
- [6K target overlay](../lookcloser_debug_outputs/freq_6k_target_crop/freq_hd/freq_overlay.png)
- [6K target high-frequency overlay](../lookcloser_debug_outputs/freq_6k_target_crop/freq_hd/high_frequency_overlay_L13_plus.png)
- [6K target stats](../lookcloser_debug_outputs/freq_6k_target_crop/freq_hd/freq_stats.json)

## Insights

The map is neither almost all minimum nor almost all maximum. The HD crop populates every level, and the 6K target crop populates levels 4-15 with only about 2% at max level.

The visual behavior matches the intended proxy: higher assigned levels concentrate around cables, rods, hard edges, and fine object details. Some brick microtexture is also classified as high frequency, so this remains a tuned detail detector rather than a semantic object-boundary detector.
