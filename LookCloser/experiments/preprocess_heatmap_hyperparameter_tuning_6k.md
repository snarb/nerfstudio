# LookCloser 6K Heatmap Hyperparameter Tuning

## What was tested

Tuned direct-image debug preprocessing on the 6K crop with visible tiny details:

- Source: `/home/ubuntu/repos/look-closer/E004_D014_graded.png`
- Crop: `[x=2560, y=1152, w=512, h=512]`
- Detail types: cable loops, dark cable bundle, horizontal metal rods, small bright labels, brick texture, hard dark/light edges.

Compared the previous patch_size=16 candidate against finer patch sizes:

- `patch_size=8`, `ssim_threshold=0.97`, `ssim_window_size=7`, `high_frequency_level=12`
- `patch_size=8`, `ssim_threshold=0.97`, `ssim_window_size=7`, `high_frequency_level=13`
- `patch_size=8`, `ssim_threshold=0.98`, `ssim_window_size=7`, `high_frequency_level=13`
- `patch_size=8`, `ssim_threshold=0.985`, `ssim_window_size=7`, `high_frequency_level=13`
- `patch_size=12`, `ssim_threshold=0.975`, `ssim_window_size=11`, `high_frequency_level=13`

All tuning runs used `train_steps_per_level=140-160`, `train_batch_size=8192`, and debug levels `[0,2,4,8,12,15]`.

## Results

| Run | Patch grid | Non-empty levels | L12+ frac | L13+ frac | Max-level frac | Visual read |
|---|---:|---:|---:|---:|---:|---|
| Previous patch_size=16, threshold 0.97 | 32x32 | 9 | 0.272 | 0.084 | 0.014 | Better than patch32, but patches are still chunky around thin cables. |
| patch_size=8, threshold 0.97, L12+ mask | 64x64 | 13 | 0.308 | 0.115 | 0.017 | Good detail resolution, but L12+ includes too much brick texture. |
| **FINAL patch_size=8, threshold 0.97, L13+ mask** | **64x64** | **13** | **0.402** | **0.125** | **0.026** | Best balance: thin structures and hard edges are highlighted without collapsing to max level. |
| patch_size=8, threshold 0.98, L13+ mask | 64x64 | 10 | 0.802 | 0.528 | 0.170 | Too strict; large parts of brick/background become high frequency. |
| patch_size=8, threshold 0.985, L13+ mask | 64x64 | 9 | 0.941 | 0.834 | 0.566 | Failed; assignment collapses toward max level. |
| patch_size=12, threshold 0.975, L13+ mask | 42x42 | 12 | 0.529 | 0.228 | 0.063 | Less precise than patch8 and still over-highlights brick texture. |

Final selected artifacts:

- Tuned high-frequency overlay on the real 6K crop: [high_frequency_overlay_L13_plus.png](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/high_frequency_overlay_L13_plus.png)
- Absolute assigned-level overlay: [level_overlay.png](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/level_overlay.png)
- Diagnostic quantile overlay: [level_overlay_quantile.png](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/level_overlay_quantile.png)
- Absolute heatmap: [level_heatmap.png](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/level_heatmap.png)
- Histogram: [freq_histogram.png](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/freq_histogram.png)
- Stats: [stats.json](preprocess_heatmap_tuning_artifacts/FINAL_6k_wires_ps8_thr097_win7_L13/overfit_hd/stats.json)

Final selected stats:

- Histogram: L3:1, L4:3, L5:9, L6:14, L7:33, L8:54, L9:189, L10:676, L11:1469, L12:1137, L13:308, L14:95, L15:108.
- L13+ fraction: `0.125`
- Max-level fraction: `0.026`
- Debug SSIM by level: L0 `0.567`, L2 `0.739`, L4 `0.771`, L8 `0.901`, L12 `0.979`, L15 `0.984`
- Full-crop SSIM: `0.984`

## Insights

The tuned map is better, but it is still a proxy, not a perfect physical frequency detector. The best visual match came from using a fine patch grid (`patch_size=8`) and displaying the high-confidence subset as L13+, while keeping the absolute `level_heatmap.png` unchanged.

The threshold should not be pushed above `0.97` on this crop with the current short training budget. At `0.98` and especially `0.985`, unresolved patches increasingly fall into the max level, and the map starts to mean "failed to reach strict SSIM" rather than "real high-frequency detail".

The preprocessing/debug defaults now use:

```bash
--patch-size 8 --ssim-window-size 7 --ssim-threshold 0.97 --high-frequency-level 13
```

For downstream preprocessing, this remains a tuned 6K default candidate that should be revalidated on additional 6K crops, because brick microtexture is still sometimes treated as high frequency.
