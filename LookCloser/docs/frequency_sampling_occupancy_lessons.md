# Frequency, sampling and occupancy: reusable findings

## What was tested

These recommendations come from measured native-crop experiments. Scene-specific recipes and full evidence live on [the Marshawn scene branch](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_final_results.md). They are not cross-scene guarantees or a new default recipe.

## Results and source-backed behavior

- Frequency levels select spatial resolution, not a fixed number of samples per ray. Step-size floors, coarse intervals and the shared full-ray budget can dominate the nominal frequency rule. Audit realized points and local detail rather than assuming more samples improve reconstruction.
- Distinguish preprocessing2D maps, the learned/runtime3D frequency grid, feature reweighting and adaptive traversal. A constant input map does not imply constant3D sampling. Audit units and resize/similarity invariance; normalized projection remains opt-in and legacy behavior remains the default. A unit-consistent formula alone did not establish better image quality in the recorded control.
- Feature-reweighting phase must restore on evaluation-only checkpoint loading, not only through training callbacks. Moving a switch that also resets RNG changes the whole event.
- Packed distortion global prefixes need FP64 to avoid cancellation across unrelated opaque rays; retain the original output dtype. Verify values and gradients against an independent pairwise oracle.
- Eval-only occupancy dilation must cover both constant-step and adaptive entry points, remain noncumulative across chunks, and restore exactly when training resumes.
- Preserve half-pixel conventions and crop principal-point shifts. Numeric AABB bounds require their world-to-box transform. The conservative mask-volume helper includes all cell corners, unknown visibility, stable selected identities and configurable margins; it is a bounding hypothesis, not physical empty-space truth.

## Insights and limits

- A lower aggregate masked LPIPS can conceal face/hair/boot regressions and exclude exterior floaters. Report PSNR/SSIM too, inspect native details and novel views, and keep numeric minima separate from visual selection. Reused validation views are not an untouched test.
- A contour ray hitting some occupied cell does not prove correct surface depth. Occlusion/out-of-FOV remain UNKNOWN. Clear-view votes, high EMA values or disconnected components are inspection candidates, not safe pruning labels. One-voxel expansion is not a universal repair.
- Match optimizer, source pixels, masks, calibration and schedules when attributing gains. A larger ray batch changes data exposure per step. Training and inference changes require separate comparisons.
- Each independent scene/frame starts cold; own resumes preserve field, optimizer, grid, counters and RNG together. Replayed work consumes the physical budget. Read a persistent cumulative counter once; sum only elapsed clocks that actually reset.
- End-to-end cost includes preparation and setup. Ordered single-pass inference timings and mixed-host clocks do not establish causal speedups. A single trajectory is not a variance estimate.
- Verify actual process identity and fresh progress rather than attributing stale progress from a previous worker. Stop admissions as well as workers; verify output/journal/stop-receipt permissions under the actual remote UID. Requested shutdown is not a negative model result.
- Still images do not establish fixed-model camera-motion stability, and that test would not establish flicker-free independently trained video frames. No claim of perfect occupancy or universal optimal parameters is supported.
