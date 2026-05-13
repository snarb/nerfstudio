# LookCloser / FA-NeRF implementation notes

## Status

This repository contains an experimental Nerfstudio implementation of LookCloser / FA-NeRF: a frequency-aware NeRF pipeline intended to preserve both low-frequency scene structure and high-frequency tiny details in a single model.

The implementation should be treated as a research prototype, not as a verified reproduction. Several parts are implemented as best-effort approximations of the paper and require empirical validation.

---

## 1. High-level idea

LookCloser / FA-NeRF is based on the assumption that the required 3D frequency of scene content can be estimated from the frequency needed to reconstruct corresponding 2D image patches.

The intended pipeline is:

1. Estimate 2D patch frequency from input images.
2. Project 2D frequencies into 3D using focal length and depth.
3. Store the resulting frequency levels in a 3D frequency grid.
4. During NeRF training, use this grid to:
   - re-weight hash-grid feature levels;
   - guide pixel sampling toward high-frequency regions;
   - adapt ray-marching step sizes in high-frequency areas.

The paper describes this as a patch-based 3D frequency quantification method combined with a frequency grid, feature re-weighting, frequency-aware sampling, and adaptive ray marching.

---

## 2. Implemented components

### 2.1. 2D frequency preprocessing

File:

```text
nerfstudio/scripts/lookcloser_preprocess.py
```

Implemented idea:

- For each training image, train a 2D Instant-NGP-style coordinate regression model.
- Split the image into patches.
- For each patch, render it using progressively more hash-grid levels.
- Assign the patch the minimum level whose reconstruction passes an SSIM threshold.
- Save a patch-wise frequency map as:

```text
<scene>/lookcloser_frequencies/<image_stem>.pt
```

Important implementation detail:

- The current saved value is a scalar frequency/resolution value, not necessarily a raw discrete level.
- Downstream code converts it back to a level using `log(freq / min_res) / log(b)`.

Expected output shape:

```python
freq_map.shape == (image_height // patch_size, image_width // patch_size)
```

approximately, depending on exact divisibility.

---

### 2.2. Frequency grid manager

File:

```text
nerfstudio/model_components/lookcloser_grid.py
```

Implemented idea:

- Maintain a dense 3D voxel grid of frequency levels.
- Convert world-space positions into voxel indices.
- Query frequency level at a 3D point.
- Update grid cells using max-reduction.
- Convert between scalar frequency and discrete frequency level.

Main methods:

```python
freq_to_level(...)
level_to_freq(...)
query(...)
update_max(...)
initialize_from_sparse(...)
update_step(...)
```

Intended role:

- Initialize the grid from sparse SfM points and their 2D observations.
- Update the grid during training from rendered depth and corresponding 2D patch frequency.

Current risk:

- The code contains an `initialize_from_sparse(...)` method, but it is not clearly wired into the training pipeline in the inspected implementation.
- If this method is never called, the frequency grid starts from zeros and the model effectively begins as a low-frequency-only grid until runtime updates change it.

---

### 2.3. Frequency-aware field

File:

```text
nerfstudio/fields/lookcloser_field.py
```

Implemented idea:

- Use a 3D hash-grid encoding similar to Instant-NGP.
- Query the 3D frequency grid for each sample point.
- Re-weight hash-grid feature levels depending on the queried frequency level.
- Pass weighted features through tiny MLPs for density and color.

Main idea of re-weighting:

- Levels at or below the queried frequency are kept.
- Levels above the queried frequency are down-weighted with a smooth factor.

Current risk:

- The formula is implemented as a best-effort approximation of the paper, but the exact matching of symbols and behavior needs validation.
- The field code must be checked for missing imports and compatibility with Nerfstudio Field APIs.

---

### 2.4. Frequency-averaged pixel sampler

File:

```text
nerfstudio/lookcloser_pixel_sampler.py
```

Implemented idea:

- Load precomputed patch-wise frequency maps.
- Bucket patches by frequency level.
- Sample pixels by first sampling frequency buckets, then choosing random pixels inside selected patches.
- Use a ramp from low to high frequency levels so high-frequency regions are sampled more often.

Current risk:

- The sampler assumes `patch_size = 32` internally.
- If preprocessing uses another patch size, sampling becomes inconsistent.
- The sampler path may not match Nerfstudio’s expected module path: the file says `nerfstudio/lookcloser_pixel_sampler.py`, but the comment says `nerfstudio/data/pixel_samplers/lookcloser_pixel_sampler.py`.
- It is not clear that this sampler is actually connected to the datamanager config used by the `lookcloser` method.

---

### 2.5. LookCloser model

File:

```text
nerfstudio/models/lookcloser.py
```

Implemented idea:

- Defines `LookCloserModelConfig`.
- Creates `FrequencyGridManager`.
- Creates `LookCloserField`.
- Implements an adaptive ray marching loop.
- Uses Charbonnier RGB loss, distortion loss, and optional sparse depth loss.

Current risk:

- The adaptive ray marching loop is custom and likely the highest-risk part of the implementation.
- The code calls `self.field.query_points(...)`, but the inspected `LookCloserField` defines `get_density(...)` and `get_outputs(...)`, not an obvious `query_points(...)` method.
- This suggests the model may not run without additional missing code or patches.
- It may bypass standard Nerfstudio sampling/rendering assumptions.

---

### 2.6. LookCloser pipeline

File:

```text
nerfstudio/pipelines/lookcloser_pipeline.py
```

Implemented idea:

- Extends `VanillaPipeline`.
- Loads precomputed 2D frequency maps into memory.
- Periodically updates the 3D frequency grid using rendered depth and image patch frequency.

Current risk:

- Runtime grid update assumes `stride = 32` when converting pixels to frequency-map indices.
- If preprocessing patch size differs from 32, grid updates use wrong patch frequencies.
- The pipeline update samples random pixels, not necessarily patch centers. The paper’s supplementary description says runtime update renders the depth of the center pixel of a training patch.
- It is unclear whether the model’s frequency grid is initialized from sparse SfM points before runtime updates.

---

### 2.7. Method config

File:

```text
nerfstudio/configs/method_configs.py
```

Implemented idea:

- Registers a `lookcloser` method.
- Uses `LookCloserPipelineConfig` and `LookCloserModelConfig`.
- Sets grid resolution to 128 and number of frequency levels to 16.

Current risk:

- The config uses the vanilla datamanager and does not obviously configure the custom LookCloser pixel sampler.
- If the custom sampler is not wired in, FAS is effectively not used during training.

---

## 3. What is definitely incomplete or uncertain

### 3.1. Sparse SfM initialization may not be active

The paper initializes the 3D frequency grid from sparse SfM points and their observations. The inspected code contains a method for this, but the training pipeline does not clearly call it.

Validation required:

- Add logging to confirm whether `FrequencyGridManager.initialize_from_sparse(...)` is called.
- Log number of sparse points, observations, and touched voxels.
- Visualize non-zero grid occupancy after initialization.

Expected:

- Before training updates, the grid should already contain meaningful non-zero frequency levels around sparse scene points.

Red flag:

- Grid histogram is almost all zero after initialization.

---

### 3.2. The pixel sampler may not be used

The code has a LookCloser sampler, but the method config appears to use a vanilla datamanager without explicitly installing that sampler.

Validation required:

- Print sampler class during training.
- Confirm that `LookCloserPixelSampler._initialize_buckets(...)` is called.
- Confirm non-empty buckets per frequency level.

Expected:

- Training should report bucket counts.
- High-frequency buckets should receive proportionally more samples according to the configured ramp.

Red flag:

- Training uses the default Nerfstudio sampler.
- No bucket initialization logs appear.

---

### 3.3. Patch size is hard-coded in downstream modules

Preprocessing exposes `patch_size`, but downstream code assumes `stride = 32` or `patch_size = 32`.

Validation required:

- Either force preprocessing to always use 32.
- Or save metadata next to each frequency map:

```json
{
  "patch_size": 32,
  "stride": 32,
  "min_res": 16,
  "max_res": 2048,
  "num_levels": 16
}
```

Then load metadata in sampler and pipeline.

Current recommendation:

- Do not change `patch_size` from 32 until metadata propagation is implemented.

---

### 3.4. Frequency map stores scalar frequency, but grid stores levels

Preprocessing saves scalar resolutions. The grid and sampler often operate on discrete levels.

Validation required:

- Check round-trip:

```python
level == freq_to_level(level_to_freq(level))
```

for every level.

- Check that frequency maps contain values corresponding exactly to known level resolutions.

Red flag:

- Frequency maps contain arbitrary values not matching the geometric level schedule.

---

### 3.5. Progressive 2D regression may not match the paper exactly

The paper defines 2D frequency as the minimum frequency where the patch reconstruction passes SSIM. The inspected script trains one full 2D model and then masks levels during patch evaluation. This is not strictly equivalent to progressive training where higher-frequency components are opened during optimization.

Validation required:

- Compare two variants on the same image:
  1. train all levels once, then mask at evaluation;
  2. progressively train with levels opened stage by stage.
- Visualize patch reconstructions and frequency maps.

Expected:

- Frequency maps should be broadly similar, but progressive training may produce more meaningful lower-level reconstructions.

Current recommendation:

- Prefer progressive training with masking during training.

---

### 3.6. Adaptive ray marching is likely not API-compatible yet

The inspected `LookCloserModel.adaptive_ray_marching(...)` calls a field method that is not clearly implemented in the inspected `LookCloserField`.

Validation required:

- Run a minimal `ns-train lookcloser` for a few iterations.
- Confirm no attribute errors.
- Confirm output tensor shapes match Nerfstudio expectations.
- Confirm loss values are finite.

Red flag:

- `AttributeError: 'LookCloserField' object has no attribute 'query_points'`.
- Shape mismatch in renderer/loss.
- NaNs in density, alpha, depth, or distortion loss.

---

### 3.7. Depth handling is ambiguous

The paper uses sparse depth supervision early in training and uses rendered depth for runtime frequency-grid updates. The current code contains optional sparse depth loss, but it is unclear whether the dataset actually provides `depth_image` and whether sparse SfM depth is converted into the batch.

Validation required:

- Check whether `batch` contains `depth_image`.
- Check how many valid depth pixels exist.
- Check whether `depth_loss` is active for the intended early steps.

Red flag:

- `depth_loss_mult > 0` but no depth supervision is ever present.

---

## 4. High-level validation plan

Use this order. Do not jump directly to full training.

---

### Step 1: Import and CLI sanity

Goal:

- Ensure all modules import.
- Ensure CLI entrypoints exist.

Expected:

- All commands succeed.



---

### Step 2: 2D overfit sanity check

Goal:

- Confirm the 2D preprocessing model can reconstruct a small crop.

Procedure:

- Use a 256 or 512 crop.
- Train the 2D NGP on one image.
- Render full crop reconstruction.
- Save:
  - `gt.png`
  - `recon_full.png`
  - `diff.png`

Expected:

- `recon_full.png` visually resembles `gt.png`.
- `diff.png` is mostly dark.
- PSNR/SSIM are reasonably high for a small crop.

Red flags:

- Reconstruction is transposed, flipped, shifted, or very blurry.
- This indicates UV or coordinate bugs.

---

### Step 3: Progressive patch sanity check

Goal:

- Confirm that increasing frequency levels progressively improves patches.

Procedure:

- Pick random patches.
- Render them at levels such as `0, 2, 4, 8, 12, 15`.
- Save `patch_mosaic.png`.

Expected:

- Low levels show coarse color/shape.
- Higher levels recover finer details.
- SSIM generally increases with level.

Red flags:

- Level 0 already reconstructs high detail.
- Level 15 is not better than level 0.
- SSIM is flat or chaotic.

---

### Step 4: Frequency map visual validation

Goal:

- Confirm 2D frequency maps are semantically plausible.

Procedure:

- Save heatmaps and image overlays.
- Inspect multiple images.

Expected:

- Smooth walls/sky/background: low frequency.
- Text, fine texture, wires, leaves, edges: high frequency.
- Histogram is not collapsed to one level.

Red flags:

- Almost all patches are max frequency.
- Almost all patches are min frequency.
- Frequency map is random-looking noise.
- High-frequency regions appear on blank areas.

Likely fixes:

- If all max: lower SSIM threshold, increase steps, increase max_res.
- If all min: raise SSIM threshold, verify SSIM implementation, verify GT/pred patch alignment.

---

### Step 5: Frequency-map compatibility checks

Goal:

- Confirm downstream consumers interpret frequency maps correctly.

Checks:

```python
import torch, numpy as np

f = torch.load('lookcloser_frequencies/image.pt')
print(f.shape, f.min(), f.max(), f.float().mean())
print(torch.unique(f).numel())
```

Expected:

- Values are within `[min_res, max_res]`.
- Unique values correspond to the discrete level schedule.
- Map shape matches the intended patch grid.

Red flags:

- Values outside expected range.
- Shape does not match image size / patch size.
- Downstream code assumes patch size 32 but preprocessing used another size.

---

### Step 6: Sampler validation

Goal:

- Confirm FAS is active and samples high-frequency buckets more often.

Procedure:

- Add logs to sampler initialization.
- Print bucket sizes per level.
- Print actual sampled counts per level for several training batches.

Expected:

- Non-empty buckets for at least several levels.
- Batch composition follows configured ramp.

Red flags:

- Sampler never initializes.
- Most buckets are empty because freq maps collapsed.
- Method config still uses default sampler.

---

### Step 7: Frequency grid initialization validation

Goal:

- Confirm 2D frequencies are projected into 3D using sparse SfM points.

Procedure:

- Verify sparse points and observations are available.
- Call or confirm call to `initialize_from_sparse(...)`.
- Save grid histogram before and after initialization.

Expected:

- Non-zero frequency levels appear in voxels touched by sparse points.
- More detailed / closer regions should often receive higher levels.

Red flags:

- No sparse observations.
- Grid remains all zeros.
- Frequencies are extreme due to wrong depth/focal scaling.

---

### Step 8: Minimal training smoke test

Goal:

- Confirm `ns-train lookcloser` runs.

Procedure:

- Run 100–500 iterations.
- Disable expensive features one by one if needed.
- Log loss values and output shapes.

Expected:

- Forward pass succeeds.
- Loss finite.
- RGB output shape matches batch image shape.
- Depth values finite and positive.

Red flags:

- Attribute errors in field/model boundary.
- NaNs in alpha, depth, rgb, or distortion loss.
- Ray marching loop extremely slow or never terminates.

---

### Step 9: Ablation validation

Goal:

- Confirm each component has the expected directional effect.

Run variants:

1. Baseline Instant-NGP / nerfacto.
2. LookCloser with only frequency preprocessing, no FAS, no adaptive RM.
3. + feature re-weighting.
4. + FAS.
5. + adaptive ray marching.
6. Full model.

Expected:

- High-frequency crops should improve most with full model.
- Low-frequency structure should not degrade severely.

Red flags:

- Full model worse everywhere.
- Improvements only come from increased capacity, not frequency-aware logic.
- High-frequency detail improves but geometry collapses.

---

## 5. Recommended initial hyperparameters

For debugging:

```text
crop_size: 256 or 512
steps_per_image: 3000-6000
patch_size: 32
ssim_threshold: 0.95
min_res: 16
max_res: 4096 for HD/4K debug
log2_hashmap_size: 19
```

For full 4K images:

```text
steps_per_image: 6000-10000
patch_size: 32
ssim_threshold: 0.93-0.96
min_res: 16
max_res: 4096 or 8192
log2_hashmap_size: 19 or 21/23 for large scenes if VRAM allows
```

Important:

- Keep `patch_size = 32` until metadata propagation is fixed.
- Tune `ssim_threshold` visually using patch mosaics.
- Do not trust one scalar metric for frequency maps; inspect overlays.

---

## 6. Known high-risk implementation areas

### Highest risk

1. `LookCloserModel.adaptive_ray_marching(...)`
2. `LookCloserField` API compatibility
3. Sparse SfM frequency-grid initialization
4. FAS sampler integration into Nerfstudio datamanager
5. Patch-size metadata consistency

### Medium risk

1. SSIM implementation and threshold calibration
2. Full-level training vs true progressive training in preprocessing
3. Frequency scalar vs frequency level conversions
4. Hash table size and collisions for large scenes
5. Runtime grid update using random pixels instead of patch centers

### Lower risk

1. Basic frequency map saving/loading
2. Level-to-frequency geometric schedule
3. Simple patch-wise bucketing if patch size is fixed at 32

---

## 7. Things I am specifically not confident about

1. Whether the current `LookCloserModel` actually runs with the inspected `LookCloserField`, because the model appears to call a method not defined in the field.
2. Whether `FrequencyGridManager.initialize_from_sparse(...)` is ever called in the actual training path.
3. Whether the custom pixel sampler is actually used by the registered `lookcloser` method.
4. Whether the current adaptive ray marching implementation is mathematically and API-wise compatible with Nerfstudio’s expected `RaySamples`, loss, and renderer behavior.
5. Whether the scale of `f3D = f2D * focal / depth` is consistent with the scene normalization and hash-grid level schedule.
6. Whether saved 2D frequency maps should store scalar frequency, discrete level, or both.
7. Whether the SSIM threshold `0.95` is appropriate for this exact SSIM implementation, patch size, and training schedule.
8. Whether the current implementation handles variable-resolution images safely across preprocessing, sampler, pipeline, and cameras.
9. Whether the frequency re-weighting formula exactly matches the intended Eq. 6 behavior in the paper.
10. Whether high-frequency improvements, if observed, come from the intended LookCloser mechanisms rather than simply from higher capacity or denser sampling.

---

## 8. Minimum acceptance criteria before trusting results

The implementation should not be considered valid until all of the following are true:

1. 2D overfit reconstruction works on a crop.
2. Progressive patch mosaics improve visibly with level.
3. Frequency overlays are semantically plausible.
4. Frequency map histograms are non-degenerate.
5. FAS sampler is confirmed active or explicitly disabled for ablation.
6. Frequency grid is confirmed non-zero after initialization or after early updates.
7. `ns-train lookcloser` runs for at least 500 iterations without NaNs.
8. Baseline comparison exists against a standard Nerfstudio method on the same data.
9. At least one high-frequency crop improves qualitatively and quantitatively.
10. Low-frequency/global structure does not regress severely.

---

## 9. Suggested debug artifacts to save

For preprocessing:

```text
gt.png
recon_full.png
diff.png
freq_heatmap_patch_grid.png
freq_heatmap_fullres.png
freq_overlay.png
patch_mosaic.png
stats.json
```

For training:

```text
frequency_grid_histogram_step_*.json
sampled_frequency_level_histogram_step_*.json
rendered_rgb_step_*.png
rendered_depth_step_*.png
high_freq_crop_comparison_step_*.png
```

For ablations:

```text
metrics.csv
crop_psnr_ssim_lpips.csv
qualitative_grid.png
```

---

## 10. Practical execution order

Recommended order for real validation:

1. Fix import/runtime errors.
2. Run 2D crop overfit.
3. Validate progressive patch mosaic.
4. Generate frequency maps for 1-3 images with overlays.
5. Run full preprocessing on a small dataset subset.
6. Confirm sampler integration.
7. Confirm frequency grid initialization/update.
8. Run 500-step training smoke test.
9. Run short ablations.
10. Run full training only after the above passes.

---

## 11. Current interpretation of implementation maturity

The implementation has the right high-level structure:

- 2D patch frequency estimation exists.
- Frequency grid class exists.
- Frequency-aware field exists.
- Pixel sampler exists.
- Pipeline update exists.
- Method config exists.

But the implementation is not yet proven correct end-to-end. The most likely current state is:

```text
preprocessing: partially implemented, needs progressive/debug validation
frequency grid: implemented, but initialization wiring uncertain
field: conceptually implemented, API compatibility uncertain
sampler/FAS: implemented, integration uncertain
adaptive ray marching: implemented, high risk
full training: requires smoke tests before trust
```

Treat this as a prototype that needs staged validation, not as a complete reproduction.

