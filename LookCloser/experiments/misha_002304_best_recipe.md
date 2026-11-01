# Misha 002304: best measured masked-person recipe

## What was tested

Native HD LookCloser volume rendering, 66 TRAIN cameras and three held-out cameras (DD, HC, MD). This is the best measured **mean person ROI LPIPS** checkpoint as of 2026-11-01, not a claim of complete artifact removal or convergence. Background reconstruction and hybrid rendering are excluded.

The selected state is `screen_v11/world_sampling_fine875512`, step **75,512**. Its last stage resumes the complete migrated 75,000-step state for 512 updates, preserving model, Adam, scheduler, scaler and RNG. This is a recorded warm continuation, not a verified from-scratch reproduction of all preceding stages. Do not combine later experimental corrections into its recipe.

Exact original inputs and resolved configuration are archived in [recipes/misha_002304_75512](recipes/misha_002304_75512/request.json):

- [Request](recipes/misha_002304_75512/request.json) and [resolved config](recipes/misha_002304_75512/config.yml).
- [Checkpoint identity and SHA256](recipes/misha_002304_75512/checkpoint.json), [runtime/source hashes](recipes/misha_002304_75512/provenance.json), and [wrapper hashes](recipes/misha_002304_75512/external_wrapper_provenance.json).
- [Selected checkpoint measurements](recipes/misha_002304_75512/selected_metrics.json), [support policy](recipes/misha_002304_75512/person_occupancy_policy.json), and [density frequency policy](recipes/misha_002304_75512/density_frequency_policy.json). Other archived policy JSONs record masks, frequency priors, sampling, depth and appearance dropout.

### Exact final-stage settings

| Component | Recorded setting |
|---|---|
| Seed | 43 |
| RGB rays per batch | 4,096, FAS strength 1, `ignore_mask=True` |
| Additional observations | 64 measured foreground depth rays + 64 qualified photometric RGB rays |
| Appearance | 16-dimensional embedding; zero-conditioning dropout probability 0.5 |
| Density | Independent density encoding; density feature reweighting strength 0 |
| Colour feature reweighting | Strength 1 |
| Hash encoding | 16 levels, maximum resolution 8,192, log2 hash size 23 |
| Known-empty density supervision | Constant multiplier 0.1; unknown/occluded observations excluded |
| Foreground opacity | Multiplier 1, foreground-only, filtered by visibility |
| Soft matte supervision | Weight 0.1, foreground multiplier 1 |
| Depth | `interval_mass`, strength 10, multiplier 0.01; no depth-guided sampling |
| Distortion regularization | 0.01 |
| Occupancy/support | World-aligned 142 × 84 × 128; `x142_y84_halo1`; one-cell tolerance |
| Adaptive integration | Coarse step 0.000125, minimum step 0.0000125, maximum 8,192 samples/ray |
| Frequency state | Inherited 3D grid frozen throughout this continuation; source `person_native_v2` maps; SSIM threshold 0.95 |
| Optimizer | Inherited Adam; configured initial LR 0.01, final 0.0001, exponential schedule over 200,000 steps; no LR restart |

Integration distances are normalized scene units, not certified metres. The FAS warmup and ramp are zero **for this continuation**; its request does not prove that FAS or frequency weighting was active from step zero of the complete ancestry. `adaptive_warmup_steps=4096` is already past at the resumed absolute step.

### Spatial coordinates and supervision

Keep the original encoding/frequency AABB when restoring this warm model:

```text
lower = [-0.0938142166, -0.0267502367, -0.0811333433]
upper = [ 0.2117285430,  0.2297806889,  0.1682692170]
```

The query/occupancy AABB differs; the original world-cell spacing is preserved:

```text
lower = [-0.0938142240, -0.0267502442, -0.0811333358]
upper = [ 0.2451472832,  0.1415981904,  0.1682692170]
```

This extends real trouser space along X and excludes the large traced phantom along Y. The source-derived support has 331,103 allowed cells, preserves the measured 3-sigma point envelopes, and passes the artificial full-TRAIN opaque-ray capacity gate with zero impossible pixels. Capacity is not a certificate of complete learned anatomy or correct hidden depths.

Training uses `world_aligned_domain_v1/data` and its matching confidence/support/depth manifests on silver-sable. Official evaluation uses `person_native_sdr_v5_KD_primary`. The recorded generation includes reviewed visibility corrections, the primary-actor KD correction and native premultiplied SDR targets. Later V8/V11 source repairs and source-only fresh initializations are **not** part of this leader.

Restore optional runtime support/density policies explicitly when rendering; loading checkpoint tensors alone does not install every wrapper hook. Frozen engine: `/home/ubuntu/lookcloser/campaigns/misha_hd_20261029/train_code_v24`. The exact wrapper hash matches `screen_code_v20/train_misha_mask_refinement.py` (also v21). Use these archived snapshots and their dependencies rather than assuming current `main` reproduces this checkpoint. This commit documents the recipe; it does not promote the experimental implementation or bundle the dataset/model/runtime.

## Results

| Eval camera | PSNR | SSIM | Whole-image LPIPS | Person ROI LPIPS |
|---|---:|---:|---:|---:|
| DD | 32.370880 | 0.903829 | 0.104997 | 0.182839 |
| HC | 29.943581 | 0.877804 | 0.136645 | 0.217571 |
| MD | 27.090361 | 0.892024 | 0.125391 | 0.263227 |
| Mean | 29.801607 | 0.891219 | 0.122344 | **0.221212** |

The fixed per-camera person ROI rectangles are in the request. Replays differ below 1e-6 LPIPS. The user prioritizes person LPIPS; the original automatic PSNR selector instead selected step 75,000 and has not been relabelled.

Step 75,512 was a preset stage endpoint, not a validation plateau stop. Person LPIPS improves 0.223101 → 0.222103 → 0.221212 at steps 75,000 / 75,256 / 75,512. An unchanged-training continuation reaches 76,024 with 0.221448 (75,768: 0.223364). These measurements do not establish full convergence. Later longer runs changed settings.

Full native reference images on olive-lark:

- [DD](/fsx/lookcloser_misha_maskfix_20261031_pavlo/best_eval_person_lpips_20261101/best_DD_HD.png), 1920 × 1013.
- [HC](/fsx/lookcloser_misha_maskfix_20261031_pavlo/best_eval_person_lpips_20261101/best_HC_HD.png), 1920 × 1013.
- [MD](/fsx/lookcloser_misha_maskfix_20261031_pavlo/best_eval_person_lpips_20261101/best_MD_HD.png), 1920 × 1012.

Render hashes are preserved in [render_provenance.json](recipes/misha_002304_75512/render_provenance.json). Original checkpoint on silver-sable:

```text
/cache/ubuntu/lookcloser/misha_maskfix_20261031/screen_v11/world_sampling_fine875512/field/lookcloser/run/nerfstudio_models/step-000075512.ckpt
```

## Insights and open defects

The [floater report](misha_floater_prevention_lessons.md) distinguishes measured interventions from the proposed preventive recipe. Broad volume restriction removes the traced large phantom; consistent visibility labels prevent specific holes; finer integration improves LPIPS. None alone guarantees a clean model on every view. Hard post-training clipping that creates new body holes is rejected.

On 2026-11-01 the user reports that most floaters have disappeared, but `best_DD_HD.png` still has **white haze beside the right/nearest arm and tiny white spots at the edge/fold of the same sleeve toward the back of the shoulder**. Record these as open user-reported defects; the cause is unverified. Earlier AB debris and a small novel-view forearm notch also prevent an artifact-free acceptance claim.

**Hold:** do not start another improvement series until the user specifies the next action. No new training or corrective experiment was launched to prepare this recipe.
