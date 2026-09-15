# View-consistent head texture: diagnosis and dynamic replay

## What was tested

Following the [exact train-pose workaround](dec5_central_train_pose_workaround.md),
separate real mesh holes from hard texture-source seams before changing the
camera trajectory. Three matched layer diagnostics raycast the unchanged mesh,
show its clay surface, source labels and boundaries, and mark recently appended
head faces. Long nose/jaw/neck lines in the inspected cases occur on old,
continuous geometry and track source changes; they are not the new cap boundary.
Crown notches and isolated under-jaw missing pixels are a separate problem.

Four train/moving-view canaries at actual times 001123 and 001193 compare:

- softened surface-incidence penalty (power 2 instead of 8), with the existing
  target-angle prior also applied to pixel fallback;
- pure target-angle quality, also disabling graph color cost (multi-part control);
- pure target-angle per-pixel source choice instead of the face-centroid label;
- power 2 with static image registration disabled.

All retain the same mesh, depth visibility gates, foreground evidence, native
texture footprint and fixed, mean-centered camera color profile. There is no
RGB source averaging, per-time exposure adjustment, or held-out RGB in rendering.
Depth arrays are byte-identical across these texture-only controls.

One held-out camera at 001193 uses the existing frozen GT-only face ROI. This
benchmark is used for variant selection: it is **not independent evidence of
improvement across 150 actor times**. No full-frame quality metrics or loss.

## Results

| 001193 variant | Face PSNR | Face SSIM | Face LPIPS |
|---|---:|---:|---:|
| Matched baseline | 30.27655 | 0.939692 | 0.076124 |
| Incidence power 2 + pixel angle prior | 30.78402 | 0.942037 | 0.072286 |
| Pure angular graph control | 30.45028 | 0.939945 | 0.083334 |
| Pure angular per-pixel choice | 30.45114 | 0.940079 | 0.083458 |
| Power 2 + zero image registration | 30.85838 | 0.943257 | 0.072251 |

The selected texture control gains 0.582 dB / 0.00357 SSIM and lowers LPIPS by
0.00387 (about 5.1%) relative to baseline. Disabling registration adds only
0.000035 LPIPS improvement over power 2: do not attribute the whole gain to it.
Pure angular policies improve self-view source selection but worsen held-out
LPIPS. The source most similar to the virtual camera is not automatically the
best source for every reconstructed surface point.

Native self-projection diagnostics independently reproduce source-mask and
four-tap depth rejection. In 001193 G/C, 9,014 visible pixels fail the footprint
check, with 11 in the diagnostic nose box; 8,942 have a zero source mask. In
001123 K/C, the corresponding counts are 23,027 / 1,187 and 22,642. These are
visibility diagnostics, not face quality metrics. Most native projection error
is about 0.001 pixel, but rare tails reach 0.17 pixel; no broader snapping or
visibility relaxation was introduced.

Evidence:

- [Layer diagnostics](/mnt/data/dec5_head_seam_layers).
- [Source-quality controls](/mnt/data/dec5_view_consistent_head_texture).
- [Per-pixel angular control](/mnt/data/dec5_pixel_angular_head_texture).
- [Held-out scores](/mnt/data/dec5_head_source_quality_heldout/metrics.json).
- [Registration-off scores and crops](/mnt/data/dec5_unwarped_head_texture).
- [Native self-visibility diagnostics](/mnt/data/dec5_native_self_visibility).
- [Full dynamic candidate](/mnt/data/dec5_incidence2_unwarped_dynamic_150).

Full-sequence evaluation uses 150 distinct actor times and unchanged meshes,
the exact previous phase+30 elevated periodic camera path, fixed virtual lens,
and no crop, stabilization or post-render image translation. Six disjoint
workers on clever-shadow completed all 150 in 721.7 seconds (about 12 minutes),
with all worker exit codes zero. This is render throughput with existing meshes,
not a claim that PatchMatch reconstruction itself took 12 minutes.

The independent audit verifies 150 unique meshes/renders/times, unchanged path,
31.51 degrees of viewing-angle span, projected fixed-landmark travel of
384 x 153.57 pixels, and actual foreground-centroid travel of 301.40 x 150.35
pixels. Camera loop step max/min is 1.01238. Six fresh depth raycasts verify
that actual rendered cameras match the saved poses. The normal-speed MP4 has
150 frames, 1080 x 1920, 24 fps, duration 6.25 seconds; no slow version.

The main agent actually inspected all 15 overview sheets, all 25 native
jaw/lipstick sheets, and all 15 sheets decoded from the encoded MP4. Reviews
were recorded incrementally with render and image hashes. Overview groups
060-079 are explicitly failed for major hand breakup; the other 130 frames
are reviewed with known residual artifacts, **not 130 artifact-free passes**.
Important remaining defects:

- 001029-001045: torn forearm, palm and fingers during hand lowering; most severe
  around 001037-001043, visible even in the encoded overview.
- 000995-001007: skin-colored rear fin/block behind lipstick in native crops.
- 001123 and neighboring times: crown notches and a neck source-color seam.
- 001191/001193/001195: isolated small black fleck beneath the jaw persists.
- Many times: thin jaw/source lines, brown ragged hair rim, open lower torso.

The movie is an inspected intermediate texture improvement with known geometry
failures, **not an artifact-free approval or an algorithmic mesh repair**.
Twelve focused tests passed (quality formula, pixel fallback, zero registration,
native footprint and fail-closed publication checks).

Download the candidate and its ordered PNG frames:

```bash
scp ubuntu@clever-shadow:/mnt/data/dec5_incidence2_unwarped_dynamic_150/video.mp4 ~/Downloads/
scp ubuntu@clever-shadow:/mnt/data/dec5_incidence2_unwarped_dynamic_150/frames.zip ~/Downloads/
```

## Insights

The earlier incidence power 8 can favor a more front-facing source over a
closer target-direction source. Softening that competition is a measured local
improvement. Eliminating the geometric incidence term entirely is worse on
the held-out benchmark. A face-centroid label can also remain valid while
some pixels would prefer another camera; changing to per-pixel choice alone
does not establish a fidelity gain.

Camera avoidance remains an allowed temporary workaround, but the 45 exact
train-pose probes did not establish a clean combined head/hand path. This replay
therefore preserves visible motion rather than claiming an untested path fix.
True crown/hand holes require further geometry work; a texture-only change
cannot close them. Existing single-frame/model defaults remain unchanged.
