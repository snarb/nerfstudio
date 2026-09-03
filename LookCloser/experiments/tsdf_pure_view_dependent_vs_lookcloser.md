# TSDF pure view-dependent texture mapping versus LookCloser

## What was tested

The experiment asks whether the current final renderer needs a NeRF at all, or
whether a continuous TSDF actor surface plus calibrated view-dependent texture
mapping is sufficient.

All variants use DEC5 frame `000899`, held-out camera
`nearest_eval_00000`, the same GLOMAP cameras, one continuous TSDF surface,
depth visibility tolerance `0.01`, and camera-local weights proportional to
`1 / camera_distance^4`. Held-out RGB is never a prediction input. No image,
person, face, or background mask, U-Net, LPIPS training loss, or appearance
embedding is used.

The compared renderers are:

1. **LookCloser base**: the existing 62-camera volumetric prediction without
   surface detail transfer.
2. **Pure TSDF mapping, no NeRF**: direct RGB reprojection through the TSDF
   surface. The operational result uses 16 sources to give this baseline its
   best measured visibility coverage.
3. **LookCloser + high-frequency transfer**: the current final renderer. Two
   camera-local sources provide the sigma-4 high-pass band; LookCloser supplies
   low-frequency colour and visibility fallback.

No metric below includes the whole image or room. The score mask is the
intersection of target TSDF first-hit support with one of three rectangles:
actor plus held object `[0, 150, 1400, 1030]`, face
`[687, 392, 1187, 892]`, and lipstick plus hand
`[687, 540, 987, 800]`. Missing texture inside the selected TSDF surface is
penalized. For masked PSNR only selected pixels are used; masked SSIM and LPIPS
use the tight bounds with identical black outside the geometry mask.

## Results

### Operational comparison, including holes inside the actor surface

| Region | Renderer | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Actor + held object | LookCloser base | **28.5329** | **0.778193** | 0.250530 |
| Actor + held object | Pure TSDF, 16 sources | 23.7410 | 0.767764 | 0.124139 |
| Actor + held object | LookCloser + HF transfer | 27.8191 | 0.773981 | **0.107312** |
| Face | LookCloser base | **27.9378** | 0.760846 | 0.248095 |
| Face | Pure TSDF, 16 sources | 23.7384 | 0.757321 | 0.119924 |
| Face | LookCloser + HF transfer | 27.1958 | **0.763327** | **0.106495** |
| Lipstick + hand | LookCloser base | 28.0233 | 0.841993 | 0.189282 |
| Lipstick + hand | Pure TSDF, 16 sources | 23.5937 | 0.866634 | 0.083570 |
| Lipstick + hand | LookCloser + HF transfer | **28.4077** | **0.873832** | **0.079649** |

The pure renderer covers `98.04%` of the target TSDF surface with 16 sources.
Increasing it from 2 to 16 sources improves actor LPIPS from `0.132516` to
`0.124139`; lipstick/hand LPIPS improves from `0.091381` to `0.083570`.

### Causal comparison on identical two-source visible support

This table excludes only pixels that neither of the same two source cameras can
see. It isolates colour representation from disocclusion filling.

| Region | Pure TSDF mapping | LookCloser + HF transfer | Delta from NeRF low-pass/fallback |
|---|---|---|---|
| Actor + held object | 24.2184 / 0.770860 / 0.113999 | 27.8821 / 0.776001 / 0.105586 | +3.6637 dB, +0.005141 SSIM, -0.008413 LPIPS |
| Face | 23.5496 / 0.756321 / 0.119322 | 27.1989 / 0.764034 / 0.106574 | +3.6492 dB, +0.007713 SSIM, -0.012749 LPIPS |
| Lipstick + hand | 23.3091 / 0.866137 / 0.084122 | 28.4163 / 0.875130 / 0.077832 | +5.1072 dB, +0.008993 SSIM, -0.006289 LPIPS |

Every cell is PSNR / SSIM / LPIPS. The pure and hybrid direct RGB variants are
pixel-identical on valid support; the improvement in the right column is
specifically the hybrid's LookCloser low-pass plus source high-pass composition.

Visual comparison order is ground truth, best pure TSDF mapping, raw LookCloser
base, and LookCloser plus high-frequency transfer:

- Face: `/mnt/data/lookcloser_dec5_5a3_final/000899_tsdf_vs_lookcloser_nerf_necessity/face_comparison.png`
- Lipstick: `/mnt/data/lookcloser_dec5_5a3_final/000899_tsdf_vs_lookcloser_nerf_necessity/lipstick_comparison_2x.png`

## Insights

1. **NeRF is not needed to recover the important high-frequency texture.**
   Pure TSDF mapping makes the lipstick, teeth, eyelashes, skin pores, hair,
   and applicator sharp. Its lipstick LPIPS is already `0.083570`, more than
   twice as good as raw LookCloser's `0.189282`.
2. **LookCloser currently provides useful low frequencies, not useful sharp
   detail.** Raw LookCloser has high PSNR but poor LPIPS. The hybrid keeps its
   target-view colour/exposure while replacing the blurred high-pass band.
3. **The current hybrid is the best measured renderer.** It wins LPIPS in all
   three actor regions and preserves nearly all of the base PSNR.
4. **A no-NeRF product path is realistic but incomplete.** Remaining pure
   mapping errors are source-visibility holes around neck, hand, and hair
   boundaries and small low-frequency colour offsets. Those can in principle
   be handled by fuller geometry, calibrated low-frequency view-dependent
   texture, confidence-aware hole filling, and temporally smooth source
   weighting.
5. **Do not remove NeRF before a path-render gate.** This held-out view proves
   spatial quality, but not temporal stability. Nearest-source changes may
   create seams or flicker in video. The next decisive experiment is a matched
   pure-versus-hybrid camera path with temporal metrics and visual review.

The practical conclusion is: **NeRF is not essential for detail, but it is
still valuable as a robust low-frequency and coverage fallback.** A cheaper
surface light field could replace it after it passes coverage and temporal
continuity gates.
