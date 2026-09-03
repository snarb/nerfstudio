# Hard TSDF selection and homography controls

## Correction (2026-09-03)

The original fine-TSDF render in this document is rejected. Halving the physical truncation band
from `0.004` to `0.002` fragmented the clothing surface, while candidate-defined surface metrics
excluded those missing pixels. The corrected pure render uses voxel/truncation `0.0005/0.004` and
an opt-in local planar completion for small enclosed depth holes. It is published at:

`/mnt/data/lookcloser_dec5_5a3_final/000899_tsdf_nearest_fill_fixed/final_render.png`

The correction changes chest coverage from `53.75%` before completion to `99.96%`; the lower
clothing hole is fully restored. The archival tables below remain useful for comparing hard source
selection with homography controls, but the former `0.0005/0.002` image must not be promoted.

## What was tested

The experiment uses DEC5 frame `000899`, the single held-out camera
`nearest_eval_00000`, the 16 evenly distributed train-camera subset and fixed GLOMAP calibration.
Held-out RGB is metrics-only. No person/image mask, NeRF, U-Net, LPIPS training loss or appearance
embedding participates in any prediction.

Four renderers were compared:

1. `TSDF nearest-fill`: one globally nearest camera owns each visible surface pixel; source ranks
   1--15 are used only where every earlier camera fails visibility.
2. `TSDF best-view`: every pixel independently selects one source using geometry, projected
   resolution and image-border support. RGB is never averaged.
3. `Homography mix4`: four nearest calibrated images are warped through one global plane and
   averaged where valid.
4. `Homography nearest-fill4`: the nearest plane warp owns its pixels; later warps fill only its
   missing field of view.

The TSDF was rebuilt from the same 62 Splatfacto alpha-median train depths at voxel/truncation
`0.0005/0.002`, versus `0.001/0.004` previously. Source-depth tolerance stayed at `0.01` after
`0.03` and `0.05` gates increased coverage negligibly and worsened actor LPIPS. Open3D
topological-hole radii `0.003`, `0.006` and `0.012` were also tested.

No metric uses the full image. Every value below is measured on fine-TSDF first-hit pixels inside
the actor/held-object, face, ear/hair, or lipstick/hand rectangle. The room is excluded.

## Results

| Region | Renderer | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Actor + held object | TSDF nearest-fill | 22.7304 | 0.762624 | **0.122847** |
| Actor + held object | TSDF best-view | **22.8032** | **0.764857** | 0.129578 |
| Actor + held object | Homography mix4 | 20.9087 | 0.670011 | 0.333598 |
| Actor + held object | Homography nearest-fill4 | 19.3492 | 0.626436 | 0.238941 |
| Face | TSDF nearest-fill | 22.9758 | **0.780714** | **0.111879** |
| Face | TSDF best-view | **23.0267** | 0.779976 | 0.114311 |
| Face | Homography mix4 | 19.5111 | 0.625293 | 0.427281 |
| Face | Homography nearest-fill4 | 18.3094 | 0.587493 | 0.266241 |
| Ear + adjacent hair | TSDF nearest-fill | 21.8864 | 0.742324 | **0.124871** |
| Ear + adjacent hair | TSDF best-view | **22.2332** | **0.742608** | 0.128968 |
| Ear + adjacent hair | Homography mix4 | 22.0667 | 0.656966 | 0.328969 |
| Ear + adjacent hair | Homography nearest-fill4 | 18.6443 | 0.598777 | 0.203591 |
| Lipstick + hand | TSDF nearest-fill | 22.5383 | 0.858343 | 0.082588 |
| Lipstick + hand | TSDF best-view | **22.7406** | **0.859934** | **0.081004** |
| Lipstick + hand | Homography mix4 | 18.6464 | 0.601039 | 0.513773 |
| Lipstick + hand | Homography nearest-fill4 | 16.1378 | 0.548100 | 0.367662 |

The fine target mesh covers `41.6146%` of the raster. Nearest-fill colours `40.6921%` of the full
raster, or `97.78%` of the mesh surface. Source rank 0 supplies `85.89%` of valid pixels and rank 1
another `13.31%`; all later cameras together supply less than one percent.

The ear LPIPS changed from `0.131011` with the old `0.001/0.004` mesh to `0.124871` with the fine
mesh. All three topological-hole radii produced exactly the same ear metrics as the unfilled fine
mesh. The holes visible below the ear are therefore not closed mesh holes; they are missing/open
silhouette support. Loosening visibility to `0.03/0.05` worsened actor LPIPS to
`0.132113/0.136000`.

## Insights

1. Hard nearest-fill implements the requested one-camera renderer: there is no RGB blending, and
   later cameras are used only for visibility holes. It is the preferred overall pure-TSDF result.
2. Per-pixel best-view selection does not solve the ear. It slightly improves lipstick and PSNR,
   but worsens LPIPS on actor, face and ear and introduces a small earring dropout.
3. The ear defect has two layers. Coarse TSDF quantization was real and is improved by the finer
   mesh. The residual ragged/black boundary comes from missing or inconsistent depth near thin hair,
   earring and silhouette geometry; neither source mixing, visibility tolerance nor closed-hole
   triangulation fixes it.
4. One global homography is invalid for this non-planar subject. Mixing four warps creates severe
   double images. Hard nearest-fill avoids blur locally but leaves wrong parallax and rectangular
   camera-FOV seams.
5. The experiment isolates both requirements for a sharp renderer: hard/local appearance selection
   and coherent 3D surface correspondence. Removing NeRF alone is viable for detail, but removing
   geometry is not.

Full-resolution renders, source-selection maps, comparison crops and machine-readable audits are
published together under:

`/mnt/data/lookcloser_dec5_5a3_final/000899_tsdf_hard_selection_and_homography`
