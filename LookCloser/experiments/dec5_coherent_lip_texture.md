# Coherent lip-region source control at native 6K

## What was tested

Separate from the unchanged 150-frame 6K video: use one train camera for an
entire lip surface region on `001083`, keeping geometry fixed and avoiding RGB
averaging. This tests whether the observed thin source transition is better
handled as a semantic region than as independently labeled mesh triangles.

`infer_train_lip_regions.py` uses the already-installed MediaPipe 0.10.21
FaceMesh model, with no download. It detects the outer lip ring on calibrated
train-only G/C, H/C and I/C RGB for two times, `001083` and `001123`. All six
landmark overlays were visually inspected. Inputs/model weights are hashed.
No held-out image, predicted render, candidate surface or evaluation ROI
defines these masks. The masks do not change geometry.

`study_coherent_lip_texture.py` requires at least two train lip-mask and
centroid-visibility votes for each surface triangle. It chooses the smallest
calibration angle to the rendering camera among the three train cameras with
at least 99% region-centroid visibility. Each changed pixel independently
passes the existing native 5461x3072 source-depth/foreground footprint test.
Invalid pixels retain the baseline; no depth is inferred or filled.

Two controls use the original contour and a uniform four-source-pixel disk
margin respectively. Both sample the original 6144x3072 PQ16 H/C image using
the frozen decoder, profiles and exposure. The delivered HD frame is never
used as their color source. Only diagnostic 6K crops are saved, not a replacement
full-frame/video artifact.

## Results

| Control | Region triangles | Region pixels | Valid native footprints | Changed RGB pixels |
|---|---:|---:|---:|---:|
| Original lip contour | 381 | 149132 | 149132 | 106151 |
| Four-source-pixel margin | 485 | 184121 | 184121 | 128321 |

All three candidate cameras had 100% centroid visibility. Both controls chose
H004_C005_1210SZ by angle, without looking at target RGB. Maximum RGB difference
from the baseline is 71 uint8 codes. These are diagnostic counts, not face
PSNR/SSIM/LPIPS or evidence of a fidelity improvement.

- [Original contour comparison](/mnt/data/dec5_coherent_lip_texture/frames/001083/comparison.png)
- [Expanded contour comparison](/mnt/data/dec5_coherent_lip_texture_margin4/frames/001083/comparison.png)
- [Actual native H/C train crop, no mesh reprojection](/mnt/data/dec5_coherent_lip_source_audit/frames/001083/native_train_crop.png)
- [Six train landmark records](/mnt/data/dec5_train_lip_regions/result.json)

Main-agent native visual review: the single-camera control changes the
highlight pattern and removes internal camera switches in its admitted region,
but transitions persist around its boundary. Expanding the region moves that
boundary; it is not a demonstrated general seam fix. Fine bright glints remain
under one-camera transport and the native source has real glossy highlights.
Their mere coincidence with a source-ID boundary did not prove they were all
artifacts. Preserve authentic detail instead of cosmetically erasing it.

The earlier source-graph diagnosis is also refined: the 49 I/B hair faces in
the inspected crop belong to global label components of 1, 2, 7 and 115 faces.
Every inspected face has three mesh neighbors: disconnected mesh adjacency is
not their explanation. H/C is geometrically visible at 46/49 centroids; only
9/49 are excluded by the 12% relative-quality cull, while 37/49 remain eligible.
All 281 I/C lip-face centroids also retain H/C as an eligible choice. Therefore
quality culling is only a partial hair hypothesis and does not explain the
lip division by itself. The lip boundary separates much larger label regions
of 41220 and 10779 mesh faces.

Raw margin-zero producer preserved at
`/mnt/data/dec5_coherent_lip_texture/config/study_coherent_lip_texture.py`;
its hash matches that run's controller binding. The later CLI supports isolated
outputs and explicit margins. No frozen production renderer/helper was edited.
Three tests cover landmark-ring topology, integer-pixel zero-weight taps and
fractional sampling. `001123` has inspected masks but no texture control yet.

## Insights

Verdict: exploratory, not promoted. A coherent semantic patch is feasible
without averaging or geometry changes, but masks alone do not guarantee a
seam-free appearance for view-dependent gloss. Native source inspection is
necessary before labeling a fine highlight as a reconstruction defect.

Next validation would need additional times/views and an objective comparison
of boundary continuity versus preserved detail; it must not silently replace
the running 6K recipe or claim an unmeasured held-out improvement.
