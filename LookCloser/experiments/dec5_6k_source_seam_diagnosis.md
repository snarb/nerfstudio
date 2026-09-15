# Native 6K source-seam attribution

## What was tested

Post-hoc inspection of the true `3456x6144` output canary `001083`, compared
with the previously delivered `1080x1920` native-source video frame enlarged
only as a diagnostic control. Production frames are not enlarged or modified.

Scripts: `diagnose_6k_source_seams.py` and `attribute_6k_mesh_label_seams.py`.
Artifacts: `/mnt/data/dec5_6k_source_seam_diagnosis`.

The first tool overlays actual native source-ID boundaries and differences
from the diagnostic nearest-enlarged HD source-ID map. The second independently
raycasts each crop against the frozen mesh, obtains its triangle IDs and checks
whether final pixel sources differ from the frozen mesh-face source labels.
Pixel-center mapping respects the single landscape-to-portrait rotation.
Source IDs are not themselves an image-quality metric or proof of an error.

## Results

Each region contains 262144 pixels. Counts below are local diagnostics, not
face PSNR/SSIM/LPIPS, nor campaign aggregate quality estimates.

| Region | Mesh misses | Different from frozen face source | Different from enlarged HD pixel source |
|---|---:|---:|---:|
| Hair | 0 | 7 | 494 |
| Ear | 0 | 0 | 0 |
| Lips | 0 | 0 | 545 |

The main agent viewed the native 1:1 A/B hair/lips crops and the retained
boundary/ID overlays. Fine hair structure is more resolved than in the HD
control; thin triangular boundaries are also easier to see. The bright thin
line at the upper lip coincides with the frozen source boundary between
`H004_C005_1210SZ` and `I004_C005_1210BA`. Neither missing geometry nor native
visibility fallback explains that particular line: both counts are zero in
the entire inspected lip crop.

The hair crop uses H/C for 260665 pixels and I/B for 1477, plus two I/C pixels.
The I/B region has 47 four-connected raster components, including thin slivers
of 1082, 194 and 143 pixels. Raster components are not mesh components.
Only seven hair pixels differ from their frozen face source, so the visible
thin source regions are predominantly inherited mesh-source assignment, not
new native-resolution visibility fallback. The ear crop has one source H/C.

- [Lips: RGB, source boundaries and source-ID controls](/mnt/data/dec5_6k_source_seam_diagnosis/001083_lips.png)
- [Hair: RGB, source boundaries and source-ID controls](/mnt/data/dec5_6k_source_seam_diagnosis/001083_hair.png)
- [Per-pixel face-source attribution](/mnt/data/dec5_6k_source_seam_diagnosis/mesh_label_attribution/result.json)

## Insights

These crops support a texture-assignment explanation for the observed fine
seams, not a depth-hole explanation. They do not by themselves distinguish
view-dependent highlights, registration offsets and camera-color residuals.
No averaging is used by this renderer; a discontinuity can arise simply by
switching from one source to another across adjacent faces.

A useful separate control is coherent source assignment across a whole lip
patch and suppression of tiny hair source islands, always retaining visibility
checks. That has not been tested here. Do not change the running 150-frame 6K
request, claim that these seams are fixed, or promote an unverified relabeling.
The source-boundary overlays are diagnostic evidence, not production masks.
