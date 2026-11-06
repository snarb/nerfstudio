# Marshawn: saved-render transmission and the earlier opacity recipe

What was tested: selected semantic R0@111316 black-background RGB and16-bit
alpha were composited over gray/checker backgrounds without loading or changing
the field. These are encoded model-domain diagnostics, not physical relighting
or new evaluation scores. The original rendered bytes and primary metrics stay
unchanged. Reusable helper: `scripts/inspect_marshawn_render_opacity.py`.

Results: substantial surrounding haze remains visible in alpha. Root inspected
the DC/HC black-actor contacts; the dark garment also has some transmission
inside its silhouette. Even after16-HD-pixel mask erosion, the following
fractions remain below the stated alpha thresholds:

| Black actor interior | alpha<.9 | alpha<.99 |
|---|---:|---:|
| DC |4.14%|45.12%|
| HC |1.61%|32.10%|
| KA |4.39%|25.83%|

No such16-pixel-eroded actor region contains alpha<.5. The gray actor has much
less interior transmission. Erosion is a diagnostic selection, not a material
label or proof that every selected hair/occlusion pixel must be opaque.

Artifacts, exact source hashes and all18 region/erosion records:
`tuning_20261106/independent_occupancy/semantic_R0_111316_opacity_review/`.

Insights: the green-clothing Luster lesson is more specific than simply
increasing distortion. Its accepted intermediate combined **opacity-profile
regularization .1** with qualified TRAIN-depth occupancy admission. The profile
term integrates T(1−T), with a separate negative-log accumulation deficit;
neither is the current distortion coefficient. The later spatial density-element
penalty .01 was rejected: it worsened LPIPS and left the skirt ghosts. See
[the original measured report](luster_000470_opacity_depth.md).

The global Luster opacity deficit cannot be transferred directly to an isolated
actor volume: background and uncertain hair rays need not terminate inside it.
A future controlled opacity test would need explicitly qualified TRAIN opaque
interiors, preserve uncertain contours/hair, exclude known bag mislabels, and
keep the original RGB targets and sampling. No opacity objective, alpha target,
mask, checkpoint or grid was changed by this diagnostic. Its evidence motivates
qualification work, not an automatic global opacity penalty or an ideal-grid claim.
# Exterior opacity at the selected checkpoints

The separate CPU helper `scripts/inspect_marshawn_exterior_opacity.py` reads
the original scalar 16-bit alpha PNGs and frozen V1 masks. It measures Euclidean
image-plane margins of 0, 3, 8, 16 and 32 HD pixels without loading a model,
changing labels, or recomputing primary metrics. Checkpoint ages differ for W32:
R0/W64 and R1/W64 use step 111316, while R0/W32 uses step 126316.

Pixels with alpha above 0.1 beyond a 16-pixel margin:

| Camera | R0/W64 | R1/W64 | R0/W32 |
|---|---:|---:|---:|
| DC | 185,261 | 235,474 | 212,542 |
| HC | 107,305 | 123,294 | 110,803 |
| KA | 74,830 | 85,273 | 72,158 |

Root inspection of the HC opacity contact confirms a broad exterior around
the actors and a disconnected floor component. W32 reduces integrated exterior
alpha in all three views, but does not consistently reduce the area above 0.1.
R1 increases both measures in all three views. These are diagnostics of saved
renders, not independent nonactor labels or physical empty-space certificates;
mask omissions, uncertain hair and occlusion still matter.

[All measurements and source hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_exterior_opacity/receipt.json)
and [HC opacity comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_exterior_opacity/HC_opacity.jpg).
Good masked LPIPS alone cannot certify a clean isolated asset. A separate
feasibility study should distinguish opacity constraints on qualified clear
background from opaque clothing constraints; neither should treat uncertain
hair or foreground occluders as an alpha target.

