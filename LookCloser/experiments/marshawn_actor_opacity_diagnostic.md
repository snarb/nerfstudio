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
