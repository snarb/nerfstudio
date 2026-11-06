# Matched15188 SH and prefix geometry audit

## What was tested

The same28 original TRAIN shirt rays were replayed on frozen full/body80/r0
control, SH-corrected, and prefix.003 checkpoints at step15188. Native traversal
uses each checkpoint's actual saved GPU core; uniform8192 and16384 integration
query the same learned field independently of occupancy. FR is explicitly1.
The loaded SH flag is verified true for the SH arm and false for prefix.
Camera rays are independently checked against the prepared source geometry.

The prefix arm fits9 point IDs/18 rays. Four nearby grouped point IDs/8 rays
are reserved from its auxiliary; duplicate66/2 rays are separately excluded.
These are all TRAIN RGB observations. The reserve is correlated local evidence,
not an independent scene test. Historical native20k explicitly supervised all28
points through its geometry objectives and is not a pure architecture control.

## Results

**Correcting SH improves geometry as well as held-out RGB, but substantial
incorrect front termination remains. Prefix suppression is much stronger on
its fitted rays and generalizes less well to the nearby reserved points.**

The table reports percentages of total ray probability under dense16384
integration. The band is±.02 prepared-world units around triangulated shirt
depth; it is not a calibrated confidence interval or a claimed metric distance.

| Rays / model | Before band | In band | After band | Transmission |
|---|---:|---:|---:|---:|
|All28: control|84.799|12.930|2.212|.059|
|All28: SH corrected|52.538|39.777|7.671|.014|
|All28: prefix.003|25.711|44.404|28.663|1.222|
|Prefix-fitted18: control|82.354|14.927|2.635|.084|
|Prefix-fitted18: SH corrected|50.568|41.116|8.297|.018|
|Prefix-fitted18: prefix.003|6.882|58.216|33.325|1.577|
|Reserved8: control|90.289|8.508|1.190|.013|
|Reserved8: SH corrected|59.851|35.443|4.702|.005|
|Reserved8: prefix.003|72.356|15.573|11.767|.305|

Native traversal reaches the same qualitative conclusion. On the reserved8,
its before/in-band percentages are74.99/21.93(control),26.70/64.32(SH), and
64.89/19.66(prefix). Native skips part of the same field's front density;
therefore native and uniform values are intentionally reported separately.
One must not equate a native-render improvement with complete field correction.

All28 rays in all3 models pass the declared8192→16384 RGB/opacity convergence
checks. The maximum change in any of the four depth-mass components is.000301
for control, .001379 for SH and .001638 for prefix. The observed differences
are much larger than these finite-quadrature changes.

At this matched gate, the fixed held-out actor metric improves from control
LPIPS.080374/PSNR24.8495 to SH.069681/26.4216. Prefix is worse overall at
.087157/24.4121 despite its constrained-shirt depth improvement. This is not
a final checkpoint selection or a claim about later saturation.

[Matched cumulative-depth plots](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/sh_prefix_matched15k_v2/comparison.png)
were visually inspected: prefix clears its fitted ray prefixes but often moves
remaining termination beyond the shirt band; its reserved rays still have
large earlier layers. SH moves more probability toward the correct band on
both subsets. [Full per-ray arrays and summary](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/sh_prefix_matched15k_v2/comparison.json)
include the excluded duplicate separately, native integration, provenance and
camera-chain checks.

## Provenance and execution

SH checkpoint SHA256:
`150d48c77d9948a475d93d3b683f24539025af3ce20819c0090e30a12ac6d336`.
It was protected by a hardlink on silver-sable before further pruning. Its full
archive remains at
`/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/independent_occupancy/sh_corrected_015188`.
The slow full-checkpoint transfer was stopped with root authorization; copied
trace arrays and receipts bind the preserved original checkpoint hash. The
remote field-source hash matches the local frozen SH source.

Prefix's immutable hardlink and copied authoritative grid are under
`independent_occupancy/prefix003_015188`. Prefix replay ran locally with explicit
read-only overlap authorization and a10GiB allocator cap. SH replay used idle
silver GPU1 while holding the shared per-user GPU lock for the entire audit;
it exited and released that slot. No optimizer or training callback ran, and
the replay verifies restoration of the copied model after temporary traversal
profile substitutions. Production sources, checkpoints and grid states were
unchanged. No speed comparison is inferred from these runs.

## Insights and next steps

SH is the stronger current general improvement: better held-out actor metrics
and more shirt-band support on the prefix-reserved rays. Its52.5% dense early
mass across this selected problem region still rules out claiming correct
geometry everywhere. Prefix-only clears constrained rays without simply making
them transparent, but one third of fitted-ray probability now lies behind the
declared band. Fitted-ray agreement alone is insufficient for promotion.

A subsequent geometry-source expansion should sample independently reviewed
opaque clothing across both actors and multiple spatial regions, keep grouped
spatial reserves, exclude floor-bag aliases and uncertain boundaries, and retain
native-field gradient checks. This is a proposal only; no new prior source,
weight, occupancy edit or training arm was installed by this audit.

Read-only feasibility filtering finds184 black-actor and124 grey-actor candidate
points with at least3 original TRAIN observations at least8HDpixels inside the
corresponding semantic mask;171 black points are beyond the initial14 IDs. No
point passes both actor identities. Candidate points are inside the accepted
AABB and above prepared floor z=.10 to avoid the floor neighborhood during
initial review. This is a source-review convenience, not an anatomical or
free-space assertion. The pool includes unreviewed surfaces and must **not** be
used directly as geometry supervision.
[Candidate pool](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/sh_prefix_matched15k_v2/candidate_source_pool.json).

Proposed sequence:

1. Use SH-corrected full/body80/r0 as the reference and test its compatibility
   with the existing18-ray prefix first. This adds one factor to the improved
   model and avoids attributing a simultaneous SH/source expansion to either.
2. Manually review about24–40 new opaque clothing points, balanced across both
   actors and separate torso/sleeve/trouser regions. Select only original
   observed cameras; reject reflections, mixed occlusions, hair/skin boundaries
   and bag/floor aliases. Require source crop confirmation for every prior ray.
3. Reserve entire spatial clothing patches before training, rather than nearby
   individual glyph points. Preserve the current reserved IDs in the diagnostic
   set, merge duplicate source pixels, and do not label the spatial reserves
   independently measured ground truth. Keep source-camera balancing and the
   optical-thickness formula fixed; rerun actual gradient checks for the new
   artifact before choosing a coefficient.
4. Compare expanded versus original sources under the same SH recipe. If a
   coefficient change becomes necessary, identify it as an additional factor
   with its own matched control. Do not add a surface-band term in this source
   comparison. Add shoe-specific constraints only after separate source review
   distinguishes real shoes from the confirmed stationary floor object.
