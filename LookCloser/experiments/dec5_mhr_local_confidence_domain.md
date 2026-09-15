# Confidence-gated local anatomical domain

## What was tested

A separate frame001193 extraction ablation following the negative
[open-edge local patch control](dec5_mhr_local_patch_candidates.md).
Same three conformed priors, anatomical band, unsafe-facet exclusions, subdivision,
surface/boundary distances, normal gate and minimum centroid distance. Only the
requirement that the closest original point project onto an actual open edge is
removed. The raw overlapping domain is explicitly **not an acceptable mesh**.
Acceptance still requires independent measured-depth support or certified local
interpolation, plus the unchanged native free-space guards. No target camera/RGB
selects the domain, and no per-frame exception is introduced.

Source mesh remains the raw COLMAP001193 mesh ending in SHA256`316dc2...b03`, not
the differently carved production mesh. Existing measured-depth and mask evidence
can be reused only with explicit geometry rebinding.

Root: `/mnt/data/dec5_mhr_local_confidence_domain`.
Separate admission root: `/mnt/data/dec5_mhr_local_confidence_admission`.

## Results

| Arm | Open-edge raw facets | Wider local-domain facets |
|---|---:|---:|
| smooth025 |11606|76507|
| smooth100 |18193|95295|
| smooth400 |24057|106807|

The full extraction audit replays subdivision, vertex normals, all geometric
thresholds and exact facet coordinates. It also verifies every open-edge control
facet remains in the wider candidate domain and every original mesh vertex/facet
is unchanged. This tests a strict domain superset, not a different prior fit.

Measured-depth admission and native visual outcomes are recorded separately;
domain generation alone is neither evidence of hole repair nor production approval.

## Insights

The preceding strict open-edge control retained99/89/66 depth-supported facets,
but closed none of the44 fixed missing pixels. It was safe under measured-depth
guards yet ineffective for the requested repair. This ablation tests whether the
open-edge projection criterion, rather than the prior itself, prevented supported
local filling. Do not bypass independent confidence gates just to increase counts.

No changes enter the active6K video. Native textured validation and temporal
transfer remain required before any positive result can be promoted.
