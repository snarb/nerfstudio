# Local anatomical patch extraction controls

## What was tested

Frame001193; three frozen [measured-conformance priors](dec5_mhr_measured_conformance.md),
not a whole-head replacement. Original raw COLMAP mesh SHA256
`316dc2b6a82c0d99a6e399d15d941a69d491896921bc82e4d88ea307bb358b03`
retained exactly. This is not the later carved production mesh.

Camera-independent proposals use neutral anatomical y135..153cm, excluding all
known intersecting or normal-reversed source facets. Four shared-midpoint rounds
reduce maximum edges to0.00075 normalized scene units. Each new vertex must lie
within0.002 of original surface,0.003 of original boundary vertices, with normal
cosine>=0.25. New facet centers must project near an actual opposite open edge
(barycentric<=0.03), and be at least0.00002 away from the measured surface.
Distances are scene units, not meters. No target camera/RGB enters extraction.

Root: `/mnt/data/dec5_mhr_local_patch_candidates`.
Builder, native reviewer, gate diagnostic and auditor are separate opt-in scripts;
existing reconstruction/render defaults and the active6K video are unchanged.

## Results

| Conformance arm | Subdivided facets | All vertices local | Raw added facets | Fixed hole remaining /44 |
|---|---:|---:|---:|---:|
| smooth025 |759808|78712|11606|44|
| smooth100 |759808|97680|18193|44|
| smooth400 |759808|108598|24057|29|

Main LLM inspected six native train RGB/clay panels and the fixed requested-hole
crop. All raw candidates fail visual acceptance: additional ragged fringe sheets
and islands appear near the lower neck. Original facial detail remains; this is
not yet a textured or depth-approved repair. The strongest arm fills15 missing
pixels in the fixed44-pixel diagnostic; the other two fill none.

Examples: [native profile](/mnt/data/dec5_mhr_local_patch_candidates/raw_review/E004_B005_1210I7.png),
[front](/mnt/data/dec5_mhr_local_patch_candidates/raw_review/M004_B005_12109O.png),
[requested hole](/mnt/data/dec5_mhr_local_patch_candidates/raw_review/requested_hole.png).
Added-facet overlays are retained separately, not all independently inspected.

Post-hoc association diagnosis of44 previously sampled prior-ray hit points:

| Arm | All vertex distances pass | All vertex boundary distances pass | Normals pass | Actual open-edge gate passes | All gates pass |
|---|---:|---:|---:|---:|---:|
| smooth025 |44|44|42|0|0|
| smooth100 |44|44|44|0|0|
| smooth400 |42|42|44|16|14|

This is nearest-facet association at stored prior hit points, not an exact replay
of candidate raycasting; its14 successes must not be substituted for15 actually
filled render pixels. Maximum vertex/surface distances for025/100 are0.000785 and
0.000796, comfortably inside the0.002 locality cap. The nearest open-edge condition
is the dominant rejection for these two controls, not failure to reach the region.

Tests cover actual opposite-edge classification (a shared interior diagonal must
not count merely because its endpoints are boundary vertices), shared-midpoint
topology, winding and facet provenance. The audit verifies retained geometry,
input hashes, original exact prefixes, anatomical exclusions and nearest-boundary
tests. It binds producer normal evidence but does not independently recompute
the full subdivided prior's vertex normals. It is not a depth/visibility audit.

## Insights

An anatomical fit close to the rim does not automatically yield an acceptable
patch. Nearest-open-edge extraction can exclude a useful nearby prior segment
when its closest original point lies inside an adjacent existing facet. Conversely,
blindly keeping every nearby prior facet would create duplicate surfaces/fringes.

Next discriminate actual two-view depth support from bounded interpolation inside
trusted observed seeds, retaining strict measured-free-space vetoes. These tests
are separate artifacts: none of the raw controls is promoted. A future general
alternative to the nearest-edge rule should require independent local evidence,
not a target-camera-specific exception. Transfer to the carved production mesh,
native RGB review and temporal testing are still required.
