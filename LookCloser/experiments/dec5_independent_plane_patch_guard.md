# Independent plane-patch depth witnesses

## What was tested

At `001037`, compare the nearer four-view FoundationStereo proposal with farther
corroborated PatchMatch depth. Photo witnesses exclude **all four prior-input
cameras and the query camera**. Up to eight are selected by calibrated parallax
(1–35°), not RGB. Native RGB uses the current fixed profiles/exposure.

`plane_patch_evidence.py` projects query-pixel rays through competing parallel
planes. Native photographed bounds, positive depth and source-center occlusion
are checked. Compare 15×15 and 31×31 windows with query-frontoparallel and
proposal-tangent normals. This is not COLMAP's internal matching cost.
The initial gate requires three identical witnesses to favor near depth under
all four configurations: NCC≥.65, near–far≥.2, query channel-std≥2, fewer than
two witnesses favoring far depth. Unavailable/flat evidence abstains.

Deterministic controls shift corroborated old-mesh skin points .024 normalized
camera-z units toward the camera. These are consistency controls, not 3D GT.

## Results

49 camera groups, 629 sampled conflicts, 726 measured controls and 15 previously
frozen diagnostic samples were scored in 29.1 seconds. The initial rule rejects
107 farther conflicting depths and **0/726 controls**; none of the 15 frozen
samples pass. Small skin patches frequently lack discriminating texture.

A disclosed **post-hoc exploratory** rule uses the 31×31 window, std≥4 and
the best of the two plane orientations **independently for each depth**. It
rejects 264/629 conflicts, 6/15 frozen samples and 0/726 controls. This is not
independent final validation. The palm remains ambiguous; several forearm
samples favor near skin/cloth context over farther clothing projections.
All seven saved witness montages were inspected.

- [Independent F/E witness example](/mnt/data/dec5_forearm_independent_patch_evidence/review/F004_E005_1210FP_2.png)
- [Palm ambiguity](/mnt/data/dec5_forearm_independent_patch_evidence/review/F004_E005_1210FP_0.png)

The exploratory qualifier was applied uniformly across all 62 query cameras and
two ray offsets. Fractional rays additionally require a compatible native-ray
hit on added geometry. Seven pruning passes converge, retaining **36,518/40,925**
proposed faces versus 37,385 for the earlier color qualifier. The final mesh
still has 2,488 original trusted-depth contradictions, all disqualified by the
new rule. Zero qualified vetoes does **not** mean zero original contradictions.

Three matched current-renderer RGB views show no convincing advantage over the
color qualifier: wrist holes and peppering remain, with some coverage lost.
All three fail whole-frame visual acceptance.

A separately identified diagnostic removes only the final PM veto, retaining
all 40,925 proposed faces. It removes much of the forearm peppering, but real
wrist ray misses, hand defects and the two-tone surface seam remain. It is not
approved geometry; its render receipts explicitly state the disabled guard.

| Change versus earlier color qualifier | H/A | E/D | Moving |
|---|---:|---:|---:|
| Qualified mesh: new / lost depth pixels | 127 / 244 | 0 / 35 | 141 / 313 |
| No-final-veto control: new / lost depth pixels | 664 / 0 | 9 / 0 | 840 / 0 |

Counts are diagnostic, not anatomical metrics. Upper 1,200 portrait rows are
unchanged in all six matched RGB renders. Original mesh arrays are preserved;
no full-frame quality metrics, source mutation, or video/default update.

- [Qualified moving crop](/mnt/data/dec5_independent_plane_patch_guard/review/moving_detail.png)
- [No-final-veto moving crop](/mnt/data/dec5_independent_plane_patch_guard/unguarded_diagnostic/review/moving_detail.png)
- [Negative visual gate](/mnt/data/dec5_independent_plane_patch_guard/visual_review.json)
- [335-hash audit](/mnt/data/dec5_independent_plane_patch_guard/artifact_manifest.json)

Eleven focused tests passed, including synthetic plane-transfer correspondence,
pixel conventions, source exclusion, invalid/flat abstention and symmetric
plane selection. The earlier NCC test now also imports independently instead
of relying on another test's import-path side effect. All six RGB workers and
other jobs terminated normally. Recheck with
`scripts/freeze_independent_plane_patch_guard.py --check`.

## Insights

Photo evidence can expose correlated wrong-layer depth agreement, but the
tested qualifier does not yet yield a clean surface. Large windows use context
across boundaries; a good score is not exact depth truth. The native-ray check
also makes pruning depend on neighboring retained faces. No rule is promoted.

Crucially, remaining raw-consensus holes do **not** prove missing neural depth.
The subsequent [multiview admission test](dec5_multiview_forearm_admission.md)
locates most of them in the earlier, reference-only eligibility rule.
