# Marshawn video: source admission and learned-grid edge audit

## What was tested

Keep the accepted common three-second AABB and selected cold-training recipe.
Check whether positive actor contours have any retained cell along their camera
rays, using exact voxel traversal rather than render-alpha heuristics. Source
admission and learned occupancy are different masks and are audited separately.
A retained cell on a ray does not establish the true surface depth or certify
all hidden geometry.

The read-only dense helper is committed in `076a1cf9`. Its exact DDA traversal
passed 250 random comparisons with exhaustive cell/slab intersection. It binds
the source RGB/masks, poses, AABB and actual exported grid snapshots. Native
canary rays use the source camera's calibrated inverse distortion; forward
roundtrip error was below 2e-12 native pixels.

## Results

### Frame 1675: genuine source-label failure, before field publication

The initial sparse witness found 55 failures in 918 rays. Every failure was
the black actor's topmost flying braid; body, feet and grey actor samples passed.
Exact traversal confirms an inside-box ray, with no transform/indexing problem.
For the best cells, 58 cameras support foreground but AB/AC/AD/BC/BD incorrectly
label the visible braid tips as known background. Those tips are 127–218 native
pixels from the existing nominal masks, beyond the 64-pixel RGB falloff; their
W64 and both UNKNOWN masks are zero. The classifier follows incorrect inputs.

| Original 1675 mask | Boundary rays | Rays with no retained cell |
| --- | ---: | ---: |
| Source base | 162757 | 6238 |
| Source exact one-cell halo | 162757 | 4701 |
| Frozen mapped source support | 162757 | 2867 |

All mapped failures are black-actor boundaries in the same 55 views; grey has
zero. Additional native contours from the five fresh hair canaries also expose
the missing support. The weak AD seed is not anatomical ground truth.

No 1675 production model was published. The old source release and all
transaction/grid/ready proofs were archived on both hosts; only the input
transaction's CPU controllers were retired. Original checkpoint and field
processes were untouched. The common AABB remains unchanged.

Root manually reviewed all five current native candidate contacts. Qualified
additions are AB blond-dreadlocks14, AC dreadlocks16, BC dreadlocks29 and BD
dreadlocks26. AD's weak dreadlocks7 is permitted only as a geometry-uncertainty
seed. Even the strong NN candidates miss the witness by 11–46 native pixels,
so foreground additions alone do not establish safe free-space labels.

The correction adds a separate regulated 128-native-pixel geometry UNKNOWN
envelope around current anchored hair/head support. This does not force density,
RGB supervision or occupancy; it prevents confident background carving near
unresolved hair. Original RGB/W64/UNKNOWN arrays remain separate. All 66 cameras
receive current native head/hair queries; previous tracking proposes a search
region only. The new source revision requires its own full grid and contour
audit before publication. Corrected-grid results are pending.

Evidence: original source campaign
`operations/witness1675_diagnostic/{HC_ray_trace.json,five_clear_camera_exact_margins.json,dense_all66/}`;
native selection `operations/1675_topbraid_ROI_v6/root_five_native_candidate_selection.json`.

### Completed frame 1670: actual learned occupancy, checkpoint 139999

The checkpoint SHA is
`ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015`;
training completed exactly 140000 iterations. All 168805 nominal actor-boundary
rays were checked against the saved real grid, without loading or modifying the
NeRF during CPU traversal.

| Mask | Retained cells | Boundary-ray failures |
| --- | ---: | ---: |
| Source base | — | 302 |
| Source halo1 / mapped support | — | 0 |
| Runtime semantic admission | 441304 | 0 |
| Saved learned core / active R0 | 37821 | 3 |
| Fresh runtime R1, clipped to same admission | 92460 | 0 |

The three R0 failures are ND black pixels (1313,535), (1314,536), (1315,536).
Current native NN labels say black=true and UNKNOWN=false. Their largest saved
occupancy EMA is .00936276, below the .01 core threshold. Root inspected actual
native GT and matched R0/R1 crops: R0 has stepped black holes at the coat edge;
R1 partially fills them. Alpha increases from zero to .0910/.1364/.1292. This
is partial coverage, not proof of opaque or physically correct depth.

All registered binary-grid aliases were restored after the render. Pipeline
tensor SHA before/after is identical, and restored R0 HD RGB is byte-identical.
The original checkpoint, config and learned EMA were never changed.

| Matched in-sample DC/HC/KA actor monitoring | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| R0 | 46.12168 | 0.991924 | 0.00094554 |
| Full R1 | 45.86622 | 0.991849 | 0.00097706 |

All six actor/view LPIPS values worsen slightly under blanket R1. Root reviewed
the actual V4 record2 full 4K R1 image: floor fragments and the hair shell remain.
The extra support does not justify global promotion on these measurements.

A narrower render-only candidate keeps every R0 cell and additionally admits
saved EMA >= .009 inside unchanged semantic admission. It uses 39278 cells
(1457 extra, +3.85%) and restores 2/1/1 cells on the three failed rays. The R0
superset property proves zero failures on the same observed boundary-ray set.
Matched actual render/metric evaluation is pending; no training setting changed.

Evidence: capped campaign `operations/contour_runtime_001670/`, including
`dense/report.json`, `ND_three_ray_proof.json`, `render_RGBA_root_replay/`,
`R1_full4K_and_HD_metrics/`, and `root_EMA_threshold_candidate_counts.json`.

## Insights

The image-derived one-cell halo prevents real edge omissions in the tested
source admission. It cannot repair confident background labels caused by a
mask missing an entire braid tip. Native head/hair uncertainty must be handled
before carving. The learned grid can independently omit low-EMA edges after
source admission passes. Blanket runtime dilation restores those edges but
currently worsens image metrics, motivating a smaller source-checked correction.
None of these tests certifies every empty or occupied cell in unseen geometry.

## Corrected1675 source admission

The current native head/hair revision and regulated128-native-pixel geometry
UNKNOWN inputs passed the source checks and ROOT's66-view visual review. An
independent review also inspected all66 current UNKNOWN/photo overlays and head
insets: bands stay near the actor's head/braids, covering0.59–2.01% of each
raster. The weak AD fallback is the widest and reaches upper shoulders; its
qualified RGB foreground addition remains zero. These are uncertainty inputs,
not forced occupied cells. The AABB and calibration did not change.

A derived-data cache error was caught before publication: the frozen DATA
builder returned early on an old `prepared_ready.json`, leaving old W64 maps
after native hair changed. AB had1477 qualified HD hair pixels with zero W64,
and its recorded native-mask SHA differed from the current archive. The old
completion controls were archived and the unchanged frozen builder actually
reran. The confidence guard was retained. The final QA checks native H/U are
inside native foreground, their calibrated projections match the saved flags,
and W32/W64 dominate projected native coverage on valid non-veto footprints.
Fractional coverage flags are not hard alpha labels; requiring them to be
inside the >=.5 hard HD mask was an invalid earlier QA premise.

| Corrected current-source check | Rays | Failures |
|---|---:|---:|
|All66 nominal actor contours, source base|165925|0|
|All66 nominal actor contours, source pad1|165925|0|
|All66 nominal actor contours, source mapped|165925|0|
|The exact2867 previously failed mapped rays, retested|2867|0|
|ROOT-selected five native hair candidate contours, base/pad1/mapped|13927|0|

The native canaries bind the current photograph, NN checkpoint/spec/receipt,
ROOT selection and camera calibration. RAW pixel-center lens inversion has a
forward-polynomial residual below2e-12 native pixels. AD remains an uncertainty
seed rather than certified hair foreground. These results check observed
contours and do not certify hidden physical depth or future learned occupancy.

The correction increases base source cells from380827 to383801 and pad1 cells
from415797 to419581. It fixes the known hair false carving without enlarging the
box or applying a blanket learned-grid dilation. No checkpoint, learned grid
or optimizer state was reused across frames or changed by this diagnostic.

Exact final bindings:

- Source marker: `522cb9ef8547a2d341a563db38661d41c59913285790d4c6c89c90140419c014`.
- Final ready: `85673fbecef7e33ee9410ebe39a93138124d3d00a678b6829f7819d6fd88aaad`.
- Admission manifest: `6111f75e4f700c4ecf4cbc25614b1b9b26cb81a660a291883bf163587e10be0b`.
- Grid archive: `776e16ea9ac7bf7e75258d1cc66c767bc63e349a67ecd351e625dca12ae553bb`.

Evidence under the SOURCE campaign:
`operations/advanced_inputs/001675/dense_contour_hair_v4_85673fbecef7/qualification.json`,
`operations/witness1675_corrected/native5/native_report.json`,
`operations/witness1675_corrected/old_failed_rays_retest.json`,
`operations/hair_unknown_manual66/independent_review.json`.

## Same-checkpoint render recovery on1670

All variants below use the completed140000-iteration model
`ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015`.
The checkpoint, MLP, EMA, training recipe and source admission stay unchanged.
The baseline has37821 learned active cells. Candidate cells are restricted to
the original source admission and the global one-cell neighborhood of R0.

The saved-EMA threshold .009 adds1457 cells but barely restores opacity at the
three failed rays (.00634/.00234/.00194); a traversal hit is insufficient to
fix the visible hole. Selecting only cells traversed by those rays adds30.
A one-cell footprint guard around these30, clipped to global R1, adds145
(+.38%). This is a local guard, not global R2. The read-only constructor is
`scripts/build_marshawn_contour_grid_candidate.py`; parallel and half-open
cell intersections, actual30/145-cell replay, immutable resume and rejection
of conflicting output revisions passed CPU qualification.

The145-cell candidate was rendered on all66 actual camera crops containing
both actors and64HD pixels of context, with no resizing. PSNR is measured on
each actual actor mask. Spatial AlexNet LPIPS and SSIM use identical neutral
RGB outside that actor and average only inside it. Equal actor/camera means
below contain132 measurements. All66 cameras are training views: these are
in-sample diagnostics, not held-out quality estimates.

| Same1670 model / render policy | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|Original R0|45.941303|.991698137|.002353679|
|Local145-cell secondary on every ray|45.925297|.991686942|.002355341|
|Use secondary only where primary alpha equals zero|45.942275|.991698671|.002351804|

Applying the secondary to every ray slightly regresses the overall mean and
some individual views, despite improving the known ND black-actor contour.
It is not promoted. Exact-empty recovery preserves every nonempty primary
ray bitwise. On these66 crops it improves ND black-actor PSNR by.128371dB and
LPIPS by.000247524; all other65 camera/actor pairs and every grey-actor score
remain unchanged. There are no paired actor-view regressions.6250 context
rays are recovered; this count includes background pixels.

ROOT inspected the actual full4K V4 record2 recovery and the native ND
RGB/alpha comparison. The known stepped black hole is reduced but a narrow
residual remains where primary alpha is tiny and nonzero. Floor fragments
and the broad hair shell remain; neither candidate certifies physically
correct geometry or a clean background. Exact-empty recovery therefore
remains an experimental inference guard, not a complete fix. A finite cached
CPU comparison of thresholds0/.001/.01/.05 is pending, with no further GPU
grid variants queued.

`scripts/marshawn_empty_ray_recovery.py` requires the same checkpoint and
camera-ray binding for both geometry caches, copies RGB/alpha/depth quantiles
only on selected empty rays, and rejects unrelated checkpoints. Actual66-view
replay passed. Float32 accumulation can exceed1 by a few ULPs (native maximum
1.00000143); validation allows1e-5 without clipping or changing source values.
`scripts/evaluate_marshawn_actor_contexts.py` binds source/prediction hashes
and measures the unchanged actor masks. Target RGB participates only in
measurement, never in selecting recovery pixels.

Evidence in the capped campaign under `operations/contour_runtime_001670/`:
`general_candidate_durability_qualification/`,
`all66_R0_Halo145_actor_contexts/`, `empty_ray_recovery/actor_metrics/complete.json`,
`empty_ray_recovery/ROOT_native_ND_empty_recovery_comparison.png`, and
`empty_ray_recovery/EmptyRayRecovery_V4_record2_4K.png`.

## Finite weak-ray comparison

The cached secondary was also tested only where primary alpha is at most
0/.001/.01/.05 and secondary alpha is greater. The decision uses no target
RGB or actor labels. Labels are used exclusively for evaluation. All unselected
RGB/alpha/depth quantiles remain bitwise unchanged, and secondary cells remain
the exact same145-cell candidate. No new grid or GPU render was introduced.

| Maximum primary alpha | PSNR | SSIM | LPIPS | Changed actor pixels |
|---|---:|---:|---:|---:|
|0|45.942275|.991698671|.002351804|13 black /0 grey|
|.001|45.942275|.991698671|.002351804|13 black /0 grey|
|.01|45.942356|.991698690|.002351907|14 black /0 grey|
|.05|45.942465|.991698771|.002351373|22 black /0 grey|

All four settings have zero paired actor/view LPIPS regressions against R0
on these66 in-sample cameras. The .05 candidate has the best mean LPIPS and
also improves PSNR/SSIM. Changes are very small globally. ROOT viewed the
native ND comparison and actual full4K image: the residual step is reduced
further, although it is not completely eliminated. This remains a candidate
requiring other-frame and temporal review before general export promotion.
The full4K predicate replaces60472 rays, including background; it does not
promise unchanged background geometry or solve existing floor fragments.

The pure helper now accepts an explicit `max_primary_alpha` in[0,.05], default0.
Its replay matched all66 cached predictions at every measured threshold;
invalid thresholds and mismatched checkpoints are rejected. Receipt:
`low_alpha_recovery_sweep/pure_helper_v2_qualification.json`. Additional
evidence: `low_alpha_recovery_sweep/complete.json`,
`ROOT_ND_native_weak_ray_comparison.png`, and `WeakRayRecovery_4K.png` in that
directory. No training setting, learned field or original checkpoint changed.
