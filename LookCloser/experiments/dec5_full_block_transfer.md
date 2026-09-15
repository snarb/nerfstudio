# DEC5: full-block TSDF transfer beyond the repaired 000971 shelf

## What was tested

The [000971 shelf experiment](dec5_temporal_shelf_repair.md) demonstrated that
per-view surface-block activation can leave a false foreground zero crossing.
Integrating every real observation over the bounded union of discovered blocks
removed that particular shelf. This is a measured-data fusion correction, not
a synthetic-view, material-mask or cylindrical shape prior.

The current wide-flight inventory **does retain the corrected 000971 mesh**:
its `untrimmed_mesh` points to the hash-bound full-block control. There is no
evidence that this known fix was lost at 000971. However, current 000995/000997
still descend from historical per-view TSDF meshes. Their lipstick/hand-region
defects are not thereby proven to have the same cause; this experiment tests it.

`run_remote_full_block_transfer.py` is a separate opt-in controller. It stages
one time per launch, validates the historical 63 JPEG hashes and per-image gains,
uses the frozen reconstruction campaign code with source-count12, and reconstructs
one shared set of 62 photometric/geometric depth maps. Both fusion arms use the
same current fuser, with only `--tensor-full-block-integration` changed.
The fixed calibration SHA is
`79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900`.

The exact CUDA COLMAP5509fffe on dev3 is checked before work; binary/data/script
hashes are recorded. A GPU lock and GPU-occupancy preflight prevent another
controller in this experiment from overlapping reconstruction. PID, depth-map
counts, GPU processes, free space and log tails are recorded every30seconds.
Only staged JPEGs travel to dev3. Source EXRs and published geometry stay unchanged.
No masks, target RGB or eval RGB enter geometry. The historical JPEG gains do not
change the fixed exposure/profiles used by current video texturing.

## Results

**Completed at 000995: no convincing lipstick improvement; not promoted.**

Local root: `/mnt/data/dec5_full_block_transfer`.
Remote root: `/fsx/oregon/dec5_full_block_transfer` on dev3.

At the launch checkpoint on2026-09-15, 000995 matches **63/63 historical staged
JPEG hashes and gains**. Export, undistortion and PatchMatch configuration completed.
The independent clever-shadow video queue was unaffected. All remote stages
and eight matched RGB renders subsequently completed with exit code zero.
000997 was not launched after the negative first transfer gate.

All 62 geometric maps are finite at 1080×1920. Positive coverage is
0.38443036576563117 mean and 0.24904947916666667 minimum. Both meshes, metadata,
compact logs and the 62 raw geometric maps were returned and SHA-256 verified.
The per-view mesh has 79,552 vertices / 153,646 triangles; full-block has
79,693 / 153,827. Both have one component. Counts alone do not certify shape.

| Matched camera | New depth pixels | Lost depth pixels | Changed RGB pixels |
|---|---:|---:|---:|
| Actual wide left arc | 133 | 125 | 2,383 |
| H/C train pose | 75 | 450 | 1,570 |
| K/B train pose | 361 | 880 | 3,234 |

These are diagnostic differences, **not PSNR/SSIM/LPIPS or quality scores**.
Main LLM inspected all six initial head/hand panels and three supplementary
native-scale lipstick panels. Fixed initial boxes missed the lipstick in K/B
and much of the head in the moving pose; supplementary crops correct coverage
without rerendering or changing the candidate. In H/C, the false patch beside
the lipstick and broadened shape remain. In K/B, both raw arms retain a blue,
clothing-textured false surface behind the lipstick, absent in train GT. The
moving crop likewise retains the rough membrane. No convincing local benefit
justifies promotion or the second expensive reconstruction.

Native panels:
[moving](/mnt/data/dec5_full_block_transfer/000995/review/moving/lipstick_native.png),
[H/C](/mnt/data/dec5_full_block_transfer/000995/review/H004_C005_1210SZ/lipstick_native.png),
[K/B](/mnt/data/dec5_full_block_transfer/000995/review/K004_B005_1210DS/lipstick_native.png).
The separate [visual verdict](/mnt/data/dec5_full_block_transfer/000995/visual_review.json)
records the negative gate and limited scope. No production mesh or video changed;
remote scratch is retained.

Before promotion, require full-resolution finite62-depth inventory; matched
per-view/full-block meshes from those exact depths; gauge-verified transfer into
the actual video coordinates; matched current hard-source RGB; native train
lipstick/hand crops; and explicit review of both local improvement and new holes.
The prior000971 result is not a pass certificate for these new times. Raw
fusion output must not silently discard the current independently applied head
or silhouette repairs when comparing against the production video.

Two focused controller tests pass: JSON-stable command tuple/list normalization,
unchanged-byte resume, and rejection of changed or nonfinite requests.
The initial000995 worker executes its frozen controller snapshot at
`000995/controller_executed.py`, whose hash matches its immutable request.
After launch, only the local `immutable` helper was corrected to normalize
captured tuples into JSON lists for future resume comparisons. AST comparison
confirms no other controller function changed. The running remote snapshot was
not overwritten. Its first uninterrupted execution is unaffected; an archived
v1 resume may fail closed on tuple/list comparison. Do not bypass request hashes.

## Insights

A correct local experiment must be tracked through later artifact ancestry;
otherwise a hypothesis about a lost fix is easy to assert incorrectly. Here,
000971's correction survived, but at 000995 the same full-block correction is
insufficient. This rejects treating every lipstick fin as the same block-update
bug. It does not establish the remaining cause: erroneous measured depths,
weak geometric constraints, and visibility/source mapping still need to be
distinguished. The native K/B clothing-colored patch is not evidence of RGB
averaging: this controlled renderer selects one hard source per surface patch.
Further work should examine the retained false surface against the actual raw
depth observations before another reconstruction sweep.

### Comparison workflow prepared while reconstruction runs

`review_full_block_transfer.py` receives only a completed paired experiment,
validates the exact retained-file inventory and all62 raw geometric map hashes,
then independently checks their shape/finiteness/coverage and calibrated camera
mapping. Geometry transfer has an all-vertex inverse round-trip check.
It prepares matched current hard-source RGB in the actual left-arc view and
two real train poses (H/C,K/B); native target aliases prevent source-mask clipping
of the target depth. Production is shown as a separately labelled reference,
because the raw fusion pair does not include its later head repairs.

Native train GT, individual native crops and three/four-way panels are retained.
No novel-view quality metric is invented; pixel-change counts are explicitly
diagnostic. Receipt validation rejects path traversal and unexpected filenames.
Three focused controller/receiver tests pass. Receive, gauge validation,
all eight new RGB renders, matched-receipt checks and visual review completed.
`finish_full_block_transfer.py` adds native diagnostic crops and a checksum seal
over retained experiment artifacts, train RGB inputs and renderer dependencies.
It does not modify geometry, exposure, camera paths or source masks.
Final seal rechecked **468 SHA-256 bindings**. The three focused tests were
rerun successfully (3/3, 1.32 seconds); all experiment workers are terminal.
