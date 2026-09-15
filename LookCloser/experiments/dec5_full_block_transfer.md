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

**In progress; no new geometry or quality result is claimed yet.**

Local root: `/mnt/data/dec5_full_block_transfer`.
Remote root: `/fsx/oregon/dec5_full_block_transfer` on dev3.

At the launch checkpoint on2026-09-15, 000995 matches **63/63 historical staged
JPEG hashes and gains**. Export, undistortion and PatchMatch configuration completed.
The remote photometric worker is live; the independent clever-shadow video
queue is unaffected. 000997 is planned as the second same-rule control and is
not launched before the first time reaches a terminal state.

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
000971's correction survived, while generalization to other affected times
remains untested. The next decision depends on the paired geometry/RGB result,
not the number of added/removed triangles or aggregate face metrics alone.
