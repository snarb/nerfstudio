# DEC5 000971: local shelf diagnosis and block-activation control

## What was tested

The reviewed 150-time candidate has a notable skin-colored triangular shelf above
the lipstick at `000971`. The existing video and its honest failure annotation
remain immutable in `/mnt/data/dec5_smooth_150_reviewed`.

Experiment root: `/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971`.
`diagnose_temporal_mesh_shelf.py` binds a manually reviewed portrait polygon to
the original mesh and render hashes, selects predominantly contained triangles,
and checks what surfaces a local deletion exposes. Selection is a hypothesis,
not independent camera evidence. No new surface is invented.

`run_temporal_full_block_control.py` separately reconstructs exactly 62 real
train depth maps with the pinned CUDA COLMAP `5509fffe`, then compares per-view
block activation with full bounded block-union TSDF integration on those same
depths. Fixed poses and the historical geometry JPEG ingest are reproduced;
the video renderer still uses its existing single global exposure and fixed
physical-camera profiles. Geometry-test JPEG gains never enter video texturing.
Worker state, depth counts, GPU, free space and compact logs are recorded every
30 seconds. No source or published result is modified or deleted.

`review_temporal_full_block_control.py` prepares matched same-camera hard-source
renders, nearby real train crops and independent native-depth footprint votes.
Mesh coordinate gauges are explicitly transferred with a point round-trip check;
no silent normalization mismatch is allowed. Here, the two focus-center roundoffs
differ by only 1.2e-7 before scale. Missing measured depth is unknown, not free space.

## Results

| Local deletion | Removed triangles | Newly missing target mesh pixels | Actual visual decision |
|---|---:|---:|---|
| Reviewed outline, ≥80% projected containment | 83 | 32 | Reject: cuts genuine surface, shelf persists, new slits |
| Seven contained faces with deeper underlying hits | 7 | 0 | Reject: shelf persists with mottled patchwork |

The seven-face subset exposes 87 pixels of existing deeper geometry, with a
median depth separation of 0.02294 normalized units. This is a raycast
diagnostic, **not a quality metric or proof of a correct underlying surface**.

Six real same-time train views were visually inspected at twice native size.
They show a compact cylindrical tube top, not the wide skin-colored shelf.
These are nearby-view references, not exact novel-view ground truth.

- [Broad removal comparison](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/shelf_polygon_v1/comparison_detail.png)
- [Conservative removal comparison](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/shelf_polygon_v2/comparison_detail.png)
- [Six real train views](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/full_block_control/real_train_reference_shelf.png)

### Matched TSDF control: selected

All 63 historical staged JPEGs and gains match exactly. Photometric and geometric
passes took 315.9 and 453.8 seconds. There are 62 scalar 1080x1920 geometric maps,
mean coverage 0.3824820633 and minimum coverage 0.2475096451, exactly matching the
historical coverage summary. No new pose optimization, masks or synthetic views.

| Same real depths and render recipe | Vertices | Triangles | Native visual result |
|---|---:|---:|---|
| Per-view block integration | 80,502 | 155,310 | Reproduces false shelf |
| Full bounded block-union integration | 80,485 | 155,256 | Shelf removed; no new lip hole |

The two fusions took 4.3 and 5.7 seconds. Representative false faces have 22–25
robust farther-depth votes and **zero** near-surface votes. The rule samples 25
native taps, requires 80% support, uses a free-space gap exceeding both 0.005
normalized units and 1% depth, and a near tolerance of 0.0015. Of the 83 faces in
the broad rejected deletion proposal, 63 meet at least three free-space and fewer
than two near votes. Unlike mesh self-visibility, these are raw train depth tests.

Full-block fusion changes the same TSDF algorithm's update domain: reliable
farther observations now update previously discovered foreground blocks too.
This suppresses the false zero crossing that per-view surface-block activation
left behind. Merely requiring extraction weight 2 did not establish actual
multi-view surface agreement for this shard. Reflection may make raw matching
difficult, but this matched control localizes the tested shelf to fusion updates;
it does not prove that specularity caused every original bad depth.

[Matched detail](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/full_block_control/comparison_detail.png),
[matched face/hand](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/full_block_control/comparison_lipstick_hand.png).

Selected 000971 uses only full-block fusion and original fixed-profile hard
train-RGB texturing. It uses neither rejected manual deletion nor a cylinder or
diffusion prior. The final 150-time video reuses 149 byte-identical old renders
with explicit receipt ancestry and substitutes this one corrected render.
The updated four-frame native review group, actual MP4 overview and nine
consecutive decoded crops were inspected. No replacement-time jump or new lip
hole was observed. Minor existing hair/shoulder fringe and skin seams remain.

Workspace: `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_repaired_v3`.
Portable video bundle: `/mnt/data/dec5_smooth_150_final`.

### Reliability correction

The initial import exited after writing all depth and transform bytes because
NFS rejected `copy2` timestamp preservation. A strict recovery script compared
all 62 imported numeric arrays with their original COLMAP arrays and verified
source-copy hashes, image links and the all-zero eval placeholder before allowing
fusion. The failed log and explicit recovery receipt remain; no depth rerun was
needed. The importer now uses byte-only `copyfile` for its provenance snapshot,
and the control's QC correctly handles COLMAP's scalar `(H,W,1)` storage shape.
Neither fix changes camera, depth or rendering numerics. Original executed source
snapshots are retained under the control's `config/` directory.

Final regression run: **70 passed, 1 deliberately skipped GPU case**. Tests cover local selection, repair guards, gauge transfer, source
prior, temporal review, receipt-equivalent reuse, TSDF free-space behavior and
an import on a filesystem that forbids metadata copying. GPU-specific testing
was deliberately skipped while the single-GPU reconstruction was running.

## Insights

Deleting only the visible front triangles is insufficient: erroneous surfaces
remain behind them. Even zero new raycast holes does not guarantee good RGB.
The false shelf must be addressed using measured depth/free-space consistency
or a separately validated shape prior, not by declaring a successful edit from
triangle counts alone. The existing 150-time candidate is preserved while this
single problematic time is tested. The measured full-block control repairs this
defect; all 150 times are retained in the final reviewed viewing deliverable.

## Reproduction

```bash
python LookCloser/scripts/run_temporal_full_block_control.py --frame 000971 --output NEW_CONTROL
python LookCloser/scripts/review_temporal_full_block_control.py prepare --control NEW_CONTROL
python LookCloser/scripts/review_temporal_full_block_control.py render --control NEW_CONTROL
python LookCloser/scripts/review_temporal_full_block_control.py compare --control NEW_CONTROL
# After actual visual acceptance, preserve every unaffected time with checked ancestry:
python LookCloser/scripts/compose_verified_temporal_mesh_video.py compose \
  --parent PRIOR_150 --replacement NEW_CONTROL/fuse-full-block_render --output NEW_150
```

The historical NFS-failed workspace was completed with
`finish_temporal_full_block_control.py` before the byte-copy fix. That recovery
is deliberately limited to its exact terminal metadata-copy failure; it cannot
bless an arbitrary failed or partial import.
