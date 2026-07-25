# Temporal stride-7 45-frame fine-tuning v2

## What was tested

Sequential hash23 LookCloser model-parameter-only transfer from accepted frame
snapshots for `007754..008048`, using three concurrent seeds42/43/44 when VRAM
permits. The fixed recipe is LR0.015→0.0001 over300000 updates, batch4096,
standard frequency maps, FAS1.0, FR0.3 and 4096-update traversal/grid warmup.

Every seed is evaluated and saved every15188 local steps through step151880.
Only hard-gate and explicitly visual-pass checkpoints are selectable. Tail
intervals continue the seeds inside the inclusive0.07-dB PSNR window until the
selected trajectory confirms a two-interval plateau.

## Results

The live source of truth is:

- campaign: `/mnt/data/lookcloser_temporal_perframe_stride7_45f_v2`;
- accepted snapshots: `/home/brans/temporal_perframe_stride7_45f/<frame>/snapshot`;
- final-validation metrics: `/home/brans/temporal_perframe_stride7_45f/metrics.csv`.

The final table and chronological visual audit will be added after frame008048
is accepted. Evaluation loss is intentionally excluded.

## Insights

The campaign is fail-closed: an unreviewed boundary, failed hard gate, invalid
snapshot config, fresh-evaluation drift, or gap in the accepted parent chain
stops progress. Complete non-checkpoint evidence is retained under `/mnt/data`;
only nonselected intermediate checkpoint files are pruned after acceptance,
with their hashes preserved in the frame pruning manifest.
