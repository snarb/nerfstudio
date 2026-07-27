# Temporal stride-7 45-frame fine-tuning v2

## What was tested

Sequential hash23 LookCloser model-parameter-only transfer from accepted frame
snapshots for `007754..008048`. The fixed recipe is LR0.015→0.0001 over300000
updates, batch4096, standard frequency maps, FAS1.0, FR0.3 and 4096-update
traversal/grid warmup.

The initial three-seed test ran seeds42/43/44 concurrently. It proved that GPU
contention dominated latency: the three initial `007761` trajectories took
6.44 hours, while a single trajectory takes approximately 2.1 hours. The
active policy therefore uses one seed43 trajectory with no parallel runs.

Evaluation and saving occur every15188 local steps. The preferred minimums are
PSNR29.7, SSIM0.668 and LPIPS0.217, in addition to an explicit native-resolution
visual pass. Each target may use at most 130% of its parent's selected step,
rounded down to a complete evaluation boundary. If the numeric gates remain
unmet at the cap, the controller selects the best visual-pass checkpoint
inside the budget, records a numeric-gate failure and budget override, and
continues the gap-free chain.

## Results

The live source of truth is:

- campaign: `/mnt/data/lookcloser_temporal_perframe_stride7_45f_v2`;
- accepted snapshots: `/home/brans/temporal_perframe_stride7_45f/<frame>/snapshot`;
- final-validation metrics: `/home/brans/temporal_perframe_stride7_45f/metrics.csv`.

Fresh snapshot-only validation results so far are:

| Frame | Parent | Seed | Step | PSNR | SSIM | LPIPS | Visual | Selection |
|---|---|---:|---:|---:|---:|---:|---|---|
| 007754 | 007747 | 43 | 212632 | 29.913879 | 0.676014 | 0.212282 | pass | all numeric gates |
| 007761 | 007754 | 43 | 212632 | 29.729786 | 0.681559 | 0.222157 | pass | 130% budget fallback |
| 007768 | 007761 | 43 | 151880 | 29.369335 | 0.685093 | 0.232447 | pass | 130% budget fallback |
| 007775 | 007768 | 43 | 136692 | 28.974316 | 0.686357 | 0.245003 | pass | 130% budget fallback |
| 007782 | 007775 | 43 | 167068 | 29.254963 | 0.685664 | 0.250975 | pass | 130% budget fallback |
| 007789 | 007782 | 43 | 197444 | 29.213392 | 0.684686 | 0.250949 | pass | 130% budget fallback |

For `007761`, the parent step produced a raw cap of276421 and a last complete
boundary at273384. Continuing seed43 through318948 did not clear LPIPS0.217;
the best post-horizon LPIPS was still above the gate. The fallback selected
step212632 inside the authorized budget. Its promoted checkpoint SHA-256 is
`9ee81df92a585953301e641d7cdda77f4278e4266cd27faae88bec033fa7c597`.
Evaluation loss is intentionally excluded.

The reported concern that `eval_img_0000.png` had changed camera ordering was
audited before promotion. The ground-truth half of each render matched
`frame_eval_00001.jpg`, `frame_eval_00002.jpg` and
`frame_eval_00003.jpg` pixel-exactly in order, and the corresponding camera
transforms were identical across frames007740, 007747, 007754 and007761. The
visible change is therefore temporal scene motion, not an eval-view swap.

For `007768`, all boundaries through the budget boundary273384 passed the
native crop visual review, but no checkpoint reached PSNR29.7 or LPIPS0.217.
The best observed LPIPS was0.227240 at step258196, while the formal
maximum-PSNR/0.07-dB-window policy selected step151880. A fresh eval loaded
only from the promoted snapshot reproduced `29.369335 / 0.685093 / 0.232447`;
the promoted checkpoint SHA-256 is
`3d8e834eaa7d637921d1a7833284115f706ae4e11943a0f8090ea32a25550381`.

For `007775`, the shorter inherited cap was197444. No boundary reached
PSNR29.7 or LPIPS0.217. Applying the unchanged maximum-PSNR then
inclusive0.07-dB LPIPS rule to visual passes selected step136692; fresh
snapshot-only validation reproduced `28.974316 / 0.686357 / 0.245003`.
The checkpoint SHA-256 is
`6b75cf75f5290c63e79ac209c1827b1ed47a128790b322353d07157737bee575`.

For `007782`, the inherited cap was167068. The final boundary was within
0.07 dB of the maximum-PSNR boundary and had the lowest LPIPS in that window,
so it was selected. Fresh snapshot-only validation reproduced
`29.254963 / 0.685664 / 0.250975`; checkpoint SHA-256 is
`e1ba488c1b367580057f64a80326aaa36085652434debb0d12cb970e854c4c14`.

For `007789`, the inherited cap was212632. Step197444 was inside0.07 dB
of the maximum-PSNR step182256 and had the better LPIPS, so it was selected.
Fresh snapshot-only validation reproduced
`29.213392 / 0.684686 / 0.250949`; checkpoint SHA-256 is
`87c3ad61b490b141c592b8a781ffa70540c055efcf9aaa8d7158185e8826e8cd`.

An API foreground process group was terminated with signal143 after two
hours while this frame was training. The last complete boundary121504 was
intact, and resume continued the same trajectory from it. Subsequent
controllers run in named detached `tmux` sessions, while the existing hourly
JSONL supervision remains authoritative, so API-call lifetime cannot terminate
the training process group.

## Insights

Single-run scheduling is faster in wall-clock time on this GPU than independent
parallel seeds. The 130% cap bounds difficult frames without silently turning a
numeric miss into success: visual review remains mandatory, and the exception
is visible in permanent snapshot provenance.

The campaign remains fail-closed for an unreviewed boundary, visual failure,
invalid snapshot config, fresh-evaluation drift, or gap in the accepted parent
chain. Complete non-checkpoint evidence is retained under `/mnt/data`; only
nonselected intermediate checkpoint files are pruned after acceptance, with
their hashes preserved in the frame pruning manifest.
