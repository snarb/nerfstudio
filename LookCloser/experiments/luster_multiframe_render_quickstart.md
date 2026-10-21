# Multi-frame frozen LookCloser hybrid: rendering handoff

Status, 2026-10-21: usable rendering pipeline, not a certified universal quality
improvement. All changes are on `experiment/luster-multiframe-transfer-20261020`;
main and the external training campaign are not modified.

## Render

On olive-lark:

```bash
cd /fsx/tmp/luster_multiframe_transfer_20261020/code
PYTHONDONTWRITEBYTECODE=1 /home/pavlo/.conda/envs/nerfstudio/bin/python \
  LookCloser/scripts/run_multiframe_transfer.py \
  --root /fsx/tmp/luster_multiframe_transfer_20261020 \
  --frames 545 --chunk 4096 --render-only --postprocess contour-s2-d1
```

Replace545 with a frame having a verified `frames/NNNNNN/context.json`, or
supply multiple frame numbers. This command renders native cameras007/012/095/150;
it does not implement an arbitrary camera trajectory. Outputs:
`frames/NNNNNN/candidates/contour_s2_d1/camNNN.png`, float RGB, confidence and
source-ID arrays. Frozen LookCloser RGB is in
`frames/NNNNNN/geometry/camNNN/rgb.npy`. The render-completion JSON lists PNG paths.

`--postprocess none` produces the original `candidates/v1` hybrid. The contour
option reduces added crease contrast in reviewed garment crops; it is not
proof that every local artifact is removed. It is a single fixed rule, not
per-frame selection using GT. Strict-core fallback is experimental and is not
a recommended release setting.

`--render-only` skips evaluation masks, GT scoring, and comparison panels.
Omit it for native baseline/hybrid PSNR, SSIM, LPIPS and review panels. Training
CSV metrics are HD and must not be compared directly with these native results.
The545 cached render-only execution completed successfully; its log is
`logs/render_only_smoke_545.log`, status `render_state.json`.

Cold frames require source geometry and CPU warps/ranking and can take well over
an hour. Cached rerenders are faster. A per-frame lock rejects duplicate
pipeline launches. Our inference uses a campaign-only GPU lease, bounded memory
and a watchdog that can terminate only its own process. Do not bypass it to
accelerate rendering while external training is running.

## Measured scope and limitations

Native heldout-person mean LPIPS reduction (cams012/095/150):

| Frame | Original v1 | Soft contour sigma2/displacement1 |
|---|---:|---:|
|540 historical reference|30.60%|30.06%|
|544|31.15%|30.57%|
|545|31.17%|30.49%|
|549|30.25%|29.79%|

Baseline-only native measurements exist for24 verified frames. This is not24
hybrid validations. Frame551 was still ranking sources at handoff; it is not
included in the result table. Newly completed external models require a fresh
verified inventory before rendering. No new LookCloser model is trained here.

Local regressions remain. Original v1 on545 was visually rejected for an added
irregular dark crease. Soft contour protection reduces that defect but does
not establish universal artifact-free rendering. Small differences near30%
are not sufficient to prefer one recipe; user priority is visible quality and
cross-frame stability. Exact measured numbers remain in separate campaign CSVs.
Do not claim the original >=30%-on-every-frame goal has been achieved.

See [full experiment](luster_multiframe_transfer.md) for provenance, tests,
negative controls and detailed evidence. Geometry, weights, calibration, AABB,
occupancy and frequency grids remain frozen. Each transferred pixel uses one
TRAIN photo; target/self and heldout sources are excluded as specified.
