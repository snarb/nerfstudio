# Constant-step occupancy eval dilation

## What was tested

The frozen HDR control uses constant-step `occupancy` traversal. Its implementation
omitted `_ensure_eval_occupancy_dilation()`, although adaptive traversal called it.
Consequently `occupancy_eval_dilation_radius=1` was a silent no-op in this mode.
The repository repair adds that existing hook before constant-step sampling.
It changes neither the field nor the training occupancy update.

## Results

An 8³ grid with one admitted central cell and a ray through its adjacent Y cell
provides a direct traversal regression. The CUDA case uses the actual nerfacc
`OccGridEstimator.sampling`, with 0.01 step size and alpha threshold zero.

| Implementation | Admitted cells | Native CUDA samples along neighboring ray |
| --- | ---: | ---: |
| Missing hook | 1 | 0 |
| Existing hook called | 27 | 75 |

The four new CPU/CUDA tests and thirteen existing occupancy diagnostics tests
passed. Repeated eval calls retain the same mask and ray intervals; radius zero
retains the original mask; `train(True)` restores the saved mask bitwise. A second
eval/train cycle also restores correctly. Tests stop after traversal to isolate
grid admission from neural rendering. No optimizer step was taken.

Test source: [test_occupancy_eval_dilation_path.py](../tests/test_occupancy_eval_dilation_path.py).
Local logs: `/tmp/marshawn_eval_dilation_tests.log` and
`/tmp/marshawn_eval_dilation_ablation.log`.
Receipt: `/fsx/tmp/marshawn_001660_hd/staged_hdr_occupancy4/eval_dilation_fix_receipt.json`.

Model source SHA256 before: `ca6d7560002db6c43f4d0e70e412bacceae3236a544be5eaf1449de7b5a018d2`.
After: `23cb0830e83f87164f6d87f372d522d0ec70afc6a020280ac30c4f21e54e6927`.
A byte comparison confirms that the hook call is the only model difference.
The active frozen HDR training package retains the original source hash.

## Insights

Default versus eval +1 comparisons now exercise different traversal masks in
both native occupancy and adaptive modes. The guard admits a one-cell margin;
it does not prove those extra cells contain geometry or certify hidden volume.
Final checkpoint comparisons need a separately fingerprinted inference snapshot
containing the repair. Existing frozen training and previous measurements remain
unchanged.
