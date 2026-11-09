## What was tested

The frozen quiet runner stops when the latest sampled1024-ray `eval_loss` fails
to improve. This can disagree with the already computed full three-view result.
The new read-only adapter supplies the unchanged plateau comparison with
`10**(-eval_all_psnr/10)`: the geometric mean of full-image MSEs for native RGB
with PSNR data_range1. It is an actual derived photometric loss, not an early-stop
disable flag. It differs from arithmetic pixel/ray MSE.

The original sampled loss/status/delta are preserved in returned fields. CSV
bytes remain unchanged; no extra eval, sampler/RNG, NN or optimizer work occurs.
Only complete finite all-view PSNR/SSIM/LPIPS rows participate. The caller still
must validate that the dataset has the intended three eval views and range1.

## Results

| Actual transition | PSNR before→after | LPIPS before→after | Sampled stop | Full-view stop on this pair |
|---|---|---|---|---|
| Cold6k→8k |29.0220→29.1284|.286736→.280348|stop|continue|
| Uniform12k→13k |29.0736→29.0888|.302972→.300930|stop|continue|

This proves criterion disagreement, not that Monte Carlo noise alone caused it.
The geometric full-view aggregate and sampled arithmetic MSE have different
weighting. Replaying the **whole** uniform CSV with strict full-view stopping
would stop earlier at10k, where PSNR regresses28.9645→28.9339 even though LPIPS
improves. The helper adds no patience, tolerance or LPIPS-dependent stop rule.

Eleven CPU tests passed: geometric-mean identity, genuine regression/plateau
stopping, sampled-regression/full-view-improvement, incomplete/nonfinite rows,
duplicate steps, missing files, unchanged CSV, and preserved default stop/native
checkpoint selection. Actual cold CSV replay continues through8k. No training
or new runtime snapshot was launched.

- [Helper](/home/pavlo/repos/nerfstudio/LookCloser/scripts/marshawn_all_view_early_stop.py)
- [CPU tests](/home/pavlo/repos/nerfstudio/LookCloser/tests/test_marshawn_all_view_early_stop.py)
- [Internal controller audit with source hashes](/fsx/tmp/marshawn_001660_hd/report/all_view_early_stop_audit.json)

## Integration and semantics

The actual frozen function is `quiet.eval_rows`, rather than `read_evaluations`.
For a reviewed future controller:

```python
from marshawn_all_view_early_stop import configure_quiet_all_view_stop
configure_quiet_all_view_stop(quiet, native.select)
```

This binds the row reader and provenance printer, retaining the existing native
all-view PSNR/LPIPS checkpoint selector. `stop_on_no_improve=True` and the normal
`last >= previous` loss comparison remain active. Do not pass
`--no-stop-on-no-improve`. Quality reports continue to use PSNR, SSIM and LPIPS;
derived/sample loss values belong only to internal controller diagnostics.
