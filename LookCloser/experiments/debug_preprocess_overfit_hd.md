# Debug Preprocess Overfit HD

## What was tested

Implemented and ran `nerfstudio/scripts/lookcloser_debug_preprocess.py --mode overfit` on `/home/ubuntu/repos/look-closer/E004_D014_HD.jpg`.

The run overfits a 2D HashGrid on a centered 256x256 crop, renders the same crop at max level, and asserts that image indexing, UV generation, row-major patch order, and rendered crop coordinates are aligned.

## Results

| Crop x | Crop y | Size | Steps | Batch size | Min res | Max res | MSE | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 192 | 256x256 | 8000 | 8192 | 16 | 2048 | 8.3038e-6 | 50.807 | 0.99750 |

Artifacts:

- [GT crop](../lookcloser_debug_outputs/overfit_hd/gt.png)
- [Max-level reconstruction](../lookcloser_debug_outputs/overfit_hd/recon_full.png)
- [Absolute diff](../lookcloser_debug_outputs/overfit_hd/diff.png)
- [Stats JSON](../lookcloser_debug_outputs/overfit_hd/stats.json)

## Insights

The overfit reconstruction visually matches the GT crop, and the diff is dark across almost the whole patch. This validates the debug script's UV, x/y, patch extraction, row-major reshape, and max-level crop rendering path for this HD crop.
