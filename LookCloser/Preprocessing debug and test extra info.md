## Preprocessing debug/test additions

Recent changes are scoped to validating 2D frequency-map preprocessing, not the full LookCloser model.

- Added standalone `lookcloser_debug_preprocess.py --mode overfit`, which overfits a 2D HashGrid on a 256x256 crop from `E004_D014_HD.jpg`, renders the same crop at max level, and writes `gt.png`, `recon_full.png`, `diff.png`, and `stats.json` under `lookcloser_debug_outputs/overfit_hd`.

- `lookcloser_preprocess.py` now supports direct image runs via `--image-path`, so HD/6K crops can be tested without a Nerfstudio dataparser.

- When `max_res` is derived from scene size, preprocessing and the model both use `round(max_res_base * scene_size)` before constructing the frequency schedule.

  