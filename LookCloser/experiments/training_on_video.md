# Training on Video

Generated: 2026-07-07 16:30:17 UTC

## Status

- Dataset: `/home/brans/temporal_perframe_stride7_45f`
- Dataset frame directories: 45
- Leader frame `007740` is reused from the static leader and is not retrained in the chain.
- Active frame: None; temporal transfer chain is complete.
- Completed video frames in report, excluding leader sanity: 44
- Remaining frames after active: 0

## Metrics by Frame

| Frame | Label | PSNR | SSIM | LPIPS | Selected checkpoint | Renders |
|---|---|---:|---:|---:|---|---|
| 007740 | leader_sanity | 29.617964 | 0.668450 | 0.231121 | `/home/brans/lookcloser_temporal_artifacts/static_lookcloser_leader_007740/nerfstudio_models/step-000106316.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_sanity/lookcloser/007740_leader_local_20260705_183929/renders_local_data_step-000106316` |
| 007747 | const5e-4 | 28.882360 | 0.682507 | 0.337838 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_const5e-4_20260705_184726/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_const5e-4_20260705_184726/renders_selected_step-000151880` |
| 007754 | const5e-4 | 29.151651 | 0.687583 | 0.338389 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007754_from_000151880_const5e-4/nerfstudio_models/step-000197444.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007754_from_000151880_const5e-4/renders_selected_step-000197444` |
| 007761 | const5e-4 | 29.191620 | 0.689342 | 0.339054 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007761_from_000197444_const5e-4/nerfstudio_models/step-000243008.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007761_from_000197444_const5e-4/renders_selected_step-000243008` |
| 007768 | const5e-4 | 29.000465 | 0.690822 | 0.343767 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007768_from_000243008_const5e-4/nerfstudio_models/step-000288572.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007768_from_000243008_const5e-4/renders_selected_step-000288572` |
| 007775 | const5e-4 | 28.862797 | 0.689290 | 0.342878 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007775_from_000288572_const5e-4/nerfstudio_models/step-000334136.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007775_from_000288572_const5e-4/renders_selected_step-000334136` |
| 007782 | const5e-4 | 28.858006 | 0.689486 | 0.342062 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007782_from_000334136_const5e-4/nerfstudio_models/step-000379700.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007782_from_000334136_const5e-4/renders_selected_step-000379700` |
| 007789 | const5e-4 | 28.919174 | 0.692387 | 0.343528 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007789_from_000379700_const5e-4/nerfstudio_models/step-000425264.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007789_from_000379700_const5e-4/renders_selected_step-000425264` |
| 007796 | const5e-4 | 28.724560 | 0.694439 | 0.341891 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007796_from_000425264_const5e-4/nerfstudio_models/step-000470828.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007796_from_000425264_const5e-4/renders_selected_step-000470828` |
| 007803 | constant0p0005 | 28.495264 | 0.695099 | 0.341415 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007803_from_000470828_constant0p0005/nerfstudio_models/step-000516392.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007803_from_000470828_constant0p0005/renders_selected_step-000516392` |
| 007810 | constant0p0005 | 28.549046 | 0.698909 | 0.337896 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007810_from_000516392_constant0p0005/nerfstudio_models/step-000561956.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007810_from_000516392_constant0p0005/renders_selected_step-000561956` |
| 007817 | constant0p0005 | 28.411179 | 0.701176 | 0.333163 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007817_from_000561956_constant0p0005/nerfstudio_models/step-000607520.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007817_from_000561956_constant0p0005/renders_selected_step-000607520` |
| 007824 | constant0p0005 | 28.090574 | 0.701069 | 0.332894 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007824_from_000607520_constant0p0005/nerfstudio_models/step-000653084.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007824_from_000607520_constant0p0005/renders_selected_step-000653084` |
| 007831 | constant0p0005 | 28.691614 | 0.701890 | 0.328936 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007831_from_000653084_constant0p0005/nerfstudio_models/step-000698648.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007831_from_000653084_constant0p0005/renders_selected_step-000698648` |
| 007838 | constant0p0005 | 28.979307 | 0.700943 | 0.322036 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007838_from_000698648_constant0p0005/nerfstudio_models/step-000744212.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007838_from_000698648_constant0p0005/renders_selected_step-000744212` |
| 007845 | constant0p0005 | 29.103031 | 0.703886 | 0.318168 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007845_from_000744212_constant0p0005/nerfstudio_models/step-000789776.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007845_from_000744212_constant0p0005/renders_selected_step-000789776` |
| 007852 | constant0p0005 | 29.191847 | 0.703517 | 0.320896 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007852_from_000789776_constant0p0005/nerfstudio_models/step-000835340.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007852_from_000789776_constant0p0005/renders_selected_step-000835340` |
| 007859 | constant0p0005 | 29.124609 | 0.705680 | 0.321062 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007859_from_000835340_constant0p0005/nerfstudio_models/step-000880904.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007859_from_000835340_constant0p0005/renders_selected_step-000880904` |
| 007866 | constant0p0005 | 29.133450 | 0.708523 | 0.318340 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007866_from_000880904_constant0p0005/nerfstudio_models/step-000926468.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007866_from_000880904_constant0p0005/renders_selected_step-000926468` |
| 007873 | constant0p0005 | 29.077347 | 0.708055 | 0.319979 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007873_from_000926468_constant0p0005/nerfstudio_models/step-000972032.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007873_from_000926468_constant0p0005/renders_selected_step-000972032` |
| 007880 | constant0p0005 | 28.994486 | 0.705243 | 0.319318 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007880_from_000972032_constant0p0005/nerfstudio_models/step-001017596.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007880_from_000972032_constant0p0005/renders_selected_step-001017596` |
| 007887 | constant0p0005 | 28.952114 | 0.704623 | 0.321098 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007887_from_001017596_constant0p0005/nerfstudio_models/step-001063160.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007887_from_001017596_constant0p0005/renders_selected_step-001063160` |
| 007894 | constant0p0005 | 28.793875 | 0.703184 | 0.320157 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007894_from_001063160_constant0p0005/nerfstudio_models/step-001108724.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007894_from_001063160_constant0p0005/renders_selected_step-001108724` |
| 007901 | constant0p0005 | 28.823349 | 0.705877 | 0.317109 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007901_from_001108724_constant0p0005/nerfstudio_models/step-001154288.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007901_from_001108724_constant0p0005/renders_selected_step-001154288` |
| 007908 | constant0p0005 | 29.102776 | 0.705958 | 0.316408 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007908_from_001154288_constant0p0005/nerfstudio_models/step-001199852.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007908_from_001154288_constant0p0005/renders_selected_step-001199852` |
| 007915 | constant0p0005 | 29.228661 | 0.708282 | 0.314868 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007915_from_001199852_constant0p0005/nerfstudio_models/step-001245416.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007915_from_001199852_constant0p0005/renders_selected_step-001245416` |
| 007922 | constant0p0005 | 29.270948 | 0.708382 | 0.310958 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007922_from_001245416_constant0p0005/nerfstudio_models/step-001290980.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007922_from_001245416_constant0p0005/renders_selected_step-001290980` |
| 007929 | constant0p0005 | 29.130028 | 0.711945 | 0.312344 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007929_from_001290980_constant0p0005/nerfstudio_models/step-001336544.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007929_from_001290980_constant0p0005/renders_selected_step-001336544` |
| 007936 | constant0p0005 | 29.192364 | 0.711830 | 0.311962 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007936_from_001336544_constant0p0005/nerfstudio_models/step-001382108.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007936_from_001336544_constant0p0005/renders_selected_step-001382108` |
| 007943 | constant0p0005 | 29.214899 | 0.712346 | 0.309114 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007943_from_001382108_constant0p0005/nerfstudio_models/step-001427672.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007943_from_001382108_constant0p0005/renders_selected_step-001427672` |
| 007950 | constant0p0005 | 29.201183 | 0.715358 | 0.309635 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007950_from_001427672_constant0p0005/nerfstudio_models/step-001473236.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007950_from_001427672_constant0p0005/renders_selected_step-001473236` |
| 007957 | constant0p0005 | 29.228802 | 0.711291 | 0.311345 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007957_from_001473236_constant0p0005/nerfstudio_models/step-001518800.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007957_from_001473236_constant0p0005/renders_selected_step-001518800` |
| 007964 | constant0p0005 | 29.252928 | 0.713290 | 0.310319 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007964_from_001518800_constant0p0005/nerfstudio_models/step-001564364.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007964_from_001518800_constant0p0005/renders_selected_step-001564364` |
| 007971 | constant0p0005 | 29.209921 | 0.714950 | 0.308614 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007971_from_001564364_constant0p0005/nerfstudio_models/step-001609928.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007971_from_001564364_constant0p0005/renders_selected_step-001609928` |
| 007978 | constant0p0005 | 29.334623 | 0.709503 | 0.305869 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007978_from_001609928_constant0p0005/nerfstudio_models/step-001655492.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007978_from_001609928_constant0p0005/renders_selected_step-001655492` |
| 007985 | constant0p0005 | 29.330635 | 0.712757 | 0.303354 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007985_from_001655492_constant0p0005/nerfstudio_models/step-001701056.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007985_from_001655492_constant0p0005/renders_selected_step-001701056` |
| 007992 | constant0p0005 | 29.356667 | 0.713455 | 0.304761 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007992_from_001701056_constant0p0005/nerfstudio_models/step-001746620.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007992_from_001701056_constant0p0005/renders_selected_step-001746620` |
| 007999 | constant0p0005 | 29.265444 | 0.715703 | 0.306695 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007999_from_001746620_constant0p0005/nerfstudio_models/step-001792184.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007999_from_001746620_constant0p0005/renders_selected_step-001792184` |
| 008006 | constant0p0005 | 29.427263 | 0.714689 | 0.302944 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008006_from_001792184_constant0p0005/nerfstudio_models/step-001837748.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008006_from_001792184_constant0p0005/renders_selected_step-001837748` |
| 008013 | constant0p0005 | 29.355431 | 0.716896 | 0.303856 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008013_from_001837748_constant0p0005/nerfstudio_models/step-001883312.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008013_from_001837748_constant0p0005/renders_selected_step-001883312` |
| 008020 | constant0p0005 | 29.382126 | 0.716843 | 0.301196 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008020_from_001883312_constant0p0005/nerfstudio_models/step-001928876.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008020_from_001883312_constant0p0005/renders_selected_step-001928876` |
| 008027 | constant0p0005 | 29.305962 | 0.716230 | 0.310034 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008027_from_001928876_constant0p0005/nerfstudio_models/step-001959252.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008027_from_001928876_constant0p0005/renders_selected_step-001959252` |
| 008034 | constant0p0005 | 29.241364 | 0.715073 | 0.311348 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008034_from_001959252_constant0p0005/nerfstudio_models/step-001989628.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008034_from_001959252_constant0p0005/renders_selected_step-001989628` |
| 008041 | constant0p0005 | 29.115667 | 0.715140 | 0.313721 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008041_from_001989628_constant0p0005/nerfstudio_models/step-002020004.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008041_from_001989628_constant0p0005/renders_selected_step-002020004` |
| 008048 | constant0p0005 | 28.942577 | 0.717268 | 0.313425 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008048_from_002020004_constant0p0005/nerfstudio_models/step-002050380.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008048_from_002020004_constant0p0005/renders_selected_step-002050380` |

## Pending Frames

None

## Video Artifacts

Eval-view temporal videos use `eval_img_0000.png` from each selected render directory, cropped to the right render half of the Nerfstudio GT|render image.

| Video | FPS | Frames | Resolution | Notes |
|---|---:|---:|---|---|
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000/eval_view_0000_fps30.mp4` | 30.000 | 45 | 1920x1080 | Realtime preview. Source container FPS was not present in the dataset metadata. |
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000/eval_view_0000_source60_stride7_timeline_fps8p571.mp4` | 8.571 | 45 | 1920x1080 | Timeline assumption for `stride=7` if the source was 60 FPS. |

High-quality eval-view temporal output root:
`/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png`

These videos use `eval_img_0000.png` from each selected frame render, crop the right render half, and save
45 full-resolution `1920x1080` PNG master frames with no temporal interpolation. The FFV1 MKV files are
lossless video containers; the CRF10 MP4 files are high-quality preview copies.

| Video | FPS | Frames | Resolution | Notes |
|---|---:|---:|---|---|
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_fps30_lossless_ffv1.mkv` | 30.000 | 45 | 1920x1080 | Lossless FFV1 from PNG master frames. |
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_fps30_crf10.mp4` | 30.000 | 45 | 1920x1080 | High-quality preview copy. |
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_fps8p571_lossless_ffv1.mkv` | 8.571 | 45 | 1920x1080 | Lossless timeline assumption for source 60 FPS / stride 7. |
| `/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_fps8p571_crf10.mp4` | 8.571 | 45 | 1920x1080 | High-quality timeline preview copy. |

High-quality eval-view verification sheet:
`/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_lossless_verification_sheet.jpg`

Camera-path previews were regenerated with the same code tree used for training
(`/home/brans/repos/nerfstudio_time_run` on `PYTHONPATH`). The earlier invalid outputs under
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048`,
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_fixed`, and the intermediate test output
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working_test` were deleted.

Working camera-path output root:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working`

| Path | Cameras | Video | FPS | Frames | Resolution | Visual check |
|---|---|---|---:|---:|---|---|
| vertical_col_c_D_to_L | D004_C014, E004_C014, F004_C014, G004_C014, H004_C016, I004_C014, J004_C014, K004_C014, L004_C014 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/vertical_col_c_D_to_L.mp4` | 24.000 | 64 | 950x548 | decoded video frames checked |
| horizontal_row_h_B_to_D | H004_B014, H004_C016, H004_D014 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/horizontal_row_h_B_to_D.mp4` | 24.000 | 16 | 956x550 | decoded video frames checked |
| horizontal_row_i_B_to_D | I004_B014, I004_C014, I004_D014 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/horizontal_row_i_B_to_D.mp4` | 24.000 | 16 | 956x548 | decoded video frames checked |
| diagonal_center_D_B_to_L_D | D004_B014, F004_C014, H004_C016, J004_C014, L004_D014 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/diagonal_center_D_B_to_L_D.mp4` | 24.000 | 32 | 952x550 | decoded video frames checked |

Contact sheets:

- Rendered JPG sequence sheet: `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/camera_path_contact_sheet.jpg`
- Decoded MP4 verification sheet: `/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working/camera_path_video_verification_sheet.jpg`

Static leader `007740` camera-path output root:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working`

| Path | Video | FPS | Frames | Resolution | Visual check |
|---|---|---:|---:|---|---|
| vertical_col_c_D_to_L | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/vertical_col_c_D_to_L.mp4` | 24.000 | 64 | 950x548 | decoded video frames checked |
| horizontal_row_h_B_to_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/horizontal_row_h_B_to_D.mp4` | 24.000 | 16 | 956x550 | decoded video frames checked |
| horizontal_row_i_B_to_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/horizontal_row_i_B_to_D.mp4` | 24.000 | 16 | 956x548 | decoded video frames checked |
| diagonal_center_D_B_to_L_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/diagonal_center_D_B_to_L_D.mp4` | 24.000 | 32 | 952x550 | decoded video frames checked |

Leader contact sheets:

- Rendered JPG sequence sheet: `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/camera_path_contact_sheet.jpg`
- Decoded MP4 verification sheet: `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_working/camera_path_video_verification_sheet.jpg`

High-quality static leader `007740` camera-path output root:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png`

This render uses full camera resolution, PNG frame sequences, and a separate inference config copy with
`max_steps_per_ray=2048` and `eval_num_rays_per_chunk=1024` for memory headroom. No training checkpoint or
model weights were modified. The PNG sequences are the master outputs; FFV1 MKV files are lossless video
containers, and CRF10 MP4 files are high-quality preview copies.

| Path | PNG frames | Lossless video | Preview video | FPS | Resolution |
|---|---:|---|---|---:|---|
| vertical_col_c_D_to_L | 64 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/vertical_col_c_D_to_L_hq_lossless_ffv1.mkv` | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/vertical_col_c_D_to_L_hq_crf10.mp4` | 24.000 | 1900x1098 |
| horizontal_row_h_B_to_D | 16 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/horizontal_row_h_B_to_D_hq_lossless_ffv1.mkv` | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/horizontal_row_h_B_to_D_hq_crf10.mp4` | 24.000 | 1915x1103 |
| horizontal_row_i_B_to_D | 16 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/horizontal_row_i_B_to_D_hq_lossless_ffv1.mkv` | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/horizontal_row_i_B_to_D_hq_crf10.mp4` | 24.000 | 1912x1098 |
| diagonal_center_D_B_to_L_D | 32 | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/diagonal_center_D_B_to_L_D_hq_lossless_ffv1.mkv` | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/diagonal_center_D_B_to_L_D_hq_crf10.mp4` | 24.000 | 1907x1102 |

High-quality decoded lossless verification sheet:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/camera_path_007740_hq_lossless_verification_sheet.jpg`

Temporal camera-path videos use one trained checkpoint per output frame, from `007740` through `008048`, without
temporal interpolation. The camera position is moved along the same central path over those 45 temporal frames.

Temporal camera-path output root:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working`

| Path | Video | FPS | Frames | Resolution | Visual check |
|---|---|---:|---:|---|---|
| vertical_col_c_D_to_L | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working/vertical_col_c_D_to_L.mp4` | 24.000 | 45 | 950x548 | decoded video frames checked |
| horizontal_row_h_B_to_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working/horizontal_row_h_B_to_D.mp4` | 24.000 | 45 | 956x550 | decoded video frames checked |
| horizontal_row_i_B_to_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working/horizontal_row_i_B_to_D.mp4` | 24.000 | 45 | 956x548 | decoded video frames checked |
| diagonal_center_D_B_to_L_D | `/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working/diagonal_center_D_B_to_L_D.mp4` | 24.000 | 45 | 952x550 | decoded video frames checked |

Temporal verification sheet:
`/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working/camera_path_temporal_video_verification_sheet.jpg`

## Notes

- Default temporal transfer LR is constant `5e-4`, selected by the LR sweep on `007747`.
- All checkpoints are intentionally preserved; selected checkpoints and hard-stop checkpoints are not deleted.
- This report was generated from completed temporal transfer artifacts.
