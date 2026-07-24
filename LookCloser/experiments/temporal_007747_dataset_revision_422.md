# Canonical 4:2:2 revision of temporal frame 007747

## What was tested

Frame `007747` was the only active temporal frame whose historical JPEGs used
4:4:4 sampling. They and their direct/estimator-normalized maps were archived
outside the dataset at `/home/brans/007747_4_4_4`.

The 69 JPEGs were independently regenerated from EXR on dev3 with
`scripts/convert_temporal_exr_to_leader_jpeg.py`. Standard maps were then
generated once on clever-shadow with `scripts/fast_freqmap.py`:

```text
steps_per_level=1000, train_batch_size=8192, eval_patch_batch=16384,
max_res=8192, patch_size=8, ssim_threshold=0.95, ssim_window=7,
lr=0.01, seed=0
```

The single resulting map set was installed on both clever-shadow and dev3/FSX.

## Results

| Check | Result |
|---|---|
| Archived 4:4:4 JPEGs | 69, sampling `1x1,1x1,1x1` |
| Archived direct maps | 66 `.pt` + 66 `.json` |
| Archived estimator-normalized maps | 66 `.pt` + 66 `.json` |
| New JPEGs | 69, sampling `2x2,1x2,1x2` |
| JPEG content manifest | `629d8dad751e248944fbc6080bc9f5c3a62dce69f00837126fde47e5e08324ed` |
| New map generation | 66 images in `4643.6 s` (`70.36 s/image`) |
| New map content manifest | `b31c4465d9bffee064514bc4c8dfb031d97ce60bc541d4ce5c238ebcd0bbe45c` |
| Map validation | 66 finite `float32` tensors, shape `135×240`, 16 valid levels, sidecars bound |
| Equal to old direct/normalized maps | `0/66` / `0/66` |
| Active stale special-map paths | 0 on clever-shadow and FSX |
| Full cross-machine audit | all JPEG and standard map bytes match for all 45 frames |
| Full dataset manifest | `d326f46d717e35cfac7204d0e01e6ddc870dc92b620535fabca68a220c0ad2c9` |

The new 007747 map distribution is consistent with adjacent canonical frames:

| Frame | mean log2 resolution | fraction `>=2352` | fraction `8192` |
|---|---:|---:|---:|
| 007740 | 11.158344 | 0.774423 | 0.123232 |
| 007747 | 11.148780 | 0.773277 | 0.120501 |
| 007754 | 11.141792 | 0.774034 | 0.116406 |

Active provenance is recorded in
`007747/dataset_revision_422.json` on both dataset copies; its SHA-256 is
`5983bc94168ded04ec6b8fe10ec01f0703417ba903115a01ced4d2b280e996e0`.

## Insights

JPEG canonicalization and map generation are one dataset-revision operation:
maps must be regenerated from the final JPEG bytes, never reused across the
conversion boundary. Historical `lookcloser_frequencies_chroma422` maps were a
valid estimator-only workaround for the archived 4:4:4 inputs, not the correct
maps for the active canonicalized dataset.

New 007747 training must use only
`/home/brans/temporal_perframe_stride7_45f/007747/lookcloser_frequencies`.
The hash23 fine-tuning follow-up is specified in `fine_tuning_task_v2.md`.
