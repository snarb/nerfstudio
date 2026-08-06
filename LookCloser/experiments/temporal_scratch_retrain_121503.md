# Temporal scratch retraining through step 121503

## What was tested

Frames `007761`, `007768`, `007775`, `007782`, `007789`, `007796`, and
`007803` were retrained sequentially from scratch with seed 42. The reviewed
single-frame recipe was reused without modifying the historical temporal
fine-tuning runner. Training ended at checkpoint step `121503`
(`max_num_iterations=121504`, update indices `0..121503`).

The scratch stages were:

- feature reweighting 1.0 through step 75940;
- feature reweighting 0.3 through step 106316;
- final continuation through step 121503.

Every saved boundary received a fresh three-view evaluation. Selection first
required an explicit native-resolution visual pass, then used maximum PSNR and
minimum LPIPS within an inclusive 0.07 dB window. No loss values are reported.

Campaign artifacts:

- `/mnt/data/lookcloser_temporal_scratch_retrain_121503`;
- hourly supervision: `supervision.jsonl`;
- per-frame boundary comparisons: `visual_review/<frame>/all_boundaries_native.png`;
- replaced accepted artifacts: `replaced_artifacts/<frame>/{snapshot,render}.tar`.

## Results

All selected checkpoints were at step 121503 and passed the visual gate.
Metrics below are from fresh `ns-eval` calls loading only each promoted
`snapshot/config.yml`.

| Frame | Old PSNR | New PSNR | Old SSIM | New SSIM | Old LPIPS | New LPIPS | Hard gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 007761 | 29.729786 | 29.725723 | 0.681559 | 0.671803 | 0.222157 | 0.216905 | pass |
| 007768 | 29.369335 | 29.788849 | 0.685093 | 0.685677 | 0.232447 | 0.227504 | fail |
| 007775 | 28.974316 | 29.692703 | 0.686357 | 0.674919 | 0.245003 | 0.230436 | fail |
| 007782 | 29.254962 | 29.754587 | 0.685664 | 0.688165 | 0.250975 | 0.226525 | fail |
| 007789 | 29.213392 | 29.612612 | 0.684686 | 0.679932 | 0.250949 | 0.233566 | fail |
| 007796 | 29.071283 | 29.822910 | 0.688657 | 0.673510 | 0.254799 | 0.236158 | fail |
| 007803 | 28.537096 | 29.572836 | 0.693037 | 0.675397 | 0.258206 | 0.225718 | fail |

The hard gate is PSNR >= 29.7, SSIM >= 0.668, and LPIPS <= 0.22. Frames that
did not clear it are recorded as `budget_exhausted_best_available`: the
authorized maximum step was reached and longer training was forbidden.

## Remaining stride-7 sweep

The same single-run scratch recipe was then applied sequentially to every
frame absent from `metrics.csv`, from `007817` through `008048`. All 34 frames
received eight boundary evaluations, native-resolution comparison against the
previous accepted frame and leader `007740`, and a fresh promoted-snapshot
evaluation. Thirty-three selections were at step 121503; `007866` selected
step 106316 because it was inside the inclusive 0.07 dB PSNR window and had
lower LPIPS.

Campaign artifacts:

- `/mnt/data/lookcloser_temporal_scratch_remaining_121503`;
- hourly supervision: `supervision.jsonl` and `manual_supervision.log`;
- per-frame metrics, three-view renders, configs, hashes, and wall timings:
  `campaigns/<frame>_scratch_seed42_maxstep121503`;
- visual comparisons: `visual_review/<frame>`.

| Frame | Step | PSNR | SSIM | LPIPS | Gate tier |
|---|---:|---:|---:|---:|:---:|
| 007817 | 121503 | 29.885515 | 0.681702 | 0.218416 | hard |
| 007824 | 121503 | 29.783222 | 0.675501 | 0.215351 | hard |
| 007831 | 121503 | 29.877871 | 0.684415 | 0.224555 | budget |
| 007838 | 121503 | 30.015127 | 0.680769 | 0.214522 | preferred |
| 007845 | 121503 | 30.032993 | 0.679316 | 0.207697 | preferred |
| 007852 | 121503 | 29.820633 | 0.684818 | 0.206944 | hard |
| 007859 | 121503 | 30.101961 | 0.686571 | 0.207080 | preferred |
| 007866 | 106316 | 29.804564 | 0.680188 | 0.217608 | hard |
| 007873 | 121503 | 29.947783 | 0.699448 | 0.212836 | preferred |
| 007880 | 121503 | 30.051983 | 0.676968 | 0.207257 | preferred |
| 007887 | 121503 | 29.961292 | 0.678227 | 0.208185 | preferred |
| 007894 | 121503 | 29.984020 | 0.689297 | 0.204013 | preferred |
| 007901 | 121503 | 29.747511 | 0.684150 | 0.200828 | hard |
| 007908 | 121503 | 29.828775 | 0.680428 | 0.203030 | hard |
| 007915 | 121503 | 29.857964 | 0.679936 | 0.206184 | hard |
| 007922 | 121503 | 29.877634 | 0.680334 | 0.204363 | hard |
| 007929 | 121503 | 29.864353 | 0.679770 | 0.200185 | hard |
| 007936 | 121503 | 29.868259 | 0.676669 | 0.201941 | hard |
| 007943 | 121503 | 29.986391 | 0.683076 | 0.205718 | preferred |
| 007950 | 121503 | 29.776541 | 0.684697 | 0.204572 | hard |
| 007957 | 121503 | 29.440752 | 0.685384 | 0.206327 | budget |
| 007964 | 121503 | 29.888968 | 0.680307 | 0.205741 | preferred |
| 007971 | 121503 | 30.054703 | 0.704958 | 0.201040 | preferred |
| 007978 | 121503 | 29.959276 | 0.678335 | 0.198962 | preferred |
| 007985 | 121503 | 29.922159 | 0.688045 | 0.199852 | preferred |
| 007992 | 121503 | 29.903355 | 0.685864 | 0.201376 | preferred |
| 007999 | 121503 | 29.765591 | 0.678030 | 0.202753 | hard |
| 008006 | 121503 | 30.020674 | 0.677296 | 0.206896 | preferred |
| 008013 | 121503 | 29.790863 | 0.684245 | 0.206727 | hard |
| 008020 | 121503 | 29.924982 | 0.675313 | 0.203261 | hard |
| 008027 | 121503 | 29.908714 | 0.677382 | 0.202715 | preferred |
| 008034 | 121503 | 29.999119 | 0.684953 | 0.198325 | preferred |
| 008041 | 121503 | 29.865732 | 0.680087 | 0.201861 | hard |
| 008048 | 121503 | 29.850962 | 0.679392 | 0.206374 | hard |

All 34 passed the explicit visual gate. Thirty-two passed every hard numeric
gate and 16 reached the preferred combined target. `007831` failed only LPIPS
and `007957` failed only PSNR; both exhausted the fixed budget and are recorded
as best available rather than silently extending training.

## Insights

Scratch training decisively removed the cumulative PSNR/LPIPS degradation seen
in the fine-tuned chain. Compared with the replaced fine-tuned snapshots,
scratch improved PSNR by up to 1.036 dB and LPIPS by up to 0.0325. SSIM did not
move uniformly and fell on several frames, so the explicit visual gate remains
necessary rather than relying on a single metric.

The selected moving-detail crops retained separated fingers and a continuous,
sharp chain. Automated full-view artifact flags were separately inspected
against ground truth and treated as false positives only after native-resolution
review.
