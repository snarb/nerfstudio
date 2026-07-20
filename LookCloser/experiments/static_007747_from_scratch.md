# Static LookCloser from scratch on 007747

## What was tested

Goal: train LookCloser from random initialization on
`/home/brans/temporal_perframe_stride7_45f/007747`, reproduce or improve the canonical 007740
leader metrics, and preserve the contacting hands/fingers/chain in eval0. The fixed visual crop is
`(700, 100, 1120, 480)`.

The initial ladder is canonical FAS1/hash23, FAS0.75/hash23, then hash24 with the better sampling
setting. E1--E6 use seed42; E7 is the explicit seed43 control. E1--E7 use the canonical
75940-update FR1.0 stage and FR0.3 continuation; E8/E9 isolate FR0.2/FR0.1 from the same verified
seed42 Stage-A parent. All runs use stable occupancy and full evaluation every 15188 local updates.
No leader weights are loaded.

## Accepted recipe

The result was accepted after the user's full-resolution visual review on 2026-07-20. The automatic
strict PSNR gate remains deliberately visible: the trajectory maximum is `0.0618 dB` below the
leader and therefore inside the requested `0.07 dB` comparison window, but the selected
LPIPS-tie-break checkpoint is `0.0871 dB` below the leader. It is selected because its LPIPS is
better and it is only `0.0253 dB` below the trajectory maximum.

The frozen coordinates are:

| Coordinate | Accepted value |
|---|---|
| Dataset | `/home/brans/temporal_perframe_stride7_45f/007747` |
| Train/eval split | filename split, 66 train + 3 eval |
| Frequency maps | `lookcloser_frequencies_chroma422`, verified 66 `.pt` + 66 `.json` files |
| Initialization | random, seed42; no leader checkpoint in training ancestry |
| FAS / hash table | `0.75` / log2 size `24` |
| Stage A | FR1.0 through cumulative step75940 |
| Stage B and tails | campaign-local continuation with FR0.2 |
| Evaluation cadence | every 15188 local updates; keep every checkpoint |
| Final selector | maximum full-eval PSNR, then minimum LPIPS within `0.07 dB` of that maximum |
| Selected checkpoint | step197444, SHA-256 `7df570ec0bfa923782d4fc187191ab8e349c5660c74dd78a2edf131bf8b253b3` |
| Visual protocol | eval0 crop `(700, 100, 1120, 480)`; fingers separated and chain continuous |

E8 was computed efficiently by branching from E4's campaign-local seed42 Stage-A checkpoint. That
checkpoint itself was trained from random initialization. The final controller below recreates the
same Stage A before continuing with FR0.2, so this optimization does not introduce leader ancestry.

## Results

Canonical leader targets:

| PSNR | SSIM | LPIPS | Eval0 contact ROI PSNR | ROI SSIM | ROI LPIPS |
|---:|---:|---:|---:|---:|---:|
| 29.840143 | 0.669203 | 0.219455 | 29.735380 | 0.773583 | 0.112038 |

The previously rejected transfer checkpoint at step227820 measures
`29.150847 / 0.770933 / 0.119176` on the fixed ROI and is visibly blurrier than the leader. It is
retained only as a negative visual reference, not as training ancestry.

Completed campaign selections:

| Campaign | Selected step | PSNR | SSIM | LPIPS | ROI PSNR | ROI SSIM | ROI LPIPS | Plateau | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| E1 canonical FAS1/hash23 | 167068 | 29.360167 | 0.671818 | **0.207288** | 29.434307 | 0.770874 | **0.108790** | yes | no |
| E2 FAS0.75/hash23 | 167068 | **29.464527** | **0.675807** | 0.226435 | **29.584771** | **0.780366** | 0.116583 | yes | no |
| E3 FAS0.75/hash24 | 151880 | **29.544245** | **0.676974** | **0.206243** | **29.687953** | **0.780542** | **0.112468** | yes | no |
| E4 hash24 + corrected maps, FW0.3 | 197444 | 29.731533 | 0.673260 | 0.203957 | 29.775435 | 0.783075 | 0.112854 | yes | no (strict PSNR) |
| E8 hash24 + corrected maps, FW0.2 | 197444 | **29.753056** | **0.673878** | **0.203921** | **29.776102** | **0.783162** | **0.112835** | yes | user-accepted selector-window + visual pass |

E1's fixed crop passed manual review: fingers remain separated, wrist-chain links remain continuous,
and the patch is not blurrier than the leader. It still fails the global numeric gate by
`0.479976 dB` PSNR and has one separate significant eval2 artifact. E1 stopped after the last two
intervals (`136692→151880→167068`) were both numeric plateau and visually unchanged.

E2 recovers `0.104361 dB` global PSNR and `0.009492` ROI SSIM over E1, but gives back
`0.019148` global LPIPS and `0.007792` ROI LPIPS. Its fixed crop also passes manual review, while a
separate selected-checkpoint full-view artifact and both PSNR/LPIPS leader gates fail.

E3 dominates E1/E2 globally on the PSNR/LPIPS selector. Its selected crop passes manual review and
the artifact gate; it is within `0.04743 dB / +0.00696 SSIM / +0.00043 LPIPS` of the leader ROI.
Global SSIM and LPIPS beat the leader, while global PSNR remains `0.295898 dB` lower.

E4 is the isolated validation of the chroma-normalized maps. At the common step60752 it improves
E3 by `0.242239 dB` PSNR. Its maximum PSNR is `29.764877` at step182256; the prescribed
PSNR-window/LPIPS selector chooses step197444. The selected checkpoint is clean and its crop passes,
but strict global PSNR remains `0.108610 dB` below the leader. Three trailing numeric plateau
intervals and two explicit no-improvement reviews confirm the E4 stop.

The FAS sweep around E4 rejects both sides of `0.75`. At step60752, FAS1.0 (E5) is
`29.1587 / 0.672446 / 0.230018`, while FAS0.70 (E6) is
`29.3865 / 0.684979 / 0.247695`; both are PSNR-dominated by E4's
`29.666052 / 0.676152 / 0.247155`. Seed43 (E7) reaches
`29.4948 / 0.678794 / 0.242731` at the same horizon and is also rejected. These ablations were
stopped as dominated after the mandatory horizon, not mislabeled as plateau.

The Stage-B feature-reweighting sweep branches only from E4's verified random-initialized Stage-A
checkpoint. FW0.1 (E9) gives `29.7541 / 0.675594 / 0.224261` at step91128 and loses to FW0.2.
FW0.2 (E8) reaches the trajectory maximum `29.7784 / 0.674460 / 0.207042` at step167068:
PSNR is within `0.0618 dB` of the leader while SSIM and LPIPS beat it. The prescribed selector
chooses step197444 because it is within `0.0253 dB` of that maximum and improves LPIPS to
`0.203921`. Fresh evaluation reproduces `29.753056 / 0.673878 / 0.203921`.

Fresh full eval and fixed-ROI renders were also archived at the three final E8 boundaries:

| Step | PSNR | SSIM | LPIPS | ROI PSNR | ROI SSIM | ROI LPIPS | Visual |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 167068 | 29.778376 | 0.674460 | 0.207014 | 29.774343 | 0.783692 | 0.114549 | pass |
| 182256 | 29.743654 | 0.674027 | 0.205168 | 29.720370 | 0.782661 | 0.114160 | pass |
| 197444 | 29.753056 | 0.673878 | 0.203921 | 29.776102 | 0.783162 | 0.112835 | pass |

Both final global intervals satisfy the numeric plateau thresholds: PSNR growth is negative then
`+0.009401 dB`, SSIM growth is negative in both, and LPIPS improves by only `0.001847` then
`0.001247`. Full-resolution side-by-side review of all three fixed crops shows no visible
moving-detail improvement over the final two intervals. Fingers remain separated and the chain is
continuous at every boundary, so the required two-interval numeric-and-visual plateau is confirmed.
The renders and signed ROI protocols are stored beside the selected evaluation under the sibling
`step-000167068`, `step-000182256`, and `step-000197444` directories.

The E8 selected fixed crop is `29.776102 / 0.783162 / 0.112835`, compared with leader
`29.735380 / 0.773583 / 0.112038`; there are no serious artifacts. Enlarged visual review passes:
individual fingers remain separated, the chain is continuous without gaps, and contact detail is
not visibly blurrier than the leader. The selected checkpoint SHA-256 is
`7df570ec0bfa923782d4fc187191ab8e349c5660c74dd78a2edf131bf8b253b3`.

The canonical E1 checkpoint at 121504 (not final; training continued because PSNR still improved)
measured `29.380341 / 0.671829 / 0.213637`. Its fixed ROI measured
`29.379259 / 0.771490 / 0.111701`. The ROI LPIPS and gradient ratio already match or beat the
leader, while global and ROI PSNR remain below it. Two significant full-view cable defects appeared
outside the ROI at this checkpoint, so it is not visually accepted.

## Insights

- The 007740 and 007747 transforms are byte-identical and both split as 66 train + 3 eval.
- None of the 66 target frequency tensors is identical to its 007740 counterpart; about 67.8% of
  patch cells change. Target level15 occupancy is 18.63% versus 12.32%, and mean scalar resolution
  is 3855.9 versus 3379.6.
- At the time of E1--E9, the target JPEGs were q95 4:4:4, whereas canonical 007740 used a different
  quantization and subsampling profile. This was a real dataset-difficulty confounder, not by itself
  proof of an incorrect frequency map. Those image hashes stayed immutable throughout the campaign;
  the later EXR-to-JPEG canonicalization is a separate dataset revision.
- Luminance mean-absolute gradient is effectively unchanged (`0.027424` on 007740 versus
  `0.027452` on 007747), but decoded Cb/Cr gradients increase from `0.003744/0.006331` to
  `0.004966/0.010001`. The high-level map increase therefore tracks the JPEG chroma representation
  much more strongly than useful luminance detail.
- The target maps have a complete `fast_freqmap.py` generation log (66 images at about 70.4 seconds
  each). The leader maps have identical declared JSON geometry but only a common copy timestamp and
  no generation log. Adjacent temporal frames generated by the fast path consistently assign about
  31.7--34.2% of cells to levels 14--15, while 007740 assigns 22.9%. A separately regenerated map
  A/B is required before treating either provenance as canonical.
- Independent one-camera fast-map recomputation shows stochastic cell-level differences but
  reproduces leader aggregate statistics. The same current recipe leaves target high14--15 at
  `28.49%` versus leader `13.12%` for that camera, confirming a data-representation effect rather
  than a swapped map.
- `build_chroma_normalized_frequency_maps.py` applies horizontal 2x Cb/Cr low-pass only inside the
  map estimator, matching the leader JPEG's 4:2:2 component ratio while preserving full-resolution
  luminance and leaving train/eval JPEGs immutable. Across all 66 maps the mean/max luminance delta
  is `1.73e-6 / 3.05e-6`; high14--15 moves from `31.95%` to `20.66%` (leader `22.90%`) and mean
  scalar resolution from `3855.9` to `3242.9` (leader `3379.6`). The original maps remain intact;
  corrected maps live in `lookcloser_frequencies_chroma422` with a source-hashed provenance file.
- A partial-normalization probe was rejected rather than promoted. On representative camera33,
  independent same-seed recomputation gives leader scalar/level/high14--15
  `3273.2 / 11.9917 / 0.2052`; full normalization gives
  `3247.0 / 11.9410 / 0.2057`, whereas strength0.90 overshoots to
  `3398.0 / 12.0394 / 0.2294`. The verified full-correction builder and map manifest remain the
  final preprocessing recipe.

## Root cause and prevention

The failure was not a camera-pose or split mismatch: the 007740/007747 transforms are byte-identical
and both datasets contain 66 train and 3 eval views. During E1--E9 the confounder was the estimator
input: target JPEGs used q95 4:4:4 chroma while leader images had a 4:2:2-like component ratio.
Useful luminance gradients stayed nearly constant, but target chroma gradients and high-frequency
map occupancy increased sharply. The estimator therefore interpreted a compression/export
difference as extra scene detail and changed FAS, frequency-grid updates, and feature reweighting
together.

For future datasets:

1. Freeze an estimator-input contract before generating maps: decoded color representation,
   resolution, chroma bandwidth, patch geometry, estimator parameters, implementation hashes and
   random seed. Maps from different contracts are different datasets and must not share a campaign.
2. Keep train/eval images immutable. If camera or export profiles differ, remove only the nuisance
   difference in a temporary tensor passed to the estimator; do not re-encode the training images.
   The 4:2:2 normalization used here is dataset-family-specific, not a universal LookCloser default.
3. Generate into a new empty output directory and retain the provenance JSON. Never silently reuse
   maps after image bytes, preprocessing code, decoder, parameters, or filenames change.
4. Before a full run, verify one `.pt` and one sidecar per train image, filename binding, tensor
   shape/finite values, source image hashes, and aggregate level statistics. Compare mean scalar
   resolution and levels14--15 as well as luminance and Cb/Cr gradients against adjacent frames or
   a trusted reference. A large map shift with stable luminance is a representation warning.
5. Recompute a representative camera with a fixed seed and run a short same-seed A/B. Promote a
   correction only when map statistics, full metrics, artifact checks and the priority crop all
   agree. Do not infer correctness from a closer histogram alone.

## Exact reproduction

Run from `/home/brans/repos/nerfstudio` on `main`. The promoted maps already exist and are the exact
inputs used by E8. To rebuild them on a fresh copy, the output directory must initially be absent or
empty; do not rerun the builder over a populated promoted directory. The explicit generation
command is:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python \
  LookCloser/scripts/build_chroma_normalized_frequency_maps.py \
  --images-dir /home/brans/temporal_perframe_stride7_45f/007747/images \
  --out /home/brans/temporal_perframe_stride7_45f/007747/lookcloser_frequencies_chroma422 \
  --steps-per-level 1000 --train-batch-size 8192 --eval-patch-batch 16384 \
  --max-res 8192 --patch-size 8 --ssim-threshold 0.95 --ssim-window 7 \
  --lr 0.01 --seed 0
```

The resulting sibling provenance file is
`lookcloser_frequencies_chroma422.provenance.json`. For the promoted set, the builder source hash is
`706eea2fc2ab14d33f1211de786c14faa63361bd6f4b4bacec922a2bd60d175e` and the ordered map/sidecar
manifest hash is `7a393919397994ccc95af871dc06c924915e58c17df0115cd81043b436aa4936`.
Builder provenance records each source-image hash; campaign preflight independently verifies all 66
map/sidecar names and hashes, tensor shape/finite values, transforms and the JPEG profile.

The final controller recipe is a full from-scratch run; no E4/E8 or leader checkpoint is passed:

```bash
/home/brans/repos/nerfstudio/.venv/bin/python \
  LookCloser/scripts/run_static_target_from_scratch.py \
  --campaign-name 007747_final_hash24_chroma422_fw02_seed42 \
  --variant custom \
  --frequency-map-dir lookcloser_frequencies_chroma422 \
  --seed 42 --fas-strength 0.75 --log2-hashmap-size 24 \
  --stage-b-feature-reweighting 0.2
```

First add `--dry-run` to inspect the fail-closed preflight and confirm that Stage A contains no load
checkpoint while Stage B loads only that campaign's step75940 checkpoint. Then run the command as
shown. Add one `--tail-intervals 1 --resume` invocation at a time, keeping every other recipe flag
identical, while the recorded plateau conditions are not satisfied. Render and inspect the fixed
crop at each boundary; record `no_improvement` only after an actual side-by-side review. Stop only
after two consecutive intervals satisfy both numeric and visual plateau rules.

The validated selected E8 artifacts are under
`/home/brans/lookcloser_007747_from_scratch_runs/evaluations/007747_fromscratch_E8_fw02/step-000197444`.
