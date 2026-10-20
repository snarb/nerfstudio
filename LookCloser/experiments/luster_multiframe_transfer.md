# Frozen LookCloser transfer across the completed Luster frames

## What is being tested

User goal (2026-10-20): at least 30% person LPIPS improvement on **every**
available frame, without newly introduced obvious geometric artifacts.
Interpretation follows the original recipe: native mean over heldout012/095/150
for each frame separately; cam007 and local spatial regions additionally must
not regress. A pooled average cannot hide a failing frame. Visual review is
required, and no metric-only run may claim success.

Worktree/results: `/fsx/tmp/luster_multiframe_transfer_20261020/code` and its
parent directory. Branch: `experiment/luster-multiframe-transfer-20261020`.
Do not merge this experiment into main without a new user instruction.
The concurrent training campaign is read-only:
`/fsx/tmp/luster_video_540_599_body80_20261018`.

The initial snapshot has 23 completed final106316 models. It verifies checkpoint
and config hashes against published evaluation markers. Incomplete/intermediate
models are excluded. The source CSV has only HD LookCloser metrics; hybrid
quality has not previously been measured there. Native baselines must be
rendered/scored under the same protocol as native transfer before calculating
improvement. The user's clarification confirmed this CSV.

The worst initial heldout HD LPIPS frames are545 (.083966),544 (.079322),
549 (.078656),551 (.076710),550 (.076364). Cam007 additionally degrades late in
the sequence:598 (.047711),599 (.044390). These values prioritize investigation;
they are **not** native hybrid scores or evidence that transfer already fails.

## Isolation and GPU policy

No new LookCloser training, checkpoint modification, calibration changes,
grid edits or occupancy dilation. No signals, locks or writes to the training
campaign. Its running processes and uncommitted worktree are untouched.
The compatible runtime is copied into this campaign's own directory.

BLAS/OpenMP use two threads (retained OpenCV helpers allow four), and inference
runs at lowered CPU priority. GPU
inference requires >=64GiB free and <24GiB total used at admission. Our Torch
allocator is capped at4GiB; an independent watchdog exits **only our process**
if its GPU footprint reaches6GiB, global use reaches24GiB, free memory falls below
64GiB, or telemetry fails. It never kills or restarts another process. The
observed historical training peak at initial inspection was8114MiB on a97887MiB
device. The watchdog covers allocations outside Torch's allocator as well;
the reserve is intentionally much larger than the observed training footprint.
LPIPS and candidate evaluation stay on CPU. There is no claim that monitoring
is a mathematical guarantee against every unrelated sudden device allocation.
The campaign-only inference lock now queues callers **before Torch import**.
`prefetch_multiframe_geometry.py` prepares544/549 while545 projection runs on
CPU; only one own renderer can allocate CUDA memory. A two-process test with
stubbed CUDA confirms both calls finish and maximum concurrent rendering is1.
Seven queue/frame-isolation tests pass. No trainer lock is acquired.

## Implementation and evaluation

- `inventory_multiframe_transfer.py`: immutable per-frame contexts, checkpoint
  verification, separate baseline CSV and source CSV snapshots.
- `LUSTER_HYBRID_FRAME_MANIFEST`: explicit process-local frame binding in
  `luster_hybrid_common.py`; historical540 defaults remain unchanged.
- `guarded_transfer_inference.py`: bounded native inference, unchanged q50
  geometry, self-only GPU watchdog. No trainer integration.
- `protocol_multiframe_transfer.py`: native per-frame source masks, person
  scores and15 evaluation-only spatial regions following each mask's extent.
  Original540 face/hand rectangles are not pasted onto moving people.
- `summarize_multiframe_transfer.py`: separate per-frame/camera/region CSV,
  missing-frame accounting and minimum-frame gain. Metric gates never substitute
  for visual acceptance.

All sources remain TRAIN-only, target/self exclusion is retained, and one
source photograph supplies output texture per pixel. Other photos can supply
scalar confidence only. Native RGB, camera-z, pixel centers and footprints
retain the accepted recipe's conventions. No target RGB/masks enter prediction.

Frames below585 are development;585 and later form a locked temporal validation
block for the first global recipe selection. No frame-specific tuning or source
assignments. Newly completed frames can be added as prospective checks.

## Results

Initialization tests: **121 passed**, including original transfer tests,
process-local frame binding, unchanged540 defaults, foreign-checkpoint cache
rejection and conservative GPU admission. Initial visual review inspected the
existing545/cam012 GT/LookCloser comparison: fine garment detail is softened;
hair and hand boundaries need close review in the subsequent native transfer.

The source-native confidence pool is frozen to the selected source union before
construction; unrelated extra geometry caches cannot silently change witnesses.
Cam007 remains excluded from all witnesses and from its own prediction. It is
allowed as a TRAIN texture source for other targets if automatically selected;
the original confidence helper's blanket source007 rejection was unnecessarily
restrictive outside540. All heldout photographs remain forbidden for construction.

Native multi-frame hybrid results are pending. No universal improvement has
been demonstrated yet. Frozen inventory and measurements live in
`preflight/completed_frames.json`, `evaluation/baseline_inventory.csv` and
`evaluation/<candidate>_metrics.csv` under the results root.

Native inference pilot on545/cam007: our GPU footprint was approximately1.9–2.1GiB,
with total device use below10GiB while the trainer continued its normal stages.
AABB rectangle culling was ineffective (98.4% of pixels retained). Conservative
saved-occupancy cell projection retained36.85% of pixels: every skipped pixel
had exactly zero alpha in the full reference. Maximum RGB difference was
1.85e-6, maximum q50 difference5.96e-8; finite depth support was identical.
These small floating differences follow different ray batches, not fitted
geometry. Reference, AABB and occupancy controls remain separate in `probes/`;
evidence is `evaluation/native_ray_culling_diagnostic.json`.

The chunk4096 control on545/cam095 took33.18s, compared with68.35s
for chunk1024 (concurrent training load varies). GPU process memory reached
approximately3.2GiB under the unchanged4GiB allocator cap. Maximum RGB
difference was1.31e-6 and q50 difference2.38e-7; finite support was identical.
Evidence: `evaluation/chunk4096_parity.json`. The sequential pilot resumes with
chunk4096; the unrelated trainer was not interrupted.

Common-protocol CPU measurements:

| Frame / candidate | Native person baseline LPIPS | Hybrid LPIPS | Reduction |
| --- | ---: | ---: | ---: |
| 540 / reused accepted v1 reference | 0.1351882741 | 0.0938206365 | 30.6000% |
| 545 / pending | 0.1475184709 | pending | pending |
| 544 / pending | 0.1381023899 | pending | pending |

These are per-frame means across012/095/150. The540 reference uses the
already accepted image, not a new global-recipe result. Its separate CSV is
`evaluation/reference_v1_540_metrics.csv`; local regressions and visual review
remain separate acceptance gates. The training campaign CSV contains baseline
LookCloser metrics only and is never modified by this experiment.

The common spatial evaluation exposes an existing cam007 fold regression:
`spatial_r3_c1` LPIPS0.0316205 ->0.0457955. The saved comparison visibly shows
irregular fold edges in the accepted540 hybrid. This is a failure of the
stricter local gate, even though heldout-person mean improves30.6%.

An unpromoted control, `contour_guard_luster_hybrid.py`, estimates coherent
appearance-edge displacement from the frozen baseline and proposed hybrid.
It attenuates the existing transfer, without moving coordinates or reading
target GT/masks. The predefined screen is sigma2/4 crossed with displacement
threshold1/2 native pixels; coherence threshold.8 and edge strength.015 are
fixed. Three synthetic checks verify identical-input preservation, local
rejection of a shifted step and preservation of detail on a flat surface.
Full multi-frame quality is pending; passing these tests is not evidence of an
improved universal recipe.

First completed screen (`contour_s2_d1`,540): heldout-person LPIPS0.0945535004,
**30.0579%** below the frozen baseline. Cam007 person improves from the accepted
hybrid's0.0673453 to0.0630166. Its problematic fold region improves from0.0457955
to0.0356025, but remains worse than baseline0.0316205. Visual inspection confirms
that some irregular fold edges remain. This control is **not accepted**:
local regression remains and the30% margin is narrow. The screen scores person
plus five predefined spatial regions; it is not a full protocol evaluation.
The summary gate now explicitly rejects incomplete metric coverage, including
a matching regression test; all10 focused multi-frame/contour/summary tests pass.

Completed540 matched screen (all four controls are unaccepted):

| Control | Heldout-person LPIPS reduction | Cam007 fold LPIPS |
| --- | ---: | ---: |
| sigma2, displacement1 | 30.0579% | 0.0356025 |
| sigma2, displacement2 | 30.4762% | 0.0417323 |
| sigma4, displacement1 | 30.2277% | 0.0380800 |
| sigma4, displacement2 | 30.5548% | 0.0431634 |

`evaluation/contour_540_screen.csv` compares the same23 measured camera/region
pairs for every control and the reference. Full protocol coverage is checked
separately. A full evaluation of sigma2/displacement1 is also scheduled.
`run_multiframe_contour_screen.py` waits for the live, identified own v1
controller and runs these same four settings on545/544/549 using CPU only.
This is a matched cross-frame check, not per-frame parameter selection.

The full sigma2/displacement1 evaluation subsequently completed: all protocol
regions are covered, heldout improvement remains30.0579%, and five local
LPIPS regressions remain (four cam007 regions and one cam150 region), versus
seven for the reference. It still fails the local acceptance gate.

All54 source geometries for545 completed and the controller advanced to CPU
projection. Geometry544 is being prefetched under the single own GPU lease.
Baseline544 metrics were evaluated on CPU alongside this work. Initial
cam007 head comparisons for544/545 are saved as
`frames/<frame>/evaluation/baseline_cam007_head_preview.jpg`: ornament outlines
are already relatively crisp, while photographic glints and eyelid contours
differ. These are baseline observations, not acceptance of an unrendered hybrid.

Storage control: a separate zstd level1 copy of545/cam059 RGB compressed from
147456128 to44642188 bytes (30.27%). Streamed decompression reproduced the exact
SHA256. `probes/cache_compression/receipt.json` records the measurement.
No source cache was removed; the result supports future lossless cache archival
if needed to retain the training disk reserve.

CPU corroboration prefetch: after all545 source geometry completed,
`precompute_source_confidence.py` started the same TRAIN-witness confidence
calculation in parallel with cubic interpolation. It uses two OpenCV threads
and reduced CPU priority. A per-source lock prevents a ranking worker from
reading/writing an incomplete shared result; receipt hashes and frozen witness
pool checks remain mandatory. A two-process test verifies serialized access
to the same source, and all seven focused lock/frame tests pass.

## Insights and next steps

First reproduce the accepted v1 transfer on weak development frames and540;
separate poor source visibility/geometric correspondence from source sharpness
and conservative admission. Develop one global selection/merge rule from that
evidence, then run the locked temporal block and every completed frame. Preserve
the earlier negative controls in
[the v1 recipe](luster_000540_hybrid_postprocessing_recipe.md).
