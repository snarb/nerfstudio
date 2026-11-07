# Best LookCloser native visibility capture: completed

## What was tested

Read-only native packed capture from the actual best LookCloser checkpoint126316,
SHA256 `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`,
under exact root-released v4 package. The frequency grid, original mapped W64
admission, R0/core/EMA, FR=.3, legacy ARM1024 and complete source field stayed
frozen. This measured capture replaces neither geometry nor labels.

Package: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v4`.
Complete receipt: `capture/complete.json`, SHA256
`fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d`.

## Results

| Result | Measured value |
|---|---:|
| Declared TRAIN cameras / usable cache archives | 63 /60 |
| Fixed TRAIN rays / actual final packed samples | 69,888 /12,993,330 |
| Whole-camera UNKNOWN records | CE, JE, NE; zero rays retained explicitly |
| Approved IA UNKNOWN overrides | Only global indices40470 and40622; original positive targets retained, weight0 |
| Native cap1024 hits | 92 rays:91 qualified positives,1 qualified negative,0 UNKNOWN |
| Runtime density-to-weight maximum error | 1.11006e-7 |
| Runtime alpha maximum error | 2.62261e-6 |
| Runtime independent RGB / actual renderer maximum error | 1.49012e-6 /1.25170e-6 |
| Independent CPU all-cache density weights / alpha / RGB error | 1.11007e-7 /3.74098e-6 /2.20515e-6 |
| Independent CPU accepted-box contribution-cap error | 3.57492e-6 |
| Runtime/CPU proof tolerance | Unchanged2e-5 |
| Elapsed source capture workflow | 155.36s |
| Peak observed GPU process memory / RSS | 2596MiB /5,671,985,152bytes |
| OOM evidence | None |
| Terminal state | Controller complete/worker return0; worker gone, GPU compute idle |

All three original model RGB and alpha renders passed source replay before TRAIN
capture. Maximum difference was one saved code. RGB changed fraction was at most
8.57763e-7 against1e-5 limit; alpha at most.000566124 against.001 limit. These are
source-model integrity checks; heldout target RGB/masks did not enter the cache.

The final receipt confirms unchanged full live pipeline, FR, source RNG, primed
nonpersistent LUT, admission and normalization method, all seven serialized source
state trees, source checkpoint and original fixed query files. Full-pipeline cache
digest is `c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636`.
Every cache uses identity world-to-box because source positions are already floor
coordinates. Physical frustum intervals and normalized spacing remain separate.

The independent CPU audit rehashed all60 cache archives and all63 original query
archives, checked every pixel/target/category, the exact two label-weight changes,
sorted packed IDs/counts/cap flags, finite sigma/RGB/weights, full-pipeline digest,
identity frame and independent emission/absorption composition. No qualified
positive had accepted-box contribution capacity below95% of its native alpha.
This is distinct from the historical136 NGP capacity-case identities, which were
retained without filtering; it does not reclassify those identities or certify
physical surface coverage.

Inspect complete evidence in `CPU_cache_audit.json`, `terminal_supervision.json`,
`capture/initial.json` and `capture/source_FHD_parity/parity.json`. The earlier v2
virtual-module guard failure and v3 mixed-precision observer failure remain
preserved in their own namespaces. The successful observer keeps native RGB and
weights untouched and accumulates their product in its naturally promoted dtype.

### CPU-only silver staging

All203 bound head inputs plus supporting source metadata were staged and rehashed
on silver:212 files verified,183 uploaded,460,389,792bytes transferred. The exact
source checkpoint was verified and hard-linked from its existing silver archive;
zero checkpoint bytes were transferred. Original NPZ and canonical contract/
capture JSON bytes remain identical and read-only.

Separate execution aliases are at
`/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/frozen_LC_semantic_head_silver_alias_v1`,
with the local review mirror at the same relative path under the local campaign.
The canonical actual-capture receipt hash remains unchanged. Executable paths
relocate C→SC, while derived adapter/label hash references are explicitly repaired
and recorded in `canonical_equivalence.json`; no numeric objective, prior, seed,
LR, grid, camera packets, labels,136 identities or cache arrays change.

Remote CPU checks verified209 release bindings, all60 cache paths, the derived
contract→adapter→label hash chain and both fresh2064 dry runs. Silver Python is
`/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python`, qualified as
Torch2.7.1+cu128/CUDA12.8 with CUDA uninitialized during these checks. The staged
controller plan targets two independent idle silver slots through local-to-remote
SSH. Staging launched no controller, training, rendering or GPU work.

The first released remote pair exposed two incomplete staging checks: the frozen
controller helper and the IA qualification proof leaf were referenced at runtime
but absent from the declared head binding list. After the controller was staged,
both workers failed at the missing qualification proof before cache loading, head
construction or Adam updates. Failed queues and all original contracts/releases
remain preserved. The exact controller and proof bytes were subsequently staged;
future staging explicitly includes both. A full executable-reference closure
then verified212 bindings, both qualification leaves, both spec/history/output
paths, canonical source endpoints, checkpoint, all60 caches and actual standalone
imports. Both output directories were absent and both workers/GPU processes
terminal. A separate recovery controller changes only job names and queue root;
its new root release remains an explicit prerequisite to execution.

Stage receipts: `frozen_LC_semantic_head_silver_stage_v1/stage_receipt.json` and
`frozen_LC_semantic_head_silver_alias_v1/alias_stage_receipt.json`; the latter has
SHA256 `9786ac1cf0c9075548087ef5f4723fd32936125fe31d187d87949d97c48fc517`.

## Insights and next steps

The direct-LC cache now supports independent actor/nonactor-head proposals on the
actual high-detail field without an NGP geometry transfer. Root and the semantic
agent received the exact cache/receipt/digest immediately on completion, and
remote input staging does not block local technical validation.

Native eval has no opacity-pruning callback, but occupancy skipping and92 legacy
cap hits still limit geometric coverage. The cache partitions original source
occlusion; physical actor extraction requires newly computed transmittance and
separately qualified sampling. Source renders still contain scattered floor/tail
density. No perfect occupancy grid, unseen-free-space result or artifact-free
asset is established by this capture. Further label proposals, including the MD
shoe review, remain separate from these unchanged original queries.
