# Native SAM3 preparation on idle silver-sable GPUs

## What was tested

An optional external adapter distributes individual native 8192×4320 camera
views to idle silver-sable GPUs. It imports the immutable campaign's SAM3
`main`, selection, component qualification and native 1008-pixel refinement.
The adapter changes source paths and identity-reference file I/O only, with an
additional read-only observer retaining current-frame person candidates.

The 512×270 boolean identity references are computed by the canonical LOCAL
`seed` function, copied byte-for-byte, and bound to their original reference
frame/archive hash. Each native PNG retains its original SHA. The remote
portable receipt and the original LOCAL source receipt are both preserved;
collection records their explicit mapping instead of pretending their paths
or JSON bytes were identical.

The exact same SAM3 overlay/checkpoint was staged once to
`SV/tools/sam3_overlay`, where `SV` is the silver-sable campaign root. The
checkpoint SHA is `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Native model code SHA is
`48f1cebeded7b9998c65783a87c4eba7accbb91cc1e5525d860f34d95ea126bb`;
component qualification SHA is
`10049fe9b448e1e0ab993acc11eb64e709cd65066ebb507674df90cd40c3d8c4`.
Frozen `V/code` and the active frame-1660 training job were not changed.

## Results

HC1665 was compared with the already completed canonical LOCAL result.

| Canary | Remote time | Refined black/grey, crew veto, UNKNOWN, crop | Identity/component/tile metadata |
|---|---:|---|---|
| Portable I/O adapter | 22.95 s | Bitwise identical, zero differences | Identical |
| Same adapter plus read-only all-person candidate capture | 28.48 s | Bitwise identical, zero differences | Identical |

Both returned the exact canonical NPZ SHA
`1e034ac1021943a288ec2177b2a9b8627e0a7165ac59ca73507ccff02a7cf782`.
The observer preserves all current native person masks and selected fighter
instance indices without another neural inference or an RNG draw. Its adapter
SHA is `f81902e04bca3e1233f8dccbc826add417899122a26e8c3f1bcba2db7cfecb16`.

Root visually reviewed whole-view and native head/feet contour comparisons and
approved the transport/math gate. Evidence is under
`V/operations/sam3_assist/canary_1665_HC/` and
`V/operations/sam3_assist/canary_1665_HC_candidates/`:
`comparison.json`, `LOCAL_vs_SG_edges.png` (first canary), returned operation
receipts, native archives and full-person candidate archives.

CPU checks verified source-qualified candidate publication, rejection of a
tampered reference binding, and refusal to overwrite an existing LOCAL mask.
The old helper source is retained under `operations/sam3_assist/code/versions/`.

## Insights and integration

This verifies transport and canonical calculation equivalence; it does not
certify the semantic correctness of every person mask. MC1665 exposed a real
crew-head inclusion in the grey actor's canonical mask. A fresh crew guard and
person-identity qualification are mandatory before `prepared_ready` publication.
An all-person candidate cannot be used as a blind veto: one MC candidate also
contained part of the actor's leg. The separate crew-guard pilot addresses that
quality gate. No bulk masks were adopted on the basis of transport equality.

`marshawn_sam3_assist_controller.py` is outside frozen science. It partitions
views before LOCAL masking starts, choosing LOCAL and remote camera subsets
from currently idle GPU capacity. Its generic fleet dispatcher uses the same
global GPU locks, physical UUID verification, inherited lease FD and PID/start/
boot identity checks as training. Running models are never preempted. Confirmed
dead jobs can retry their own view; unreachable/live jobs do not duplicate.

The controller writes `assignment.json` first. The producer can run the listed
LOCAL cameras concurrently. After assist finishes, the canonical LOCAL ALL66
command verifies/skips completed cameras and handles any unassigned/fallback
views, publishing its normal batch receipt. Only afterward may the mandatory
crew guard and data builder run. The assist receipt explicitly has
`prepared_publication_allowed: false` and states that crew qualification is
still required.

At the measured 28.48 s/view, 58 remote views across seven otherwise idle GPUs
require roughly four minutes of worker time, plus input staging and collection;
this is an estimate, not a measured full-frame result. The one-time checkpoint
transfer is already complete. The producer's prior 20–30 minute native stage
can therefore benefit substantially while spare GPUs exist. Once all training
GPUs are occupied, the partition keeps preparation on LOCAL instead of
preempting a training model.

Full-native candidates appear at
`frames/001665/person_candidates/HC.{npz,json}` for the qualified canary;
their JSON carries the original LOCAL source receipt hash, while the portable
remote JSON and operation proof remain in `assist_provenance`.

## Candidate-only and early RGB follow-up

The candidate-only adapter executes the canonical model/global-prediction
prefix, captures the native input to selection, and stops before boundary
refinement. HC1665 took 14.51 seconds and returned the exact same all-person
candidate NPZ SHA as the full observer run:
`aa34118d73b007c86f09e38e8b6c2b1ed1c7753749874e895363916685915f58`.
It does not produce or replace a LOCAL actor matte. Identity selections remain
explicitly unqualified hints for the crew bootstrap.

Both ALL66 candidate-only batches, frames 1665 and 1670, subsequently completed
and passed source/array checks. Their source-mapped artifacts and aggregate
receipts are in `frames/<frame>/person_candidates_sg/`. GPU indices 1/5/6/7
were used; training on GPU0, crew-pilot reservations 2/3 and export GPU4 were
left available to their owners. The second batch streams individual native
PNGs to workers as they arrive, overlapping the approximately 4.5 GB/frame
transfer with inference instead of imposing a whole-frame transfer barrier.

`marshawn_rgb_frequency_early.py` staged unchanged same-frame HD RGB and the
exact eventual `images/train_<camera>.png` paths for frames 1665/1670, then
generated immutable specs through the frozen frequency helper. Its minimal
`ready.json` explicitly says RGB_ONLY_UNTRAINABLE; no masks, admission,
prepared/frequency-ready or dispatch marker is made. After both candidate
batches finished, it launched 120 independent frozen-recipe map fits on the
same four spare GPUs. The six DEV3-owned cameras AA/AB/AC/AD/BA/BB are excluded
for each frame. Completed maps can be reused after crew qualification because
their actual RGB SHA, source frame/camera, output paths and fitter recipe are
the same as the final pipeline. The main model queue does not wait for the
early controller's 120-map aggregate completion.
