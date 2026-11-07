# Marshawn video: source preparation, frequency scheduling, native union V5

## What was tested

Campaign: `/fsx/tmp/marshawn_video_1660_2015_24fps_20261107`.
All additions here are operational helpers outside the immutable model-code seal.
The field recipe and per-frame cold-start requirement are unchanged.

- Exclusive future-source producer: frames1675–2015, step5; frozen RGB/native decoder, four CPU workers,256GiB minimum disk reserve. Each frame becomes independently available after all66 source receipts and RAW/RGB/native hashes verify.
- DEV3 owns only six frequency cameras (`AA AB AC AD BA BB`) per frame. Silver owns the other60. Every map uses that frame's RGB hash and the frozen preprocessing recipe; no frequency optimizer state is transferred.
- Source-only Silver backfill writes same-frame RGB inputs and frequency caches. It never publishes ready/prepared/admission/dispatch controls or replaces richer frame inputs. Existing frame models retain their physical-GPU leases.
- Native SAM3 transport canaries preserve exact source PNG, canonical worker, checkpoint, coarse identity arrays, component qualification and native boundary-tile math.
- V5 observes each original refined actor mask before the canonical main function clears the black/grey overlap. Observation preserves the original canonical output. A separate recorded operation then conserves the current two-actor union, keeps labels disjoint and records ownership uncertainty independently of geometry/RGB UNKNOWN.

## Results

| Check | Result |
|---|---|
| Portable full native HC1665 canary | Refined arrays and metadata equal; native archive byte-identical |
| Read-only current-person observation | Canonical native arrays unchanged; candidate-only global instances byte-identical |
| Candidate-only1665+1670 | All132 camera outputs source-verified; no live actor masks overwritten |
| Early RGB frequency1665+1670 |120 Silver maps completed; reserved DEV3 six excluded |
| V4 timeout regression | Multi-wave queue survives total elapsed time exceeding60s; only actual queue inactivity allows fallback; UNKNOWN owners protected |
| Explicit native/frequency drain | Actual broker nonce acknowledgment followed by fresh remote supervisor/worker death proof; no process preemption |
| Fresh V5 MC1675 canary | Canonical post-clear archive exactly `c020dffb…`; resolved archive exactly approved `be5c01c8…`; zero differing pixels across all six arrays;84,720 overlap pixels restored |
| V5 legacy identity input | MC/HC/DC/KA1675←1670 references exactly unchanged |
| V5 next identity input | MC1680←1675 excludes all84,720 uncertain native pixels before original `small()`; original archive SHA remains bound |
| CPU V5 invariants | Original post-clear replay, conserved actor union, independent veto/UNKNOWN preservation, uncertainty-only identity exclusion and idempotence pass |
| One-call parent death | Top-level CLI terminated; detached factory and existing batch owners continued without duplicate native jobs |
| Actual MAT compatibility | All66 legacy native input aliases verified; existing candidate specs preserved |
| Broker V2 fairness regression | With field GPUs0–4 occupied,40 map completions on each5/6/7 over10 CPU/native cycles; natural drain ≤60s for90s fits |

The full fresh V5 frame1675 capture completed all66 views: every held-original
post-clear archive is byte-identical and every current actor union is conserved.
All six source contact pages were visually inspected; additions follow actor
contacts, waists, arms, calves and boots. This contact-scale inspection does not
replace native/W64/source qualification or the root publication gate.

The durable capture evidence is stored under
`operations/sam3_assist/v5/frame001675/all66_controller`.
Its final `all66_review_ready.json` permits source review only. It never installs
live masks or publishes training inputs. The producer/source-guard owner performs
the later reviewed installation, crew/fragment checks and DATA preparation.

## Insights

The canonical destructive overlap clear can erase pixels supported by both
current actor predictions. On MC1675 it removed visible foreground at actor
contacts and the leg/boot. V5 preserves that current union while separating
uncertain actor ownership from background/crew uncertainty. Uncertain labels
must not become later identity anchors.

Source preparation and frequency fitting can run ahead of semantic review:
their source RGB bytes remain valid during a mask-quality HOLD. Publication
stays behind semantic/admission qualification.

Speculative reservation throughout CPU DATA/review can starve map fitting once
field models occupy Silver3/4. The versioned brokerV2 instead reserves native
slots for explicit drain nonces or actual queued Silver-native work. Existing
fits finish naturally before partition; independent leases continue to arbitrate
all eight Silver GPUs. V2 was activated after the66 captures completed. The old controller/watchdog
and staging descendants were proved dead; remote running map workers were
preserved and adopted. All eight slots are eligible under existing leases.
The handoff receipt is `operations/future_rgb_sg/v1_to_v2_handoff.json`.

Operational entry for the producer:

```text
operations/sam3_assist/code/marshawn_mask_assist_entry_v5.py --campaign CAMPAIGN --frame FRAME
```

Final isolated-input marker:

```text
operations/sam3_assist/v5/frameFRAME/native_inputs_ready.json
```

The shared V5 recipe remains disabled for future frames until the root records
the full66 source-review release in `operations/sam3_assist/v5/full_release.json`.
This is the current internal review gate; it does not require another user action.
