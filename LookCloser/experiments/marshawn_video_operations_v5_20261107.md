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

The next factory qualification is restricted to frame1680. Its versioned
`marshawn_mask_assist_entry1680_v5.py` requires the root-reviewed1675 semantic
and W64 gate, checks the released semantic/prepared/visual hashes and rejects
other frames. The original V5 release remains unchanged. Missing gate,
out-of-scope frame and changed-prepared-input regressions pass; a fresh1680
factory run is still pending this source-review gate.


## MAT/BAG scheduling

The original material supervisor tried GPUs4/3 only after observing less than
1GiB usage. Continuous map fits could occupy every retry window. The versioned
advanced dispatcher now publishes a source-verified all66 MAT request, chooses
one idle or proven campaign-map GPU in preference order4,3,7,6,5,2,1,0, and waits
for a broker nonce acknowledgment. FIELD, native inference and unknown owners
are excluded. The existing MAT model/worker accepts the selected GPU through
its existing UUID/global-lease interface; its scientific processing is unchanged.

BrokerV3 removes only that requested slot from new frequency jobs. Existing
workers finish naturally and retain their inherited lease; all other free slots
continue fitting. Source-op and alias-proof hashes bind each request. Completion
requires the actual current-source material batch, rather than a stale PID.

CPU regressions pass for both fields0–2/maps3–7 requesting4 and
fields0–4/maps5–7 requesting7: the90s fit drains and other map slots advance.
An actual inherited-flock child test verifies MAT cannot acquire early, can
acquire after the fit exits and does not block other slot scheduling.

The controlled V2→V3 handoff proved old LOCAL control/staging processes dead and
preserved all five active remote fit workers with identical PID/start/boot.
Evidence: `operations/future_rgb_sg/v2_to_v3_handoff.json`.
BrokerV3 watchdog2265295/controller2265296 runs configSHA`3ad75a0d…`, bound to
advanced dispatcher`d2589001…` and helper`44c7931b…`. Actual first1680 MAT intent
is still pending native-input availability; no production MAT inference was
claimed by these CPU regressions.

### Hard140k revision and ten-slot execution

The user replaced the unlimited budget with a strict140000 attempted iterations
per frame. Final models now live under
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107` (C), with its Silver
counterpart SGC. Root sealed the scientific cap revision as `d873cd61…` and
qualified exactly steps0–139999; optimizer, warm/ARM/FR math and LR schedule
remain unchanged. Frames1660/1665 restarted cold after archived overcap runs.
Frame1670 resumed its own122000 checkpoint with root-verified tensor/optimizer/
EMA/grid/RNG equality and completed exactly140000 attempts (last step139999).
Its selected checkpoint SHA is `ccd0da52…`. No other frame receives this state.

The operational fleet controller uses all eight Silver GPUs, Olive0 and DEV3
L40S0. Active plan is
`/fsx/tmp/marshawn_fleet10_140k_20261107/video_plan.json`, SHA`74c92390…`;
its initial watchdog/controller are2270474/2270475. Host claims precede input
publication and bind source-ready, safe grid, inventory, plan and physical GPU.
Every host trains using its original local C/SGC paths; model YAMLs are not
relocated. DEV3 write permissions are qualified per owned frame only.

Bootstrap copies include only regular, same-frame DATA. Absolute operational
frequency specs from SOURCE were found during failed pre-training setup and
replaced with target-root specs. CPU cached validation checked all66 maps for
all three bootstrap frames on C and SGC, preserving ready/admission/map bytes.
Evidence is `C/operations/input_materialization/cached_frequency_CPU_validation.json`.
No gradient update happened in those failed setup attempts.

### All66 frequency ownership migration

A fixed six-camera reservation on DEV3 could block every future frame once
DEV3 starts a field model. The new `marshawn_frequency10.py` uses all66 cameras
on transient free slots across all ten GPUs. One durable owner exists per map;
all completed maps reach the canonical SOURCE Silver cache before publication.
The existing Silver command/spec/job/receipt namespace remains unchanged, so
MAT capacity detection and running jobs can be adopted safely.

The old DEV3 CPU batch parent stopped at its completed1735/AB map boundary;
the current GPU fitter finished naturally. Parent/fitter death and global lease
release were verified before missing six-camera tasks became eligible elsewhere.
Completed maps were preserved. Old Silver CPU owners2270516/2270517 retired
without signalling any scientific fitter. Adoption imported74 valid legacy-six
maps and4554 all-camera tasks. Evidence:

- `SOURCE/operations/future_frequency/exclusive_service_retired.json`
- `SOURCE/operations/frequency10/legacy_SG_CPU_retirement.json`
- `SOURCE/operations/frequency10/adoption.json`
- `SOURCE/operations/frequency10/release.json` (SHA`f91e34e4…`)

Unified watchdog/controller2294301/2294303 started with config bound to the
final advancedV3 source `3de949a7…`. First actual tick recorded559 complete,
7 running and3988 pending maps. Silver0/2 ran capped fields; Silver1/3/4/5/6/7
fit maps at99% utilization; DEV3 fit its unified1715/DE map. Olive remained
physically leased by root's requested hybrid-render session. No permanent GPU
role remains. Ready field demands take precedence; current short map fits
finish naturally. The first released1675 source was sealing in advancedV3,
while1680 remained correctly blocked on its source guard.

A stale1680 native drain nonce left after a streaming CPU handoff was retired
only after all66 actual NN jobs and the source-bound completion marker passed.
This restored map progress on six otherwise idle Silver GPUs. The later CC1680
hair ROI completed on GPU4; no stale ROI reservation was introduced.

### Input storage

The versioned decoder policy lowered only the free-space reserve from256GiB to
192GiB, retaining the8GiB per-frame budget. Existing source-ready hashes and
exclusive source ownership remain unchanged. No RAW, mask, checkpoint or
qualified evidence was deleted. New decoder watchdog2288279 adopts completed
source receipts; the previous decoder CPU owners retired at a completed-frame
boundary. FSX had227GiB free at the all66 handoff check.
