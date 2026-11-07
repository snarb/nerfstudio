# Marshawn video: current-frame occlusion and safe admission dispatch

## What was tested

The selected first-frame recipe treats opaque mats and the bag as UNKNOWN for
occupancy clearing. Fresh video-frame preparation initially covered people only,
which could incorrectly clear actor geometry hidden by these props. The video
input controller now recreates this visibility policy from each current image
before admitting a frame to cold training. All implementation lives outside the
sealed campaign source, under `operations/`; model math and hyperparameters are
unchanged.

Fresh SAM3 inference uses each camera's native 8192×4320 image for `floor mat`
and `bag`, with current-source hashes and a 128-native-pixel UNKNOWN margin.
Reviewed first-frame regions serve only as search/identity priors. They are not
reused as current masks or as evidence that space is empty. Uncertain mat
visibility conservatively disables clearing by that camera. CE, JE and NE remain
unqualified for clear votes; all66 cameras still train.

The added layer is **admission visibility UNKNOWN only**. RGB, actor labels,
W64 training weights and hybrid source-eligibility UNKNOWN remain unchanged.
The wrapper imports the frozen `MaskCamera`, complete-cell classifier and
`WholeCellAdmission` implementation. Its rule remains two cameras proving the
complete projected cell known background, followed by exactly one128-grid halo.
Every frame constructs this grid independently; no model or learned grid is
transferred.

## Results

| Check |1665 |1670 |
|---|---|---|
| Fresh native prop evidence |66 cameras |66 cameras |
| Final frequency maps |66 source-bound maps |66 source-bound maps |
| LOCAL and silver final ready SHA |Identical |Identical |
| Frozen `load_support` verification |Pass |Pass |
| Current actor silhouette ray checks |921 samples, zero failed rays |Pass |
| RGB, actors, W64, hybrid UNKNOWN |Byte unchanged |Byte unchanged |
| Dispatch ordering |Verified gate, then exact paused parent resumed |Verified gate, then original stage publisher |

1665 additionally checked four manually identified current-frame shoe-center
proxies from HC/DC/KA rays; all four corresponding cells were retained. These
approximate triangulations are visible-source checks, not measured depth ground
truth. The 1665 admitted grid retains452704 of2097152 cells after the single
halo. Conservative occluder handling therefore still permits substantial clearing.

1670 is the first completed transaction through the new autonomous dispatcher.
Its final ready SHA is
`416f20a6e222cad5407ce100d66d88b9717f0ec6e76fe33ed6bf4f659b3067db`;
gate manifest SHA is
`4538d7e365e78ad65a2d888b24fb95c34597857a134b047ed65c05807654a69c`.
The original frozen publisher subsequently dispatched the frame, and cold
training started. The1660 and1665 inputs were not altered by this transaction.

Evidence roots:

- `/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/advanced_inputs/001670/dispatch_complete.json`
- `/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107/operations/admission_visibility/001665/`
- `/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107/operations/admission_visibility/001670/`

## Autonomous transaction and recovery

The dispatcher accepts only the new source-bound `native_semantic_ready` marker,
after the separate CREW guard has completed. It stages rich RGB/semantic inputs
without publishing dispatch, waits for all66 current-image frequency maps,
validates every cached map, and performs the frozen final frequency seal on CPU.
It gathers the maps and their original metadata back to LOCAL, copies the exact
recipe provenance, refreshes the LOCAL inventory and requires identical final
ready hashes on both hosts. Only then does it construct and verify the current
occlusion-aware grid, save the pre-dispatch proof and call the original frozen
stage publisher.

The service uses source/config hashes, a controller flock, a detached watchdog,
PID/start-time/boot identities and remote-worker adoption. GPU workers hold the
same global physical-device lease as the training fleet, including an inherited
lock descriptor. An uncertain SSH result cannot launch a duplicate worker.
Dispatched frames and frames with cold-training provenance are immutable.
The obsolete late frequency-push dispatcher was retired before this service
started. Future source decoding, mask production and frequency fitting can run
ahead without publishing an unsafe training marker.

Initial dispatcher SHA256:
`1275775a9afd40612743baf86b810a77d7257171a7f3ee779eb34134f2d1ee97`.
Released config SHA256:
`0dd4e5dd9951a7be0bb20d029815166af461072f34f1e52d9c2a3312854840d9`.

## Insights and limitations

Opaque props must veto clear votes independently of their training-loss policy.
Using only person masks can make cells behind a mat appear empty. Separate
visibility UNKNOWN preserves the selected training recipe and hybrid pixel
eligibility while preventing that unsupported inference.

SAM3 sometimes includes opaque floor tiles in a broad mat mask. This makes the
grid more conservative; it is not evidence of a minimal physical occupancy
grid. Source silhouette-ray tests and shoe proxies detect some serious errors
but cannot prove the hidden surface depth or perfect occupancy. Uncertain views
must retain cells rather than invent clear evidence. Training/render quality
still requires the independent final visual gate.

## 1675 source-quality hold before training

Manual review found missing pixels in the grey actor's far leg in MC after the
first1675 preparation marker had been emitted. The preparation and input
dispatch controllers were paused immediately. No1675 field dispatch, training
specification, cold provenance, runtime or learned weights existed. The three
already-running models were unaffected.

The input controller was stopped during rich-input staging. Both paused owner
identities and all descendants were checked: no live rsync or SSH child remained.
The partial1675 control metadata was archived on both hosts with matching
hashes, while native RGB, frequency caches, semantic arrays and independent
RAW-bound MAT/BAG evidence were preserved. A fresh source correction and review
must precede rebuilding the data and issuing a new marker. The stale suspended
controller stack must be replaced by a fresh process, not resumed.

The quiescence and LOCAL/silver quarantine receipts are under
`operations/advanced_inputs/hold_1675_20261107`. This hold is an explicit source
quality gate; successful structural source checks alone do not establish that
every actor boundary is complete.

## Dynamic MAT capacity and coordinated restart

The released second operational dispatcher cooperates with frequency brokerV3.
Fresh66-camera RAW/task/receipt verification precedes a durable MAT scheduling
request. GPU preference is4,3,7,6,5,2,1,0. A candidate must be idle with a free
global lease, or contain only a known campaign frequency worker whose exact
command and receipt PID/start-time/boot/GPU-UUID match. FIELD, native-network,
unknown and untracked owners are excluded.

The broker acknowledges the request nonce and withholds only that slot from
new frequency fits. An existing fit finishes; MAT then uses the same physical
UUID and global inherited lease as the fleet. Other slots continue fitting.
If a FIELD/NN claims the slot before MAT starts, the stale request is retired
and a new eligible slot/nonce is selected. No worker is preempted. A request
completes only when its actual source-bound native batch exists.

The CPU integration test covers FIELD0–4 with MAP5–7: MAT requests7, waits for
the broker acknowledgement, and leaves5/6 eligible. It also covers a FIELD race
on7 followed by retargeting6. The broker's separate90-second drain and real
inherited-flock tests passed. A read-only physical SG probe correctly excluded
the three running FIELD workers and identified the five owned MAP workers.

The old paused dispatcher/watchdog were retired after proving their exact
identities and zero live descendants. BrokerV3 adopted the five existing fit
workers without changing PID/start/boot. The new dispatcher then started with
1675's source marker still withheld: actual status is source-wait, zero errors,
and no1675 field dispatch. Existing models remain untouched.

Released source SHA256:
`d2589001a4287bdc61993dc6a0e639dfbe83606f46b7372674365682e4313e2d`.
MAT helper SHA256:
`44c7931b458572a37ee17ab3de5ab1c420b5acecf5c5a07ebdfd123805c59239`.
Authorized config SHA256:
`678bd220a324b758b866bbff6d08ac233abb7cc57a4a4ea8e86373aa091d1660`.
Release and qualification receipts are under `operations/advanced_inputs/`.

## Hard140k production revision and input routing

The former dispatcher was retired with its old model revision. The released
V3 dispatcher writes new model inputs only to
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107` or its SG counterpart.
It retains source preparation and frequency caches in the original campaign.
Its source SHA is
`3de949a7666a5f149c70f875712f86c6215abe76429144f3b53c5439dc60983d`;
the released config SHA is
`df6f36d425d9938ad9544ae22b85a01dd8d932a5f7c25ff6466216d79b99a0c4`.
Actual watchdog/controller PIDs at release were2294318/2294319. The receipt is
`operations/advanced_inputs/release_v3_hard140k_actual.json`.

For each future frame, final LOCAL/SG frequency inventories must have identical
ready SHA before current MAT/BAG visibility, frozen whole-cell admission and
positive actor-ray checks. Only then can the capped fleet reserve a training
host and publish that host's dispatch marker. The first1675 transaction passed
rich-source staging and identical final ready; its admission gate was still
running at this report update. No grid perfection or1675 training completion
is implied by these intermediate checks.

The new host materializer copies regular same-frame DATA and admission inputs,
never a model. Operational `frequency_specs` contain host paths and are excluded
from copying, regenerated with the unchanged target-host function, and checked
against all66 cached maps. Scientific RGB, semantic, map, ready and grid bytes
remain bound to their original source proofs. Fleet reservations are excluded
from MAT capacity even before CUDA allocation. All caches are frame-local;
there is no cross-frame learned state transfer.

The1675 positive-ray gate then rejected55/918 observed actor rays; no production
assignment or model was created. An independent exact DDA audit of HC's top
braid ray crossed142 cells and found none admitted in base, pad1 or the mapped
grid. Replaying the unchanged full-cell classifier reconstructed the base grid
exactly. Its best cells had58 positive-support views but five KNOWN-BG votes
from AB/AC/AD/BC/BD. Current-image review identified missing flying-braid tips
in those source supports. This was a source-mask failure, not a reason to
bypass the gate, add dilation or change the AABB.

The old1675 release/ready/staging/control metadata and entire SG admission
were archived under `operations/advanced_inputs/001675/archived_before_fiveview_hair_v53`.
The exact CPU dispatcher/watchdog were retired with zero staging children;
RGB, all frequency maps and running models were preserved. Fresh source hair
qualification and a clean source transaction are required before1675 can train.
The diagnostic contact is
`operations/advanced_inputs/001675/cell_audit/HC_ray_best_cells_contributors.jpg`.

## Corrected 1675 admission and ordinary production release

Current native head/hair qualification now covers all 66 cameras. Geometry-only
UNKNOWN is stored separately in `data/admission_unknown_hair`; its manifest is
bound by `prepared_ready.json`. The gate verifies every current native image,
processed RGB, raster and proof hash before adding this layer only to its
local `MaskCamera.UNKNOWN` copy. Original geometry UNKNOWN, RGB-source UNKNOWN,
W64 and actor arrays remain unchanged by the gate. Qualified new actor pixels
were separately approved as current source evidence. The AABB and field recipe
did not change.

The corrected source-only release passed the frozen support loader, sparse
actor-ray checks,165,925 nominal contour rays across all 66 cameras and13,927
native proposal-contour rays in the five problem views. Every tested base,
pad1 and mapped ray was retained. AD remains an uncertain geometry seed rather
than certified foreground. These checks do not certify hidden physical depth
or later learned occupancy.

Actual final ready SHA:
`85673fbecef7e33ee9410ebe39a93138124d3d00a678b6829f7819d6fd88aaad`.
Admission manifest SHA:
`6111f75e4f700c4ecf4cbc25614b1b9b26cb81a660a291883bf163587e10be0b`.
Source grid SHA:
`776e16ea9ac7bf7e75258d1cc66c767bc63e349a67ecd351e625dca12ae553bb`.
The publish-suppressed canary and dense receipts are in
`operations/advanced_inputs/v4_actual_hair_source_grid_canary.json` and
`v4_actual_hair_dense_qualification.json`.

An external launcher wraps the unchanged V4 transaction with a mandatory,
read-only all 66 contour qualification before **each future frame's** training
host reservation. Current source/grid/audit hashes must match; known failures
block publication. The corresponding proof SHA is included in the new C
dispatch receipt. The launcher does not alter field/grid construction math.
Its positive real 1675 test reached a stub publisher; wrong ready and wrong
audit-code bindings were rejected without creating a model or assignment.

Ordinary production was then released with config SHA
`059e33f34b4c8abbd3322a8860d25179a3187932f06d0143c2e89973c3fb5b55`.
Watchdog/controller 2320635/2320636 published 1675 to olive GPU0 after the real
dense prerequisite. Trainer 2322568 started with step 0, zero occupied cells,
zero EMA and empty optimizer state; its actual limit is 140000 updates.
The field inputs use the new C revision and final ready above. No previous
frame's weights, learned occupancy or optimizer state was transferred.

A later read-only audit of the already-completed capped 1660/1665 models is
recorded under `C/operations/video_fleet10/private_profile_v1/other_frame_CPU`.
Frame 1665 retained all 173,185 current nominal contour rays in both source
and learned grids. Frame 1660 had 426 source-pad contour misses, reduced to
109 in whole-cell mapped admission; learned R0 had the same 109, across
grey-labelled shoe-tip contours in IA/MA/MB/MC/NA/NB/ND. Every failed ray
intersected the AABB. Native TIFF samples show points at the visible toe edge.
For one MA ray, the exact reviewed 1660 admission inputs reproduced all 53
inside-cell decisions; no cell survived. A finite ray extension beyond the
box did not reveal stronger outside-box support. These observations establish
a nominal contour/support disagreement, not its physical depth or cause.
No source, box, checkpoint or completed model was changed. The private
render guard rejects this unrepairable 1660 candidate and retains R0.

The opt-in render candidate remains separate from production. Its geometry
adapter applies the same own-frame policy to target and donor geometry,
preserves separate R0 caches, and binds checkpoint/camera/profile/grid hashes.
Only primary alpha at most .05 within a conservative projection of added
cells can receive a secondary query. Eight actual-camera CPU ROI tests
(including the 4K video camera) had zero missed ray/cell intersections.
Actual GPU sparse-query equivalence and other-frame/temporal visual checks
remain required before delivery promotion. The fixed IC/JC/KC/LC/MC smooth
focal reference yields scale 2.9 at the start and 4.26772055 at the native
MC endpoint, avoiding a nearest-source focal switch.

The subsequent finite same-GPU numerical qualification passed on SG GPU7
(UUID `GPU-9c2ea572-537d-66a7-6df3-41c3d8914d39`, worker153690). On the
341×342 native ND crop, only665 secondary rays were queried. Compared with
the same model's full145-cell render, merged RGB differed by at most1.49e-7,
alpha by2.98e-7, and all depth quantiles matched exactly; PNG code difference
was zero. Two full-reference selection decisions differed only by tiny
rounding outside the conservative ROI. The complete pipeline tensor SHA
`578088c722e7d588bf87e69d42bde0cd7f2dd713a31fcb587b5f08dfcb4f276e`
matched before/after; both registered grid aliases, EMA and admission were
restored. A repeated R0 render differed by at most2.56e-6 float RGB. The GPU
lease was released before receipt gathering. This validates sparse query
mechanics, not a new quality result or temporal safety. The private profile
V3 SHA is `3fdefd89949652025d71fdb2ae10eedf8bc92d89909c225f7a9b217dec2f1491`.
Other-frame hybrid/temporal review remains required before promotion.

Frame1680 then completed the first automatic post-canary source transaction:
all66 source-bound maps, current native MAT/BAG and head uncertainty, frozen
support verification and mandatory dense contour qualification passed. The
C dispatch binds ready `beac1e9fbe45a373a911688a43fdba187fb350955f7d014926d76fc95de44ede`
and admission `e58f253a72689193430cea579bf7a48b573118ec4cc5fd4d6be4bda8c12f52b6`;
the fleet assigned DEV3 GPU0 without model-state transfer. Initial runtime
startup is supervised by the existing fleet.
