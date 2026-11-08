# Clever-shadow frequency-worker feasibility

A finite read-only inspection found `ubuntu@clever-shadow` reachable. It has no
FSX mount and approximately136GB free root storage. No packages, files, jobs,
GPU allocations or fleet/broker configuration were changed.

The closest existing environment is `/home/ubuntu/sam3env/bin/python`:

| ABI property | Clever | Qualified SG reference |
|---|---|---|
| Python |3.11.17|3.11.16|
| Architecture / SOABI |x86_64 /cpython311|same|
| glibc |2.39|2.39|
| Torch |2.7.1+cu128|2.7.1+cu128|
| CXX11 ABI |true|true|
| Compiled CUDA targets |includes sm120/compute120|same|

Clever has no installed tinycudann. Its libstdc++ provides GLIBCXX3.4.33,
while the existing SG SM120 TCNN extension requires at most3.4.32 and glibc2.33.
A private copied binary overlay is therefore ABI-plausible without compilation,
but was not copied or runtime-qualified. The frozen source is14.84MB, the SG
TCNN extension53.72MB and one HD input approximately3MB: the minimal base is
about71.6MB before exact dependency closure. Whole RAW/video/model staging is
unnecessary for independent2D frequency fitting.

The candidate also lacks tyro, rich, pytorch_msssim, scipy and jaxtyping.
The frozen fitter imports the nerfstudio dataparser registry and its parser
chain, so the full exact import closure must be qualified; silently selecting
the alternative SSIM fallback would change the recipe. The existing CA4D Docker
image and host CA4D virtual environments also lack TCNN/nerfstudio/tyro, giving
no ready-to-run container shortcut.

Conclusion: not ready as-is; a small private overlay plus actual source-bound
one-image canary is a plausible separate operational step. This inspection did
not establish GPU execution compatibility or add an eleventh worker. The
existing ten-resource campaign was unchanged. Evidence is saved at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/clever_readonly_feasibility_20261108/report.json`.

## Private ABI closure and actual canonical fit

The subsequent authorized private-overlay canary qualified the exact frozen
fitter on Clever without installing packages or changing its SAM3 environment.
The verified closure contains6,936 files (680.30MB), including the copied SG
SM120 TCNN binary. Python minor versions differ, while the CPython3.11 SOABI,
Torch2.7.1+cu128, CUDA12.8 and CXX11 ABI match. CPU-decoded image tensors and
seed/backend contracts match the SG reference. The exact `pytorch_msssim`
implementation is active; no SSIM fallback was substituted.

A single current-source1690HC canonical fit took72.59s on Clever, with1.424GB
peak allocated GPU memory and the existing4GiB cap. It used the unchanged
`.95` SSIM threshold,16×1000 stages,1024-pixel evaluation batches, stride8,
hash23, resolution16–8192 and seed42. No other frame's state was loaded. The
private map is not a main-campaign map.

One necessary same-physical-SG repeat took75.72s and showed that raw frequency
maps are nondeterministic even on the reference host:

| Comparison against original SG map | Equal level | Within one level | Mean absolute level difference |
|---|---:|---:|---:|
| SG repeat |45.215%|74.335%|0.9658|
| Clever |44.666%|74.927%|0.9636|

Clever's difference is consistent with this observed within-SG variability;
one image and one repeat do not establish a statistical equivalence bound.
Large differences in exponential resolution units alone were not a valid
host-failure test. All three results, full per-stage SSIM curves and CPU seed
contracts are retained. The original campaign map was unchanged.

Evidence:
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/clever_frequency_canary_20261108/canary_review.json`
and
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/frequency_variability_control_20261108/comparison.json`.

## Candidate nonshared frequency backend V13

The candidate adds Clever only to the2D frequency broker. The ten field slots,
training plan, models, optimizer, map recipe and source preparation are
unchanged. A logical map owner is persisted before staging one SHA-verified
HD image/spec. The existing frozen supervisor then acquires the physical UUID
lease and records launch intent before the worker starts. Old SG/olive/dev3
names, specs, dispatch roots, worker PIDs and leases are preserved. Clever
remains eligible during field demand because it is not a field host; existing
NN, MAT and field priorities still apply to their original slots.

Completed Clever inputs, maps, metadata and actual supervisor/spec receipts
are SHA-verified and archived on SG NVMe before publication to the two
canonical source caches. Publication creates files atomically, refuses foreign
existing bytes and resumes after a partial publication. The private root is
capped at48GiB and requires32GiB free; exceeding either limit disables new
Clever assignments while the original resources continue. The frozen fitter
emits no large checkpoints, so there are no checkpoint duplicates to evict.
Inputs, maps, receipts and SG archive pointers are retained. No originals,
RAW files,3D checkpoints or renders are removed.

An isolated backend test reused the already-qualified same-image frequency
cache, avoiding an unnecessary new numerical fit. It actually ran the frozen
worker through the global lease, lost the SSH acknowledgement after detached
launch and adopted the same supervisor/worker. Injected losses after SG
archive completion and SG publication also resumed identical bytes. Both real
main1690HC maps and metadata remained unchanged. CPU tests reject wrong
source bindings, corrupted archives, namespace traversal, symlink escapes and
foreign existing outputs. Independent scheduler review verified field demand,
NN/MAT exclusions and safely ignored non-request JSON. A read-only preflight
checks every existing materialized job digest against V12 before handoff.

The backend currently uses bounded synchronous SSH/RPC operations, so a host
outage can lengthen a broker tick; running scientific workers continue under
their own durable supervisors. Host failures fail closed without changing the
other hosts' scientific job specifications.

The review bundle is
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/frequency13/candidate_ready.json`;
actual adoption/publication evidence is
`operations/frequency13/qualification/backend_canary_result.json` relative to
the same campaign. Source copies are in
[`scripts/marshawn_frequency_nonshared`](../scripts/marshawn_frequency_nonshared/).
At this checkpoint V13 is a qualified **candidate**, with main-job authorization
false. No active broker replacement or main map claim was performed.

## ROOT-approved activation and first actual main map

ROOT approved V13 and wrote the active config with main-job authorization
true (SHA `4152cbbec5e68eebf9f8173637a8973f593fab4a172191a33f28ea69b47a09c5`),
bound by `operations/frequency13/root_release.json`. The candidate and its
original hashed proofs remain unchanged.

A fresh separate-path adoption preflight and quiescent state archive preceded
the CPU-only handoff. Exact old V12 watchdog/controller identities were retired;
V13 watchdog2391759/controller2391761 adopted all four then-live fits with the
same worker/supervisor PID, boot/start identity and scientific spec. All old job
names were preserved. No scientific process or GPU lease was signalled,
released or restarted. Receipts are in
`operations/frequency13/handoff_20261108_0130/`.

The first actual main assignment, frame1845/cameraED, fitted from scratch in
72.37s with1.424GB peak allocated memory, seed42 and no previous-frame state.
It completed on Clever's qualified physical UUID without OOM. The full SG
archive was verified before identical map/metadata publication to canonical SG
and LOCAL caches. Map SHA is
`ac9f5024a2ca91c2792e92452f484cffb23ec085b1b9f36cebb0c4195da3ede9`.
The source-bound actual proof is
`operations/frequency13/first_actual_main_map.json` (SHA
`85697aa606840a911504ecd99e75c414b78e627162caaff78aaa7f8f29158d46`).
This activates an eleventh frequency-only GPU; the ten field slots and training
recipe remain unchanged.
