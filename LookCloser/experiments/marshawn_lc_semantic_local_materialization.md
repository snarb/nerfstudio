# Local best-LookCloser capture materialization

## What was tested

Materialize a completely local, reviewable read-only capture launch on olive
GPU0, preserving `frozen_LC_semantic_capture_plan_v1` unchanged. New package:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v2`.
No GPU/context/model construction, source edit, fitting, head transfer or export
is authorized by this preparation.

## Results

| CPU verification | Result |
|---|---|
| Real frozen TuningParser | 63 TRAIN and3 heldout cameras |
| All69,888 fixed pixel-center rays | Origin and pose error0; independent direction maximum error1.1920929e-7 |
| CUDA context/model | CUDA_initialized=false; no model constructed |
| Local Torch build | 2.7.1+cu128, CUDA12.8 |
| Config edits | Four path nodes plus shared data-path alias; every nonpath tagged scalar/type unchanged |
| Direct silver/local input hashes | Original YAML, transforms, ready, box, semantic-admission source manifest identical |
| Olive process state during preparation | Prior worker2150545/watcher2151107 absent; GPU0 has no compute process |
| Olive physical UUID | GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46 |
| CPU tests | 19 passed: v1 adapter plus local gating, actual-import byte binding, resource guard, source-only RGB/alpha parity and YAML alias/numeric preservation |
| Standard controller dry run | Valid single local job, GPU0 only; no controller or worker launched |

The exact checkpoint was staged from silver using read-only byte ranges into
private partial files, assembled in source order, SHA256 checked, then published
with an atomic no-overwrite hard link and made read-only. The original source
archive remains untouched. Required SHA256:
`340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`.
The final materialization receipt and manifest record the completed copy; no
partially copied checkpoint is accepted by execution.
CPU-only checkpoint inspection confirmed step126316,52 pipeline tensors and all
seven serialized state trees, with pipeline digest
`c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636`.

The new `local_source/config_local_paths_only.yml` derives from the untouched
original silver YAML. `path_materialization.json` records each before/after
pathname and the structural proof. Data references resolve to the local campaign
`data` directory; semantic_data and semantic_manifest resolve to its existing
identical source artifacts. The output directory resolves inside the new v2
namespace. The exact local checkpoint is selected by the existing eval callback.
No scientific number, feature strength, camera transform, sampling option,
frequency option or geometry value changes. Source specs and parent manifests
remain byte-identical provenance, including their historical path strings; the
runtime never follows those historical paths. Archived grid loading receives the
explicit local checkpoint override.

### Executable local launch and supervision

The package contains an unchanged copy of the standard frozen
`marshawn_tuning_controller.py`, its exact controller plan, the reviewed observer,
local worker and a root-gated launcher. The host entry has no SSH address, so all
controller/worker execution is local. One job waits for idle GPU0 and takes the
standard global GPU lock. Its direct worker verifies the supervisor receipt,
physical UUID, single visible logical cuda:0 and actual compute-process UUID.

The launcher is dry by default:

```bash
/home/pavlo/.conda/envs/nerfstudio/bin/python /fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v2/code/launch_marshawn_lc_semantic_capture_local.py --root /fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v2
```

After root reviews the final manifest/code/controller-plan hashes and creates a
separate exact `root_release.json`, adding `--execute` launches the standard
detached quiet controller. The false template does not authorize launch. Worker
stdout/stderr go to `queue/jobs/LC_best126316_native_capture/worker.log`;
controller logs remain under `controller`. The worker output is the new `capture`
directory. Existing outputs are never overwritten or resumed implicitly.

The standard supervisor records process liveness, GPU processes, OOM evidence,
free disk and progress every5s. An independent thread inside the worker checks
RSS, available RAM, GPU process memory/exclusivity and disk every5s throughout
checkpoint/model loading, source FHD parity and capture. A resource violation
writes a failure receipt and terminates only its own process. Limits remain
8GiB Torch allocator,10,240MiB actual GPU process,12GiB RSS and1800s wall time;
startup requires8GiB available RAM and20GiB free disk. Runtime guards stop below
2GiB available RAM or5GiB free disk. Parent supervision remains active and can
record a compact check with the launcher `--status` option.

### Gates before any TRAIN capture

The local worker first verifies all sealed inputs, the exact checkpoint, frozen
scientific code and source runtime. The unmodified source eval loader constructs
the pipeline in inference mode, without image/mask caching or optimizers. Its
entire state must match the source checkpoint. Actual core/active/EMA/admission
must match the archived GPU snapshot. The restored source FR must already be.3;
it is asserted rather than rewritten. The discrete LUT is primed and source RNG
restored before immutable snapshots.
Actual project/helper import paths and bytes are checked against the packaged
code map, frozen1495-file source map and bound helper map before model setup,
after source FHD rendering and at completion. Explicit config-path assertions
also verify every executable data/semantic/checkpoint/output reference.

The worker then renders complete DC/HC/KA native model RGB and alpha images and
compares them with the archived original source renders, using the existing
campaign replay policy: at most one saved code, RGB8 changed fraction<=1e-5 and
alpha16<=.001. It reads model reference renders, not heldout target RGB/masks.
All six comparisons plus full pipeline/RNG/LUT invariance must pass before the
first TRAIN packed archive is written. A failure retains only diagnostics.

Thereafter the v1 temporary-forward observer captures original native final
samples, physical frustum intervals, sigma/RGB/native weights and ordering, with
independent density-to-weight, RGB and alpha proofs. Every parsed floor-camera
pose/raster/intrinsic and ray chain is checked again. The two exact approved IA
positive records become UNKNOWN by weight0 only in the new cache. Original
queries, masks and targets remain unchanged. Completion repeats all frozen-state
and checkpoint/query-byte checks.

## Insights and next steps

This local package resolves the previous missing-checkpoint and inaccessible
silver-path problems while keeping v1 evidence intact. Root review and a
separate exact release remain required before any GPU work. Actual pipeline,
original FHD source parity and packed GPU composition are runtime gates and are
not claimed as passed by CPU preparation.

The cache remains native visibility supervision, not complete physical geometry.
Native eval lacks opacity pruning but still skips occupancy and may clip legacy
ARM tails at1024 samples. Physical actor export needs independent full-path
density sampling and recomputed transmittance. Sparse semantic queries cannot
certify a perfect occupancy grid or artifact-free hair, feet and hidden geometry.
