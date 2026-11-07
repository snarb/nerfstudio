# LookCloser capture import guard: v2 failure and CPU v3 correction

## What was tested

Root released and launched v2 once. Its worker failed before checkpoint loading,
model construction, source FHD parity or TRAIN capture: the import proof resolved
the synthetic `torch.ops.__file__='_ops.py'` against the package working directory
and rejected the resulting nonexistent project pathname. This was an observer
guard bug, not a radiance-field, checkpoint, OOM or rendering failure.

The failed v2 namespace, code, manifest and root release remain unchanged.
Worker2173139 and controller2173127 reached terminal failure; GPU0 became idle.
No FHD parity receipt or TRAIN NPZ was produced. Evidence remains in
`frozen_LC_semantic_capture_plan_v2/failure_diagnosis.json`, the standard worker
log/receipt, controller state and resource checks.

New CPU-only package:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v3`.
It reuses the immutable v2 checkpoint, path-only YAML and all original scientific
code, queries, qualified UNKNOWN overrides and FHD model references. Only the
import guard and launcher worker filename change. No numeric config, sampler,
field, feature strength, mask or geometry change occurs.

## Results

| Check | Result |
|---|---|
| Original CPU tests | All19 retained and passed |
| New adversarial CPU tests | 17 passed |
| Synthetic Torch modules | `torch.ops='_ops.py'`, `torch.classes='_classes.py'` skipped as virtual relative metadata |
| Named project/helper modules | Relative, nonexistent or unbound absolute files fail closed |
| Physical unbound Torch file inside package | Rejected; no blanket Torch exemption |
| Real worker import order from package cwd | 169 actual project/helper file bindings passed |
| CPU preflight CUDA/model state | CUDA_initialized=false; model_constructed=false; checkpoint not loaded |
| Source artifacts | Exact immutable v2/source assets reused; no scientific recopy or edit |

The corrected rule determines whether `__file__` is an existing absolute physical
file before considering its directory. Named project modules must prove such a
file and its bound bytes. Other modules enter the project proof only when a real
absolute pathname lies within a trusted source/helper/package root. Virtual
relative metadata is never resolved against cwd.

The CPU import replay follows the actual worker order through its pre-model
binding gate: isolation helper, Torch/NumPy, native capture adapter, tree digest,
semantic support, eval utilities and trainer RNG helpers. It runs from the new
package directory with GPU visibility empty, and verifies that both synthetic
Torch namespaces are present. TCNN's existing CPU import path reports unavailable
hardware and defers network use; no GPU backend, model or CUDA context is created.
Actual GPU loading and all scientific parity gates remain pending root release.

Readable worker/launcher differences are saved as `worker_diff.txt` and
`launcher_diff.txt`. The new namespace contains the exact standard local
controller plan, false root-review template, CPU import proof/bindings, test
receipt and executable launch vectors. The source capture schema and all
full-pipeline/core/EMA/admission/FR/LUT/RNG checks remain unchanged.

## Insights and next steps

The original CPU tests checked hand-built helper modules but missed Torch's
dynamic virtual module metadata and the complete eval-import sequence. The new
regressions reproduce that failure boundary and test both sides of the trust
rule. They do not establish GPU source FHD or native packed composition parity.

Root must review the final sealed v3 package and create a separate exact release
before source FHD plus fixed TRAIN capture. No head fitting or export is
authorized. Native cap1024/occupancy omissions remain explicit; unseen geometry
is not certified free.
