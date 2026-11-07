# HD63 frequency-map activation

## What was tested

Root released and launched 126 independent frequency-map fits on the existing 63 TRAIN HD images. This is the frequency stage after the structural experiments; it does not train a NeRF or establish which maps improve rendered detail.

The two factors reuse the frozen fitter and 1000 updates per level, 16 levels, full 1920×1012 RGB:

| Factor | SSIM threshold | Learning rate |
|---|---:|---|
| `lr_decay` | 0.95 | 0.01→0.001 per level; retain Adam state |
| `threshold093` | 0.93 | Constant 0.01 |

Both use patch/stride 8, resolution range 16–8192, two features per level and log2 hash size 23. The resulting scalar maps are 126×240; the final four image rows remain outside the patch grid, matching the original implementation. Each camera's two factors use the same host, physical GPU, seed and initial sampling sequence.

## Prepared result and validation

The immutable package is `/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_activation_v1` (`A` below); its silver copy is `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/frequency_HD63_activation_v1`. Root launched the reviewed package on 2026-11-07; controller PID 2131639. Release SHA256 is `ad1c7f0aa8e9f249b8725cd199646891e65dd270612c4caac55de8db4c450e3d`. The initial check found three completed fits and three active paired fits, zero failures/OOM, and correct parent leases and physical GPU UUIDs on all three lanes. A later concurrent launcher invocation returned “Controller already alive”; no duplicate worker/controller was created. `A/manual_supervision.jsonl` and `A/first_lanes_verified.json` record the check.

- Activation SHA256: `74ed2f44a9f95f93b1b3913ca6b1ff4983435c0c9b794a57c90fa47d31573e50`.
- Frozen worker: `13a3d723cb5462eeadead9df5862cd57eba37c5b351a5e9d417f0a5be524d91e`.
- Frozen preprocessing: `729f10f26b24b97f56adcc3c62a6722a78c4ee5d59d5b0b8dd963fde6561ffad`.
- Existing global-lease controller: `2e7a1be53794882ffc3c0a6a82fff3cc24070b36fbf7593e65771772e0dadf39`.
- Successful earlier actual-4K factory receipt: `d7bbc11107d78defb366ed6135c6d2cf0410323335c8bd926f45588fb0c6a777`. This proves the generalized execution path; these new jobs use HD input and the HD update budget.

All 2100 bound files were independently rehashed on each host. New host-path specifications passed the frozen worker's CPU validation: local 42, silver 84, no Torch import or GPU work. Existing scientific tests and completed fits were not repeated. Original prepared specifications, RGB, calibration, masks, existing production frequency maps and frozen code are hash-bound and remain unchanged. Only host paths and output destinations differ in the new specifications.

The local controller dispatches 42 jobs to local GPU0 and 84 jobs to silver GPUs4/7. A camera is assigned round-robin to one lane, with both factors kept on that lane. Both environments currently report Torch 2.7.1+cu128, CUDA build 12.8 and matching SSIM package source. Runtime/package and GPU identities are recorded in `evidence/runtime_{local,silver}.json`; this is not a bitwise reproducibility claim across hosts.

The fitter's existing 4 GiB allocator limit and fixed 16000 updates per fit remain unchanged. Prior HD fits took about 76 seconds each, suggesting roughly 55–70 minutes for three lanes with process and validation overhead. This is an estimate, not a wall-time guarantee. The controller checks progress/processes/GPU status, uses physical UUIDs and one global lease per worker, and receives additional hourly manual supervision. Fleet workers validate the actual parent lease and do not acquire a second lock.

## Release, failure and completion protocol

Preparation requires a separate root release before launch. `A/root_release.template.json` contains the exact schema; approval sets `approved_map_fits=true` and binds the activation hash above. The local-origin command is:

```bash
python3 /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_activation_v1/scripts/launch_marshawn_frequency_activation.py --release /fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD63_activation_v1/root_release.json
```

The launcher rechecks both hosts' complete binding sets, declared GPU identities and disk availability before invoking the unchanged dispatcher. State lives in `A/controller`; fit receipts/results live under each host's `frequency_HD63_activation_v1/host/jobs`. The local controller uses established SSH credentials; no silver self-SSH or credential copying is needed.

Completed prefixes and failed outputs are immutable. A failed fit is not automatically retried or resumed halfway through its levels; a retry needs a separate reviewed generation. SSH uncertainty is not treated as proof that a worker stopped. Existing durable claims prevent duplicate launches.

After all jobs finish, `validate_marshawn_frequency_activation.py` performs CPU-only per-host validation of all source/result hashes, dimensions, finite scheduled labels, censoring metadata, update counts, learning-rate contract and lease/UUID receipts. `reconcile_marshawn_frequency_activation.py` joins exactly 126 successful results and checks that each camera's factors shared host/GPU, initialization, sampling prefix and final RNG state. Neither helper assembles data, writes READY, changes production maps or launches a NeRF. Assembly and downstream comparison are separate review steps.

## Insights and limits

Earlier baseline-repeat fits showed substantial patch-label variability; the lower threshold generally selected coarser labels, while the decay effect differed by camera. A full paired inventory measures those map changes but cannot itself demonstrate a 3D rendering benefit. Selection requires subsequent controlled NeRF experiments after a separately reviewed assembly. EVAL images are excluded from fitting.

Reusable preparation, binding, launch, path validation, runtime inspection and result reconciliation helpers are in `scripts/*marshawn_frequency_activation*.py` plus `inspect_marshawn_frequency_runtime.py`. The scientific fitter is unchanged.

All126 fits completed with zero failed jobs or OOM evidence. The dispatcher and workers exited and released all three GPUs. Both CPU validators and the126-job/63-pair reconciler passed; final validation SHA256 is `b2c16b591b91e71c999b9e9e18b91409ddcde20a9ba24ed0f12a361359a1a174`. Every new same-camera pair has identical initialization, first-eight draw hashes and final RNG state. Silver's84 complete result sets were copied and rehashed into `A/results_silver` for FSX durability.

Root then authorized isolated CPU dataset preparation. `C/frequency_HD63_assembly_v1/assembly.json` has SHA256 `2f0e18a5a0cfa99168ce6d45903e0aa95506459bfa6c3c628ac9e0c45a394713`. Each dataset changes exactly126 existing TRAIN map/metadata files and adds63 explicit censor sidecars. RGB, masks, original confidence, poses and accepted box are unchanged; portable relative links point to the original data. EVAL maps were absent in the baseline and remain absent. No EVAL fit occurred. The original dataset and fit outputs are unchanged.

| Actor / map | Mean level | Changed actor patches vs frozen baseline | Exact censored fraction |
|---|---:|---:|---:|
| Black / baseline |13.843|—|Unknown|
| Black / decay |13.906|40.00%|56.77%|
| Black / SSIM.93 |12.786|59.40%|29.03%|
| Grey / baseline |12.654|—|Unknown|
| Grey / decay |12.568|62.86%|23.27%|
| Grey / SSIM.93 |11.219|77.11%|7.25%|

These are equal-camera means over complete8×8 patches with at least90% original actor-mask coverage. Full-image exact censor fractions are31.27% for decay and11.35% for.93. Baseline level15 does not reveal whether its legacy fit converged; its censor fraction is not inferred. The legacy baseline is also not a new repeat-matched fit, so map disagreements include fitting variability.

Six TRAIN source/map contacts—AA,DA,GA,JA,MA,OA—were inspected across hair, clothing and feet. The fixed palette is viridis level0 purple→15 yellow, identical for all panels. SSIM.93 generally coarsens actor levels; decay varies with camera/material. Maps also describe background patches and are not segmentation. Source panels and upper/middle/lower crops are under `C/frequency_HD63_assembly_v1/review`; `visual_review.json` records their scope. No3D quality benefit has yet been measured.

The two ready hashes are decay `f91d1354ffcf00ad0757176a7dc1d2e5f1f77b0cae19f5ec587d9e2b4d454fe6` and.93 `2f5c0bbf93affc2bd97798f65e538d4a8e4e8f28879d848e399706e8496e64f6`. They are prepared inputs, not training authorization. [The CPU assembler](../scripts/assemble_marshawn_HD_frequency_variants.py) verifies the original inventory before and after assembly and preserves the fitted map bytes. Subsequent trials must start cold from seed42 with the same quarter-step/λ.1 recipe and original confidence; maps affect both FAS ray proposals and frequency-field updates.
