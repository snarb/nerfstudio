# Best LookCloser native semantic capture: CPU proposal

## What was tested

Prepare read-only packed TRAIN visibility capture on the actual best LookCloser
field, before considering a separate actor/nonactor head. No fitting, density
editing, mask editing, new labels, source-image color work, or GPU work occurred.

The selected source is `quarter_distortion_rawmin`, exact step 126316, checkpoint
SHA256 `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`.
It retains quarter coarse step `.0015625`, distortion `.1`, original mapped W64
admission, R0, body80/RGB W64, FR `.3`, `trunc_exp`, correct SH and legacy ARM
cap1024. The local paper motivates preserving its frequency grid, reweighting,
and adaptive intervals; this adapter adds no alternative sampler.

Package: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v1`.
Its manifest SHA256 is
`9f023343bc47dd53249642badce10bf64cec156e0290e77f34bd325f07bc98bc`.
Both the manifest authorization and release template authorization are false.

## Results

| CPU check | Result |
|---|---|
| Frozen scientific generation | All1495 files match manifest `ad32a034c492314162fac8d966be2215c834393a1bb984beb05f1eed8f9c43b3` |
| Qualified fixed TRAIN queries | 69,888 rays;63 declared cameras,60 usable; CE/JE/NE have no queries |
| Heldout camera exclusion | DC/HC/KA absent from fit-query membership |
| Coordinates | All63 LC floor poses equal old NGP pose transformed once into floor coordinates, maximum error exactly0 |
| Raster/intrinsics | Exact across all63 cameras |
| Class-head coordinate transform | Identity; applying old NGP `native_world_to_box` again would be wrong |
| Exact IA overrides | [1019,567], index40622 and [1013,562], index40470 retain original positive target but receive weight0 in the new cache only |
| Source checkpoint availability | Absent locally; source bytes must be obtained from silver or execution must occur there |
| Adapter CPU tests | 13 passed, CUDA visibility empty |
| Packaged CLI dry run | Reports authorization=false, GPU_started=false |

The test command was:

```bash
CUDA_VISIBLE_DEVICES='' /home/pavlo/.conda/envs/nerfstudio/bin/python -m pytest -q tests/test_marshawn_lc_semantic_capture.py
```

CPU tests exercise noncontiguous ray IDs, empty rays/entire batches, physical
versus normalized interval units, independent serial emission/absorption
weights, RGB/weight/spacing/ray-order/delta corruption, duplicate field calls,
frozen-mode gates, and exact restoration of an instance-overridden forward after
an exception. These tests do not establish actual GPU source parity.

The existing native DC model render was inspected as a720px thumbnail. It shows
the two actors but also extensive scattered density extending below/left of
their feet and ragged boundaries. This is visual evidence of remaining source
artifacts, not evidence that a semantic head will remove them safely.

![Existing source DC model thumbnail](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v1/source_DC_thumbnail.png)

### Native path and interval semantics

The frozen source's `adaptive_ray_marching` invokes `self.field(ray_samples)`
exactly once after its native sampler/refinement/fallback hooks when samples are
nonempty, and zero times when empty. A temporary field.forward wrapper captures
those exact final samples and field outputs while the unmodified native
`get_outputs` supplies packed indices and weights. The original Python binding
is restored in `finally`, including preexisting instance overrides.

The adapter proves final spacing-array identity, packed shape/order, original ray
origins/directions, midpoint XYZ, physical deltas, nonoverlapping ordered
intervals, segmented sigma-to-weight composition, accumulated alpha, black RGB
composition and actual renderer RGB. It reads physical distances from
`RaySamples.frustums.starts/ends`. Normalized `packed_spacing_starts/ends` are
saved separately solely for alignment; they are never interpreted as distances.
An unwrapped native render of the same chunk is also compared at runtime.

In eval, `FrequencyAwareVolumetricSampler.get_sigma_fn` returns None. Nerfacc
therefore has no density callback for opacity pruning. The source threshold is
also0. This native eval path is not opacity terminated. Nevertheless, occupancy
skipping and the original legacy ARM cap can omit geometry. Its legacy allocator
rescales requested dt then clips sample ranks at1024; the final tail can be
discarded. `cap_hit` and per-ray sample counts are saved. A cap hit is a saturation
diagnostic; it does not quantify discarded intervals. No cap/sampler fix is made.

### Concrete runtime package for root review

The three new scripts are copied into the package's `code/` directory and bound
by the manifest. Execute the packaged runner only. The normal CLI is a dry run;
`--execute` requires a separate root receipt with `capture_authorized=true`, the
exact manifest hash and exact code hash map. The false template grants nothing.
Authorization/input verification happens before Torch/source/checkpoint load.

For local execution after hash-exact source checkpoint availability and root
review, the existing globally leased UUID controller must spawn this argv as its
direct worker:

```text
/home/pavlo/.conda/envs/nerfstudio/bin/python
  /fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v1/code/run_marshawn_lc_semantic_capture.py
  --plan /fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v1/manifest.json
  --release <separate-root-reviewed-release.json>
  --execute
```

The controller must set an exact physical UUID in both `CUDA_VISIBLE_DEVICES`
and `EXPECTED_GPU_UUID`, and supply `LOOKCLOSER_GPU_LEASE_RECEIPT`. The inherited
isolation helper verifies its actual immediate supervisor, process start/boot
identity, UUID environment and CUDA properties. The worker uses logical cuda:0
only, checks actual compute-process UUID and exclusivity per camera, and rejects
foreign processes. No local silver self-SSH is used.

Resource limits:128 rays/chunk,8GiB Torch allocator cap,10,240MiB actual process
GPU cap,12GiB RSS cap,1800s alarm; require8GiB free RAM and20GiB free disk before
loading. Progress/resource receipts are written after each camera. The worker
fails if available RAM drops below2GiB. Root supervision remains required. Silver
can use the same sealed package bytes with `--campaign` pointing at its matching
campaign root and a confirmed Torch2.7.1+cu128/CUDA12.8 interpreter.

The strict existing native_A eval loader is reused in inference mode, avoiding
image/mask caching. Only source-root path relocation, exact-checkpoint selection
and sourceR0 eval-dilation selection occur in memory; scientific files are never
edited. The full loaded pipeline must match the source checkpoint digest, and
actual core/active/EMA/admission must match the archived GPU snapshot. Source
FR=.3 is reinstated, its exact discrete LUT is primed, and source RNG is restored
before immutable capture snapshots. Parsed camera pose/raster/intrinsics and an
independent pinhole pixel-center ray chain must pass on every chunk.

Each camera archive contains original qualified pixel/target/category rows,
only the two approved in-memory weight0 overrides, actual XYZ/sigma/RGB/physical
intervals/native weights/packed indices, normalized spacing, native RGB/alpha,
sample counts/cap flags, identity world-to-box and source-model digest. Empty
camera records remain explicit. Completion requires unchanged full pipeline,
RNG, primed nonpersistent LUT, semantic admission/normalization method, all seven
serialized checkpoint states, source checkpoint and original query hashes.
No source optimizer, callback, occupancy update or frequency update runs.

## Insights and next steps

This package is ready for root code review, not GPU release by this subtask.
Checkpoint availability and actual GPU import/camera/array/RGB/alpha/state parity
remain mandatory runtime gates. Root should inspect the complete receipts before
authorizing any separate head-fitting proposal.

Native contribution supervision is conditional on existing source occlusion.
It cannot certify hidden actors, false floor geometry, clean hair/contact
boundaries, or a perfect occupancy grid. Physical actor extraction requires an
independent full-path density evaluation and newly computed transmittance after
semantic thinning; the packed inference cache alone is insufficient. No class
head transfer from the softer NGP field, fitting, physical export, or geometry
change is included here.
