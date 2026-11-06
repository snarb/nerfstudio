# Separate prefix-only candidate: preparation and technical validation

## What was tested

The candidate adds one TRAIN-only visible-prefix optical-thickness term to
the unchanged full/body80/r0 LookCloser recipe. It uses native softplus density,
128 camera-balanced rays×64 stratified samples, and a private stateless seed
derived from optimizer step. No positive surface band, MVS, mask-absence term,
occupancy carve, activation change, SH change or sampler retuning is added.
The current production campaign and its frozen source remain unchanged.

Candidate root:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/prefix_candidate_20261106`.
Actual candidate source is `code_v2`; final proposed production specification is
`spec_prefix003.json`. Root reviewed the evidence and authorized production;
the separate controller launched the candidate on local GPU0. The opt-in worker preserves
the full historical FR trajectory and plateau stopping, with no iteration cap.

The source artifact uses9 reviewed opaque lettering points,18 original TRAIN
rays across AA/AB/BA/AC/AD. All points have3–6 original triangulation views;
only the2 visually reviewed observations per point supply prefix supervision.
Points63/64 and71/72 form nearby groups reserved together as8 diagnostic rays.
Point66 is excluded because it shares both rounded source pixels with65;
their world separation is.0002413, consistent with nearby correspondence
sampling rather than a large contradictory depth. Thus no duplicate source
pixel is double-weighted. The reserve is correlated local TRAIN evidence,
not an independent scene or native-model depth test.

The original track graph, archive hashes, current RGB/semantic hashes, rigid
floor transform and accepted AABB are checked. Current camera origins agree
with transformed original observations within1.11e-16; original float projected
UV agrees within4.55e-13 pixels. Prefixes stop at leastmax(.01,.05×depth) before
the witness and are clipped/inset inside the AABB. This margin is policy,
not calibrated depth uncertainty. Review excludes hair, silhouettes, floor,
bags and person overlaps. No additional clothing rays were admitted without
manual qualification.

## Results

The full4096-ray native mixed-precision zero-update probe used the actual
unmodified cold LookCloser field and candidate data. Geometry gradients were
finite at the native initial scaler65536; model state hash was exact before
and after, and no parameter gradient buffers or optimizer updates were left.

| Prefix coefficient | Auxiliary/RGB geometry gradient norm |
|---|---:|
|.001|3.091%|
|**.003**|**9.273%**|
|.01|30.911%|

The chosen proposed coefficient is.003, the only tested value in the declared
5–10% band. Peak allocated memory was5.85GiB. The earlier128-ray overlap probe
was a technical preliminary; timing was not treated as a benchmark.

The initial2-attempt save/resume smoke is explicitly **insufficient**: Adam
state remained empty while GradScaler backed off. It was retained as failed
optimizer-resume qualification, and a separate32-attempt output was used.

| Real technical checkpoint | Attempted step | Adam state entries | Successful Adam steps | Scaler |
|---|---:|---:|---:|---:|
|Before intentional pause|15|4|13|8192|
|After actual process restart|31|4|29|8192|

There were3 initial AMP skips and16 successful updates after restart. Saved
NN values and both Adam moments are finite. The initial full-training gradient
overflow source was not localized by the geometry-only zero-update probe;
native GradScaler behavior was preserved rather than changing the recipe.
The established finite scale then supported29 successful optimization steps.

Saved prior policy reproduces identical CUDA step16 samples across the two
checkpoints and leaves global CPU/CUDA RNG unchanged. This checks the stateless
private policy; it does not claim bitwise equality of independent TCNN training
runs. All three HD RGB/alpha outputs were produced and inspected. Their32-attempt
warmup fog is not a quality result or candidate promotion.

Seven CPU tests pass, including analytic/finite-difference softplus gradients,
source/coordinate/margin checks, no global-RNG effects, actual checkpoint policy
roundtrip and exact full trainer-config parity after normalizing the new parser's
serialization-module name. The original frozen worker is byte-identical to the
base used for the separate candidate worker.

- [Full4096-ray gradient receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/prefix_candidate_20261106/probe_cold4096/probe.json)
- [Actual Adam/CUDA resume receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/prefix_candidate_20261106/technical_resume_verification.json)
- [Qualified geometry manifest](/fsx/tmp/marshawn_001660_hd/tuning_20261106/prefix_candidate_20261106/geometry_v2/manifest.json)
- [Technical render inspection](/fsx/tmp/marshawn_001660_hd/tuning_20261106/prefix_candidate_20261106/technical32_overview.png)

## Insights and next steps

The isolated candidate passed root review and is training. Its effect on actor
quality is unmeasured. Compare matched
historical gates with full/body80/r0 using the existing fixed actor metrics,
held-out renders, admitted/skipped density traces, prefix mass, shirt-band mass
and residual transmission. Evaluate the4 reserved point IDs separately from
the fitted9 and excluded duplicate. Prefix-only may remove fog without creating
the correct surface; depth and opacity evidence must accompany LPIPS.

After root authorization, `release.json` binds the source, production spec,
geometry, gradient probe and optimizer-resume receipts. Controller2046193,
supervisor2046206 and trainer2046211 launched `full_body80_r0_prefix003` on
local GPU0. The separate controller records process, progress, GPU, OOM and
disk checks every30seconds. All preceding technical processes exited before
this production launch. The parent retains campaign supervision and visual gates.
