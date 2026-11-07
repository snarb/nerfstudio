# Four existing SOURCE support variants, CPU feasibility

Tested existing corrected occlusion_v2 W32/W64 × gate0/pad1, each with an explicit L∞ geometric buffer .002 and separate numerical epsilon 2e−7. Native R0 occupancy/traversal, models, masks and training were unchanged. W32/W64 here are native RAW support widths, not the earlier RGB-loss-width experiment.

| Source | Cells | Whole-LC cover¹ | Positive XYZ / pixel-depth | Minimum front / back ray clearance | ±.02 interval retained, minimum | Positive TRAIN rays still missed |
|---|---:|---:|---:|---:|---:|---:|
| W32_gate0 | 65,970 | 91,703 | 109/109 | 0.010377 / 0.027907 | 75.943% | 730 |
| W32_gate_pad1 | 91,703 | 117,736 | 109/109 | 0.021702 / 0.044641 | 100.000% | 91 |
| W64_gate0 | 78,056 | 105,049 | 109/109 | 0.019062 / 0.038401 | 97.656% | 308 |
| W64_gate_pad1 | 105,049 | 131,911 | 109/109 | 0.028016 / 0.063362 | 100.000% | 0 |

¹ Whole-LC cover counts are the existing conservative cell mapping of each undilated source mask; they do **not** include the additional pointwise .002 buffer. No mapped grid was installed.

All 109 observations represent only 41 distinct XYZ: 39 clothing points plus two shoe anchors in six views. All points and calibrated pixel-depth positions survive every option. Point5568 retains its inherited hair/chest occlusion ambiguity; no opacity-one target is inferred. These sparse tests do not certify complete hair or foot silhouettes. Along-ray clearance is an exact cell-union intersection, not a 3D uncertainty bound.

The existing 4,657,323-ray audit was inherited, and only its known misses were retested because positive geometric expansion is monotone. W32/pad1 still misses 91 rays: IA65, LA18, LB7, MC1. These overlap views with previously uncertain bag contacts; neither real actor clipping nor harmless false labels may be assumed. W64/gate0 misses308; W32/gate0 misses730. **None of these tighter variants is cleared for training or carving.**

The fixed source126316 A native profiles lose mean original clothing mass0.5220%,0.4578%,0.4620%,0.2149% respectively; no measured native ±.02 clothing-band mass is removed. Dense16384 profiles remove0.0169178 and0.0009815 total band weight for W32/gate0 and W64/gate0 respectively, versus total97.3787; both pad1 variants remove0. These are restrictions/integrals at saved sample positions, not new native rendering performance and not physical-free-space proof. All115 supplied rays were checked; DC12 remain posthoc EVAL diagnostics with no witness depth and never become TRAIN targets.

Provenance: all63 support archives were rehashed; W32 support is a subset of W64, with byte-identical valid/UNKNOWN arrays, shared corrected-positive sources, mat source, transforms, AABB and floor frame. Actual semantic and RAW archive hashes were verified. SourcePad1 is independent of native occupancy dilation. Outside-domain UNKNOWN remains admitted; depth support intervals are clipped to the field AABB.

Validation: three CPU tests cover parallel/face rays, analytic-interval/point-lookup equality, monotone buffer growth and outside-UNKNOWN distinction. CUDA remained uninitialized. Initial v1 stopped only because saved trace arrays had not yet been fetched; v2 completed; v3 additionally binds the existing ray-audit checkpoint/pose/query identity. No model query or label mutation occurred.

[Full source/hash report](assets/marshawn_source_support_fourway_20261107/report.json), [saved-position integrals](assets/marshawn_source_support_fourway_20261107/saved_sample_mass.json). Source artifacts: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/source_support_fourway_CPU_v3`.


## All91 remaining W32/pad1+.002 rays: existing source review

All91 pixels are present in four connected clusters, with no omitted pixels: [IA65](assets/marshawn_source_support_fourway_20261107/missed91/IA_cluster01.png), [LA18](assets/marshawn_source_support_fourway_20261107/missed91/LA_cluster01.png), [LB7](assets/marshawn_source_support_fourway_20261107/missed91/LB_cluster01.png), [MC1](assets/marshawn_source_support_fourway_20261107/missed91/MC_cluster01.png). Pages pair HD context, corrected UNKNOWN, native RAW photographs and original black/grey/added-hair masks; every red footprint maps the same calibrated HD pixel. Semantic coverage uses the actual AREA-prefilter plus calibrated remap, separately from the geometric footprint outline.

All four pages were visually inspected. The missed clusters lie on tips/edges of a separate dark floor object consistent with the previously inspected bag region, not on visible actor hair. This observation is not a new label or permission to carve. At nearest RAW centers89/91 pixels are original grey+nominal, two are outside nominal; none are black or added hair. All91 exact HD pixels are outside the current corrected UNKNOWN, though UNKNOWN surrounds the previously reviewed interior. This explains why unchanged positive masks can conflict with a tighter volume; root source qualification remains required before changing any contract.

[All91 exact coordinates, original/current classes, calibrated RAW coverages and source/pose hashes](assets/marshawn_source_support_fourway_20261107/missed91/review.json). No model, GPU, relabeling, grid change, training change or V1 modification was used.

Root subsequently inspected all four native pages and qualified these91 clusters as visible distal bag-tip/floor, not apparently actor material;89 originalgrey labels are false positives. This is a diagnostic semantic review, not optical alpha GT. Frozen labels remain unchanged. No whole-ray negative is authorized: an opaque object does not make the space behind it empty. The91 misses therefore are not demonstrated actor losses, but this does not certify the entire tighter geometry. Exact review receipt: `C/independent_occupancy/W32_pad1_missed91_source_review_v2/root_manual_review.json`.


## Prepared read-only W32 comparison, not launched

Separate generation `C/independent_occupancy/source_mask_W32_counterfactual_v1` reuses exact original quarter+.1 source126316 (`340e794…cd641`). Phases: original mapped A, known W64SourcePad1+.002, W32SourcePad1+.002, restored A. Each saves three full FHD RGB/alpha views and124 declared rays (103TRAIN clothing,12posthocDC,9contour), native plus8192/16384 dense profiles. The91 bag pixels are not negative supervision or additional training data. Original coarse traversal/core/EMA/frequency remain fixed; only the source point density lookup differs. This can diagnose contour cost and artifact reduction; it cannot promote an inference-only asset.

The controller requires an idle physical UUID under the shared exclusive lease, bounded8GiB allocator/10GiB process/900s; termination cleans only its child. Source A parity, known W64 B full-render/native9 parity, actual admitted/excluded density identity, seven serialized states, pipeline, RNG/LUT and restored native arrays are checked. Original1495 science files are unchanged. Five CPU tests passed (three geometric, two exact-release fail-closed); all1495source,8helpers and23 artifacts were independently rehashed on silver without CUDA. Final policy SHA `c63ffb131d84e9cdf44cb6c7867ba24a73f0573b7729f5b12c2d609c3b1c690d`.

[Prepared policy](assets/marshawn_source_support_fourway_20261107/W32_readonly_preparation/policy.json), [CPU binding](assets/marshawn_source_support_fourway_20261107/W32_readonly_preparation/CPU_binding.json). `release.template.json` remains false; no controller or GPU query was launched. The separate final CHILD continuation replay policy122848 remains unchanged.
