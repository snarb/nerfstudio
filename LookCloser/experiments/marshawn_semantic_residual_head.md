# Conditional frozen-coarse semantic residual preparation

## What was tested

CPU-only preparation of [FrozenResidualActorGrid](../scripts/marshawn_semantic_residual_head.py). It copies the learned coarse XYZ logits, accepted bounds and identity floor transform into frozen buffers and adds a separately initialized zero fine XYZ logit grid. Inside the inclusive box,
`p = sigmoid(trilerp(coarse_logits) + trilerp(residual_logits))`;
outside, `p=0`. Both interpolations use Z/Y/X sampling coordinates, border padding and `align_corners=True`. The sole parameter is the residual. No regularizer, target change, prior change or selector is introduced.

This is an optional hypothesis after the direct-LC baseline visual gates, not an approved experiment recipe or measured improvement. The local [paper](../Paper%20LookCloser.md), §4.2, discusses frequency-guided feature resolution but does not propose this semantic residual architecture. Native RGB, density, packed intervals, weights and LookCloser frequency/admission state are outside this helper. Baseline head/fitter and the physical render loop are unchanged.

The accepted box is approximately `[-.84,-.21,-.07] .. [.15,.42,.63]` in source floor coordinates, not certified metres. The helper copies actual saved floating-point bounds; it does not round them. Baseline source step 126316 SHA256 is `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`. The sealed capture receipt SHA256 is `fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d`; its 63 records comprise 60 usable cameras and CE/JE/NE UNKNOWN, 69,888 rays and 12,993,330 native packed points, with 92 cap-hit rays. No cache or label file was changed. Only the previously approved IA UNKNOWN overrides 40470/40622 remain admissible; the proposed MD shoe/FD hair omissions have not been incorporated.

## Results

[CPU receipt](assets/marshawn_semantic_residual_head_20261107/receipt.json) records 16 residual tests plus 9 existing baseline contracts. All 25 pass with CUDA uninitialized. The largest residual test grid is 9³; memory counts for large grids are analytic. Tests establish:

- Bitwise zero-residual equality of coarse probabilities, native-weight rendered classes and their unchanged class objective on synthetic mixed-opacity/UNKNOWN/empty-ray examples.
- Independent analytic XYZ affine-logit checks in anisotropic source bounds, closed faces and nextafter outside samples, and zero outside gradients. A sharp coarse knot is preserved exactly even when coarse and fine knot planes are not nested.
- Positive and negative observations can refine separate fine supports inside one coarse cell. Local positive observations update only their eight reached fine basis nodes; disjoint unobserved supports retain the exact coarse prediction. Coarse copies and native XYZ/RGB/sigma/weight arrays stay unchanged and receive no gradient.
- Ordinary serialization restores head, dense Adam and private sampling generator exactly; four subsequent updates match bitwise. Initialization consumes no global Torch RNG.

No model was loaded, fitted, rendered or exported, and no GPU operation was launched. These synthetic contracts provide no hair/floor/shoe quality evidence; actual baseline and candidate renders remain the required visual gate.

FP32 memory, with MiB=`2²⁰` bytes:

| Node resolution | Nodes | Grid alone | Independent grid Adam minimum | Frozen128 + residual inference | Frozen128 + residual Adam minimum |
|---|---:|---:|---:|---:|---:|
|128³|2,097,152|8 MiB|32 MiB|—|—|
|256³|16,777,216|64 MiB|256 MiB|72 MiB|264 MiB|
|512³|134,217,728|512 MiB|2,048 MiB|520 MiB|2,056 MiB|

Adam minimum counts parameter, gradient and two FP32 moment arrays; hierarchical totals add the frozen 8 MiB coarse buffer. It excludes scalar step, metadata, an original source-head copy retained by the caller, packed caches, coordinate/interpolation activations, temporary optimizer tensors and allocator reservation. PyTorch Adam foreach temporaries can add material memory. With extent `.99/.63/.70`, node pitch is approximately `.007795/.004961/.005512` at 128, `.003882/.002471/.002745` at 256 and `.001937/.001233/.001370` at 512, in source units. Nodes span both endpoints; native 128 cells use a different pitch.

## Insights

Freezing a coarse head avoids resetting unobserved body regions to a low class prior. Initially zero residual preserves every coarse output by the same interpolation arithmetic followed by addition of zero. Under fresh ordinary Adam, basis nodes that never receive a gradient remain zero. Adjacent points sharing a trained basis node can change; previously trained nodes may continue moving through Adam momentum even when absent from a later packet. This is local support preservation, not a guarantee to preserve every unlabelled hair or shoe boundary.

The 12.99M points are quadrature contributions to 69,888 ray labels, not 12.99M independently supervised 3D labels. Many labels share occlusion ambiguity, and low-weight hair samples can have little influence. Positive rays can push background points toward actor; finer grids can memorize ray tubes or create holes between observations. A wrong coarse negative remains wrong in unobserved fine cells. The unapproved shoe/hair source proposals therefore cannot be treated as recovered supervision, and cap-hit rays still need their existing diagnostic treatment.

128/256/512 endpoint lattices are not nested: 127 does not divide 255 or 511. Keeping the original coarse interpolation explicitly avoids an approximate upsampled replacement. The existing single-grid node/cell bridge cannot be applied blindly to sampled sums: coarse knot extrema may lie between fine knots. A future conservative bridge could bound each field separately over a native cell and add their maximum-logit bounds, or evaluate their joint knot partition; that bridge is not implemented here.

Soft class suppression can still leave high-sigma floor opaque. Any physical selector and new transmittance must be assessed with actual source hair/shoe/clothing observations and comparable floor/bag/crew renders. Higher resolution alone cannot supply missing evidence or certify empty space. A future controller must bind the chosen learned coarse checkpoint and source/cache/label hashes, save exact optimizer/RNG/sampling state and pass GPU numerical/resource plus physical visual gates before integration.
