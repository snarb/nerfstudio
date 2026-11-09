# Canonical Marshawn PQ preflight

## What was tested

The signed take contract uses52.13360318135631 nits per linear RGB unit,
floor0.005nits and content peak4000nits. Vendor `encode_tier` performs
`PQ(clip(gain*linear, floor, peak))`, with the ST2084 reference fixed at10000nits.
Actual vendor AP1 primaries use D60 despite a misleading `ap1-d65` metadata tag.
The canonical relative AP1 EXR master has no gain4 or scalar normalization baked in.

Frozen LookCloser HDR utilities instead add the black value and clamp at10000.
Changing flags alone does not reproduce the signed transfer. The uninstalled
[preflight helper](../scripts/marshawn_canonical_hdr_preflight.py) implements the
signed floor/peak transfer separately; no accepted model or package was changed.

## Results

Five CPU tests passed: actual vendor function agreement below6e-8, true
floor/peak clamp gradients, differentiable raw-density capture, extreme-negative
surface gradients, unsupported-policy rejection and exception cleanup.

The existing surface objective is not directly compatible with frozen
LookCloserField: it does not populate the inherited raw-density cache and has
no `average_init_density`. Its actual density is `trunc_exp(h0+1)`.
A temporary geometry-MLP hook exposes `h0` through a plain external accessor
with `average_init_density=e`. No field cache, module or parameter is replaced.

Actual CUDA forward/backward used the full frozen16-level/hash21/max8192 field,
scene_loose bounds and original validated44938 observed intervals. Sigma agreed
with `exp(h0+1)` exactly; the surface forward value matched direct density
integration exactly. Encoding and geometry gradients were finite and nonzero;
the color network had no gradient. Parameters, buffers and global CUDA RNG were
unchanged; the hook was removed. There were zero optimizer updates.
Receipt: `/fsx/tmp/marshawn_001660_hd/canonical_hdr_preflight/actual_field_probe.json`.

Train-only actual vendor AP1-Y statistics select explicit head initialization
0.0201105075 and scale30.89463468, avoiding automatic Rec709 luma coefficients.
The complete physical control remains a held proposal:
`/fsx/tmp/marshawn_001660_hd/canonical_hdr_preflight/proposal.json`.

A separate native encoded-PQ numerical control passed independent prelaunch
review. Full AA/DC/HC/KA EXR frames equal vendor uint16 PNG values divided by65535
exactly, retaining20k–42k distinct channel codes. The real128-sample training
batch matched EXR exactly. All66 camera metadata fields agree except image
extension; all six geometry-prior NPZ files are byte-identical. The fixed empty
background is the actual quantized floor988/65535. Native NN/model/optimizer
configuration remains unchanged; the controller caps22000 attempts and pauses
in place after the saved2k checkpoint for decoded visual review.
Receipt: `/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/independent_review.json`.

## Insights

Encoded-PQ numerical compositing is a distinct control, not physical linear
radiance integration. Its preview must apply the EOTF and actual AP1-D60 color
transform; raw PQ code channels must not be displayed as ordinary sRGB.
The physical HDR proposal still needs the canonical encoder wired into both
loss and metrics, and an explicit verified geometry-metadata bridge. It is on
hold; no physical HDR training was launched. The accepted SDR release remains
the fallback, with its checkpoint and archive unchanged.
