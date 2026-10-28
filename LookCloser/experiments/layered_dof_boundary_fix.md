# Layered DoF: remove bright/dark subject-edge rims

## What was tested

The old presentation pass blurred already-composited RGB using mixed-image depth.
Partially covered edge pixels retained sharp old background while their neighbors
were defocused. Controlled frame00001 ablations kept RGB, alpha and checkpoint
fixed. Merely separating layers while retaining mixed-image blur radii failed;
using the background layer's blur radii removed the tested rims. Linear-light
composition also removed them: gamma alone and straight/premultiplied confusion
were not established as the cause.

## Results

The distant-background variant was checked on48 native3000x4096 flight frames.
It convolves a complete clean background with a circular aperture of radius24
native pixels before composing the original in-focus premultiplied foreground.
Opaque pixels are unchanged (at most one8-bit rounding code for recovered layers),
alpha/topology is unchanged, and changed-pixel masks accompany the validation.
Hair, hands, lace and gaps were inspected, including indices0/1/2/4/10/14/19/20/21/37/44/45/47.
This accepts the boundary composition fix, not opaque hem geometry, every matte,
or the complete192-frame movie/47-to48 mode transition.

- [Causal ablation](/home/pavlo/renders/luster_boundary_diagnosis_20261027/00001_arm_causal_ablation.jpg)
- [Independent frame14](/home/pavlo/renders/luster_boundary_diagnosis_20261027/00014_arms_before_after_v3.png)
- Native48-frame evidence: `/fsx/tmp/luster_dof_native_sequence_gate_20261028/`.
- Original campaign: `/fsx/tmp/luster_arc_variants_20261025/A`.

## Usage and insights

`scripts/layered_dof.py` provides reusable NumPy/OpenCV composition; use
`compose_distant_in_focus` when the subject must remain sharp and a constant
far-background blur is appropriate. Radius is in output pixels;24 is the tested
native-resolution setting, not a universal default. For varying background depth,
`compose_in_focus` requires background-only CoC (blur-radius map), not mixed depth.
`extend_background_coc` is an optional nearest-background extrapolation, not a
recovered metric-depth map. All operations return the selected compositing space.

```python
from scripts.layered_dof import compose_distant_in_focus, linear_to_srgb

# HxWx3 floating RGB in [0,1]; alpha is HxW coverage.
# foreground_premult has alpha applied and NO old background contribution.
linear_rgb = compose_distant_in_focus(
    foreground_premult, alpha, complete_background, radius_pixels=24,
    source_space="srgb", composite_space="linear",
)
rgb = linear_to_srgb(linear_rgb).clip(0, 1)
```

If layers were originally accumulated in sRGB, conversion must unassociate the
foreground, decode, then reassociate. Decoding premultiplied sRGB directly is
incorrect. In that same original blend space, recover the foreground from a
known composite as `raw - (1-alpha[..., None])*original_background`; do not pass
raw RGB as premultiplied foreground. Verify reconstruction and valid layers
before rendering. Alpha from a radiance field can include floor: it is not
necessarily a person matte. An incorrect matte or missing clean background is
not repaired by this operator. No erosion, component deletion, color/height
selection, retraining or geometry modification is used.

The archived-frame adapter reproduces the variable-CoC ablation, records input
hashes and change masks, and refuses to overwrite its output directory:

```bash
python scripts/diagnose_luster_layered_dof.py \
  --campaign /fsx/tmp/luster_arc_variants_20261025/A \
  --index 1 --out /fsx/tmp/luster_dof_recheck_frame1
python -m pytest -q tests/test_layered_dof.py
```

This adapter requires retained campaign float buffers and its archived
`code/compose_arc.py`; it is not a general RGB-only movie converter. The core
operator has no campaign dependencies. Historical depth-DoF code is retained for
reproduction, but should not be used as the recommended boundary-safe method.
Opaque turquoise false folds exist before DoF and remain a separate unresolved
reconstruction issue.
