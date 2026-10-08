# Frame 000470: does local opacity hardening repair the hanging cord?

## What was tested

The no-prior 30k field followed by a 1k local-sampling continuation
(`radius=.005`, `step=.001`) still renders the lower hanging braid too faintly
on held-out cam095. A forward-only counterfactual multiplied the **existing**
field density inside a narrow tube around the train-only triangulated cord
segment. Factors `2`, `4`, and `8` and tube radii `.003`/`.005` were screened;
one variant affected only the lower 45% of the segment. The learned field,
RGB samples, sampling pattern and GT images were unchanged. No source masks
were used. The diagnostic source is
[`screen_luster_cord_density_scaling.py`](../scripts/screen_luster_cord_density_scaling.py),
and the [receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_density_scale_20261008/frozen_r005_31k_v1/result.json)
records checkpoint and fit hashes, exact crop scores, and unchanged field
fingerprint.

## Results

| Frozen density multiplier | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR |
|---|---:|---:|---:|---:|
| Native field | 19.9177 | .7276 | .09083 | 26.2408 |
| ×2, radius .003 | 19.5676 | .7298 | .08767 | 26.1231 |
| ×4, radius .003 | 19.0467 | .7176 | .08823 | 25.3360 |
| ×8, radius .003 | 18.4460 | .7004 | .09637 | 24.2486 |
| ×4, radius .005 | 18.7923 | .7055 | .09271 | 25.1728 |
| ×8, radius .005 | 18.0625 | .6788 | .10509 | 23.9270 |
| ×4, lower segment only | 19.6707 | .7244 | .08256 | 26.0181 |

The [cam095 comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_density_scale_20261008/frozen_r005_31k_v1/cam095_comparison.png)
shows more contrast in the upper and middle braid as density grows, but the
lower braid still fades into the skirt. The native row exactly matches the
saved checkpoint's ordinary cord crop metrics, verifying the counterfactual
parity. The factor-4 lower-only variant improves LPIPS while lowering PSNR and
still leaves a visible lower break.

## Insights

Simply forcing the existing volume to be more opaque is not sufficient; it
amplifies misplaced or wrong-colored density as well as the cord. The two
landmark segment projects near the braid in multiple original sharp cameras,
but inspection of the saved projection overlays shows it is not centered on
every twist or lateral view. A future hard prior should rely on more precise
multi-view cord geometry and color evidence, with uncertainty where the
thin object is blurred or occluded. The present density multiplier is a
diagnostic only, not a candidate renderer.
