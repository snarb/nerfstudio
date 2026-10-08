# Frame 000470: local cord sampling during sharp-view continuation

## What was tested

The earlier sharp-28 continuation to 32k retained a visibly faint lower belt
cord. A frozen local sampler (`radius=.005`, fine step `.001`) improved the
held-out cam095 cord without changing field parameters. This experiment asks
whether using the same sampler **during** further training adds to that
forward-only benefit.

Two 32k→33k runs resume the identical sharp-28 checkpoint and optimizer
state, use the same seed and 28 training cameras, and evaluate at 32.5k/33k.
The control keeps ordinary sampling. The candidate changes only local cord
ray sampling; it does not add source masks, depth supervision, an opacity
objective or occupancy admission. The first 32 actual pixel/RGB batch hashes
match exactly. Both jobs were supervised to completion without OOM.

## Results

| Training sampler and renderer | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam095 cord LPIPS | cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary train / ordinary render | 32500 | 19.8516 | .81650 | .22070 | 21.2556 | .08419 | 26.8188 |
| Ordinary train / **local render** | 32500 | 19.8526 | .81654 | .22068 | 22.2929 | .06780 | 27.1692 |
| Local train / local render | 32500 | 19.8485 | .81657 | .22094 | 22.2898 | .06761 | 27.2263 |
| Ordinary train / ordinary render | 33000 | 19.8238 | .81679 | .22055 | 21.5752 | .08533 | 26.9574 |
| Ordinary train / **local render** | 33000 | 19.8248 | .81683 | .22053 | **22.6756** | .06923 | 27.3411 |
| Ordinary train / local render + occupancy `.003` | 33000 | 19.8248 | .81686 | .22052 | **22.6863** | **.06694** | 27.4103 |
| Local train / local render | 33000 | 19.7956 | .81603 | .22152 | 22.5719 | .06945 | 27.3386 |

The 33k ordinary-trained checkpoint was independently reevaluated under the
same local renderer as the candidate; the forward-only
[32.5k receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control32500_frozen_local_r005/result.json)
and [33k receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_frozen_local_r005/result.json)
record unchanged field fingerprints. At 32.5k the held-out cord PSNRs match
within `.004 dB`. At 33k, ordinary training plus local rendering exceeds
local training by `.104 dB` on held-out cord PSNR and by `.029 dB` globally.
The first 1000 optimization steps took `55.94 s` for the control versus
`62.62 s` for local training, about 12% slower.

The frozen 33k control was also rendered with `.003` cord occupancy admission.
Its [receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_local_r005_occ003/result.json)
again proves unchanged field weights. The small gain in held-out SSIM and
LPIPS is consistent with prior occupancy screens. The
[GT/source32k/native33k/combined33k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_best33k_cam095_comparison.png)
shows that the middle braid is stronger, while the lower braid still merges
with the skirt.

## Insights

The local sampler fixes an inference visibility problem, but training with it
does not improve the learned field over ordinary training on this continuation.
The 33k control rendered locally with admitted occupancy is the strongest
tested cam095 cord PSNR so
far. The lower braid remains visibly softer and less continuous than GT, so
this is an incremental gain rather than an artifact-free result. The mandatory
full-scene checkpoint selection across the sharp-28 lineage still favors
30.5k: its locally rendered `19.9804 dB` exceeds 33k's `19.8248 dB` by more
than the `.07 dB` LPIPS tie band. The 33k checkpoint is designated only as
the best **cord-specific** tested variant.
