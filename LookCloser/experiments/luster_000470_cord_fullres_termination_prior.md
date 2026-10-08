# Frame 000470: termination-mass prior after native-size RGB continuation

## What was tested

The [native-size cord gate](luster_000470_cord_fullres_gate.md) improved some
held-out cord metrics and reduced a face patch but did not restore the lower
braid. To test whether accurate source detail changes the prior's effect, two
32.5k→33k continuations used the **same** 4096px-trained 32.5k checkpoint,
the same 28 sharp training cameras, identical transferred frequency maps,
local cord sampling (radius `.005`, step `.001`), and native-RGB evaluation at
stride two. The only model-control difference was `cord_ray_regularization`:
zero versus `.003` with the train-only, two-landmark `termination_mass`
objective. The prior encourages ray termination inside `±.003` normalized
depth of the approximate 3D cord line, as described in the earlier
[objective experiment](luster_000470_cord_termination_mass.md). Neither arm
used source masks or held-out pixels for training. All 32 recorded ordinary
pixel-index and RGB-batch hashes matched, as did all source-code hashes.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS | 500-step seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No prior | **19.70369** | .81483 | **.23199** | **23.95539** | **.76829** | **.10119** | **26.61852** | **.81212** | **.08408** | **32.77** |
| Termination mass `.003` | 19.69888 | **.81497** | .23200 | 23.71776 | .75606 | .10585 | 25.93474 | .79522 | .09118 | 38.98 |

The prior lowers held-out cam095 cord PSNR by `.238 dB`, SSIM by `.0122`,
and worsens LPIPS by `.0047`; training cam097 cord PSNR falls `.684 dB` and
LPIPS worsens `.0071`. Global PSNR falls `.0048 dB`. The
[GT/no-prior/prior crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_termination003_comparison.png)
shows that the prior makes some cord sections thicker without separating the
lower fibres. Face metrics are nearly unchanged (held-out cam095 face PSNR
`27.181→27.189 dB`, LPIPS `.07805→.07775`). Both runs finished without
OOM; the prior adds roughly 19% optimization time for these 500 steps. The
[prior selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_termination003_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_termination003_033000/supervision.jsonl)
pin its checkpoint and GPU state.

## Insights

More source pixels do not make this approximate straight-line depth/opacity
prior reliable enough for the lower braid. Its pressure redistributes density
but harms both held-out and sharp training-view cord detail, consistent with
the previous 1920px result. This candidate stops at the 33k visual gate.
The no-prior native-size model remains the better of the two paired 33k
checkpoints, though neither fully reconstructs the braid. The fitted line
has only two train-view landmarks and no measured PatchMatch depth on the
cord; improving its local geometry or multiview correspondence is a more
defensible next step than increasing this prior coefficient.
