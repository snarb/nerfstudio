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
After the strong prior failed, a third arm used coefficient `.001` from the
same checkpoint, changing only that coefficient. Its first 32 RGB/pixel
batch and source-code hashes also matched the other two arms.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS | 500-step seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No prior | 19.70369 | .81483 | .23199 | **23.95539** | **.76829** | **.10119** | **26.61852** | **.81212** | **.08408** | **32.77** |
| Termination mass `.001` | **19.71749** | **.81503** | **.23133** | 23.85759 | .76336 | .10219 | 26.29884 | .80366 | .08800 | 39.18 |
| Termination mass `.003` | 19.69888 | .81497 | .23200 | 23.71776 | .75606 | .10585 | 25.93474 | .79522 | .09118 | 38.98 |

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

Reducing the coefficient threefold recovers most of the damage but still
lowers cam095 cord PSNR by `.098 dB` and train cam097 cord PSNR by `.320 dB`
relative to no prior; both cord SSIM and LPIPS also worsen. Global
PSNR/SSIM/LPIPS improve slightly (`+.0138 dB`, `+.00020`, `-.00067`), but
those small whole-frame gains do not repair the user's priority. The
[GT/no-prior/weak/strong comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_termination_strength_comparison.png)
shows no distinct lower braid for either coefficient. The
[weak-prior selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_termination001_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_termination001_033000/supervision.jsonl)
record terminal review without OOM. Both prior arms were stopped after the
500-step gate.

A read-only ray audit compared exactly the same 4K cam095 pixels in both
fields, capturing the original field samples and volume weights rather than
querying them again. On 65 fixed rays within `.002` of the fitted line and
with control band mass above `.5`, the prior increased mean termination
mass within `±.003` of the line from `.837` to `.913` and reduced mass behind
it from `.110` to `.034`. Yet the conditional RGB of that band against GT
worsened from `22.739` to `22.213 dB`; full rendered RGB on those rays
worsened `22.367→21.687 dB`. On the 34 lower near-line rays, the prior
raised band mass `.783→.902` and rendered RGB PSNR `22.301→22.453 dB`,
but band colour quality still declined and the full cord crop regressed.
This is a local diagnostic, not a separate checkpoint selection metric.
The [paired summary](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_ray_layers_paired_summary.json)
pins the fixed ray indices and the two [raw no-prior](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_ray_layers_no_prior/result.json)
and [prior](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_ray_layers_termination003/result.json)
audits. No held-out pixels entered training.

## Insights

More source pixels do not make this approximate straight-line depth/opacity
prior reliable enough for the lower braid. Its pressure redistributes density
but harms both held-out and sharp training-view cord detail, consistent with
the previous 1920px result. Both coefficients stop at the 33k visual gate.
The ray audit confirms that the prior actually moves termination weight to
the desired depth; it fails because the resulting colour and neighboring
cord pixels degrade, rather than because it has no geometric effect.
The no-prior native-size model remains the better of the three paired 33k
checkpoints, though neither fully reconstructs the braid. The fitted line
has only two train-view landmarks and no measured PatchMatch depth on the
cord; improving its local geometry or multiview correspondence is a more
defensible next step than increasing this prior coefficient.
