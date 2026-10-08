# Frame 000470: local-sampler repeat and independent density branch

## What was tested

The earlier sharp-28 local-sampler control was repeated from its pinned 32k
checkpoint to check whether the apparent cord benefit of the new termination
prior was stable. Both use the same 28 cameras, train-only two-landmark fit,
local ray radius `.005`, step `.001`, source RGB, and first 32 pixel/RGB batch
hashes. The repeat used newer code with added opt-in controls and was split
into 32k→32.5k and 32.5k→33k processes; the original was one 32k→33k
process. These are therefore a variability check, not a bitwise replay. No
source masks or held-out depth were used.

The separate-density hypothesis was then tested from a conversion of the
same 32k checkpoint. The converter copied the density hash encoding/MLP and
their optimizer moments, preserving the original colour branch and all
original parameter tensors bit-for-bit. Split control and split
termination-mass candidate use identical configuration except the `.003`
prior coefficient; their first 32 RGB/pixel batches match exactly. Each was
trained for 500 steps and evaluated at 32.5k. The candidate is a short gate
for whether geometry updates can be isolated from learned colour.

## Results

| Shared-field training | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original local only | 32500 | 19.8485 | .81657 | .22094 | 22.2898 | .77768 | .06761 | 27.2263 | .84148 | .06825 |
| Repeat local only | 32500 | 19.8175 | .81618 | .22095 | 22.2399 | .77699 | .06758 | 27.1891 | .84151 | .06846 |
| Termination mass `.003` | 32500 | 19.8350 | .81702 | .22109 | 24.1803 | .78729 | .07182 | 26.6763 | .82814 | .07401 |
| Original local only | 33000 | 19.7956 | .81603 | .22151 | 22.5719 | .78013 | .06945 | 27.3386 | .84131 | .06755 |
| Repeat local only | 33000 | 19.8024 | .81671 | .22057 | 24.4260 | .79707 | .07006 | 27.3295 | .84150 | .06758 |
| Termination mass `.003` | 33000 | 19.8052 | .81718 | .22097 | 24.2186 | .78862 | .07212 | 26.8668 | .82988 | .07251 |

At 32.5k, the new prior raises held-out cord PSNR by `1.94 dB` against the
repeat but makes its LPIPS `.00424` worse; training cam097 cord PSNR falls
`.51 dB` and LPIPS worsens `.00555`. At 33k, the repeat itself overtakes
both depth-prior variants on all three held-out cord metrics. The
[GT/original/repeat crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_local_repeat_33000.png)
shows real variation in learned cord colour, but neither repeat reconstructs
the GT lower braid. The [GT/repeat/density-prior/mass-prior
crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_localonly_priors_33000.png)
also shows the lower strands stay blended. A narrow evaluation-only strip
along the projected fitted line gains `+.86 dB` on rows 1340–1384 at ±4 px
for termination mass over the repeat at 32.5k, but this is not sufficient
visual recovery.

The earlier original local-only and density-band comparison remains a
valid matched pair: their source-code manifests are identical and both have
the local sampler. The repeat and termination-mass pair likewise share the
newer source-code manifest. The substantial `22.57→24.43 dB` held-out cord
variation between original and repeat at 33k, despite matching initial
pixel/RGB hashes, means small single-run gains should not be promoted.

| 32.5k split training | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS | 500-step seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Split, local only | **19.8508** | .81656 | .22129 | **22.2803** | **.77849** | **.06721** | **27.1868** | **.84160** | **.06895** | 46.65 |
| Split, termination mass `.003` | 19.8429 | **.81661** | **.22119** | 22.1633 | .77133 | .06893 | 26.7225 | .82977 | .07435 | 53.45 |

Both split arms finished without OOM. The [GT/control/prior
crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_split_mass_32500.png)
shows essentially the same lower blur, with no recovery of separate strands.
The split prior worsens all three cord metrics on both cameras. Its global
PSNR also falls `.008 dB`; the tiny SSIM/LPIPS changes do not counter the
cord regression. Splitting density alone costs about 47% more optimization
time than the shared-field control (`46.65` versus `31.82 s` for 500 steps);
adding the prior to the split branch costs a further 15%.

The [split conversion](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_split_initial_032000/conversion.json),
[control selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_split_control_032500/selection.json),
and [prior selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_split_termination003_032500/selection.json)
pin the exact fields and parameters. The conversion was checked for tensor
identity, but this turn did not run a separate exact forward-parity audit of
the two 32k models; previous split-field work established that parity on an
earlier lineage.

## Insights

The existing depth priors can alter actual termination weight and improve
some pixel error, but their effect on visible braid detail is not robust.
The globally best person checkpoint is still sharp-28 30.5k. The split
experiment rejects independent density as a remedy for this cord: extra
geometry capacity and depth pressure cost time and hurt both the held-out and
sharp training-view braid. The candidate stopped at the 32.5k visual gate.
Work should return to local multiview correspondence and colour detail on the
thin braid while preserving the normal no-prior person baseline.
