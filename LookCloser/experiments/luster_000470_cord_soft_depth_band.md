# Frame 000470: softer train-only cord depth-band prior

## What was tested

The previous `0.01` cord-depth opacity prior hurt both the held-out and
training-view cord crops. A milder `0.003` coefficient was tested over the
same sharp-28 32k→33k continuation as the zero-prior local-sampling control.
Both arms resumed the identical checkpoint and optimizer state, used the
same seed, 28 cameras and local ray sampler (`radius=.005`, step `.001`), and
shared the first 32 pixel/RGB batch hashes exactly. The candidate alone
added 32 train-only rays per step around the two-landmark cord line, asking
for `.99` opacity in a `±.003` depth band and low density in six probes in
front. No image masks were used, and the held-out cam095 supplied no prior
ray. Both runs finished without OOM.

## Results

| Training | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-prior control | 32500 | 19.8485 | .81657 | .22094 | 22.2898 | .77768 | **.06761** | **27.2262** | **.06825** |
| Cord band `.003` | 32500 | **19.8513** | **.81728** | **.22065** | **24.0604** | **.78214** | .07578 | 26.4249 | .07633 |
| Zero-prior control | 33000 | 19.7956 | .81603 | .22152 | 22.5719 | .78013 | **.06945** | **27.3386** | **.06755** |
| Cord band `.003` | 33000 | **19.8082** | **.81742** | **.22068** | **24.2345** | **.78785** | .07368 | 26.7808 | .07325 |

The standard global checkpoint rule selects 32.5k for both arms. The
[GT/best prior-free 33k/matched control/prior comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_cord_soft_prior_4way.png),
[32.5k GT/control/prior crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_ray003_32500_cam095_comparison.png)
and [33k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_ray003_33000_cam095_comparison.png)
show a somewhat denser middle braid, but the lower portion still loses its
distinct strands and blends into the skirt. The large cam095 ROI gain is not
all on the cord: in a diagnostic strip following the approximate projected
3D line from rows 1250–1385, the 32.5k PSNR changes by `+0.75 dB` at a
`±4 px` width, by `−0.25 dB` at `±6 px`, and by `−0.22 dB` at `±8 px`.
For rows 1340–1385, the corresponding changes are `+0.95`, `+0.58`, and
`+0.82 dB`. This strip is an evaluation diagnostic only, not a dataset mask;
the line approximates the braid rather than tracing each twist.

At 32.5k, held-out cam095 face PSNR improves `28.0546→28.1104 dB`, and
the hand and clothing also improve. The prior run takes `73.22 s` for its
1000 optimization steps versus `62.62 s` for the matched zero-prior local
sampler, about 17% longer. The [candidate selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_ray003_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_ray003_033000/supervision.jsonl)
record the completed run.

## Insights

The weaker prior can improve RGB agreement near the cord, but it is not yet
an artifact-free recipe. It worsens held-out cord LPIPS and the training-view
cam097 cord despite modestly improving global metrics and portions of the
held-out centreline. The full-scene best checkpoint remains the earlier
sharp-28 30.5k candidate (`19.9804 dB` with local rendering), exceeding
this run's selected `19.8513 dB` outside the `.07 dB` LPIPS tie band.
The learned surface still lacks the GT braid texture in the lower half.
Further work should target reliable local colour/geometry instead of merely
increasing a geometric penalty or claiming that the large ROI PSNR proves
the transparency artifact is solved.
