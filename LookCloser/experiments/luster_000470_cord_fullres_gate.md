# Frame 000470: sharp-camera selection and native-resolution cord gate

## What was tested

The 28-camera, mask-free, no-prior local-sampling control was continued from
the same 32k checkpoint to 32.5k. A paired arm added only cam100 to its
training-camera allowlist. Cam100 has a resolved lower hanging cord in its
original JPG but a partly occluded upper section; it was a test of whether
another clear view repairs the faint lower braid. Both arms used 1920px
processed RGB, radius `.005`, fine step `.001`, and the same training recipe.

The original 165-camera frame was also decoded into an unmasked native-size
dataset (`4096×3000` or `3000×4096`) with the same 162/3 train/eval split,
normalized poses, intrinsics scaled to the original size, and scene AABB.
Decoded RGB was checked pixel-for-pixel against the source JPG for cams025,
026, 095, 097, and 100. A cheap initial training gate moves each existing
1920px frequency map to the native-size patch grid by bilinear interpolation
of its *locations*, preserving its fitted scalar-resolution values. This is
explicitly **not a native-size frequency fit**. The
[transfer receipt](/home/pavlo/renders/luster_000470_opacity/data_fullres_4096/frequency_transfer_audit.json)
records all 162 source-map hashes and the new transforms hash. The same 32k
field is first evaluated unchanged under this native-size RGB protocol, then
continued for 500 steps on the same 28 sharp cameras. Evaluation uses every
second native pixel; all compared native-size checkpoints use that protocol.

## Results

| 1920px 32.5k arm | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original 28 clear cameras | **19.81747** | **.81618** | **.22095** | **22.23988** | **.77699** | **.06758** | 27.18905 | **.84151** | .06846 |
| Add cam100 | 19.80843 | .81589 | .22115 | 22.22165 | .77580 | .06947 | **27.19510** | .84075 | **.06722** |

The [GT/control/+cam100 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/cam100_32500_comparison.png)
shows no reconstruction of the lower strands. Adding the camera slightly
worsens held-out cam095 cord PSNR, SSIM, and LPIPS. The arm finished without
OOM; the 500-step optimization segment took 31.84 seconds. Its
[selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp29_cam100_local_032500/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp29_cam100_local_032500/supervision.jsonl)
pin the checkpoint and GPU state.

The [native/1920px GT crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/native_vs_1920_gt_cam095.png)
shows that the downscale visibly smooths the twisted fibres, though the
1920px source still contains a distinct lower strand. Resolution is therefore
a plausible contributor, not proof that preprocessing alone caused the
rendered fade.

| Native RGB, every second pixel | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unchanged 32k field | 19.71464 | .81295 | .23575 | 22.24991 | .74452 | .10861 | 26.46424 | .80739 | .08983 |
| 32.5k, 500 steps at 1920px, frozen eval | 19.73415 | .81412 | .23440 | 21.92338 | .74291 | .10896 | 26.71170 | .81380 | .08718 |
| 32.5k, 500 native-RGB steps | 19.73291 | .81507 | .23267 | 23.94279 | .76467 | .10377 | 26.66791 | .81164 | .08386 |
| 33k, 1000 steps at 1920px, frozen eval | 19.71884 | .81466 | .23399 | 23.89605 | .76299 | .10986 | 26.82263 | .81362 | .08596 |
| 33k, 1000 native-RGB steps | 19.70369 | .81483 | .23199 | 23.95539 | .76829 | .10119 | 26.61852 | .81212 | .08408 |

Native RGB raises held-out cord PSNR by `1.69 dB` and improves SSIM/LPIPS
against the unchanged 32k source; whole-view metrics also improve. The
[GT/source/native-trained crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres_transfer_32500_comparison.png)
still shows the lower twist as a smooth blue region rather than separate
fibres. The run finished without OOM (`11.5/97.9 GB` GPU usage at its peak
supervision check), with 32.58 seconds of optimization. Its
[selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_transfer_032500/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_transfer_032500/supervision.jsonl)
record both checkpoints and the terminal review gate. A frozen evaluation of
the *1920px-trained* 32.5k control under exactly this native-RGB protocol
shows a `+2.02 dB` held-out cam095 cord advantage for native training, with
SSIM `+.0218` and LPIPS `-.0052`, while global PSNR differs by only
`-.0012 dB`. The [four-way GT/source/1920-trained/native-trained crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/native_vs_lowres_train_32500_comparison.png)
shows a somewhat stronger middle cord with native training, but the lower
fibres remain blended. The [frozen evaluation receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/frozen_lowres32500_on_fullres/frozen_evaluation_receipt.json)
pins the ordinary checkpoint and shared native-RGB protocol. The
low-resolution control and native-resolution continuation have identical
source-code file hashes; their images/frequency maps are the intended
differences.

Another 500 native-RGB steps barely change cam095 cord PSNR (`+.013 dB`)
but improve its SSIM/LPIPS. Global PSNR declines by `.029 dB`; because this
is within the `.07 dB` tie band, the specified checkpoint rule selects 33k
for its lower global LPIPS. The [GT/32.5k/33k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres_32500_33000_comparison.png)
shows no recovery of distinct lower fibres. The continuation
[selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_transfer_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_transfer_033000/supervision.jsonl)
show completion without OOM, with 32.77 seconds of further optimization.
The ordinary 1920px-trained 33k continuation almost catches up in the
held-out cord PSNR when evaluated on native RGB: `23.896` versus `23.955 dB`.
Native training retains better cord SSIM (`+.0053`) and LPIPS (`-.0087`),
but global PSNR is `.015 dB` lower and train-cam097 cord PSNR is `.20 dB`
lower. The [GT/1920-trained/native-trained 33k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/native_vs_lowres_train_33000_comparison.png)
shows a modest middle-cord contrast improvement, not resolved lower fibres.
Thus the large 32.5k PSNR gap was transient rather than a stable `+2 dB`
resolution benefit.

The [held-out cam095 face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/native_vs_lowres_train_33000_face.png)
shows a small angular patch on the chin in the 1920px-trained 33k render;
it is much less conspicuous after native-size training. Face PSNR improves
`27.086→27.181 dB` and LPIPS `.08170→.07805` under the shared native-RGB
evaluation. Clothing in the same view moves the other way (`24.161→23.922`
dB and LPIPS `.08592→.08992`), so the gain is not uniform across the person.

### Complete occupancy on the native-trained 33k checkpoint

The [frozen grid audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_voxel_audit_only/voxel_audit.json)
finds 12 cells intersecting the radius-`.003` conservative tube around the
train-fitted cord. Eight were already occupied. The existing sampled
admission would add all four missing cells; the twelfth cell it omits was
already occupied. A separate frozen screen forced all 12 cells active and
saved the cam095 render, though its full six-view evaluation was interrupted
by SIGTERM before global metrics were written. The
[GT/original/full-cover cam095 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_cover_cord_comparison.png)
still has the same smooth lower braid. From the saved 8-bit crop, PSNR rises
only `23.955→24.037 dB` (lower half `23.065→23.201 dB`); this is a
*quantized-image diagnostic*, not the normal float evaluation metric. The
[pixel comparison receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_cover_partial_analysis.json)
pins both render hashes and shows that only four cam095 opacity pixels
changed by at most one 8-bit unit. Thus the visible lower-cord fade is not
caused solely by unoccupied grid cells at native resolution either.

## Insights

The extra sharp cam100 alone does not fix the lower braid. The native-size
continuation yields modest held-out cord SSIM/LPIPS gains and a small visual
improvement in the middle cord at 33k, but does not solve the lower fade. A
single continuation is not enough to separate resolution effects from all
training-trajectory variance.
Its transferred frequency maps make this a cheap screen, not a conclusive
evaluation of a fully refitted native-size recipe. The second 500-step gate
does not recover the lower braid, so more steps with this unchanged objective
are unlikely to be the efficient next experiment. A native frequency fit or
more accurate cord geometry should be assessed before adding a stronger
depth/transparency prior.
The previously tested [complete occupancy cover](luster_000470_cord_voxel_cover_and_layers.md)
and its native-resolution repetition admitted all cells intersecting the
fitted cord tube and did not restore the braid; extending the occupancy
warmup is not the next lever.
