# DEC5: preserving original texture support after cross-time mesh completion

## What was tested

Transfer of the frozen [exact-original-surface fallback](dec5_inferred_visibility_backoff.md)
to the completed [all-radius controls](dec5_mhr_radius_seed_transfer.md) at
001083 and 001195, each in current cinematic, F/E train and old moving stress
views. The preceding turn verified and delivered the native 6K file but did not
advance geometry repair. This experiment addresses one obstacle to using already
improved meshes: inferred appendages can unnecessarily remove texture support
from original, unchanged target intersections.

`transfer_original_surface_backoff.py` invokes the existing frozen producer,
two CPU workers, with verified baseline/candidate render receipts. Recovery
requires the same original triangle, depth within 1e-7, barycentrics within
1e-6, exact original mesh prefixes and identical cameras/color/source recipe.
Only black/no-source candidate pixels can reuse their verified baseline train
reprojection. New geometry, valid candidate colors, and all other source IDs
remain unchanged. No GT, fitting, source RGB averaging or geometry editing.

The new independent array audit rejects changes outside the admitted pixels,
incorrect restored colors/source IDs and invalid baseline sources. All recovery
and remaining-regression components receive native crops; none are top-k hidden.
These are 1080×1920 diagnostic stills, **not a replacement for the actual
3456×6144 video**. The original all-radius meshes are referenced, not rewritten.

Root: `/mnt/data/dec5_mhr_surface_backoff_transfer`.

## Results

All six cases finished successfully. Counts compare against their matched
production geometry/render, not against the older nearest24 completion.

| Time | View | New black before → after | Added geometry hits | Added hits still uncolored |
|---|---|---:|---:|---:|
| 001083 | Current moving | 2 → 0 | 0 | 0 |
| 001083 | F/E | 0 → 0 | 2 | 0 |
| 001083 | Old moving | 0 → 0 | 0 | 0 |
| 001195 | Current moving | 1 → 0 | 0 | 0 |
| 001195 | F/E | 0 → 0 | 112 | 0 |
| 001195 | Old moving | 6 → 0 | 115 | 7 |

The nine recovered pixels are all unchanged original-surface intersections.
Every other RGB/source pixel is bit-identical to the raw completed render.
Candidate depth and geometry are unmodified, including every added hit. Neither
time loses an original depth hit in these views. Zero new black pixels is a
specific regression result, **not** proof of correct texture or artifact freedom.

Replaying the fixed 73-ray 001195 under-jaw inventory gives **73 → 1 misses**
for original → completed mesh, with 72 colored fills before and after fallback.
Thus the existing substantial local hole repair survives removal of its texture
regressions. This experiment did not newly invent or enlarge that geometry repair.
The one missing ray remains missing; no RGB inpainting hides it.

- [Native fixed-hole comparison](/mnt/data/dec5_mhr_surface_backoff_transfer/fixed_hole/jaw_native.png)
- [Old-moving whole head/neck](/mnt/data/dec5_mhr_surface_backoff_transfer/001195/old_moving/review/head_native.png)
- [Numerical inventory](/mnt/data/dec5_mhr_surface_backoff_transfer/result.json)
- [Visual verdict and hash audit](/mnt/data/dec5_mhr_surface_backoff_transfer/visual_review.json)

Main agent actually inspected all 21 saved comparison panels: six head/neck
contexts, eight recovery components, six uncolored components and the fixed-hole
crop. Shoulder/hair-margin single-pixel regressions revert. The original dark
puncture under the jaw is visibly reduced in the completed mesh. Seven newly
intersected pixels at the clothing margin remain untexturable; the fallback
correctly does not paint them using an unrelated original surface. Ragged
hair/neck/shoulder edges and inherited texture issues remain. **Partial local
improvement; not approved as an artifact-free temporal reconstruction.**

Nine focused tests pass (seven transfer-array checks, two original geometric
selection tests). Final sealing rehashes inputs/retained outputs and frozen
renderer helpers. This is not a held-out metric comparison; PSNR/SSIM/LPIPS
are not inferred from pixel counts. No original data, video or defaults changed.

## Insights

The same failure and narrow remedy transfer across actor times: an inferred
surface may suppress sources of a still-visible original surface. Keeping the
measured/original and inferred layers distinct avoids sacrificing that original
texture when confidence in the new occluder is insufficient. Reusing a matching
point's old source is not permission to copy old colors onto newly exposed
geometry or to bypass real source occlusion generally.

The next deployment decision needs a genuinely temporal mesh/completion cohort
and renderer integration of this layer-aware visibility rule. Two sparse times
cannot establish flicker-free video. The unresolved mesh objective also includes
incorrect outward contours and the lipstick membrane: append-only completion
cannot remove those. Repeating support-count variants or declaring the nine
recovered pixels the full objective would not address those requirements.

Reproduce with the existing repository venv, `OMP_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, `OPENCV_IO_ENABLE_OPENEXR=1`:

```bash
python scripts/transfer_original_surface_backoff.py \
  --studies /mnt/data/dec5_mhr_radius_transfer_001083 \
            /mnt/data/dec5_mhr_radius_transfer_001195 \
  --output NEW_PRIVATE_ROOT
python -m pytest -q -o addopts= tests/test_transfer_original_surface_backoff.py \
  tests/test_original_surface_texture_backoff.py
```

The experiment-specific finalizer targets the recorded root: `prepare` generates
the fixed-hole panel; `seal --confirm-reviewed` is only run after actual manual
inspection of every listed image. It does not automatically grant a visual pass.
