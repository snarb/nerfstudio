# DEC5 000995: object-instance masks versus fin depth protections

## What was tested

Bounded, train-only semantic diagnostic of the remaining lipstick-adjacent fin.
The previous neutral-color silhouette could include nails/skin and damaged a
finger in a rejected control. This pilot instead uses prompted instance masks;
**no geometry or delivered video changes**.

[SAM 2's official image-prediction implementation](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)
is pinned to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, with official
SAM2.1 Hiera Large weights SHA256
`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
Five small dependency packages live in a private target directory; the main
environment, COLMAP, calibration and model defaults are unchanged. Inference
uses CUDA bfloat16, TF32 off, no mask postprocessing, and three candidate masks.

Native 240×300 portrait crops come from real 000995 train cameras H/C, K/B,
I/C, J/C, J/A with the exact frozen display response. The existing diagnostic
fin centroid only centers the crops. Manually inspected positive/negative
points and boxes are saved in `prompts_v1.json`; they describe visible lipstick,
not hidden geometry. Held-out RGB is never loaded. Enlarged review images are
diagnostics, not higher-resolution source observations.

Root: [/mnt/data/dec5_lipstick_instance_mask_000995](/mnt/data/dec5_lipstick_instance_mask_000995).
Code: `scripts/study_lipstick_instance_masks.py` and
`scripts/audit_lipstick_instance_witnesses.py`.

## Results

All five original crops and all fifteen SAM candidates were actually inspected.
Candidate 2 has the highest model score in every camera and follows the exposed
pink tip, barrel and lower metal without broad finger/nail/clothing inclusion.
Some candidate-1 masks include speckles on nails or adjacent skin. Selected
boundaries and occluded finger contact remain uncertain; SAM scores are not
independent shape accuracy. [Selection/review receipt](/mnt/data/dec5_lipstick_instance_mask_000995/mask_review.json).

| Train view | Selected mask area, pixels | SAM predicted score | Near-depth sample observations outside lipstick |
|---|---:|---:|---:|
| H/C | 3,207 | .9531 | 96 / 96 |
| K/B | 3,246 | .9375 | 63 / 63 |
| I/C | 3,078 | .9492 | 87 / 87 |
| J/C | 2,940 | .9336 | 87 / 87 |
| J/A | 2,900 | .9688 | 31 / 31 |

The existing manually diagnosed fin cohort contains 86 triangles, each queried
at three vertices and its centroid. The last column uses the original observed
depth deltas with near tolerance `.0015`, and a two-native-pixel semantic boundary
band. These are **364 camera/sample observations, not 364 unique points**.
All are outside the selected lipstick masks; none are ambiguous boundary samples
or outside the saved crop. This does not assert that the corresponding depth
measurements are wrong: they can correctly observe a different object.

For representative face 48941, the two previously established near protections
are its vertex 1 in I/C and J/C. They project **40.03 / 45.99 native pixels outside
the lipstick mask**, respectively. Actual native RGB witness panels were viewed:
both locations are by the fingertip/nail, not on the metallic tube. The other
three samples of that representative face are not near-supported in those views.
[I/C witness](/mnt/data/dec5_lipstick_instance_mask_000995/witness_audit/I004_C005_1210BA.png),
[J/C witness](/mnt/data/dec5_lipstick_instance_mask_000995/witness_audit/J004_C005_1210I4.png).
Only these two annotated witness panels were visually reviewed here; all five
are retained. This extends the previous numeric near-protection attribution,
not a claim that the entire fin is now safely removable.

The first inference attempt failed while drawing an overlay: SAM's thresholded
mask array was float 0/1, unsuitable as a NumPy boolean index. Its partial
`sam_v1/`, log and exact executed script are preserved. The corrected adapter
requires finite binary values before conversion to bool; `sam_v2/` completed
normally. Native crop bytes and prompts stayed unchanged. Seven focused tests
cover rotation mapping, API binary-mask conversion/rejection and signed-distance
orientation. No PSNR/SSIM/LPIPS or full-frame quality claim is made by this
semantic/geometry diagnostic.

## Insights

Learned object masks are more useful here than neutral-color thresholds, but
the important result is causal: some apparent "lipstick fin" protections are
measurements of the **finger**, not the lipstick. A nearby measured vertex can
protect a triangle spanning toward unsupported space; it does not establish
that the entire triangle is a valid barrel surface.

Do not blindly cut the selected cohort with a lipstick cylinder or negate all
non-lipstick support: either can cut genuine finger geometry. The next geometry
test should distinguish visible tube from hand/occlusion and qualify local
surface support, rather than treat every close-depth sample as support for the
same object. The masks are a disclosed semantic prior, not measured unseen
geometry, and are not promoted into the existing 6K delivery.
