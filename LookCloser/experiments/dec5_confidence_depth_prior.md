# Confidence-gated learned depth pilot

## What was tested

Completed 2026-09-14: **no learned variant is promoted**. This is an isolated experiment, not a replacement
for the selected COLMAP geometry backend or a change to production defaults.
Artifact root: `/mnt/data/dec5_confidence_depth_prior`.

Two difficult times, `001083` and `001123`, use their original, unrepaired TSDF
meshes. The same 16 physical train cameras supply cached DA3 Large 1.1 at process
resolution 1008. Camera poses, intrinsics, exposure and camera color profiles stay
fixed. All three held-out cameras, including `F004_B005_1210O9`, are excluded from
inference, confidence, region construction and candidate selection.

Matched controls test a local plane, a locally aligned DA3 residual, and
that residual with additional cross-view learned-depth agreement in both input
orientations. Only missing
mesh rays in manually traced train-image skin/hair regions can propose additions;
all original vertices and triangles are retained exactly. Boundaries must agree
with measured PatchMatch depths in at least three other train cameras. A depth
vote requires valid observed depth, depth agreement, return reprojection and
nontrivial parallax; camera count, frustum overlap and silhouette votes alone do
not qualify. Learned additions remain inferred geometry, not new observations.

The original campaign cleaned its dense depth rasters. The parallel source-count
study is reproducing the original 12-source-per-reference recipe on the pinned
COLMAP build and retaining the new rasters for this independent confidence gate.
Both 62-map sets passed shape, uniqueness and local/remote checksum checks.
The maps are independent of DA3 but share calibration and PatchMatch evidence;
they are not independently surveyed ground truth.

An additional preprocessing control rotates the sideways sensor images upright
before DA3, with matching rotations of calibration, then rotates predicted depths
back. It uses the same model and camera membership, without downloading or training
a specialized face/body model.

## Results

### Initial visual and geometry checks

The real train images show a true cast shadow under the jaw and a genuine
chin/shoulder background gap in some views. A missing ray does not establish
missing anatomy. The original TSDF already has crown/side-hair notches.

| Time / train reference | Face-skin pixels / mesh misses | Hair pixels / mesh misses |
|---|---:|---:|
| 001083 / E004_B005_1210I7 | 35,864 / 0 | 198,261 / 4,298 (2.17%) |
| 001123 / G004_A005_121071 | 24,829 / 0 | 147,111 / 2,279 (1.55%) |

These are camera-conditional, manually bounded diagnostic regions, not complete
actor anatomy or ground-truth geometry. Zero train-view skin misses does not
rule out a novel-view hole.

[001083 native-source miss overlay](/mnt/data/dec5_confidence_depth_prior/001083/original_region_misses.png),
[001123 overlay](/mnt/data/dec5_confidence_depth_prior/001123/original_region_misses.png).

DA3 inference completed for both times and both orientations. The initial
landscape passes took 2.16 / 1.47 seconds including preprocessing/API conversion;
peak allocated GPU memory was 12.80 / 12.87 GiB. Loading, RGB staging and compressed
artifact writes are excluded. The uncorrected learned depth was biased by roughly
0.012 normalized units across original mesh hits, so raw depth mixing is unsuitable.

[001083 orientation/shape diagnostic](/mnt/data/dec5_confidence_depth_prior/001083/orientation_native.png),
[001123 orientation/shape diagnostic](/mnt/data/dec5_confidence_depth_prior/001123/orientation_native.png).
These preliminary images align to the original mesh only and are explicitly not
independent depth validation. An affine fit restricted to the narrow head range
can collapse its scale in 001083, another reason to require observed-depth anchors.

### First completed time: 001083

All 62 reproduced geometric maps passed remote and local hash checks. Camera
normalization matches the inference cameras exactly. The independent confidence
analysis took 20.18 seconds on CPU. The skin region has 35,857 trusted pixels and
the hair region 139,041; median supporting other observed maps are 46 and 24,
respectively. These are depth/reprojection votes, not frustum counts.

| Variant | Added hair pixels / triangles | Moving-view newly visible pixels | Held-out hair PSNR / SSIM / LPIPS |
|---|---:|---:|---|
| Original TSDF | 0 / 0 | 0 | 20.534222 / .648494 / .222177 |
| Observed-boundary plane | 1,534 / 2,967 | 928 | 20.557842 / .648827 / .221479 |
| Observed-boundary DA3 | 4 / 6 | 0 | 20.534222 / .648494 / .222177 |
| DA3 plus multiview gate | 0 / 0 | 0 | 20.534222 / .648494 / .222177 |
| Upright DA3 plus multiview gate | 0 / 0 | 0 | 20.534222 / .648494 / .222177 |

Every arm preserves the original vertices and triangles exactly and adds zero
skin pixels. Held-out skin scores are identical: **28.390865 / .956807 / .023525**.
The two stricter learned meshes are byte-identical to the original mesh, so their
identical-camera RGB is reused with explicit ancestry receipts rather than claimed
as an additional render.

The learned boundary-fit median residual is **.011442**, versus **.000529** for
the plane; the fixed gate is .001 normalized units. Upright input improves the
learned depth's MAE on trusted hair anchors from .006210 to .002741, but still
does not yield a passing multiview addition. Local casts across the crown/side
boundary show learned depth transitioning to the background at different places
than observed PatchMatch geometry.

Eight deterministic 12x12 pseudo-holes per region test local depth shape on
strongly supported interior pixels. Median per-patch MAE in normalized units:

| 001083 region | Plane | DA3 | Upright DA3 |
|---|---:|---:|---:|
| Skin | .00002338 | .00005080 | .00004564 |
| Hair | .00018259 | .00015953 | .00024314 |

DA3 is slightly better than the plane in this small hair-interior control, but
does not transfer that advantage to the actual silhouette holes. These withheld
PatchMatch depths are consistency targets, not independent geometric ground truth.

[Native crown failure patch](/mnt/data/dec5_confidence_depth_prior/001083/failure_patches/component_31_native.png),
[side-hair failure patch](/mnt/data/dec5_confidence_depth_prior/001083/failure_patches/component_17_native.png),
[moving-camera comparison](/mnt/data/dec5_confidence_depth_prior/001083/moving_view/clay_comparison_native.png),
[held-out hair comparison](/mnt/data/dec5_confidence_depth_prior/001083/evaluation/hair_comparison_native.png),
[skin comparison](/mnt/data/dec5_confidence_depth_prior/001083/evaluation/face_skin_comparison_native.png).

Native review: the principal crown notch and brown hair fringe remain. The plane
provides a small coverage improvement, not a complete repair. It newly occludes
one old-surface pixel by more than .001 normalized units in the moving view; no
chin-to-shoulder bridge appears. No learned candidate is promoted.

### Transfer time: 001123

The same frozen thresholds produced the following result. Independent confidence
analysis took 19.59 CPU seconds. There are 24,390 trusted skin pixels and 107,703
trusted hair pixels; median supporting other observed maps are 51 and 20.

| Variant | Added hair pixels / triangles | Moving-view newly visible pixels | Held-out hair PSNR / SSIM / LPIPS |
|---|---:|---:|---|
| Original TSDF | 0 / 0 | 0 | 19.262018 / .611655 / .272979 |
| Observed-boundary plane | 130 / 224 | 50 | 19.262241 / .611661 / .273058 |
| Observed-boundary DA3 | 145 / 305 | 110 | 19.262018 / .611655 / .272979 |
| DA3 plus multiview gate | 4 / 14 | 1 | 19.262018 / .611655 / .272979 |
| Upright DA3 plus multiview gate | 23 / 53 | 14 | 19.262018 / .611655 / .272979 |

All skin scores are identical: **28.525261 / .971832 / .011777**. Every original
vertex/triangle and every evaluated skin RGB pixel remain exactly unchanged.
Median boundary-fit residuals are .000529 for the plane and .001151 for DA3.
Upright input is not consistently better: after observed-depth alignment its
trusted hair MAE is .003648 versus .003192 for landscape input.

| 001123 region, eight 12x12 pseudo-holes | Plane median MAE | DA3 | Upright DA3 |
|---|---:|---:|---:|
| Skin | .00003842 | .00007905 | .00006150 |
| Hair | .00020212 | .00020884 | .00021221 |

[Native held-out hair comparison](/mnt/data/dec5_confidence_depth_prior/001123/evaluation/hair_comparison_native.png),
[skin comparison](/mnt/data/dec5_confidence_depth_prior/001123/evaluation/face_skin_comparison_native.png),
[moving-camera clay comparison](/mnt/data/dec5_confidence_depth_prior/001123/moving_view/clay_comparison_native.png).

Whole-head and moving-view review still shows the large crown notch and brown
fringe. The plane / local DA3 add 50 / 110 visible moving-view pixels, but neither
repairs the principal defect. They occlude 11 / 1 old-surface pixels by more than
.001 normalized units; the stricter variants occlude zero. No catastrophic
anatomical bridge appears in the two checked moving views.

The unchanged learned-arm ROI scores have a limitation: their small changes are
outside the fixed hair polygon. At 001123 the local / strict / upright DA3 arms
change 89 / 0 / 12 pixels in the full held-out image, and zero inside the hair or
skin masks. The report therefore does not treat unchanged ROI scores as proof
of unchanged geometry or use them alone to reject/accept a candidate.

Scores use
exact masked RGB PSNR; SSIM and AlexNet LPIPS use the tight region bounding box,
with both images zero outside the fixed manual GT-only mask. These are the
isolated comparison's metrics, not scores for the separately delivered video.

## Insights

The hair crown is the clear completion target in the inspected train views;
cheek/forehead skin is a preservation control. Cast shadows and open silhouettes
must survive. Independent observed-depth gating correctly prevents most of the
learned hair/background boundary extrapolation. The surviving additions are too
small to resolve the visible problem. A local plane is slightly useful at one
time, but its effect does not transfer strongly to the other time.

Keep trusted COLMAP geometry. These results do not justify enabling this DA3
completion or replacing COLMAP with a human/face prior. They also do not prove
that every learned prior will fail: only one cached model, two orientations,
two times and this conservative boundary/multiview protocol were tested. A
specialized face/body model, replacement of weak existing surfaces, local
alignment in every supporting view, watertight stitching and a full temporal
sequence were not tested. The appended patches are inferred surfaces, not new
measurements; no production geometry or model default changed.

Reproduction entry point: `scripts/study_confidence_depth_prior.py`. Commands
`stage`, `prepare_geometry`, `infer`, `infer_portrait`, `preview`, and
`inspect_orientation` completed, followed by `analyze`, `render`, `moving_view`,
`failure_patches`, `score` and `audit`. `analyze` requires each time's
`real_depth_input.json`; `render` uses the unchanged hard-source renderer under
separate output roots. Inference uses the local `.venv` Python with
`PYTHONPATH=/home/brans/deps/Depth-Anything-3/src` and `HF_HUB_OFFLINE=1`.
Two layout/orientation coordinate checks and saved-PLY prefix checks passed.
The first loader attempt rejected a singleton channel before creating geometry;
its failed log is retained. All rejected candidate workspaces are preserved.

[Final 20 region scores](/mnt/data/dec5_confidence_depth_prior/metrics.json),
[artifact audit](/mnt/data/dec5_confidence_depth_prior/audit.json),
[frozen experiment protocol](/mnt/data/dec5_confidence_depth_prior/experiment_request.json),
[001083 evidence](/mnt/data/dec5_confidence_depth_prior/001083/analysis.json),
[001123 evidence](/mnt/data/dec5_confidence_depth_prior/001123/analysis.json).
