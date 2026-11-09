# Silver-sable LookCloser hair campaign

## What was tested

User-authorized budget: 16 wall hours on eight GPUs (at most 128 aggregate
GPU-device hours), separate from the prior 16-GPU-hour campaign. Host verified:
eight RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each; persistent boot
4.8 TB and volatile `/cache` 12 TB. No unrelated processes are stopped.

The first gate compares frozen HD and foreground/background LookCloser on the
same 16-pose path, same native train RGB, source order and background fallback.
Next, same-hardware fresh controls vary subject quotas and point budgets, plus
foreground/background empty-space strength. Only material measured gains justify
extensions. Dataset 001055 has 59 train and three excluded validation cameras;
reserved test RGB is absent from the transfer bundle.

## Results

Bootstrap and data migration completed; all 615 data/model payloads verified. The proven gs2 environment, private
nerfstudio/dependencies and CUDA 12.6 are transferred with SHA256 verification;
working source environment is not modified. Conda packaging ignores stale NumPy
package metadata overwritten by pip; destination imports/real-field execution
must pass before training. Source and destination write permissions differ:
source clever-shadow can write the object prefix; silver/dev3 cannot. No new
bucket/account configuration is required from the user.

Single-command fleet status:

```bash
ssh ubuntu@silver-sable 'python3 /home/ubuntu/lookcloser/code/LookCloser/scripts/gpu_experiment_fleet.py status --root /home/ubuntu/lookcloser/campaigns/hair_20261028'
```

Add `--json` for a machine-readable snapshot. It includes every registered job,
physical GPU memory/utilization, progress, PSNR/SSIM/LPIPS when available, elapsed
GPU-device hours and OOM evidence. Five controller tests pass: reject busy GPU, reject over-budget plans before
launch, distinguish complete/setup-failed, reserve a GPU while its controller is
alive before allocation, and enforce the shared wall deadline.

## Insights

The preceding foreground/background screen improves hair LPIPS 5.2% and holes
67.7%, but increases halo 56%. It needs path and visual validation. The additional
known-background continuation does not improve separation and is closed. Native
input resolution alone had no material gain; new trials must improve geometry
rather than repeat input-resolution changes in isolation.

Bootstrap gate passed on the destination: torch 2.9.1+cu128, NumPy 1.26.4,
OpenCV 4.11.0, eight CUDA devices, GPU tensor computation and the NumPy bridge.
The packed NumPy files initially failed import; reinstalling the **source's exact
1.26.4 version** in the destination environment fixed this. LookCloser imports
pass. TCNN uses the proven SM90+PTX binary on SM120, with a performance warning;
it is not presented as a native Blackwell-optimized build.

At **2026-10-28 14:38:18 UTC**, eight independent frozen-path workers launched.
Shared wall deadline: **2026-10-29 06:38:18 UTC**. Initial device allocation and
controller/process snapshots are retained in `supervision_checks.jsonl`; the
first CUDA extension build is serialized by the normal PyTorch cache lock.


The first eight workers completed (four HD, four FG/BG path chunks), without
OOM, using **0.679 conservative GPU-device hours** including initial compilation.
Sixteen poses per arm were merged after checking controller success, payload
SHA256, frozen checkpoint identity and identical source inventory.

| 16-pose proxy | HD baseline | FG/BG 5k | Change |
| --- | ---: | ---: | ---: |
| Visibility toggles / baseline anchors | .002724 | .001519 | -44.2% |
| Depth exposure dropouts / baseline anchors | .016986 | .013441 | -20.9% |
| Source switches / baseline anchors | .024718 | .024655 | -0.3% |
| Present semantic hair / baseline anchors | .205235 | .186456 | -9.2% |

[Path metrics](assets/dec5_silver_sable/fg_path_metrics.json).
The last row is **not GT recall**: source-projected semantics can also label
spurious backdrop geometry. Still, reduced presence makes a stability-only
claim insufficient. All 16 decoded frames in both arms were visually inspected:
FG/BG reduces protruding crown fragments, but residual contour/background chips
and face visibility holes remain. This is a head-ROI diagnostic, not a finished
whole-frame render. Physical-time stability is not established.

At **14:44:24 UTC**, eight trainings started under the same controller:

| GPU | Case | Hair / other person | Point budget | Design |
| --- | --- | ---: | ---: | --- |
| 0 | HD FAS control | 0 / 0 | 2^18 | Fresh full LookCloser |
| 1 | HD subject90 | .60 / .30 | 2^18 | Same-hardware control |
| 2 | HD hair80 | .80 / .15 | 2^18 | Higher hair quota |
| 3 | HD person95 | .45 / .50 | 2^18 | Higher nonhair subject quota |
| 4 | HD subject90 points2x | .60 / .30 | 2^19 | More field queries |
| 5 | Native subject90 points2x | .60 / .30 | 2^19 | Native RGB plus more queries |
| 6 | FG empty001 | .60 / .30 | 2^18 | Warm 20k native; empty-space weight .01 |
| 7 | FG empty020 | .60 / .30 | 2^18 | Warm 20k native; empty-space weight .20 |

Fresh runs allow 60k steps with validation progress stopping; warm screens allow
10k. Each has a two-hour controller cap, subordinate to the shared wall deadline
and aggregate budget. Native RGB checkpoint selection remains PSNR with the
.07-dB LPIPS tie-break; geometry candidates are labeled separately. Training
supervision continues, with full controller/worker/GPU snapshots in the campaign.


### Domain bug found before new geometry selection

Visual inspection traced a large gray face cutout to the **old receiver ROI**,
not necessarily missing field density. The inherited 001083 fragmented domain
excluded 28.7%, 40.9% and 35.0% of the 001055 reference face/neck pixels, while
covering all reference hair pixels. Old face-regression numbers therefore do
not establish quality across the whole face.

`build_dec5_convex_roi.py` creates a **new versioned domain**: the filled convex
hull of all original train-projected ROI pixels, with an exact superset check.
No target image or semantic reference constructs this mask. A subsequent
reference-only coverage audit finds **zero excluded face/neck or hair pixels**
on all three validation cameras. Original domains/metrics remain unchanged.
[Audit](assets/dec5_silver_sable/roi_domain_audit.json),
[red overlay: subject pixels excluded by the old domain](assets/dec5_silver_sable/old_roi_face_exclusion.jpg).
Training inputs/objectives are unaffected. All new geometry comparisons use
`roi_4k_hull_v2` and the corresponding common background fallback; old and new
face metrics must not be compared as if the rendering protocol were identical.

Frozen geometry controls also compare expected depth and conditional 10%-mass
front depth with the median, using **the same statistic in source visibility**.
The extra conditional depth comes from existing native packed adaptive weights;
it adds no field queries or training. Five analytic packed-depth tests pass,
including nonuniform subdivision, layer attenuation and front quantiles. No
extra density peak is automatically certified as a real strand.

The background wrapper had a bookkeeping inconsistency: final `best.pt` and
progress included the initial attached RGB function, but `selection.json` still
contained the trainer's pre-rebase selection. The wrapper now writes all three
consistently. The two completed silver warm screens were reconciled from their
complete history after checking the best checkpoint step; weights are unchanged
and previous metadata is retained in reconciliation receipts.


Native RGB + doubled point budget stopped at 20k without progress: PSNR
16.203 / 15.991 / 15.971 at 4k / 10k / 20k, with deteriorating SSIM/LPIPS.
Rendered 4k and 20k outputs were inspected: severe grayscale volumetric haze.
This is a negative result, not a candidate to extend. An exact native 1x control
was added on the same silver GPU to distinguish budget sensitivity from a
migration/hardware issue before interpreting the effect. Core field/model/grid/
RNG/trainer helper sources match the proven source snapshot; pipeline differences
are equivalent prefix handling and removal of a duplicate no-grad decorator.

The background training-mask audit also found a physical-scale inconsistency:
older native warm trials eroded **16 native semantic pixels**, while the fitted
background component used **16 HD semantic pixels**. Their lambda comparisons
remain matched to each other, but are not a strict same-mask comparison with the
background fit. Subsequent warm trials must use `semantics_hd` for background
supervision, with calibrated sampled-ray coordinates mapped to that raster.
Subject quota semantics remain native when training native RGB. The audit now
records semantic raster and erosion radius explicitly. The background provider
also accepts its exact fitting dataset hash (HD), as well as its preauthorized
native serving dataset; no unverified camera system is admitted.


The same-hardware native 1x reproduction reaches **20.608 dB** at 10k versus
20.453 dB for the earlier clever native control. This rules against a general
catastrophic migration failure as the explanation for the 2x gray-haze result;
it does not prove deterministic equivalence across hosts.

The first-path evidence archive has been copied directly from silver to FSx;
source and destination SHA256 both match
`762f70a2c9b78de6fdfb69715b1171dc0f8056b6e8309a5715432e86ce63543a`.
No bucket configuration is required to launch, monitor or preserve the campaign.


### First repaired-domain FG result and a geometric constraint track

FG/BG 5k in the repaired domain gives hair LPIPS .188506, face/neck LPIPS
.111574 (the old domain yielded .280881). Hair metrics remain almost unchanged,
consistent with the audit that the old mask already covered all hair pixels.
At matched source/target depth statistic:

| FG/BG statistic | Hair PSNR | Hair SSIM | Hair LPIPS | Face LPIPS | Halo | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median | 22.72403 | .39584 | .188506 | .111574 | .002405 | .983428 |
| Expected | 22.55772 | .38954 | .185141 | .115391 | .002740 | .984020 |
| Conditional 10% front | 22.14331 | .40064 | .176723 | .112937 | .001762 | .984117 |

Front depth improves hair LPIPS **6.25%** and halo **26.7%** versus the FG median,
but lowers PSNR and slightly worsens face versus that parent. Acceptance must
use the common repaired HD baseline and a path gate; no promotion yet.

`build_dec5_empty_space_grid.py` computes 59-view confident-background votes on
a 256^3 normalized voxel grid, with the exact HD eroded masks and calibration.
A three-view rule marks 79.56% of the bounding volume empty; an eight-view rule
marks 74.36%. **Zero of 627** high-confidence train-SfM hair witnesses are rejected
by either raw rule. These are fitting witnesses, not independent 3D GT.
`dec5_empty_space_gate.py` further erodes known-empty voxels in 3D, then attenuates
frozen field density with the trilinear mask. It preserves parameters/occupancy,
requires matched frame/calibration, and rejects a grid if >0.5% of the reliable
hair witnesses would be carved. Source and target density gates must match.
Two CPU-tensor tests verify XYZ convention, interpolation, unknown exterior and
nonmutating density/RGB outputs. A frozen three-camera experiment is running;
no model is trained from this constraint before a measured RGB/geometry gain.

The HD foreground/background follow-up uses the **exact HD fit semantics** for
background empty-space supervision and warm-starts HD geometry, while retaining
the same frozen background weights and 60/30 subject quota. It has completed;
its auxiliary final geometry is being evaluated alongside its native RGB best.

The repaired HD baseline is now available. Its median gives hair LPIPS .198838,
face .113712, holes .010972, halo .001541, recall .920119. Its front-10% depth
improves hair LPIPS to .188130, but holes rise to .012471 and recall is .944974.
Against this **front-depth HD baseline**, FG/BG front depth still improves hair
LPIPS **6.06%**, holes **71.9%**, and recall **3.91 pp**, with essentially identical
face LPIPS (.112937 versus .112941). Halo differs by only about 1.1% (.001762
versus .001743). Thus the combined gain is not explained solely by depth-statistic
selection. Matched 16-pose median/front exports for both fields are running in
the repaired domain, with source geometry reused from verified frozen exports.

The finished quota/native controls do not beat the original 20k HD geometry:

| Native-RGB-selected field, median | Hair LPIPS | Face LPIPS | Holes | Halo | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original HD 20k | .198838 | .113712 | .010972 | .001541 | .920119 |
| Longer HD 60/30 subject | .205274 | .112848 | .011495 | .001533 | .914662 |
| HD 80/15 hair-heavy | .209006 | .114412 | .011333 | .001467 | .921416 |
| HD 45/50 person-heavy | .201542 | .112316 | .012457 | .001511 | .917993 |
| Same-host native 60/30 control | .198313 | .114918 | .012619 | .002081 | .919815 |

Increasing the hair quota or extending training does not establish better depth;
the high-res reproduction also fails the material geometry gate. Close these
branches rather than refining adjacent quota values. All rows use the same UHD
GT transfer, repaired ROI, source RGB and background fallback. RGB checkpoint
selection was performed before this geometry comparison by the required rule.
