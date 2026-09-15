# DEC5: cross-frame test of all-radius measured support

## What was tested

Transfer of the [001193 all-radius seed control](dec5_mhr_radius_seed_support.md)
to already independently audited, per-time priors at001195 and001083. The former
is adjacent in time; the latter has a substantially different head pose. No pose,
geometry or measurements are copied between frames. These transfers reuse the
original per-time2px/100-step silhouette priors, **not** the newer001193 dense43
fit: they isolate the seed-neighborhood rule on existing reconstructions.

`transfer_mhr_radius_seed_control.py` validates the original reusable completion
config, input hashes, geometry and admission audit before invoking exact-count
path/config adapters of the frozen all-radius producer and auditor. Its generated
source hashes are retained in each request. All numerical gates remain unchanged:
same actual measured seed pool, radius .003, normal dot .5, minimum8 seeds,
hull containment, fit conditioning, leave-one-out and offset tolerance .0005.
Every semantic candidate vertex is tested; no target crop selects fitting input.

Both original per-frame priors remain frozen. The only changed certificate rule
uses all radius/normal-eligible measurements instead of the nearest24 cap;
old passes can be lost. All62 cameras and both native ray lattices still gate
final added faces. Original mesh vertices and triangles are preserved exactly.
No source data, COLMAP, model defaults or delivered6K video is changed.

Inputs: `/mnt/data/dec5_mhr_reusable_001195` and
`/mnt/data/dec5_mhr_completion_001083`.
Outputs: `/mnt/data/dec5_mhr_radius_transfer_001195` and
`/mnt/data/dec5_mhr_radius_transfer_001083`.

Per frame the matched RGB experiment renders baseline production geometry,
previous nearest24 completion and all-radius completion. All share fixed
exposure, camera profiles and current incidence2/unwarped hard-source rendering.
Views are actual cinematic camera, F/E train camera, and the earlier elevated
moving stress camera. Frames and two view workers run concurrently on CPU;
these1080×1920 diagnostics are not a replacement for the user's6K delivery.

```bash
python scripts/transfer_mhr_radius_seed_control.py produce \
  --source AUDITED_COMPLETION --output NEW_ROOT
python scripts/transfer_mhr_radius_seed_control.py audit --output NEW_ROOT
python scripts/transfer_mhr_radius_seed_control.py render --output NEW_ROOT
python scripts/review_radius_seed_transfer.py --root NEW_ROOT
```

Use the existing repository venv and `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OPENCV_IO_ENABLE_OPENEXR=1`. Existing outputs are not overwritten. No fitting
or prior model downloads occur.

## Results

| Source frame | Queried vertices | Nearest24 → all-radius certified | Newly pass / lose pass | Initial → final all-radius faces | Native rounds | Produce seconds |
|---|---:|---:|---:|---:|---:|---:|
| 001083 | 17,657 | 9,011 → 9,538 | 574 /47 | 21,065 →20,975 | 2 | 26.19 |
| 001195 | 180,069 | 114,238 →122,215 | 8,385 /408 | 229,299 →229,004 | 4 | 75.34 |

Both numerical audits pass, including exhaustive seed-distance selection
(independent of the producer KD-tree), every certificate and124 final native
checks per frame. The existing certificate/veto mathematics are reused, not
claimed as separate derivations. More verified facets do not establish visible
repair; completed RGB comparisons and actual manual review are required below.

Initial implementation checks:15 tests pass across exact source adapters,
reference-hole/regression accounting, the radius certificate and prior reusable
completion/transfer interfaces. These tests do not establish image quality.

### 001083: no visible or numerical RGB benefit

All9 renders complete. The all-radius and nearest24 variants have **bit-identical
RGB and depth** in current moving, F/E and old moving views. Added facets are
therefore not a visible repair in those cameras. This equality is checked on
actual arrays, not inferred from aggregate coverage. The receipts identify
different actual mesh paths/hashes, so this is not a reused-render shortcut.

Main LLM viewed all three native triplets at full saved resolution. No visible
new improvement or regression appears between nearest24 and all-radius. Existing
neck/shoulder fringe, hair outline contamination and the underside protrusion
remain. The prior completion already added two black contour pixels relative to
production in the current moving view; all-radius does not fix them.

Enclosed-miss inventories for nearest24→all-radius are3→3 current moving,
10→10 F/E and5→5 old moving. The larger neutral-projected head/neck crop used
here differs from the earlier face-landmark crop, so F/E's count10 must not be
compared with the old report's count0 as a regression. It is unchanged within
this matched test. These are not confirmed anatomical-hole counts.

### 001195: small side-view gain with a texture regression

All9 renders complete. Compared directly with nearest24:

| View | New geometry / uncolored | Lost geometry | New black RGB | Changed RGB pixels | Common-depth changes >.003 |
|---|---:|---:|---:|---:|---:|
| Current cinematic | 0 /0 | 0 | 0 | 17 | 0 |
| F/E | 8 /0 | 0 | 0 | 35 | 21 nearer |
| Old moving stress | 0 /0 | 0 | 4 | 11 | 0 |

The eight F/E new hits close eight of23 enclosed misses in this diagnostic
crop. Maximum nearer depth changes by .014909 normalized units; all21 affected
pixels lie inside portrait box[708,1179,716,1191] at the neck edge and are fully
included in the inspected native side-effect crops. This is a changed first-hit
surface, not a measured per-vertex displacement or a guaranteed anatomical gain.
Current-moving max common-depth change is .001385; old-moving .000655.

The original73 fixed under-jaw diagnostic rays from the001195 locality evidence
still have73 misses in production, **one miss in nearest24 and one in all-radius**.
Thus this new rule does not improve the previously demonstrated72-pixel repair;
the eight new F/E hits are a different local change. The800 enclosed misses in
the larger old-moving review crop include other components and must not be
misrepresented as the original anatomical puncture.

Main viewed all three native triplets and all four generated side-effect
panels. Small F/E neck-edge punctures diminish, but the ragged edge remains.
The two overlapping old-moving panels include all four new black pixels:
(544,1047),(544,1048),(537,1071),(537,1075). They lie near hair/neck margins,
not an approved missing-background region. Current head/hair silhouette and
skin appearance have no obvious broad improvement. This is **mixed local
transfer, not promoted**. No full temporal validation or artifact-free claim.

All18 matched renders and both audits terminate successfully. The final focused
suite passes65 tests, including the prior geometric guards and new transfer/review
tests. Per-frame visual verdicts and final hash checks are retained next to the
results; numerical audit approval does not override their non-promotion verdicts.
Final verification rehashes1,135 bound files, confirms all18 render receipts use
the intended frame/mesh/recipe, verifies001083 exact RGB/depth equality and the
001195 fixed73-ray counts, and checks that the delivered6K SHA256 is unchanged.

## Insights

This is a cross-frame test of evidence selection, not a new silhouette fit or
an automatic batch repair. Additional measurements both grant and revoke
certificates. Compare all-radius against the matched nearest24 result, not only
against the original holed surface, to avoid attributing previous improvements
to this new rule. Enclosed ray misses are diagnostic components, not ground-truth
anatomical labels; no PSNR/SSIM/LPIPS or full-frame quality metric is claimed.

The rule improves local support selection at001193 but is not a general cure:
001083 renders are unchanged, and001195 exchanges a few closed side-view pixels
for a new texture regression in another camera. Do not launch a150-frame rerender
or claim that increased certified-face counts remove the video's artifacts.
Further mesh work must address incorrect/missing candidate shape and object-layer
ownership (especially the lipstick fin), not keep enlarging the support set.

One separate, still untested direction is to distinguish support for a queried
surface from corroboration of a nearby observed point. The
[lipstick near-protection study](dec5_lipstick_near_protection.md) established
that two near observations have28 corroborating views each; that does **not**
make the fin query itself supported by28 views. A future free-space conflict
test must make that distinction explicit and preserve genuine hand/tube surfaces;
this transfer experiment does not justify relaxing their deletion guards.
