# Canonical physical HDR: BODY preservation and DC brown veil

What was tested: read-only frozen canonical AP1-linear LookCloser checkpoints2k and4k, with the exact original BODY128 selection and three predeclared DC pixels `(1600,250)`, `(1700,450)`, `(1400,300)`. The typed adapter loads the positive HDR field, verifies actual exported native mask/EMA/bounds bitwise, and uses no objectives, callbacks, optimizer or near-plane changes.

Results:

| Checkpoint | Known native misses /9022 | BODY misses /495 | BODY prefix mass >10% /128 | BODY band mass ≥50% /128 |
|---|---:|---:|---:|---:|
|2k `1fdb9865…5381` |19|6|0|126|
|4k `b1bcb58e…70d9` |41|5|0|126|

These counts agree across actual native sampling, temporary all-cells sampling and8192/16384 uniform profiles. Train BODY geometry remains strongly anchored. All12 GT/HDRfield crop pairs per checkpoint were actually viewed: coherent actors with blur/texture artifacts, more obvious2k colour banding. The BODY probe samples fitted train witnesses; it does not cover held-out DC's near-camera volume.

The DC veil has a strong near-camera contribution:

| DC pixel |2k mass before .02 |4k mass before .02 |4k dominant-weight depth |
|---|---:|---:|---:|
|1600,250|.57575|.98609|.0115|
|1700,450|.36673|.92395|.0115|
|1400,300|.40601|.89739|.0115|

Native near is .01 prepared-world units. Actual packed weights agree with captured density integration within4e−8. Prefix premultiplied RGB and opacity are integrated on the **same native samples**, including partial cut cells. Analytic suffix `(C−Cprefix)/Tprefix` agrees with independent recomposition; no step-phase change or gain fitting occurs.

At4k, pixel1700,450 has full AP1 RGB `[.02588,.02116,.01396]` against GT `[.01173,.01085,.00841]`. Removing the prefix through .05 diagnostically yields `[.01157,.01054,.00918]`. Pixel1600,250 has full RGB `[.03038,.02572,.02183]`, while its .05 suffix is `[.00737,.00633,.00483]` against GT `[.00892,.00693,.00668]`. Pixel1400,300 retains colour mismatch after removal. This is substantial causal evidence for near-camera emission/attenuation in the DC veil, not evidence that clipping alone solves the full image.

The effect differs from the earlier KA near-prefix test, where native clipping explained only a small part of the floor deficit and brightened already-bright controls. No global near-plane change follows. Actual source-camera hardware support, visibility and minimum receiver clearance must be inspected before changing any camera-specific integration domain.

Evidence:

- [Matched summary, full hashes and per-ray RGB](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/canonical_BODY_DC_comparison.json), SHA `e52d78baf84d0cea4e68f5f034483fc561dacce429388d37bd67e6ff2f1f63fd`.
- [2k BODY audit](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/geometry2000_BODY_DC/body128/audit.json), [4k BODY audit](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/geometry4000_BODY_DC/body128/audit.json).
- [2k DC prefix/coordinates](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/geometry2000_BODY_DC/DC_three_native_prefix.json), [4k DC prefix/coordinates](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/geometry4000_BODY_DC/DC_three_native_prefix.json).
- [Actual manual BODY review](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/canonical_BODY_DC_manual_review_occupancy.json).
- [Executed reader source](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/canonical_geometry_reader_source.py).

Insight: fitted BODY train-ray preservation and a held-out near-camera veil can coexist. Preserve this distinction when evaluating occupancy and colour calibration. Neither a successful BODY gate nor fewer known voxel misses certifies clean novel-view geometry. Existing checkpoints, input data, native masks and model parameters remain unchanged.


### Near .052 and actual one-halo control

At unchanged canonical4k weights, the explicit protected mask (`actual core UNION9022 observed-point cells`, then ONE halo) eliminates known-point misses but degrades all3 views: signed-PQ PSNR36.10855→33.35653, SSIM.944219→.939781, LPIPS.162612→.179327. In the common SDR display, PSNR27.71348→25.00699. The actual contact was inspected: black traversal tiles diminish while a broad veil appears. This policy is not a visual-quality release. Both masks and NN/EMA were restored bitwise after the control. Evidence: `canonical_linear_HDR_PQMSE_STAGED/gate4000_near0052_guarded_HD/{receipt.json,comparison.json,all3_GT_default_guarded.jpg}`.

The same3 predeclared DC rays were traced with this protected mask at near .052. Through .3world their prefix opacity is at most2.253e−6; dominant depths1.9465/1.6975/2.0335. Near-prefix high-HDR RGB contributes about0.55–1.64% of total red, while opacity is negligible. These3 rays do not support a new near-clip remedy for the broad halo veil. This result does not describe all image pixels. No further near setting was applied. Exact same-sample suffix diagnostics and temporary-mask restoration are bound in `near0052_guarded_DC_prefix/{DC_three_native_prefix.json,summary.json}`.

### Near-resume6k frozen geometry gate

Exact checkpoint `15268fe947b6425951b389e27c02dbba06055a77b1b1b5a14def2d23021082bb` was audited with the SAME128 visible-body rays. Native/allcells/uniform8192/uniform16384 all have0/128 prefix mass>10% and126/128 surface-band mass≥50%. Native grid misses36/9022 known-point cells, including4/495 body cells, versus4k41/5. The12 body GT/pred panels were actually viewed; bodies remain coherent with fine-detail blur. The three DC rays have zero sampled prefix through .1 at near .052; one has opacity.518, so background traversal holes remain a separate issue. All nine previously fixed KA pixels in existing HD buffers have opacity≥.999819; these are renderer outputs, not independent floor-depth truth.

Read-only outputs: `canonical_linear_HDR_NEAR052_RESUME4k_STAGED/geometry6000_BODY_DC/`. Model/grid hashes were unchanged, no optimizer/callback or worker resume occurred. Known-ray support and point coverage do not certify hidden occupied/free geometry or a clean floor.


### Whole-grid6k numerical audit after failed7k forward

The immutable6k snapshot was queried at ALL2,097,152 voxel centers in its actual128³ field AABB, chunks≤65,536. Both eval/no-autocast and training-mode/autocast geometry paths returned finite rawTCNN h0, geometry embedding and FP32 activated density everywhere. Rawh0 spans−241.25..11.7421875; density maximum341870.5625 is finite. No center has h0+1>88. All16,777,216 center/direction combinations (eight normalized cube-corner directions) also returned finite rawcolor and positiveAP1RGB. Rawcolor absolute maximum401.25, eval activated maximum12176.0.

Highest-density cell[70,82,25] world[.341469,.947353,−.090098] belongs to nativecore;σ341870.56. Fivecells exceed65504σ, allnativecore; density is correctly FP32, so this exceeds half precision range without causing forwardInf. Of8197cellsσ>88,8196 are admitted by nativecore and exactly-onehalo, one lies outside that halo. These are numerical/traversal labels, not physical occupied/free truth. HighestRGBcell[19,106,97],world[−2.46224,2.35320,1.55885] has h0−241.25,σ0 and is outsidecore/halo.

The same1024 extremecells (union of highest density-logit and rawcolor samples) were queried in both precision modes and eight directions. TCNN rawgeometry/rawcolor remainFP16 and are identical acrossmodes; density remainsFP32. RGBactivation isFP16 outsideautocast versusFP32 underAMP; highestRGB12176.0 versus12172.486, allfinite. This verifies the precision difference but does not demonstrate it caused the unavailable7k failure.

NN/core/EMA digest was bitwise unchanged, no callbacks or optimizer occurred. Complete boundreport: `canonical_linear_HDR_NEAR052_RESUME4k_STAGED/wholegrid6k_numeric_LOCAL_v3/audit.json`; compact extrema with rawgeometry, rawRGB and activated outputs, cellIDs/worldXYZ and admission labels: `extreme_and_bad_cells.npz`. Executed helper bytes are frozen there. Initialcomplete scan is preserved; a presentation-only v2 negative-stride export failure is separately archived and corrected in v3. Center and eight-direction samples cannot certify every continuous location/direction or reconstruct failed7k weights.
