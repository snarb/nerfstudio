# Finite floor-band scene prior

What was tested: a positive opacity likelihood within a finite train-reviewed
floor region, using a plane fitted to 16 existing sparse observations. This is
a soft scene assumption beyond the measured points, not floor depth ground
truth. Samples cover continuous triangle area and a ±0.02 world-unit normal
band; there are no negative visibility constraints or occupancy edits.

Results: preparation and zero-update checks passed for weight0.0001 on the
immutable random-background10k parent. Complete model, Adam, scheduler,
scaler65536 and RNG state restored exactly. Both the initial64-ray batch and
the actual14,966-ray adaptive batch produced finite combined AMP gradients;
no optimizer updates occurred. The initial0.01 scan overflowed and remains
archived. This is not a render-quality result or training authorization.

Artifacts: [/fsx stage](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/preparation_receipt.json),
[actual proof](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/actual_zeroopt_floor_probe_LOCAL_smallweights/floor_prior_probe.json),
[approved domain](/fsx/tmp/marshawn_001660_hd/floor_band_prior_PROPOSAL/root_geometry_review.json),
[launch plan](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/launch_plan.json).

Insights: weight0.0001 supplies a finite localized density gradient without
dominating the existing RGB objective. It does not establish correct floor
depth. Any trial must inspect the prior boundary, the outside control ray,
all three held-out views and existing body geometry at the required12k gate.
The full-state continuation has a conservative total attempt cap21,999;
production required the separate root review recorded below.

The root approved one continuation after reviewing the complete source, data
and proof. It paused before12001 after2000 attempts,1999 Adam updates and one
native AMP skip. The saved12k model is finite; the total charged family budget
is13,000 attempts at this gate. After the root reviewed these images and ray
audits, it resumed the same worker with the existing default stop policy.

| Actual all-three-view evaluation | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Random-background10k parent |29.14361|0.810825|0.275139|
| Finite floor prior12k |29.2514|0.812390|0.272212|
| Selected prior14k, fresh final evaluation |29.23558|0.813125|0.269608|

[Matched whole views](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12k_all3_GT_parent_new_safe.jpg)
and [wide KA boundary crop](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12k_KA_prior_boundary_margin.png)
use GT, parent10k, prior12k and safe13k columns. Manual review found no obvious
new sharp boundary seam, but broad floor mottling remains. The critical black
ROI improves LPIPS0.47210→0.43655 while PSNR falls26.668→26.074; the wider KA
margin is almost unchanged in LPIPS. These are fixed native PNG crop metrics,
not checkpoint-selection scores. Actual ray tracing confirms that several
formerly deep layers move into the assumed band; this is compliance with the
scene prior and does not independently establish physical depth.

The phase finished normally at the14k plateau. The PSNR difference from12k
is within0.07dB, so improved LPIPS selects14k. Both worker and controller
exited; the last logged14160 plus a conservative10-attempt allowance gives
a total family bound15,170. There is no20999 checkpoint or hard-cap terminal
evaluation to perform.

[Selected14k whole views](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate14k_all3_GT_parent_new_safe.jpg)
and [black-floor ROI](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate14k_KA_blackROI.png)
retain visible mottling. The black ROI LPIPS improves0.47210→0.43475 versus
the10k parent, while PSNR falls26.668→26.263. The wider prior-margin crop
is nearly unchanged in LPIPS. This is useful geometric improvement, not
acceptance of clean rendering or proof of physical hidden geometry.
The [terminal receipt](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/finished_receipt.json)
binds the exact selected checkpoint, fresh metrics and attempt budget.
