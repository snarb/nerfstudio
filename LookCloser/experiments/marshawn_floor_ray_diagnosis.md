# Marshawn floor patches: learned field versus traversal

## What was tested

Nine fixed rays on the accepted PNG R0 step8000 model with its exact witness-union halo1 candidate, and on the direct-PQ step6000 control. Five eval pixels cover visible floor patches/adjacent regions; four train pixels are local controls. Floor classification uses actual AA/GA/OA train-source crops. No eval RGB was used to fit the reference or select physical floor labels.

The reference is the earlier manually vetted 15-SfM-point plane: normal `(0.0297973,-0.1303632,0.9910184)`, offset `0.1682054`. It describes approximately planar **thin floor covering**, with genuine folds and view-dependent reflections. It is not structural-floor depth truth or the raised crashmat. The 15 samples include near-duplicates. All five eval targets fall outside their XY convex hull; this extrapolation is explicitly recorded. The normal-distance band is ±0.03 prepared-world units, converted to ray-distance width using `abs(normal·direction)`.

Source inspection rejected a pole-base query, a query close to an actor, and a panel-transition query. Actor-occluded AA projections for GA controls and the GA projection for the OA control are explicitly excluded from source evidence. Parent independently inspected all three source pages and confirmed these decisions.

## Results

Every actual packed ray reproduces its saved HD pixel: maximum error `3.87e-7` for PNG and `5.96e-8` for PQ. Each run restores the original grid and verifies the full model-state hash. The counterfactual opens all cells temporarily. Uniform 8192/16384 samples bypass the grid and show convergence; these sample counts are not training iterations.

| Test | PNG | PQ |
|---|---:|---:|
| Max native/all-cells RGB difference | 0.0000068 | 0.005571 code |
| Max uniform N/2N termination-band mass difference | 0.000246 | 0.000190 |
| Max uniform N/2N RGB difference | 0.0000617 | 0.0000308 code |

The RGB scales differ between models and must not be compared directly.

| Query | Hull | PNG native termination mass after reference band | PQ all-cells termination mass after reference band |
|---|---|---:|---:|
| HC1050,800 (patch) | outside | 0.293 | 0.616 |
| HC1150,820 (adjacent crease) | outside | 0.223 | 0.470 |
| HC1250,800 (cleaner adjacent) | outside | 0.442 | 0.536 |
| DC1150,820 | outside | 0.368 | 0.633 |
| KA1250,850 | outside | 0.744 | 0.750 |
| AA550,650 control | inside | 0.602 | 0.765 |
| GA1350,500 control | inside | 0.625 | 0.647 |
| GA1450,550 control | inside | 0.406 | 0.488 |
| OA1450,750 control | inside | 0.526 | 0.608 |

The PNG OA control also has 0.251 termination mass before the reference band (uniform: 0.242). Actual profiles show broad or multiple termination components, including controls inside the support hull. They are not a single opaque surface confined to the approximate band.

Evidence: [selection and caveats](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/selection.json), [source provenance](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/source_provenance.json), [PNG actual traces](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/PNG/audit.json), [PQ actual traces](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/PQ/audit.json), [convergence comparison](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/comparison.json). Six profile pages and raw per-sample RGB, density, interval and weight arrays are retained in the same directory. All three source pages and all six final profile pages were manually inspected. Density plots leave skipped intervals blank, and cumulative plots preserve their constant-transmittance gaps.

## Insights

Opening the grid does not materially change the accepted PNG pixels at these queries. Another occupancy halo cannot remove these particular floor patches. The learned field contains broad termination components relative to the approximate reference; the PQ control also has some traversal differences, but opening its grid leaves the broad tails.

This is not proof that thickness alone causes the visible patches. The cleaner adjacent HC query has a larger below-reference tail than the patch query. Genuine reflections, folds and reference uncertainty remain confounds. A matched paper-distortion experiment is a justified concentration test; its success must be judged by the actual floor crops and profiles. The evidence does not justify a hard floor carve, a global empty-space label, or fitting a new positive floor-depth prior from this sparse plane. No model training, density edit or persistent grid change was performed.

## Matched linear-distortion step8000 follow-up

The separately trained PNG control finished under its default plateau rule. The same sealed nine rays were then traced against its actual core0 grid, all cells, and uniform8192/16384 integration. No additional model training was performed by this audit. Actual saved HD crop RGB is reproduced within `5.96e-8`; the model and original step configuration are restored exactly.

The uniform16384 comparison isolates field changes from traversal: the old rendered candidate uses a guard, whereas the new default is core0. Median width ratio for the **central80%** termination interval (q10→q90) is `0.06274`, a 93.7% reduction. This width ratio is independent of the plane offset. New widths are approximately 0.0064–0.0168 prepared-world normal-distance units. Eight queries place nearly all termination in the approximate ±0.03 band. AA550,650 instead concentrates below that reference: uniform mean height −0.067 with 0.987 mass after the band. Concentration can choose a different depth; sparse-plane uncertainty remains.

The actual default grid misses 47/9022 known point cells, including 4/495 subject points (IDs3731,4020,7875,8170). A separately derived inference candidate uses **actual exported GPU core UNION9022 observed point voxels, then exactly ONE halo**. It preserves core, EMA, NN and checkpoint. It admits 893294 cells and covers all9022/495 points; this is conservative traversal admission, not a physical occupancy certificate. Ordinary halo alone still misses IDs511 and2813. [Candidate receipt](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/witness_union_halo1_candidate/receipt.json).

On the nine rays at step0.003, native/all-cells RGB differs by at most0.004830; candidate/default differs by at most0.003823. Against converged uniform16384 integration, maximum RGB differences are 0.007574 default, 0.005418 all-cells, and 0.006852 candidate. New uniform8192/16384 convergence is much tighter: RGB0.000130, band mass0.0000567.

### Read-only integration-step test

| Step | Candidate mean RGB error vs uniform16384 | All-cells max RGB error vs uniform16384 |
|---|---:|---:|
| 0.003 | 0.002697 | 0.005418 |
| 0.0015 | 0.001529 | 0.002895 |
| 0.00075 | 0.000785 | 0.000677 |
| 0.000375 | 0.000632 | 0.0000880 |

Eight candidate rays converge similarly; AA550,650 retains an approximately0.00536 difference because the candidate excludes some field admitted by the all-cells reference. That residual is not integration-step error. The all-cells/uniform field itself is not geometry truth.

Actual native HC/DC/KA crops were rendered with the same candidate at steps0.003 and0.000375. [Crops and raw buffers](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/quadrature_crops/receipt.json): all three were manually inspected. Finer integration changes edges and small speckles, while broad residual pattern and softness remain. Mean RGB changes are0.00201/0.00305/0.00327; edge maxima reach0.0948/0.1223/0.1249, so nine isolated pixels understate edge errors. Average error to GT is essentially flat for HC/DC and slightly lower for KA. This supports using a finer inference step for numerical accuracy, but does not establish that quadrature or another guard removes the remaining floor mottling.

Evidence: [field-width comparison](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/comparison_old8k.json), [actual candidate rays](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/candidate9/audit.json), [step convergence and per-sample traces](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/quadrature/audit.json). All three before/after field-profile pages were manually inspected. Concentrated termination does not imply zero density behind the visible surface, nor a physically perfect occupancy grid.

### Provisional full-HD guarded download

The protected step8000 field was rendered at full1920×1012 for all three eval views at inference steps0.00075 and0.003. Both use the same witness-union, one-halo mask. Native metrics use unchanged fixed-scalar SDR source PNGs; output PNGs contain predictions only.

| Inference | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original default core0, step0.003 | 28.709667 | 0.796946 | 0.317703 |
| Protected guard, step0.003 | 28.648159 | 0.796585 | 0.318721 |
| Protected guard, step0.00075 | 28.668568 | 0.799247 | 0.321135 |

The finer step improves PSNR/SSIM over the matched guarded coarse render but worsens LPIPS; per the requested fallback rule, the top-level provisional download retains guarded step0.003. The step0.00075 variant is preserved separately. The guard costs0.0615dB relative to default core0 while restoring coverage of all known9022/495 point cells; neither fact certifies the physical volume.

[Provisional PNGs, raw buffers and manifest](/fsx/tmp/marshawn_001660_hd/native_png_linear_distortion_STAGED/current_best_guarded_HD/manifest.json) are separate from the previous accepted ZIP. All three selected native-HD images were manually viewed. The central actors are opaque and recognizable; residual floor/mat patterns and blur remain. Raw NPZs contain actual model RGB codes plus opacity/depth, not physical linear radiance or GT composites. Original NN/EMA/grid/config/checkpoint state and files are preserved. This output awaits root's manual review before transfer and carries no clean/final-grid claim.

## Train-view appearance and low-LR gate

Seventeen native train-camera patches at six fixed reference neighborhoods were rendered from the protected8k field and compared separately with AA/GA/OA source GT. Actor-occluded secondary views remain excluded. [Six source/render contact pages and descriptive residuals](/fsx/tmp/marshawn_001660_hd/floor_ray_diagnosis/DISTORTION/view_dependence/audit.json) were manually inspected. Real floor appearance varies strongly with view: warm yellow GA reflections can correspond to neutral AA/OA neighborhoods, and the model generally follows that variation. Exact dense correspondence and a physical BRDF are not established by the approximate-plane projections.

Residuals mix lost wrinkle/edge contrast with local color bias; they are not a single global exposure or WB offset. Median descriptive8px-lowpass residual-energy ratio is0.343 over low-gradient central patches (not an orthogonal energy decomposition). Per-patch channel biases typically lie around0.003–0.02 encoded RGB and change by location/channel. A new source WB/matrix correction is not supported by these contacts.

The actual8k checkpoint still holds Adam LR0.0083178292: its scheduler span is200000 despite the shorter training cap. A reduced actual optimizer LR is therefore a justified controlled test; it does not guarantee geometry preservation because RGB and density share base features. Root chose to repeat the failed Patch11+DSSIM0.01 phase with only LR changed to0.0005, keeping the parent immutable.

The first9k low-LR phase checkpoint was rendered at the same native HC/KA crop bounds, using core0 and step0.003 for both parent and phase. [Matched contacts and measurements](/fsx/tmp/marshawn_001660_hd/native_png_patch_dssim001_LR0005_STAGED/gate9000_matched_floor_contact.json) assert exact GT equality and were both manually inspected.

| Crop | Parent PSNR / SSIM / LPIPS | Low-LR9k PSNR / SSIM / LPIPS |
|---|---|---|
| HC | 31.7151 /0.851889 /0.260931 | 31.6630 /0.856893 /0.267984 |
| KA | 30.4239 /0.822591 /0.327466 | 31.3070 /0.828050 /0.328821 |

KA floor color is slightly closer to GT; HC is broadly similar. Fine wrinkles remain blurred, and LPIPS worsens in both crops despite SSIM improvements. No gross new cloud or transparent floor appears in these two crops. Four fixed floor pixels change expected depth by at most0.000555 prepared-world units with opacity approximately1. These are model-depth comparisons, not independent surface truth or full-body validation. This supports an early continuation review only, not acceptance as clean. Geometry did not resume the worker; root and occupancy own the whole-view gate and continuation decision.
