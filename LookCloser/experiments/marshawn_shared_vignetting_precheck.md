# Independent-card radial photometry test

Existing1700 even paper cards fit a shared radial cos^k model, with one offset per camera. No new matches, target1660 eval RGB fits, corrections or NN updates. Exact native lens undistortion supplies normalized ray radii.

| Validation RMSE, EV | Existing global gains k=0 | Fixed k=4 | Free k=5.918 |
|---|---:|---:|---:|
| 1700 odd paper | .23945 | .09362 | .06076 |
| 1740 odd paper | .24669 | .09708 | .06168 |
| 1740 separate cardstock | .14822 | .06644 | .03891 |

The shared free coefficient is positive and repeatable: card-block95% interval[4.32,6.68], leave-one-camera-out slopes5.63–6.13. The interval exceeds the expected0–6 range; no bound was forced. Only10unique physical fit cards across4target cameras are available. Focal lengths differ by0.5–0.64%, and sampled cards occupy a limited sensor region.

KA remains diagnostic only: one partial edge card, excluded from every fit. Fixedk4 predicts−.561/−.566EV radial differences at1700/1740, close to its unexplained−.633/−.622EV relative to the existing pose-only offset. This supports the radial hypothesis but cannot identify a global KA correction or remove floor BRDF ambiguity.

Detailed immutable measurements, model/validation scores, bootstrap, coverage, source hashes and the manually inspected plot are in [summary.md](/fsx/tmp/marshawn_001660_hd/report/shared_vignetting_card_PRECHECK/summary.md) and [audit.json](/fsx/tmp/marshawn_001660_hd/report/shared_vignetting_card_PRECHECK/audit.json). The k=0 fit was independently checked to reproduce all four prior equal-card gains to <1e−12EV. Helper: `scripts/audit_marshawn_shared_vignetting.py`. No corrections are accepted/applied.

Separate offsets remove camera-constant exposure confounding: camera-centered OLS still givesk5.910. Allsix actual segment headers identify the same NIKKOR Z28mm f/2.8 atF8/focus3970mm, with distinct lens serials. Individual slopes4.25/4.28/7.26/5.97 and sparse JA/LA coverage retain copy/BRDF uncertainty. The exact linear-domain correction candidate and double-EV warning are in `per_pixel_formula_NOT_APPLIED.md`; nothing was applied.

A separate joint fit on the existing26,030 TRAIN scene observations /5,287 tracks gives shared unconstrainedk3.67354; originalfive disjoint TRACK-fold fits range3.652–3.697. Held-out mean-fold medianEV is.11147(k0),.05044(fixed4),.04798(free); RMSE.22279/.12974/.12911. The p90 tail does not improve over4. Fixed4's remaining partial radial slope implies slight overcorrection (unweighted effectivek3.72867), so the scene data do not justify increasingk towards the independent cards'5.92. These datasets remain separate; material/coverage effects prevent universal lens calibration claims. No images, corrections or NN were produced by this followup. [Detailed untouched-fold comparison](/fsx/tmp/marshawn_001660_hd/report/shared_radial1660_train_PRECHECK/free_k_track_test.md).

The NEWjointk4 pose-predictedKA offset must be checked afresh; OLDscalar agreement was radially confounded. Independentcard11paper now implies a residual−.290/−.275EV at1700/1740, cardstock−.239/−.235. On the same12/11physicalpaperpoints, KA differs fromLB by−.205/−.195EV although normalizedradii are almost equal(.735/.729): changingcommonk4→6 explains only−.007EV. A pose-gain bias around.25EV is plausible, but onepartialcard cannot establish a global correction. TRAINcard11itself deviates up to.16EV from its globalcameraoffset; angular BRDF/copy-specific shading remain. ProperAP1Y→709Y differences<.0015EV exclude a luminance-domain explanation. No new gain was applied. [Cross-check and limitations](/fsx/tmp/marshawn_001660_hd/report/shared_vignetting_card_PRECHECK/new_joint_offset_crosscheck.md).
