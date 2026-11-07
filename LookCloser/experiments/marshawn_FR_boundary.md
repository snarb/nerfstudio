# Feature-reweighting boundary: same-state diagnostic

Quarter and focus checkpoint75940 full states are protected before the original75941 switch. No training schedule or production model was changed.

`final_weights = 1 + FR * (analytical_weights - 1)`: reducing FR from1 to.3 weakens frequency damping. Shared density/RGB can therefore change immediately, before an optimizer update.

| Arm | Step | FR | Actor | PSNR | SSIM | LPIPS |
|---|---:|---:|---|---:|---:|---:|
| quarter | 45564 | 1 | all | 31.61045 | 0.835400 | 0.02001966 |
| quarter | 45564 | 1 | black | 30.29326 | 0.800047 | 0.02114498 |
| quarter | 45564 | 1 | grey | 32.92765 | 0.870752 | 0.01889435 |
| quarter | 60752 | 1 | all | 31.55859 | 0.831921 | 0.01949879 |
| quarter | 60752 | 1 | black | 30.22120 | 0.795936 | 0.02063522 |
| quarter | 60752 | 1 | grey | 32.89599 | 0.867906 | 0.01836235 |
| quarter | 75940 | 1 | all | 31.44405 | 0.829468 | 0.01938236 |
| quarter | 75940 | 1 | black | 30.14764 | 0.793096 | 0.02052269 |
| quarter | 75940 | 1 | grey | 32.74046 | 0.865840 | 0.01824203 |
| quarter | 91128 | 0.3 | all | 31.45281 | 0.828438 | 0.01934326 |
| quarter | 91128 | 0.3 | black | 30.14073 | 0.791561 | 0.02038282 |
| quarter | 91128 | 0.3 | grey | 32.76489 | 0.865314 | 0.01830369 |
| focus | 45564 | 1 | all | 31.56194 | 0.831404 | 0.02112244 |
| focus | 45564 | 1 | black | 30.43050 | 0.801460 | 0.02287078 |
| focus | 45564 | 1 | grey | 32.69338 | 0.861348 | 0.01937409 |
| focus | 60752 | 1 | all | 31.44666 | 0.826865 | 0.02098508 |
| focus | 60752 | 1 | black | 30.29198 | 0.795814 | 0.02285829 |
| focus | 60752 | 1 | grey | 32.60133 | 0.857916 | 0.01911188 |
| focus | 75940 | 1 | all | 31.36096 | 0.823755 | 0.02101182 |
| focus | 75940 | 1 | black | 30.23255 | 0.792149 | 0.02290000 |
| focus | 75940 | 1 | grey | 32.48936 | 0.855360 | 0.01912364 |
| focus | 91128 | 0.3 | all | 31.27765 | 0.821241 | 0.02132436 |
| focus | 91128 | 0.3 | black | 30.19187 | 0.789258 | 0.02305624 |
| focus | 91128 | 0.3 | grey | 32.36344 | 0.853225 | 0.01959247 |

Quarter91128 slightly improves aggregate LPIPS after the switch, with black improving and grey worsening. Focus worsens both actors. These two historical observations do not isolate FR from subsequent optimization or the historical seed reset.

Authorized read-only test: each immutable75940 state, FR1 → FR.3 → restored FR1; three FHD renders and sealed103 TRAIN/12 DC rays, native plus whole-field dense reference. Fixed native grid, no occupancy/frequency update or optimizer. Seven serialized states, live pipeline, global/private RNG and nonpersistent LUT are checked/restored. Native arrays must restore bitwise; saved RGB8/alpha16 use the pre-existing max1-code, changed-fraction1e-5/.001 and all6/view/actor metric1e-5 replay tolerances.

Any future actual continuation would use fresh75940 source full state and matched75941 seed reset for both arms, preserving Adam, scheduler, main sampling, occupancy and frequency state; only FR schedule differs. Private focus RNG, if used, must remain exact. No continuation is launched or approved by this report.

Evidence: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/quarter_FR_boundary/`; probe generation `FR_boundary_same_state_v1`.

## Same-state result (v3)


No optimizer step, frequency refresh or occupancy update. Both models use their own immutable75940 state. FR.3 weakens the analytical frequency damping relative to FR1.

| Model | FR | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| quarter | FR1 | 31.444047 | 0.829468 | 0.01938208 |
| quarter | FR03 | 31.083726 | 0.823344 | 0.02073437 |
| quarter | FR1_restored | 31.444047 | 0.829468 | 0.01938249 |
| focus | FR1 | 31.360956 | 0.823754 | 0.02101192 |
| focus | FR03 | 31.350705 | 0.823860 | 0.02096339 |
| focus | FR1_restored | 31.360955 | 0.823754 | 0.02101185 |

| Model | Group | Native band FR1% | Native band FR.3% | Dense band FR1% | Dense band FR.3% |
|---|---|---:|---:|---:|---:|
| quarter | old_prefix_supervised | 85.5046 | 89.6095 | 81.2569 | 85.2262 |
| quarter | new_prefix_supervised | 84.9371 | 88.4525 | 82.2611 | 85.8323 |
| quarter | old_local_reserved | 90.1033 | 91.2481 | 83.7044 | 84.8414 |
| quarter | new_whole_patch_reserved | 95.3252 | 96.5316 | 92.2690 | 93.4538 |
| focus | old_prefix_supervised | 82.8809 | 83.7580 | 81.4933 | 82.4074 |
| focus | new_prefix_supervised | 95.8024 | 95.8647 | 94.0846 | 94.3089 |
| focus | old_local_reserved | 84.3906 | 87.8161 | 82.7778 | 85.4791 |
| focus | new_whole_patch_reserved | 98.1526 | 98.2062 | 97.4903 | 97.6689 |

Immediate FR perturbation is a function-jump diagnostic with the learned occupancy held fixed. It does not predict which independently continued training schedule will yield better detail or geometry. Dense profiles expose density changes hidden by occupancy traversal; skipped density is not physical truth.

All seven serialized states, actual pipeline/global RNG and private focus state are unchanged. Nonpersistent feature LUT is restored. Native115 ray arrays restore bitwise; FHD RGB8/alpha16 images use pre-existing max1-code and changed fractions1e-5/.001, all6/view/actor metrics1e-5. V1 strict byte-image failure and interrupted v2 are preserved separately.

Both state probes passed all restoration/source guards, with no optimizer, occupancy or frequency updates. Quarter used198s and focus208s; peak process memory2596MiB under the8GiB allocator/10GiB process/900s limits. GPU4 lease was released. The v1 strict byte-image mismatch (at most1code) and interrupted v2 are preserved; v3 uses the already established campaign replay policy, not a newly loosened quality tolerance.

Quarter immediate FR switch worsens LPIPS6.98% and both actors, while native and dense witness-band locality improves. Focus immediate LPIPS slightly improves0.23%; its later degradation is therefore not explained by an immediate FR jump alone. Current original quarter training recovered to LPIPS.01923172 at111316, below its75940 value. A trained paired continuation is required before selecting an alternative schedule.

Personally inspected all6 GT/FR1/FR.3/restored1 contacts. QuarterFR.3 increases the brightness/density of broad exterior remnants, particularly HC/KA; focus differences are small. Gross hair/shoe disappearance is not apparent, but neither is clean.

Mature focus75940 reveals actual native cap saturation7/103 clothing rays (0/12 DC), versus quarter0/115. Focus all-cells saturation30/103 is separately diagnostic. Maximum dense tail beyond native end is.00480 atFR1 and.00393 atFR.3. This blocks any blind focus+quarter+distortion triple; the completed cap2048 counterfactual removes those hits but gives no visible or LPIPS benefit. No trained cap change was made; see `marshawn_focus_native_cap.md`.

Artifacts: `C/independent_occupancy/FR_boundary_same_state_v3/{comparison.json,report.md,quarter_*_comparison.jpg,focus_*_comparison.jpg}`; all native/dense arrays remain on silver. Initial paired FR continuation proposal is `C/independent_occupancy/quarter_FR_boundary/paired_continuation_proposal.json`.

## Separate paired continuation: CPU preparation only

`C/quarter_FR_continuation_candidate_v1` contains an isolated frozen worker/factory and an idle-exclusive technical supervisor. No technical or production release has been issued at this preparation gate. The parent quarter75940 checkpoint and all original1495 source files are unchanged; only two new files implement the continuation. Source inventory SHA `12e85211b74967c6438d933d6651e7e181f4e9eef013603ad2804e8dd6280523`. Technical policy SHA `6abcac3f28052605419736e330e46e2c8de0aa02d4b6143d84a622eeded3333c` binds17 artifacts.

The sole scientific configuration difference is `pipeline.feature_reweighting_after_switch`: original .3 versus hold1. Both arms preserve the original seed42 reset at75941, source Adam/LR/scaler/sampler/frequency/core/admission and cap1024. Technical32+32 attempts cross75941 and resume75972→75973, checking actual nonempty Adam updates plus Python FR switch/strength and nonpersistent LUT state. A future fresh4000 continuation would evaluate76440/76940/77940/79940 and must observe the actual original frequency refresh at76800. Technical checkpoints cannot become production sources.

CPU anchor validation proves all seven full-state hashes, actual core/EMA/active and R0 admission equality. The config diff contains only the declared FR factor after path normalization. Seventeen CPU tests pass; no GPU context or optimizer objects were created during preparation. The supervisor requires a separate hash-bound root release before zero/technical work and production remains separately closed. Future hold1 replay must use the actual child FR contract: the historical auditor's hardcoded .3 after75940 would be wrong for that arm.
