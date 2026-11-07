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

Authorized read-only test: each immutable75940 state, FR1 → FR.3 → restored FR1; three FHD renders and sealed103 TRAIN/12 DC rays, native plus whole-field dense reference. Fixed native grid, no occupancy/frequency update or optimizer. Seven serialized states, live pipeline, global/private RNG and nonpersistent LUT are checked/restored. Restored FR1 images and native arrays must match.

Any future actual continuation would use fresh75940 source full state and matched75941 seed reset for both arms, preserving Adam, scheduler, main sampling, occupancy and frequency state; only FR schedule differs. Private focus RNG, if used, must remain exact. No continuation is launched or approved by this report.

Evidence: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/quarter_FR_boundary/`; probe generation `FR_boundary_same_state_v1`.
