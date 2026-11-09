# Independent multi-card KA calibration diagnostic

The earlier single-card result described only the restricted KC-reference roster, not KA's entire field of view. Full-source inspection found three additional physical markers A/C/D. Bounded1700 feature/pattern matches verify allthree againstKB, with LB/LA corroboration; fixed homographies also pass1740 without rematching. Pixel mapping explicitly converts undistorted4K to native distortedTIFF before linear photometry.

Before reading newRGB, the plan froze primaryKB's existing TRAINjointEV, fixedk4, fitA+C1700, heldoutD andallthreecards1740. Reference-only white selection, conservative geometry/quality masks and equal-card robust scalar fitting give KA EV.213411 versus currentposeprediction.102370: **delta+.111041EV**, not+.25. No WB/k adjustment or1660evalRGB fitting.

| Independent check, PRIMARYKB | Needed KA EV | Frozen-fit residual EV |
|---|---:|---:|
| HeldoutD1700 | .237968 | −.024558 |
| A1740 | .205911 | +.007499 |
| C1740 | .209548 | +.003862 |
| HeldoutD1740 | .235081 | −.021670 |

SecondaryLB/LA validate without retuning; their paper residuals are about.011–.046EV. Adjacentstock is diagnostic, not certified Lambertian/coplanar material. The OLD11edgecard remains a failed global extension: its paper would needEV.359862/.361937 and still differs by−.146/−.149EV after the frozenfit. The newcards cover radii~.50–.66; whole-image calibration is not certified and local floor mottling is not solved by this result. No gain, image, config or NN was changed.

[Detailed fit/heldout report](/fsx/tmp/marshawn_001660_hd/report/KA_multicard_ROI_inventory/multicard_single_EV_diagnostic.md), [actual mask/point selection](/fsx/tmp/marshawn_001660_hd/report/KA_multicard_ROI_inventory/photometric_reference_selection_1700.jpg), [distinct1700patternpairs](/fsx/tmp/marshawn_001660_hd/report/KA_multicard_ROI_inventory/bounded1700_pattern_pairs.jpg), [frozen1740validation](/fsx/tmp/marshawn_001660_hd/report/KA_multicard_ROI_inventory/frozenH1740_pattern_validation.jpg). Root review precedes any application.

The subsequent two-model diagnostic reuses only A+C1700: fixed4 gives EV.213411; a KA-specific radial exponent gives **k4.734867, centre EV.074550** with KB unchanged. Two physical anchors determine two parameters; conditioning42.51 and k/EV bootstrap correlation−.994 make this insufficient for whole-camera adoption. D/1740/OLD11 are now development evidence. D barely discriminates the models; OLD11 common-pattern linear-mean residual improves from−.134 to−.040EV, but dark/white contrast is nonmultiplicative and partial-pattern/registration caveats remain.

A new B/E holdout at verified global1880/local628 failed the frozen homography gate because of occlusion; no RGB photometry or forced match followed. No gain/image/config/NN changes were applied. [Exact parameters, radii, heldout failure and contrast report](/fsx/tmp/marshawn_001660_hd/report/KA_multicard_optical_models_NOAPPLY/summary.md).
