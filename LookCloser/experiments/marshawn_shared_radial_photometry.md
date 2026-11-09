# Fixed shared radial correction plus joint camera EV: train-only diagnostic

Reused original gain4 linear Rec709 samples from44938 calibrated track observations. The exact previous frozen lower-contrast mask retains26030 observations of5287 physical tracks from63 TRAIN cameras. All63 original EXR SHA256 values match the previous sampler. Evaluation images and the lone KA card were not used to fit anything. Five unchanged physical-track folds keep every observation of one point together.

For fixed k=0 or4, fit the **original** scalar objective: log(actual Rec709 luminance), original Huber IRLS, point nuisance brightness and camera log-gain with zero mean over63 cameras. Add the analytic radial EV before fitting. k=0 reproduces every old camera EV **exactly**, maximum difference0. The k4 corrected pixel formula is

`base_gain4_linear * 2**(2*log2(1+r²) - 0.4574742245269814 + camera_EV)`

where `r²=((u-cx)/fx)²+((v-cy)/fy)²`. Prepared images are already undistorted; COLMAP centers use raster pixel(x,y)→u=x+.5,v=y+.5. Radius² from those UV agrees with independently computed camera-space direction ratios to3.33e−16. The existing UV are fitted XYZ reprojections, **not independently detected feature UV**; this consistency check is not independent calibration validation.

| 1660 scene tracks | k0 | fixed k4 |
|---|---:|---:|
| Full corrected median absolute residual, EV | .11051 | .05029 |
| Full corrected p90, EV | .35352 | .17829 |
| Five held-track median ranges, EV | .10768–.11508 | .04960–.05107 |
| Equal-track median residual, EV | .12165 | .05518 |
| Camera EV range | −.53958…+.50453 | −.32612…+.29083 |

New camera EV minus old camera EV ranges−.41428…+.43212. Maximum fold-to-fold camera-EV standard deviation is.01391. Eval camera EV is predicted with the **previous fixed k6/power3 pose interpolation**, not selected again: DC−.13310, HC+.04627, KA+.10237. No eval RGB fit.

The radial field is globally centered by its mean over the same accepted26030 TRAIN observations (.4574742245EV), and63 camera EVs have zero unweighted mean. These are distinct gauges: observation-weighted total correction mean is−.02158EV, versus old scalar−.09852EV. The difference is about+.077EV, rather than an uncontrolled+.457EV global brightness shift. No per-image normalization, automatic exposure or white balance is used.

The manually reviewed floor-covering subset is small:7 tracks/41 observations (242,263,3738,3739,3800,4766,8290; hardware-uncertain8246/8353 excluded). Pooled residual median .21208→.12049EV and p90 .48811→.38626EV improve, but two small held-track folds worsen. Directional floor reflectance, folds/creases and patch footprint errors remain; this is not a Lambertian-floor certificate.

**Separate evidence, not pooled into the 1660 fit:** the existing physical-card study at1700/1740 reports fixed-k4 versusk0 median errors .04465 versus.09909EV on1700 held odd paper cards, and.03509 versus.08551EV on1740 paper. Its per-camera offsets/held card split are a different experiment. It supports this restricted shared-radial hypothesis; it does not identify every lens/aperture or make the lone KA card a global calibration.

Independent review of root's `scripts/marshawn_radial_photometry.py` passed. All pixels of three preview EXRs exactly match an independently computed gain4-unscalar × radial × NEW scalar expression, and all PNG bytes exactly match fixed Reinhard+sRGB encoding. Pixel+.5, positive inverse-cosine sign, global centering and source master were checked. Old scalar is not applied twice. No full dataset or NN was produced by this diagnostic.

Artifacts: `/fsx/tmp/marshawn_001660_hd/report/shared_radial1660_train_PRECHECK/{audit.json,fit_k0.json,fit_k4.json,accepted_samples.npz,independent_root_builder_review_geometry.json}`. Fit_k4.json uses the agreed builder schema and binds source files. **Measured consistency justifies a matched trial; no certified flat-field or promised spot-free rendering is claimed.**
