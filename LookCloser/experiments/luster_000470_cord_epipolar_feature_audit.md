# Frame 000470: epipolar audit around the lower belt cord

## What was tested

We checked whether an obvious camera-calibration error could explain why the
lower hanging cord disappears. The read-only [audit script](../scripts/audit_luster_cord_epipolar.py)
extracts SIFT features from the original full-resolution JPGs near the cord in
training cameras 025/026/027/097/098. It matches each camera pair, uses an
independent RANSAC fundamental matrix to reject unrelated matches, then
measures the surviving matches against the COLMAP PINHOLE calibration with a
Sampson residual in pixels. Held-out cam095 appears only in two diagnostic
pairs; its pixels never enter training or the cord fit. No masks were used.

The audit also counts features near the lower portion of the previously fitted
two-landmark cord segment and requires a correspondence to lie near that segment
in **both** images before calling it a lower-cord match. The [result JSON](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/epipolar_feature_audit_v1.json)
pins hashes of the calibration, landmark fit, and source images.

## Results

| Source pair | Independent inliers | COLMAP median residual (px) | 90th percentile (px) | Lower-cord inliers |
|---|---:|---:|---:|---:|
| 025–026 | 977 | 0.198 | 0.653 | 0 |
| 026–027 | 838 | 0.197 | 0.613 | 0 |
| 025–027 | 566 | 0.203 | 0.747 | 0 |
| 097–098 | 356 | 0.266 | 0.809 | 1 |
| 025–097 | 178 | 0.423 | 0.914 | 0 |
| 026–098 | 220 | 0.248 | 0.677 | 0 |
| 027–097 | 139 | 0.244 | 0.558 | 0 |
| 095–097 (held out) | 198 | 0.234 | 0.744 | 0 |
| 095–098 (held out) | 91 | 0.376 | 1.382 | 0 |

Each original image has 22–76 detected features within the defined lower-cord
band, but almost none form a verified cross-view match. This is consistent with
the earlier [PatchMatch audit](luster_000470_cord_guided_stereo_prior.md), which
found no qualified lower-cord depth among 30 projected candidates.

## Insights

The camera model agrees with independently matched **surrounding scene**
features to subpixel median accuracy; a large global pose error is unlikely to
be the sole cause of the missing braid. The cord itself remains unverified:
these residuals do not establish its local calibration, correspondence, or
surface depth. Do not impose a hard cord-depth prior from this audit. A short
continuation with only the five well-observed source cameras can instead test
whether conflicting training views are a major contributor while preserving
the pretrained all-camera field.
