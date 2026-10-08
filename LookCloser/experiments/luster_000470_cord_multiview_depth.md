# Frame 000470: train-view depth agreement on the hanging cord

## What was tested

Generic COLMAP PatchMatch leaves most lower-cord pixels without qualified
depth. To check the two-landmark 3D segment independently, a read-only
photometric sweep used the original undistorted JPGs and COLMAP PINHOLE
cameras. For six points along the segment, a ray from **training cam097** was
moved ±`.027` normalized scene units in depth. Its 3×3, 5×5, or 9×9 mean
color was compared with projections into training cams098, 096, 026 and 027.
The per-depth score is the median across cameras with valid projections.
Held-out cam095 supplied no observation or score. The [script](../scripts/audit_luster_cord_multiview_photometric.py),
[hash-qualified receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/photometric_train_only_v1/result.json),
and [cost curves](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/photometric_train_only_v1/cost_curves.png)
preserve the calculation. No source masks were used.

## Results

Best depth offset from the original two-landmark segment, in normalized scene
units (negative is nearer cam097):

| Segment fraction | 3×3 | 5×5 | 9×9 |
|---:|---:|---:|---:|
| .10 | +.00181 | +.00226 | +.00226 |
| .25 | +.00181 | +.00226 | +.00045 |
| .40 | −.00226 | −.00226 | −.00271 |
| .55 | −.00045 | −.00045 | −.00136 |
| .70 | −.00045 | −.00045 | +.00045 |
| .85 | 0 | 0 | 0 |

The aggregate minimum is within ±`.003` at all six points and is reasonably
stable to patch size. Individual cameras disagree more strongly: at fraction
`.70`, their 5×5 minima range from about `−.010` to `+.008`; one camera is
within `.001` of the fitted line. The [native-resolution train/eval projection crops](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/midline_native_audit)
show that the straight line lies near, but is not centered on every twist of
the braid. Original source orientations also differ: cams025–027 are portrait
and cams095/097/098 are landscape, and each was cropped using its own width
and height.

## Insights

The aggregate color evidence supports the existing line as an **approximate**
cord depth. It does not certify a precise surface: a repeated braid texture,
blur, changing view angle and nearby teal fabric can produce competing color
matches. In particular, the lower section lacks a tight per-camera consensus.
No hard depth prior was created from this scan. It explains why further
decreasing ray step below `.001` and blindly increasing density are unlikely
to solve the remaining blur and transparency; the next trained intervention
must be checked against held-out cord appearance and full-person metrics.
