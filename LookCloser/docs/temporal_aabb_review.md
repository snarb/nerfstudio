# Temporal box review

Freeze the accepted single-frame box before deriving a sequence envelope. A box
must be stored together with its coordinate transform: floor-aligned min/max
numbers cannot be applied directly to the previous world coordinates.

For Marshawn the recorded rate is 120 fps; 24 fps is the timecode rate. The
two-second real-time interval beginning at global frame 1660 has checks at
1660/1720/1780/1840/1900. The 48-frame 24-fps sequence uses 1660 through 1895 in
steps of five; frame 1900 checks the end boundary. The source TIFF mapping is
global frame minus 1252. No frame interpolation or per-frame color refit is used.

The reusable geometry is in `actor_box_geometry.py`, `mask_visual_hull_aabb.py`,
and `aabb_visualization.py`. The scene adapters are:

- `prepare_marshawn_temporal_sources.py`: frozen Log3G10/RWG decode, fixed
  color corrections, calibrated HD images and native undistorted search crops.
- `segment_marshawn_temporal_samples.py`: local SAM3 native-raster candidates.
  The encoder still resizes internally; output raster resolution is not an
  assertion of perfect hair boundaries. Every instance selection needs review.
- `derive_marshawn_temporal_box.py`: explicit actor IDs, artificial crop-edge
  checks, full-camera mask footprints, floor-frame adaptive visual hulls,
  synchronized three-view SIFT anchors, and a common envelope with a configurable
  minimum face margin and outward grid rounding.
- `audit_marshawn_temporal_box.py`: complete selected mask-ray coverage and
  source-anchor consistency. Ray intersection alone does not establish depth.
- `render_temporal_aabb_review.py`: source photographs with depth-tested box
  lines, plus top/front/side and 3D views. Its depth uses a silhouette hull and
  floor proxy, not measured depth or a newly trained radiance field.
- `render_aabb_review.py`: actual frozen-field clipping on black, including
  floor-aligned boxes. Use it only for the frame that checkpoint represents.

The floor frame uses 14 previously photo-qualified floor witnesses, excluding a
crease and a hardware point. Floor is Z=0. The original X direction is projected
onto the floor; the remaining axis completes a proper rotation. Physical units
are not asserted. The saved transform includes the offset, residuals and source
hash. A future training dataset must transform every camera and geometric prior
consistently; this review does not mutate the existing calibration or model.

Silhouette intersections can produce distant aliases when other views are
occluded or outside the camera. Components require synchronized source-feature
anchors; nearby detached pieces are retained for contour safety. Discarded
components remain unresolved, and must **not** become free occupancy cells.
An anchor-bearing component may still be too broad, requiring additional views.
Native search crops that touch an actor mask must be expanded before accepting
the resulting box. Crop boundaries are not camera field-of-view boundaries.

The common box encloses all retained cell extents and anchors before adding the
margin. Sampling five poses plus a margin is not a mathematical guarantee for
every intermediate frame. Inspect rapid motion near extrema separately. If an
actor lands on a mat, an enclosing AABB cannot also exclude that mat throughout
the interval; separate loss weighting or masks will be needed later.
