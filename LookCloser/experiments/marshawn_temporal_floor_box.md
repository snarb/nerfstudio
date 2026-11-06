# Marshawn: one floor-aligned box for two seconds

What was tested: frozen accepted frame1660 box; common envelope at global1660/1720/1780/1840/1900 (120fps acquisition, real-time half-second checks). A 24fps output uses 48 source frames1660..1895, every fifth frame. Source previews1765/1795 were also inspected near the lift; these are additional visual checks, not native-mask audits.

Results: original accepted box is unchanged in `configs/aabb_review/marshawn_001660_mat_trim.json`. The common candidate is stored with its rigid world-to-floor transform in `configs/aabb_review/marshawn_1660_1900_floor.json`: min[-0.84,-0.21,-0.07], max[0.15,0.42,0.63]. Floor alignment rotates the previous up direction by7.997 degrees. Fourteen photo-qualified floor points give max fit residual0.00464 prepared units; physical meters are not asserted. Minimum margin0.025 plus outward0.01 rounding; native mask padding64 pixels.

Frame1660 uses existing66-view reviewed masks. New sampled poses use13 cameras each, with19 at1900. Local SAM3 candidates preserve native output rasters; the encoder internally resizes. Actor IDs were reviewed manually; truncated AC/NE1900 crop regions were expanded and rerun. Native masks remain imperfect and are not released as final training mattes.

Separate per-person carving avoids mixing identities before 3D reconstruction. Synchronized source SIFT tracks qualify components, with nearby detached pieces retained. Sparse counts are23/12/39/59 across the four later poses; the raised grey actor has only one three-camera anchor. Anchors are part of box construction, so their containment is a consistency check, not held-out validation. Twelve geometry/overlay regression tests pass.

The grey actor lands on the thick mat by1.5s and remains there at2s. The common box therefore cannot exclude the mat. The original accepted single-frame box is preserved separately.

Insights: this is a conservative common candidate, **not a proven minimum-volume box**. An occlusion shadow behind the grey actor at1900 still enlarges xmin. Five sampled poses plus margin do not mathematically certify every intermediate frame. Silhouette cells discarded during component qualification must not be marked free in an occupancy grid. Source overlays use approximate hull/floor depth; frozen-model clipping applies only to1660. No training, camera recalibration, occupancy edits, or source color refitting occurred.

Artifacts: `/fsx/tmp/marshawn_001660_hd/two_actor_isolation/temporal_2sec_floor_v1/` (geometry_identity_v2, review_identity, floor_frame.json, reviewed_identities.json, freeze_receipt.json). Earlier geometry variants are rejected diagnostics.
