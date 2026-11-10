# Frame923 bat discontinuity

## What was tested

User identified a gap in the bat in final GC FullHD. Compare original native PNG, production cap10, adaptive cap16 and uniform4096 samples/ray, using the same exported checkpoint. Render an isolated120×210-pixel ROI; the comparison enlarges this ROI for inspection, without changing training RGB.

## Results

[Matched bat crops](marshawn_908_1268_diagnostics/bat923/bat_compare.jpg). The gap remains in both denser renders; this is learned field geometry/appearance, not repaired by denser inference. At FullHD pixel(547,330), model depth is1.2275 model units versus~1.173 on intact neighbouring bat segments: the ray reaches a surface behind the missing bat segment. Accumulation is~1 because the coat behind it remains opaque.

Original GC RGB shows an intact bat, and its [actual soft alpha](marshawn_908_1268_diagnostics/bat923/alpha.png) preserves this section. Projecting a point interpolated between intact neighbouring bat surfaces into69 in-canvas cameras finds6 masks below128. The point is estimated from the learned surface, not independently triangulated, so projection uncertainty remains. [RGB projection checks](marshawn_908_1268_diagnostics/bat923/projected_views.jpg) show visible bat pixels in JE and KD with alpha0. JE is548 native pixels from labelled foreground, with RGB confidence0; the frozen background prior consequently penalizes opacity on this bat-intersecting ray. KD is27 pixels from foreground, so its background penalty is protected, although its RGB confidence is reduced. This is a concrete incorrect empty-space constraint. Its causal contribution to the GC gap still requires the controlled ablation below. A cable crossing the bat in CB and motion blur in several views are additional possible contributors; do not claim sole causality yet.

## Fix under validation

New isolated worker `scripts/marshawn_sequence/worker_prop_guard.py` excludes conservative projected bat capsules from empty-background-prior eligibility. It changes no RGB, alpha, masks, positive opacity targets or sealed numerical modules. Geometry uses extrapolated intact GC bat points with96 native pixels of padding; this is a conservative background-protection region, not a foreground segmentation.

Capsule numerical checks passed for normal, fully outside and degenerate segments. Cold independent923 trial stops at16,000 iterations for comparison against the original early trajectory. Hard attempt ceiling remains150,000. Durable Silver service `marshawn923-prop-guard.service`, GPU5, Restart=on-failure. Experiment root `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/diagnostics/bat923/guard_trial`. Production models and global sparse24 queues remain unchanged. Inspect8k/16k GC bat crops and retained opacity/depth before promoting any repair. This trial is not a verified fix or a completed production model.
