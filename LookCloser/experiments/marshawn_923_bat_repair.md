# Frame923 bat discontinuity

## What was tested

User identified a gap in the bat in final GC FullHD. Compare original native PNG, production cap10, adaptive cap16 and uniform4096 samples/ray, using the same exported checkpoint. Render an isolated120×210-pixel ROI; the comparison enlarges this ROI for inspection, without changing training RGB.

## Results

[Matched bat crops](marshawn_908_1268_diagnostics/bat923/bat_compare.jpg). The gap remains in both denser renders; this is learned field geometry/appearance, not repaired by denser inference. At FullHD pixel(547,330), model depth is1.2275 model units versus~1.173 on intact neighbouring bat segments: the ray reaches a surface behind the missing bat segment. Accumulation is~1 because the coat behind it remains opaque.

Original GC RGB shows an intact bat, and its [actual soft alpha](marshawn_908_1268_diagnostics/bat923/alpha.png) preserves this section. Projecting a point interpolated between intact neighbouring bat surfaces into69 in-canvas cameras finds6 masks below128. The point is estimated from the learned surface, not independently triangulated, so projection uncertainty remains. [RGB projection checks](marshawn_908_1268_diagnostics/bat923/projected_views.jpg) show visible bat pixels in JE and KD with alpha0. JE is548 native pixels from labelled foreground, with RGB confidence0; the frozen background prior consequently penalizes opacity on this bat-intersecting ray. KD is27 pixels from foreground, so its background penalty is protected, although its RGB confidence is reduced. This is a concrete incorrect empty-space constraint. Its causal contribution to the GC gap still requires the controlled ablation below. A cable crossing the bat in CB and motion blur in several views are additional possible contributors; do not claim sole causality yet.

## Fix under validation

New isolated worker `scripts/marshawn_sequence/worker_prop_guard.py` excludes conservative projected bat capsules from empty-background-prior eligibility. It changes no RGB, alpha, masks, positive opacity targets or sealed numerical modules. Geometry uses extrapolated intact GC bat points with96 native pixels of padding; this is a conservative background-protection region, not a foreground segmentation.

Capsule numerical checks passed for normal, fully outside and degenerate segments. Cold independent923 trial stops at16,000 iterations for comparison against the original early trajectory. Hard attempt ceiling remains150,000. Durable Silver service `marshawn923-prop-guard.service`, GPU5, Restart=on-failure. Experiment root `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/diagnostics/bat923/guard_trial`. Production models and global sparse24 queues remain unchanged. Inspect8k/16k GC bat crops and retained opacity/depth before promoting any repair. This trial is not a verified fix or a completed production model.

### Coexecution correction

The initial trial acquired the same numeric-device GPU5 lease as production. Frame938 preparation succeeded, but its neural worker exited on `BlockingIOError` before training; the agent retained938 for retry and began the next sparse24 frame1038. Trial was gracefully checkpointed at zero-based step5792 (SHA256 `18f95d28cd7f4ddb01784305a6fa477f4ca3c3314c56eeba86fa4d36fd958026`) and restarted on the same physical GPU via its UUID in an explicitly separate coexecution lease namespace. No sealed code, model recipe, source data or production neural state changed. Capacity check:~23GiB used of~96GiB on GPU5 before coexecution; expected second training process fits. Verify both memory and live processes during subsequent supervision. This operational mistake delayed938; it was not a training-data/model failure. Do not report938 complete or removed from scope.

### First8k visual gate: background protection alone is insufficient

[Matched original/guard8k default-cap native crops](marshawn_908_1268_diagnostics/bat923/guard8000_default_compare.jpg): the bat discontinuity remains visible in both. Thus the JE mask error is a real contradictory empty-space constraint, but removing it alone is not a demonstrated repair. Do not promote this experiment into production. Original8k saved RGB crops exist for both default and cap10; use equal steps and equal caps for comparisons, and restrict native comparisons to queried person ROIs (the black strip above the ROI is unqueried diagnostic canvas).

Second isolated cold trial: `worker_prop_rgb_exclusion.py` uses the same background guard, plus omits only RGB observations within the projected prop capsule in CB. CB source crop visibly contains a sharp cable crossing the motion-blurred bat. This is a hypothesis test of conflicting observations, not an established sole cause. Other CB pixels and all72 camera memberships remain unchanged. No synthetic RGB, no positive-opacity geometry target. Original fixed RGB normalizer is retained, explicitly defining omitted observations without rescaling other gradients. Capsule-membership test passed.

Service `marshawn923-prop-no-cb.service`, Silver physical GPU7 via UUID, own coexecution lease namespace. Root `diagnostics/bat923/guard_no_CB_prop_trial`. Independent cold run to16k, hard ceiling150k. First trial continues to16k for a matched control. Inspect8k and16k bat continuity and actor metrics before promoting either change. Background-guard-only and no-CB-prop runs are diagnostic trials, not completed production models.

Completed8k cap10 gate confirms the same discontinuity: [comparison](marshawn_908_1268_diagnostics/bat923/guard8000_cap10_compare.jpg). Equal-actor/equal-view metrics are in-sample: original PSNR30.929717, SSIM0.738227, LPIPS0.200044; background-guard PSNR30.985705, SSIM0.738424, LPIPS0.200550. This provides no LPIPS improvement and no visual repair at this gate. Continue the declared16k gate and the separate CB-observation ablation.

At16k the matched default-cap native RGB comparison still shows the same bat gap: [original/guard16k](marshawn_908_1268_diagnostics/bat923/guard16000_default_compare.jpg). Background protection alone therefore fails both declared visual gates and must not be promoted as a bat repair. Finish the committed cap10 evaluation/journal normally; continue the CB-observation ablation. No production recipe change is justified by this result.

### CB omission8k gate: insufficient

[Matched8k comparison](marshawn_908_1268_diagnostics/bat923/noCB8000_cap10_compare.jpg): the no-CB-prop variant still has the bat gap. In-sample actor LPIPS0.200101 vs original0.200044 and guard-only0.200550. Thus the single CB observation is not a sufficient explanation or repair. Complete its16k gate normally; do not promote the ablation.

Further diagnosis: fit image lines to sharp native source bat crops in JE/NB and intersect their calibrated backprojection planes, independent of learned density. [Reprojection check](marshawn_908_1268_diagnostics/bat923/triangulated_axis.jpg) includes GC/KD as cross-views; line lies within the bat in the shown diagnostic segment, with residual sensitivity to highlights/taper and blur. [Axis measurement](marshawn_908_1268_diagnostics/bat923/triangulated_axis.json) is a diagnostic, not a promoted geometry target or a guaranteed exact centerline. Probe learned density along this independently measured line next. No claim of a working repair yet.
