# DEC5 forearm: fixed-rule transfer to 001029 and 001037

## What was tested

Transfer the [001033 clipped-plane canary](dec5_forearm_confidence_prior.md) to
two separately selected neighboring times, without changing geometry thresholds
or running DA3 again. This is a small transfer test, not a 150-frame repair or
proof of temporal stability over the whole clip.

Artifact root: `/mnt/data/dec5_forearm_plane_transfer`.
Fresh 62-view geometric-depth controls are supervised separately under
`/mnt/data/dec5_forearm_temporal_transfer/controls/{001029,001037}`. They supply
confidence, not replacement baseline meshes: each baseline remains the original
unrepaired TSDF for that time.

Votes come from distinct observed camera-depth estimates, not statistically
independent ground-truth measurements; the cameras share image content and the
stereo algorithm. This caveat applies to every confidence count below.

The reference is G_A. Bare-skin polygons are traced separately from each time's
real train RGB in G_A/H_A/H_C, inset from the cuff. Only those pose-dependent
semantic bounds change. They were frozen before completed depth controls or
candidate geometry were inspected. No held-out image enters geometry, clipping,
texture synthesis, or selection. Camera profiles/exposure and the matched camera
poses are fixed across variants; H_A provides a constant physical-camera view.

RGB uses the same **local ablation renderer** as 001033: direct
`render_smooth_temporal_mesh_video.render_one`. It does not install the published
video's `install_source_masks` foreground-eligibility wrapper or the
`temporal_texture_view_prior.install` calibration-angle source-label prior.
Thus matched cameras/profiles do not make these colors production-equivalent.
Geometry and confidence checks are unchanged by those color wrappers; production
source-selection/color integration requires a separate exact-wrapper canary.

![001029 frozen train-only skin bounds](</mnt/data/dec5_forearm_plane_transfer/001029/skin_masks_native.png>)

![001037 frozen train-only skin bounds](</mnt/data/dec5_forearm_plane_transfer/001037/skin_masks_native.png>)

All original vertices and triangles are preserved. Trusted plane anchors must
agree with the original mesh and three other observed depth maps, using the
unchanged `0.001` camera-z tolerance, 1.5-pixel return reprojection, and 1-degree
parallax. Candidate pixels are original reference misses within 100 pixels of
an anchor, inside all three reviewed skin regions, and free of trusted skin-map
free-space contradictions. No measured interior vote is required: additions are
explicitly **inferred**, not measured truth. The same attachment clipping guard
removes only appended triangles hiding old pixels supported by at least three
other depths by more than `0.001`, in the moving and three train cameras, for at
most four passes. See the immutable `protocol.json` for all fixed values.

The separate transfer helper was first replayed on 001033 using CPU only. Both
plane and clipped-plane outputs match the original pilot **byte-for-byte**;
`algorithm_replay.json` records the matching hashes. The original pilot artifacts
were not modified. This checks implementation transfer, not performance at new
times.

## Results

The train previews show different difficulty: 001029 has a smaller forearm
defect and much intact skin, while 001037 has a larger lower-forearm gap and
visible hand motion blur. Both are retained as selected; neither was replaced
after looking at reconstruction results.

| Time | Trusted G_A anchors | Original G_A misses | Plane proposals | Added triangles before → after clip | Moving-view new pixels |
|---|---:|---:|---:|---:|---:|
| 001029 | 18,242 | 1,136 | 1,136 | 2,456 → 2,456 | 1,219 |
| 001033, prior pilot | 10,533 | 7,812 | 5,657 | 11,459 → 11,454 | 6,714 |
| 001037 | 383 | 10,605 | 4,216 | 8,186 → 7,549 | 5,173 |

Before clipping, 804/3,960 proposals at 001029/001037 have **zero** agreeing
measured interior depths. All original vertex/triangle arrays remain intact.
001029 requires no clipping. At 001037 the original plane hides 321 supported
moving-view and 584 supported H_C pixels; clipping removes 637 appended
triangles over two passes. This is a substantial failure of the unclipped plane,
not merely a tiny contour adjustment. The fixed half-pixel renderer guard then
reports zero supported-old visibility flags in all four review cameras.

### A second sampling lattice exposes a preservation failure

That half-pixel result is not universal preservation. A separate non-mutating
check on **integer camera rays**, matching the observed-depth diagnostic grid,
finds residual old pixels hidden beyond the same `0.001` threshold and supported
by at least three other depth maps:

| Time | G_A | H_A | H_C |
|---|---:|---:|---:|
| 001029 | 0 | 0 | 0 |
| 001033 replay | 0 | 2 | 0 |
| 001037 | 0 | 3 | 136 |

The 001033 replay changes no original artifact. The v1 rule is deliberately
retained unchanged: **001037 fails the stronger preservation check**, despite
passing its original half-pixel guard. A future guard must cover both lattices
or otherwise address subpixel coverage; this report does not silently retune v1.

All newly visible/front-facing pixels at 001029 and 001037 stay inside the three
frozen skin polygons on the integer diagnostic grid: zero outside pixels.
The 001033 replay has five H_A pixels at most `sqrt(2)` pixels outside its inset
polygon, with none more than two pixels outside. These manual inset regions are
not exact ground-truth silhouettes; passing them does not prove interior shape.

### Native appearance and the truncation mechanism

![001029 native moving RGB](</mnt/data/dec5_forearm_plane_transfer/001029/evaluation/moving_rgb_comparison_native.png>)

![001037 native moving RGB](</mnt/data/dec5_forearm_plane_transfer/001037/evaluation/moving_rgb_comparison_native.png>)

001029 is a useful local success: the two near-cuff holes largely close without
a new broad surface distortion. Existing texture seams and small contour flaws
remain. **001037 is only partial improvement**: its visible plate ends at a flat
edge well above the cuff, leaving much of the forearm missing. It is not a
successful complete forearm repair and must not be promoted on coverage alone.

The plane hypothesis at 001037 projects 4,510 of its 10,605 candidate points
outside H_C's image; another 1,776 are in that image but outside its inset skin
polygon. H_A rejects 1,422 inside-image points by its skin polygon. In total,
the three-view gate rejects 6,388 points, and a trusted free-space veto rejects
one more. Thus **image coverage becomes an artificial end of the patch**, not an
anatomical cuff boundary. These projection counts diagnose the fitted plane;
they are not ground-truth visibility of unknown anatomy. The rule and masks
were not changed after this finding.

![Three sparse times, fixed physical H_A camera](</mnt/data/dec5_forearm_plane_transfer/three_time_H_A_native.png>)

Full H_A manually traced forearm-skin regions; train-view checks, not held-out
generalization:

| Time | Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| 001029 | Original | 21.373 | 0.82678 | 0.23121 |
| 001029 | Clipped plane | 27.708 | 0.86383 | 0.14286 |
| 001037 | Original | 12.496 | 0.23608 | 0.71758 |
| 001037 | Clipped plane | 15.113 | 0.40590 | 0.68648 |

On the originally missing H_A skin subsets, PSNR is `8.93751 → 26.31312 dB`
at 001029 and `11.92368 → 14.55342 dB` at 001037. All PSNR/SSIM/LPIPS records
are retained in the per-time `metrics.json`; full-skin metrics are primary.
The missing-only region at 001029 is disconnected, so its tight-box
zero-outside-mask SSIM/LPIPS can be dominated by zeros and should not be read as
whole-forearm quality. Rendered half-pixel missing counts are 1,105/10,225;
integer diagnostic counts differ slightly, as reported above.

Held-out F_B does not see the actual added patch at either new time. At 001029
its visible upper-forearm ROI and entire RGB image are unchanged:
`26.72005 dB / 0.91493 / 0.06025`. At 001037 the forearm is below frame, so its
requested-region metrics are **N/A**; no hand/wrist substitute was scored.
Although the F_B depth images have no new visible geometry, the raw renderer
changes 2,023 full-image RGB pixels at 001037 while recomputing source visibility
and labels. Neither held-out result validates the filled interior or establishes
global RGB preservation.

Twelve full native RGB renders completed; summed per-render time is 309.3 s
with two concurrent frame workers (not a serial wall-time benchmark), around
9.5 GiB total GPU use. Ten metric triplets plus two explicitly unavailable region
records are retained. Source RGB, both complete 62-depth controls, metadata,
camera profiles/exposure, mesh prefixes, and render completion hashes pass the
artifact audit. Physical-camera pose differences from 001033 are at most
`1.21e-8` normalized matrix units. Numeric audit success is kept separate from
the negative sampling/visual findings.

## Insights

The fixed rule transfers well to the small 001029 defect, but does **not** pass
the harder 001037 test: incomplete flat truncation and integer-lattice trust
violations remain despite better train RGB metrics. The original 001033 positive
canary also had a small sampling-lattice caveat. No production default or full
video was changed.

This motivates a separately frozen follow-up, not reinterpretation of v1:
check both pixel lattices, and distinguish out-of-image uncertainty from an
in-frame semantic disagreement. Any relaxed visibility requirement must still
have multiple real skin-view limits and veto genuine observed contradictions.
The apparent success of a filled rectangle is not recovered anatomical truth.
Three sparse times with manually traced pose-specific regions do not establish
automatic mask transfer or full-clip temporal stability.

Reproduction: `scripts/study_forearm_plane_transfer.py`, commands `stage`,
`freeze`, then per time `analyze --frame FRAME`, `clip_plane --frame FRAME`,
`geometry_review --frame FRAME`, `semantic_review --frame FRAME`, `evaluation_inputs --frame FRAME`,
`evaluation_mask --frame FRAME --polygon ...`, `render --frame FRAME`,
`score --frame FRAME`, `audit --frame FRAME`, and finally `summarize`. For a new
reproduction use one fresh `--output /absolute/new/root` on every command; the
default preserves this study. The depth reader refuses
incomplete controls. GPU rendering is started only after the parent depth jobs
release the GPU. Use the parent repo `.venv/bin/python`, EXR support enabled,
and two OMP/OpenBLAS threads. No training or neural inference is launched.
