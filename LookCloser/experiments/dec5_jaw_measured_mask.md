# DEC5 jaw: measured foreground evidence versus a semantic veto

## What was tested

The [four-time transfer](dec5_jaw_repair_transfer.md) traced part of the remaining
001193 skin speck to a D004_D005_1210LZ mask veto. This control checks independent
query-camera observations before changing that mask. Production data and video
remain unchanged.

`diagnose_jaw_veto_depth.py` checks native query depths with other-view depth,
roundtrip, parallax and calibrated 5x5 color evidence. At three unique rejected
query pixels, measured depths have 13, 9 and 4 color-qualified other views. One
nearby farther observation has zero color-qualified witnesses. Most remaining
rejected pixels have missing query depth, not observed empty space. These counts
refer to distinct cameras, not independent ground-truth samples.

`build_measured_foreground_override.py` constructs a separate mask for D/D,
using exactly the same rule at 001083/001123/001193/001195:

- Consider valid measured query depths within 24 native pixels outside the old mask.
- Require three other depth-compatible foreground masks, with existing 1.5-pixel
  roundtrip / one-degree parallax and 5x5 chroma <= .04, RGB error <= .12.
- Dilate those measured seeds by the existing mask's four-pixel radius; clip all
  additions to the 24-pixel band. Never delete an old foreground pixel.

No candidate mesh, target view, held-out RGB or diagnostic skin polygon defines
these seeds. The selected physical camera is a focused ablation, not a claim of
rig-wide mask calibration. The same identity/rule is used on all four times.
Four-pixel halos are an inferred tolerance, not measured foreground at every pixel.

`study_jaw_repair_transfer.py --mask-override-root ROOT` changes only the geometric
semantic gate. Raw 3D notch proposals, sample depth requirements and strict final
124-ray guard remain unchanged. **Renderer source masks stay original.** This
keeps texture-source admission separate from the geometry experiment.

## Results

| Time | Certified seeds / added mask pixels | Previous / new added mesh triangles |
|---|---:|---:|
| 001083 | 4415 / 12078 | 6 / 6 |
| 001123 | 3311 / 10869 | 0 / 0 |
| 001193 | 7255 / 17190 | 39 / 52 |
| 001195 | 6731 / 17072 | 44 / 44 |

The 001083/001123/001195 PLYs are byte-identical to the previous control. Their
verified RGB is explicitly reused, not claimed as new renders. Seven fresh
001193 renders cover original/repaired moving and F/E pairs, previous/new D/D,
and the fixed held-out view. Compared with the **previous repair**, RGB changes
0 pixels in moving, 39 in F/E and 0 in D/D. The local gain is not visible in the
current movie camera.

The fixed F/E diagnostic skin ROI including the real neck edge has depth misses
45 in production, 35 after the previous repair, and **22** now. Its interior-only
ROI changes 30 -> **17**. Both annotations and all earlier comparisons are kept.
These are local coverage diagnostics, not face quality metrics. The remaining
speck and jagged edge are directly visible: **partial improvement, not a full fix**.

![Native F/E comparison](/mnt/data/dec5_jaw_measured_mask_control/matched_review/F004_E005_1210FP_detail.png)
![Veto-camera comparison](/mnt/data/dec5_jaw_measured_mask_control/matched_review/D004_D005_1210LZ_detail.png)
![Measured seeds and bounded mask additions](/mnt/data/dec5_measured_foreground_override/001193/jaw_native.png)

Main-agent review inspected all four native mask/jaw overlays, the depth/color
diagnostic, moving and D/D whole heads, F/E and D/D native details, and the fixed
skin diagnostic. No new obvious defect attributable to the 13 added triangles
was seen in those views. Hair/fringe, neck boundaries and the broader video's
hand/forearm/lipstick defects are not repaired by this control.

Held-out 001193 is byte-identical to the original baseline. Fixed face PSNR /
SSIM / LPIPS remain **30.276545 / .939692 / .076123**. This is only a one-view
non-regression check. No full-frame quality metrics, loss or main CSV edits.

All four independent audits replay semantic votes and bounded mask expansion,
check original geometry, and pass 124 fresh observed-depth ray checks each.
No new nonmanifold edge or component: counts remain 95/78/51/63. Twenty-five
focused tests pass, including no-seed no-op, original-mask preservation,
bounded additions and rejection of out-of-band seeds.

Roots:

- `/mnt/data/dec5_jaw_veto_measured_evidence`: independent query/color diagnostic.
- `/mnt/data/dec5_measured_foreground_override`: four frozen mask controls.
- `/mnt/data/dec5_jaw_measured_mask_control`: meshes, reused/new RGB, review and audit.

Original masks, source EXRs, prior controls and published video remain unchanged.
No new full video is warranted by an invisible moving-view change. These helpers
are opt-in; existing model/renderer defaults are unchanged.

## Insights

A binary semantic veto can discard locally useful geometry even when measured
depth/color provides contrary evidence. The bounded correction removes part of
that restriction without simply accepting a 61:1 silhouette vote. It does not
prove every original mask disagreement is erroneous or every inferred cap has
the right anatomy. Missing evidence, unsupported proposal interiors and absent
larger proposals still limit reconstruction. The next substantial step should
address supported **surface shape/coverage**, not further blanket mask dilation
or repeated full-video rendering of these tiny gains. The full goal remains open.
