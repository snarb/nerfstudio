# Production-mesh transfer of guarded MHR completion

## What was tested

The positive [raw-mesh experiment](dec5_mhr_silhouette_patch_admission.md) did
not by itself establish compatibility with the cinematic mesh: the movie uses
a subsequently carved/repaired mesh. This control repeats the same candidate
generation and confidence admission on that actual production base for `001193`.
It does not append the already accepted raw-mesh triangles blindly or restore
the pre-carving mesh.

- Frozen cinematic request: `/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free/request.json`.
- Production base: `/mnt/data/dec5_expanded_head_repairs_preserve_carving/001193/mesh.ply`,
  SHA-256 `46718391862351a38212e5602af4b685d424f48c2ab2f9099517af5cfa63d1f0`.
- Raw mesh **not** used as the base: SHA-256
  `316dc2b6a82c0d99a6e399d15d941a69d491896921bc82e4d88ea307bb358b03`.
- The normalization metadata is identical. No extra pose/scale alignment is
  applied to the final 100-step MHR prior. That fit reached its iteration cap;
  it is not claimed converged or usable as a whole-head replacement.
- Same generic anatomical domain, inherited/current unsafe-parent exclusion,
  subdivision, local-distance/normal/centroid gates, 62-view measured depth
  checks and certified local interpolation as the raw-mesh experiment.
- Original production vertex/triangle prefixes remain exact. This does not
  imply inferred additions can never overlap a previously carved region;
  they must independently pass the same admission checks.
- RGB uses the **current production policy**: incidence power 2, target-angle
  prior before clipping and in pixel fallback, native footprint guard, zero
  registration, frozen camera profiles/exposure, unchanged texture masks,
  hard source selection. It is not the old power-8 diagnostic renderer.
- Matched CPU-only baseline/strict/interpolated renders: the historical moving
  stress pose plus train cameras F/E, M/B, C/E. The moving pose is used only for
  post-hoc evaluation, never fitting or admission. CPU/CUDA pixel equivalence
  is not claimed.

Root: `/mnt/data/dec5_mhr_production_patch_001193`.
Helpers: `scripts/run_mhr_production_patch_control.py`,
`scripts/review_mhr_production_patch_control.py`,
`scripts/localize_mhr_production_patch_side_effects.py`,
`scripts/audit_mhr_production_patch_control.py`.

These native-HD diagnostic comparisons are separate from the requested
**3456×6144 delivery video**, which continues unchanged under
`/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1`. This experiment is not a
substitute HD delivery. At the cinematic time `001193`, the presentation already
uses real, time-varying train RGB; changing this mesh would not alter that ending.

## Results

Candidate generation produced 691,136 proposals; 77 unsafe anatomical parent
facets were excluded. The production base has 61,860 vertices / 120,073
triangles. Both retained candidates have 417,880 vertices (including unused
candidate vertices): strict has 348,723 triangles, interpolated 355,594.
The large triangle increment is densely subdivided local inferred surface,
not that many newly observed scene features or a compact watertight repair.

| Native moving stress view | Baseline | Strict | Certified interpolation |
| --- | ---: | ---: | ---: |
| Missing pixels in fixed hole ROI | 45 | 1 | 0 |
| Filled ROI pixels receiving train RGB | 0 | 44 | 45 |
| Filled ROI pixels with source ID 255 | 0 | 0 | 0 |
| New geometry pixels, whole diagnostic image | 0 | 89 | 94 |
| New geometry pixels receiving RGB | 0 | 83 | 88 |
| RGB-changed pixels, whole diagnostic image | 0 | 1,259 | 1,342 |
| Source changes at depth-stable geometry | 0 | 111 | 135 |

Whole-image pixel counts above are diagnostic change inventories, **not
full-frame quality metrics**. No PSNR, SSIM or LPIPS is computed or claimed here.
The ROI baseline is measured on the actual production mesh (45), not copied
from the raw-mesh result (44).

The main LLM viewed all four native RGB comparisons, requested-hole/M/B/G/B
clay comparisons, interpolation branch-difference and occlusion crops, black
pixel side-effect crops and four residual-hole train-mask overlays: 32 saved
images in the explicit review inventory. The puncture closes while the natural
under-jaw shadow remains. The pre-existing jagged neck/hair silhouette remains;
this is not a claim of an artifact-free head or anatomically correct unseen surface.

| Other native train view | New geometry pixels strict / interpolated | New pixels with RGB strict / interpolated | Nearer occlusions > .003 strict / interpolated |
| --- | ---: | ---: | ---: |
| F/E | 5 / 14 | 5 / 14 | 105 / 110 |
| M/B | 192 / 197 | 191 / 196 | 6 / 6 |
| C/E | 103 / 107 | 103 / 107 | 0 / 1 |

Depth differences use normalized scene ray-depth units, not metres. The largest
new nearer occlusions are .015071 moving, .016917 F/E, .012538 M/B and .006890
C/E (interpolated); they are not rounding errors. Inspected crops localize them
to the original puncture, jaw/neck rim and pre-existing thin fringe. No obvious
new large fold appears in these comparisons. No previously hit ray is lost.

Both moving variants introduce four black RGB pixels: three on the pre-existing
neck/hair rim and one at the collar. Six newly hit moving pixels and one newly
hit M/B pixel remain uncolored at the collar. These were explicitly localized
and viewed; they are not hidden by the successful hole-ROI count. F/E and C/E
introduce no black RGB pixels. This is local improvement, not zero side effects.

Native moving RGB: `/mnt/data/dec5_mhr_production_patch_001193/admission/rgb_review/old_moving_native.png`.
Native clay: `/mnt/data/dec5_mhr_production_patch_001193/admission/native_clay_review/`.

Validation already completed: candidate arrays and candidate PLY byte-exact
replay; 3,746,940 sample vote/footprint checks; 196,887 vertex certificates;
248 final native camera/offset checks, all with zero trusted free-space vetoes.
The admission audit ran concurrently with RGB rendering: its `rgb/` inventory
is a historical snapshot, not a final retained-output seal. The dedicated final
audit must bind the terminal outputs separately.

All 12 RGB renders terminated successfully. The final diagnostic audit passed:
205 bindings / 213 retained files, actual production mesh prefixes, all final
render receipts, current texture policy, source inventory, finite depth maps,
native dimensions and visual-review hashes. Seal:
`/mnt/data/dec5_mhr_production_patch_001193/final_seal.json`.
Seven new unit tests cover recipe mismatch rejection, explicit non-mutating
base rebinding/restoration (also on failure), and distinct black-pixel side
effects. Together with the existing admission/patch tests: **14 passed**.
Unit tests do not establish anatomical correctness.

### Separate residual under-chin hole

The F/E train view still has a separate 30-pixel hole at portrait bbox
`[682,1148,696,1150]`, unchanged in both candidates. This was selected post hoc
from the rendered comparison, never supplied to construction.

All 30 rays hit safe facets of the full prior. One is removed by the unchanged
centroid gap; 29 correspond to raw proposals, but **zero pass the silhouette
admission**. Veto cameras are A/B (13 of those ray-facets), A/C (29), B/B (26)
and C/B (4). Repetitions of a facet across rays are retained in these counts.

The main LLM inspected the original train RGB beside the projected candidate
samples and masks. The samples touch/cross the actual jaw outline onto
background; this is not an obvious erroneous interior hole in segmentation.
All four veto cameras participated in the fit. The fit permits a soft 2-pixel
outside margin, while admission accepts no available outside-mask sample.
Bilinear signed mask distance at the proposal samples reaches 1.881, 2.023,
2.780 and .663 pixels respectively. Bilinear SDF and nearest-pixel mask votes
are different measurements at subpixel boundaries. Soft penalties, vertex-only
fit sampling and the non-converged iteration cap also prevent interpreting the
2-pixel setting as a hard bound.

Evidence: `residual_hole/result.json`, `residual_hole/mask_attribution/` and
`residual_hole/margin_audit.json` under the experiment root. No mask, confidence
gate or candidate parameter was relaxed to fill this residual hole.

## Insights

The observed local benefit is not restricted to the unprocessed COLMAP mesh:
the same method closes the requested production-mesh puncture with the current
texture policy. It is a confidence-gated completion hypothesis rather than a
replacement of measured COLMAP geometry by a learned whole-body surface.

Added geometry can change visibility and graph source labels at already present
surface pixels. It is therefore not texture-neutral even though colors and
texture policy are frozen. Native train-view and signed-depth reviews remain
necessary before broader integration. The independently fitted
[`001195` transfer](dec5_mhr_transfer_001195.md) is not evidence of a 150-frame
rollout.

The next bounded hypothesis is aligning the fit's silhouette tolerance with
strict admission (offset 2→0, keeping the normalization/weights unchanged),
not allowing inferred skin to extend into observed background. That control
has separate artifacts and must repeat fitting/topology/visibility checks;
this report does not claim it has succeeded. Neither this experiment nor the
new hypothesis modifies the active frozen 6K delivery.
