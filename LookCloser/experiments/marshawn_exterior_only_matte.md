# Marshawn: external silhouette boundary policy

## What was tested

The user reported internal black holes near2seconds and an intermittently missing
bat, and explicitly restricted MatAnyone to external boundaries without overlaps.
The previous region-based reductions are replaced, not layered over old output.

Neural input is the original strict render. A filled union silhouette defines the
outside-connected background; enclosed holes and interfaces between touching
people are never treated as exterior. MatAnyone can reduce opacity only in a
3pixel contour band. Deep interiors, thin structures detected by morphological
opening, reviewed head/hair/prop boxes, and conservatively expanded intersections
of actor bounding regions are protected. Every protected pixel is checked for
exact equality before any separate restoration. Overlap boxes are conservative
manual annotations, not claims of perfect instance segmentation.

A separate restoration addresses holes that already exist in the strict density
gate: inside a closed union envelope, in possible standing-actor overlap and
within torso world heights0.18–0.32 model units, confident original raw samples
may restore pixels with lost strict opacity. This is explicitly separate from
MatAnyone, which is forbidden to reduce any interior. Restore counts are saved
per frame and targeted contact frames44–52 are reviewed.

The14photographic frames use one union foreground mask for both people. Their
interior alpha is exactly1; MatAnyone only supplies the external16native-pixel
band (12pixel outward allowance). The real narrow elbow opening retains its
unfilled geometry. There is no per-person alpha multiplication at sleeve overlap.

## Validation

`test_exterior_only_matte.py` contains four tests: enclosed holes protected,
thin bat protected, touching-figure seam protected, edits confined to the outer
band. All passed on the deployed environment. Actual70frame receipts record
protected-area equality and separately count raw-torso repairs. Geometry/timing
remain the previous70-degree path,84frames,3.5s,24fps,FullHD.

This intentionally permits residual contour haze rather than relaxing the
interior/overlap/prop protection. Existing missing model geometry cannot be
reconstructed by this alpha policy.

## Artifacts

Code: `scripts/marshawn_sequence/exterior_only_matte.py`.
Silver: `operations/flying_exterior_only_20261112/` under the existing campaign.
`policy_receipt.json`, per-frame repair masks, comparison panels, frame hashes and
encoded-video validation preserve provenance. No training or source-file edits.

Final encoded output validated: 84 frames at 24 fps; SHA256
`c147b5897b21556f961357273a4c2e38b0e59201e81cf23ba542a3c48e26dd16`.
Visually inspected comparisons at indices 0, 10, 20, 35, 44, 46, 48, 50,
52, 69, 70, 76, 83, plus the larger index-50 comparison. Conservative
protection retains more contour haze than the previous body cleanup; no claim
that all residual reconstruction artifacts are eliminated. At indices 44–52,
separate raw repairs changed 69–1049 pixels per frame. These are heuristic
restorations, not ground-truth geometry.

[Delivered video](https://drive.google.com/file/d/1VxwHr-Sb7OQgrU0DFGdnElIf7_cbR-R-/view).
