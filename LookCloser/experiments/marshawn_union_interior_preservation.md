# Occlusion-safe foreground compositing

## What was tested

The exterior-only MatAnyone revision still used depth-gated RGB as its protected
input. That invariant was insufficient: internal content was already removed by
the earlier sample-density gate. A single narrow depth interval per pixel cannot
preserve multiple visible layers at motion-blurred bat/hat occlusions.

Direct raw/strict comparison reproduced a missing bat and jacket patch at output
index 5 (source 933), and black bat outlines at indices 13–16. These defects occur
before MatAnyone. Protecting strict pixels therefore preserves the defect.

`restore_union_interior.py` instead restores original integrated RGB inside the
union of high-confidence strict and MatAnyone foreground. MatAnyone is a
positive-only guide here and cannot remove strict foreground. At distance >=5px
from this union's exterior, RGB equals raw RGB exactly. At distance <=2px, the
previous exterior cleanup remains exact. The intervening 3px band blends sources.
No blanket hole filling: real gaps excluded by both guides remain gaps. No
per-person multiplication, fixed torso height ranges, or component-size rejection
are applied to restored interior RGB. The original renders and checkpoints remain
unchanged. Photographic tail is unchanged from the exterior-only revision.

## Validation

Two regression tests verify restoration of a simulated depth-gate hole, positive-
only MatAnyone behavior, and preservation of a real enclosed background gap.
Per-frame assertions verify exact raw RGB in every selected interior and exact
previous RGB outside the restoration region.

## Limitations

This is conservative image-space compositing, not recovered semantic geometry.
A silhouette error shared by both guides, or missing geometry in the raw model,
cannot be repaired by this method. Exterior haze remains possible. The reported
shoulder loss at exactly 3.0s was not reproduced in side-by-side source/SAM/output
inspection of photographic indices 71–74; do not claim that specific observation
resolved without locating it.

Both regression tests passed. All 70 neural frames passed both pixel-equality
assertions. The H264 output decoded as 84 Full HD frames at 24 fps (3.5s).
SHA256: `c6afe2e2d55e02718edd219fb8247e45f0587bf133a111016c1d8f22769ff221`.
Inspected the full contact sheet and enlarged raw/previous/new patches at indices
5, 14, 50, 51, 64, 67. Bat/jacket deletion at 5 and black occlusion outlines at
14 and 50–51 are removed in these inspected patches. Remaining gaps shared by
both guides are deliberately not filled; this is not a claim of perfect matting.

[Video](https://drive.google.com/file/d/1-CJU3tdGPJMCCSHDY4WUVb8BfvlytwoJ/view).
[Raw / previous / corrected patches](https://drive.google.com/file/d/1XGoJtXN4QC16NT4AMuINMBJ7wHH-Gs8m/view).
Silver output: `operations/flying_union_interior_20261112/`.
