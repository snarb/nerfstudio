# Preserved scene experiments

Scene-specific code, measurements and artifacts remain on separate branches. Main retains reviewed shared implementation changes and general findings. Branch publication preserves evidence; it does not turn unfinished or rejected experiments into accepted defaults.

|Scene / context|Branch|Entry point|
|---|---|---|
|Marshawn1660 native63|[scene/marshawn1660-native63-20261109](https://github.com/snarb/nerfstudio/tree/scene/marshawn1660-native63-20261109)|[Final results, frozen recipe, complete metric register](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_final_results.md)|
|DEC5 native detail / hair / geometry|[scene/dec5-native-detail-20261109](https://github.com/snarb/nerfstudio/tree/scene/dec5-native-detail-20261109)|[Saved research workspace and inventory](https://github.com/snarb/nerfstudio/blob/scene/dec5-native-detail-20261109/LookCloser/experiments/dec5_saved_workspace_20261109.md)|
|Luster face diagnostic source|[scene/luster-face-diagnostics-20261109](https://github.com/snarb/nerfstudio/tree/scene/luster-face-diagnostics-20261109)|[Diagnostic preservation and limits](https://github.com/snarb/nerfstudio/blob/scene/luster-face-diagnostics-20261109/LookCloser/experiments/luster_face_diagnostics_saved_20261109.md)|
|Accepted Luster hybrid|[experiment/luster-000540-hybrid-20261017](https://github.com/snarb/nerfstudio/tree/experiment/luster-000540-hybrid-20261017)|Existing selected hybrid history; neural weights/data remain external|
|Luster multiframe transfer|[experiment/luster-multiframe-transfer-20261020](https://github.com/snarb/nerfstudio/tree/experiment/luster-multiframe-transfer-20261020)|Existing checked render-only handoff history|

## Shared versus experimental

Main includes packed-distortion precision, checkpoint feature-reweighting restoration, noncumulative eval occupancy dilation, explicit frequency-unit controls/tests, conservative mask-derived AABB and GPU workflow helpers. Legacy frequency projection remains default; experimental normalized projections are not declared quality improvements.

DEC5 retains its experimental ROI density branch, view-independent appearance controls, hair compositing overrides and scene sampling adapters with the corresponding scripts, tests and evidence. Generic-looking code with scene-specific dependencies has not been silently moved into the shared default path. Negative outcomes and partial pilots remain labeled in the topic reports.

The DEC5 inventory accounts for the pending local workspace states during this save. The separate Marshawn inventory records117 saved arm names/496 metric gates, not117 accepted production recipes. Data, neural checkpoints and large external assets referenced by reports are not represented as being inside Git.
