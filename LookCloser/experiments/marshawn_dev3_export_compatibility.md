# DEV3 export compatibility and first-six moving preview

## What was tested

The exact released GPU shader (`6603aa06…`) initially failed before model
loading on DEV3 because Torch2.4 does not expose `device_properties.uuid`.
A separate operational entry adapter retains the original environment and
supervisor/lease checks, then obtains the device of Torch's actual current
CUDA context with `cuCtxGetDevice` and queries that device's UUID. It compares
this with the inherited supervisor lease. It does not assume physical ordinal0,
alter model state or modify the shader. Unknown contexts, multiple visible
devices, wrong environment, foreign supervisors and lock-FD mismatches fail
closed. The private preview also needed a UID1000 write ACL on its new0004
render-output directory; no DATA/model permissions were broadened.

## Results

The actual finished1680 checkpoint `11d4ff0c…`, selected at139999 after140000
updates, passed the native HC archive check with maximum one RGB code
difference. Its full path target and16 donor geometry packages completed on
DEV3's physical L40S. The pipeline tensor SHA before/after rendering is identical.
The unchanged logged `current_train_step=0` and `FR=.3` are not a warmup render:
the shader calls `pipeline.eval()`, and the adaptive warmup branch in the frozen
model applies only while `self.training` is true. No step was forced.

The narrow CPU contract test uses a simulated current-context device7 and
requires the UUID query to receive device7, ruling out a hidden ordinal0
assumption. The actual frozen supervisor invocation and full geometry qualify
the real path. Source-bound proof:
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/video_dev3_uuid_v5/qualification_actual.json`
(SHA `8417f83c…`). ROOT approved release `f8f36759…`.

The final launcher was atomically updated, preserving its previous manifest.
An independent check verified all27 source hashes and the unchanged field-plan
SHA. The new entry imported and passed its release guards under the actual
running campaign controller's environment before and after activation. No
final72-frame export process was running during the update. The live
[launcher manifest](/fsx/tmp/marshawn_fleet10_140k_20261107/postprocess_command.json)
has SHA `4ca257d6…` at this revision. Shader6603, CPUshader512,
profile2bd9, native saved R0 and the explicit709 encoder remain unchanged.

A separate finite preview rendered the genuine selected1660/1665/1670/1675/
1680/1685 models at their respective moving-path records0–5. Models and YAMLs
stayed on their training hosts. At most one GPU per host was used, with
physical leases and field/native priorities; current fits finished naturally.
One exact existing canonical geometry package was reused, and the remaining
packages were produced by the released shader. All six packages have native
HC parity≤1 and matching before/after tensor SHAs. The temporary GPU drain
request is complete. Final delivery markers were not written.

Preview root:
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/video_partial_first6_v1/`.

- `Marshawn_first6_lookcloser_24fps_0p25s_4K.mp4`
- `Marshawn_first6_hybrid_24fps_0p25s_4K.mp4`
- `first6_fullframe_page1.jpg`, `first6_fullframe_page2.jpg`
- `first6_native_faces_manual_page1.png`, `first6_native_faces_manual_page2.png`
- `first6_native_feet_page1.png`, `first6_native_feet_page2.png`

The clips contain exactly6 frames at24fps,3840×2024, duration0.25s, with the
qualified709-limited/sRGB-transfer encoding. PNG masters remain unchanged.
Face and foot contacts use native pixel crops without resampling; full-frame
contacts are thumbnails. The earlier approximate head locators remain as
separate diagnostics; the face contacts were located manually on the actual
images. `complete.json`, `contacts_complete.json` and
`manual_faces_contact.json` bind the files and source PNG hashes.

## Insights

Manual inspection retains both actors, their hair and feet inside the full
canvas, with no gross color change. Hybrid detail changes are subtle. The
existing floor fragments and silhouette shell change visibly between the cold
models. This is an early0.25-second inspection, not proof of temporal stability
across the full72-frame sequence or a clean physical occupancy grid. No further
shader/grid variation or background cleanup was introduced.

Operational sources are preserved in
[`dev3_uuid`](../scripts/marshawn_video_export_geometry_cpu/dev3_uuid/) and
[`partial_preview`](../scripts/marshawn_video_export_geometry_cpu/partial_preview/).

ROOT subsequently inspected both full-frame pages, both native-face pages and
the second foot page. The pipeline/framing/color check passed with the same
limitations: material floor/shell variation, artifacts above the flying hair,
and soft native faces. Its exact artifact-bound review is
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/root_partial6_visual_review.json`
(SHA `f07d00cdc560ae4e256918f3ad3455d9c6ce001a6f971ed583d8019245377eb9`). No further tuning branch was opened.

## Read-only eventual-control adoption audit

A subsequent audit extracted the prospective spec construction directly from
the frozen final router and applied the real protected Bridge write logic
without writing canonical controls. The eventual olive/dev3 spec SHA is
`bb0ccc8c8206f902c2028527a079ec98b9f23a392c947780cda4976ac82657fa`;
SG is `a55e1e63f5939cf1a5a61b2670597914e0f8ad0593d56de8290eca95bc475d41`.
The current olive control is already byte-identical to its future form.
All six canonical geometry packages pass the exact full-path, native selected
checkpoint/completion, ready/transforms,16 donor row/mask/semantic, focal-anchor
and every package-file hash check. No repair is needed.

Canonical index0 intentionally has no active completion marker. The old marker
for spec `73a17f36…` is preserved under the durability schema-migration archive;
the only spec delta is `source_visibility_adapter_sha256`. The current migrator
is an actual no-op because that key is already present. Final export will
CPU-render index0 under `bb0ccc8c…` rather than adopt/restamp the old marker.
Its archived PNG hardlinks remain safe because the frozen saver writes a new
temporary PNG and replaces the public path atomically. The main CPUexec ledger
contains no stale intents. No canonical file or renderer state was changed.
Proof:
`operations/video_final_control_binding_audit_20261108/report.json`
under the capped campaign, SHA `7108a1ac…`.

The six measured CPU transfers took162.9–173.5s each (mean169.1s), giving about
67.6min for72 equal frames at three workers before later-view/IO contention.
ROOT deferred a new rolling-export coordinator to preserve the qualified final
route. A separate [existing face/clothing score audit](marshawn_hybrid_face_cloth_score_audit.md)
records the small face LPIPS tradeoff of the shared profile without opening a
new rendering branch.
