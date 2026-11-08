# Marshawn final encoder color and runtime audit

What was tested: the sealed CRF15/yuv420p MP4 encoder versus one explicit
BT709 limited-range matrix/tag control, using three existing3840×2024 PNGs.
No renderer, model, source data, image size or hybrid parameters changed.

The default olive nerfstudio FFmpeg2.8.6 cannot start: it needs
`libx264.so.138`, while that environment supplies ABI164. The existing separate
`lookcloser-jpeg` FFmpeg/ffprobe5.0.1 toolchain works. Absolute binary paths were
verified under actual fleet-controller2270475's environment. Its54 resolved
shared libraries and complete version strings are recorded; no installation,
symlink workaround or global environment change was made.

| Existing encoder | Explicit single control |
|---|---|
| No matrix/range/primaries/transfer metadata | BT709 matrix/primaries, limited range, sRGB transfer13 |
| Measured luma fit consistent with limited601 | Explicit limited709 conversion |
| Three frames,24fps,3840×2024 | Same frames/rate/dimensions/CRF15/yuv420p |

Interpreting the untagged baseline as HD709 changed smooth nonblack source pixels
by mean0.514 RGB code,99th-percentile3 codes, maximum10 codes. This is a measured
ambiguity on the fixture, not a claim that every player chooses that matrix.
The actual comparison showed no gross color shift. Both encodes remain lossy:
mean full-frame RGB PSNR was41.35dB baseline versus41.19dB explicit control,
and nonblack mean absolute RGB error was about3 codes in each. PNG masters
remain authoritative; this is a compatibility/runtime fix, not a claim of better
codec reconstruction quality.

On the same fixed500×600 PNG-reference crop, averaged across the three frames:

| Decode | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|
| Baseline/default |36.204|0.90996|0.04216|
| Explicit709/default |36.074|0.90885|0.04511|
| Baseline interpretedHD709 |35.995|0.90935|0.04603|

These quantify codec RGB parity with original PNGs, not model quality or
held-out reconstruction. No encoder parameter sweep was performed.

[Actual fixture comparison](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/encoder_color_audit_v1/actual_4K_fixture_comparison.jpg).
The measured two-encode receipt and luma proof are in
`operations/encoder_color_audit_v1/{receipt,luma_matrix_analysis}.json` under
that campaign. The repeated same single control under the actual launcher
was byte-identical, and retained expected ffprobe metadata and RGB decoding.

The versioned OPS-only adapter preserves PNG/source/path checks, rejects foreign
completed movies, binds owned prelaunch intents and inherits an encoder flock
through FFmpeg. Restart can adopt only a matching completed owned artifact.
Four crash points, real inherited locks, modified PNG rejection and the exact
frozen route's encode call passed CPU tests. The120-frame/5-second control-plane
test uses explicitly labelled mocks; actual120-frame movies still await all
model/renders and final visual review. Real verification includes stream tags,
frame count, dimensions, duration and RGB decoding to the last frame.

At this report's qualification point, candidate encoder SHA was
`63db4c593cd90e45b2dcefbec486a7839b24b912f3ab1d5b9bc3eebdbc6e7383`,
wrapper SHA `b95925f8596d6fdd6a4089b62a922ae982f539ba689ff992eeaa88552d3e7ff6`,
and qualification SHA
`ab9b8fd06b6dcccefda51ae9d976904ee55da730f3e9ccafae12a04c3ca9bf5f`.
The candidate is held for concrete ROOT release; fleet hook390ef99d… was
unchanged during the audit. Activation is a separate OPS-only posthook update.
Qualified source archives are in
[`encoding`](../scripts/marshawn_video_export_geometry_cpu/encoding/README.md).

ROOT subsequently approved and activated this OPS-only final encoder. Independent
read-only verification passed every active-hook source hash and the release/tool
library guard under the actual fleet2270475 environment. The fleet plan remains
74c92390…, with no duplicate activation or early encoder/model launch.

Active fleet posthook SHA:
`06f2541bf7cb8d53e7b95029e4f5cff97d995f61fba870386b83e908b604c32c`.
ROOT encoder release SHA:
`175dd1c6dbf6feca983e7bc8d369176abf28d0b69d4b1601a8059b7063f0922a`.
The old390ef99d… hook and the immutable candidate were preserved. Activation and
independent guard receipts are under
`operations/video_geometry_cpu_encoding_v4/{activation,independent_activation_verification}.json`.
The final encoder will run autonomously after all72 models and120 image records
are complete; full-video visual review remains required before delivery.
