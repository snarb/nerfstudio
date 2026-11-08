# Final MP4 encoding adapter

The external OPS adapter makes RGB→YCbCr conversion and MP4 color signalling
explicit, while retaining the qualified render shaders and original PNG masters.
It uses the existing working absolute FFmpeg/ffprobe5.0.1 binaries, BT709 matrix
and primaries, limited range and sRGB transfer13. Source PNGs already use the
sRGB transfer; the adapter performs no gamma transform or image resize.
H264 CRF15/yuv420p/24fps remain lossy. PNG masters are authoritative.

The production deployment candidate is
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/video_geometry_cpu_encoding_v4`.
These repository files are byte-identical source archives of that deployment.
The route and qualification scripts resolve paths from the deployed OPS layout;
run the approved deployment, not these archival paths directly.

The adapter preserves original path/PNG completion checks, binds source-receipt
hashes and verifies dimensions. Each MP4 has a source-bound prelaunch intent,
owned temporary filename and completion sidecar. An inherited flock remains
held if the coordinator dies while FFmpeg runs. Restart adopts only the same
source/profile artifact; foreign completed files are rejected. Interrupted own
temporaries are archived. Final verification checks all120frames/5seconds,
24fps,3840×2024, expected color tags and actual RGB decoding through the last
frame. It records lossy RGB parity measurements at first/middle/last frames.

The CPU tests include the exact frozen route's encoder call AST, a clearly
labelled mock120-frame/5-second flow, source changes, foreign existing movies,
four injected coordinator crashes and actual inherited-file-descriptor locking.
The mock metadata does not represent an encoded120-frame movie. A separate
actual3-PNG control was encoded under the real fleet launcher's environment and
matched the original same control byte-for-byte. Tool binaries, resolved shared
libraries and version strings are bound; the missing ROOT release fails closed.

Activation requires the concrete ROOT release and a one-time atomic fleet
posthook change. It does not edit sealed code, training inputs, shader math or
hybrid parameters. See
[`marshawn_video_encoding_color_audit.md`](../../../experiments/marshawn_video_encoding_color_audit.md).
