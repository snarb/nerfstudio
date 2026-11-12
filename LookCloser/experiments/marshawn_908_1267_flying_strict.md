# Marshawn — strict flying camera and native BC slow motion

## What was tested

Inference-only delivery requested after the sparse24 campaign was stopped.
Frozen preferred checkpoints, including the approved repairs, remain unchanged.
No training or dense-frame preparation is restarted.

The video preserves the source action interval908–1267 at120fps, extending the
output to3.5seconds at24fps:70 NeRF images908,913,…,1253 and14 actual original
BC photographs1254,…,1267. The latter play at one fifth speed. No RGB temporal
interpolation is used. The calibrated BC camera pose and intrinsics are reached
at output index69; index70 begins the0.5833-second slow-motion section.

The final trajectory uses shape-preserving spherical cubic splines through train-camera centers
GC→FB→EB→DD→CD→BC and a rotation spline through their orientations. Its horizontal
arc is70.04degrees. A cosine time easing stops camera motion at the endpoint.
Focal length/principal point smoothly move from the ensemble framing to a medium
BC crop. The first attempted126-degree rear/upper route had large view-dependent
holes and was rejected after inspecting eight pilot renders.

Strict rendering recomputes its opaque connected support at each novel camera.
Anchors require alpha>0.98,10,000pixels and120pixels height. Detached components
whose median3D distance from the same frame's GC foreground surface exceeds
0.35m are rejected. This is a component check, not projection of a GC silhouette:
connected occluded surfaces are not individually discarded. The existing
semantic admission remains active. Sample density is gated before integration
within0.15m of support depth, with0.10m falloff, and3+5pixel image-space support.
The GC-specific partial-actor border exception is not applied to moving views.

BC source crops come directly from the16-bit native PQ images, using the same
fixed decode/display curve as preparation. There is no preliminary source RGB
resize. The animated window changes2074×1167→1920×1080 and moves upward140native
pixels, giving approximately8% zoom. A variable window in a fixed1920×1080video
requires one output sampling operation; this uses Lanczos, ending with an exact
1:1crop. The source image itself is never globally downsampled.

The photographic tail is explicitly GT, not neural output. Its alpha comes from
the original8K SAM soft mask, stabilized with a five-frame motion-aligned median
and a2pixel bidirectional-flow consistency gate. RGB remains the actual source
photograph. No AABB clipping is applied to this tail. This is a different matte
source from the NeRF section, which does not apply SAM to output RGB.

## Results and validation

Delivery files are on silver-sable under:
`/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/flying_strict_low_20261112/`.

- `path.json`: every calibrated pose, intrinsics, source time and native crop.
- `neural/NNN/`: raw/strict images, depth/opacity arrays and checkpoint receipts.
- `gt_sources/`: native overscan source crops, original alpha and extraction provenance.
- `gt/`: motion-stabilized native photograph crops, plus an index69 endpoint reference.
- `delivery_receipt.json`: image/video hashes, frame count, codec and timing.
- `validation.json`: source timeline, proper rotations, native crop ray parity,
  exact pose/intrinsics at the splice, and checksum verification of all70 used models.
- `all_frames_contact.jpg`, `hair_temporal_contact.jpg`, `pilot_contact.jpg`: review evidence.

No held-out quality metric is claimed for novel views or the mixed GT tail.
The second pilot set removed the obvious rear-angle holes. Full-frame review
found one remaining detached island at1178; reducing component support distance
from0.45m to0.35m rejected it. The final path descends through DD/CD before
returning to BC: a matched comparison at1198–1228 made the top AABB hair cut less
conspicuous than the higher DB/CC route. It does not restore missing hair.
The final shot retains headroom above the dreadlocks. Some pre-existing
reconstruction defects on peripheral actors/limbs and thin colored contours can
remain; this is inference cleanup, not a recovered perfect semantic surface.

## Reproduction

Campaign tools: `flying_path.py`, `flying_render.py`, `flying_gt.py`,
`flying_delivery.py`, `flying_encode.py`, `validate_flying_delivery.py`.
`flying_extract_gt.py` records the original WS08 native extraction operation.
Prepared source crops are preserved on Silver, so replay does not require WS08.

After `path.json` and `gt_sources` are present, run with the campaign Python:

```bash
python tools/flying_delivery.py --root "$CAMPAIGN" --output "$OUTPUT"
python tools/flying_encode.py --output "$OUTPUT"
python tools/validate_flying_delivery.py --root "$CAMPAIGN" --output "$OUTPUT"
```

The batch runner locks its output, uses eight GPUs, retries failed subprocesses
once, validates path/renderer hashes before reusing frames, and refuses assembly
on incomplete output. Detach with stdout/stderr redirected for crash durability.
The H.264 encoder uses `imageio-ffmpeg` installed only in the output's `codec/`
directory; the neural environment is not changed. Training STOP markers remain.

Final video: [Google Drive](https://drive.google.com/file/d/1kA9tJue9g9nbU0xCFHEtBmth5ZIkXvdq/view).
H.264/CRF16/yuv420p,84decodedframes,1920×1080,24fps,3.5seconds. SHA-256:
`f669863889b2a51990a4c340e57c25338b901b5acdc97a3a7261fe612c15be07`.
All84frames were reviewed as contact sheets; consecutive14GT hair close-ups and
matched high/low camera crops were inspected. This is not a claim of flawless
matting or a real-time playback review. Seventy weight hashes were checked in
the preceding same-session high-path audit; the low-path validator explicitly
references that audit instead of pretending to rehash unchanged weights again.

A separate user-requested MatAnyone2 test runs in an isolated environment. It
is not applied to this delivery without a quality gate; initial tests improved
the outer halo but retained foreground-labelled background in a hair gap.
