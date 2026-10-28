# Luster fly-through: depth-based defocus

**Historical recipe, superseded for boundary-safe defocus:** mixed RGB/depth can
retain bright/dark rims at partial coverage. Use the [validated layered DoF
fix](layered_dof_boundary_fix.md) when clean foreground/background layers are
available. The export instructions below remain applicable.

## What was tested

Separate version of the existing 314-frame, 24fps, 1500×2048 hybrid video.
Script: `scripts/render_luster_depth_dof.py`.
Output directory: `/fsx/tmp/luster_gt_flythrough_20261022_v11_dof`.

The DoF pass does not read any silhouette masks. Input is v4 unblurred RGB,
plus the v8-minus-v9 hybrid appearance residual in flight frames. Existing
flight synthesis/transfer upstream used alpha/contour support; this experiment
replaces **defocus**, not those upstream reconstruction methods.

All-frame relative disparity estimated with [Depth Anything V2 Large](https://github.com/DepthAnything/Depth-Anything-V2), input short side756.
It is not calibrated metric depth or a physical aperture simulation. No training.
Focus reference is the farthest median from five narrow central body rectangles;
focus and depth scale are smoothed over time. Adjacent normalized disparity maps
are aligned by DIS flow and blended conservatively only where depths agree.
A broad asymmetric depth interval preserves the girl's depth extent. Six blur
levels (Gaussian sigma2–24px) interpolate continuously according to disparity.
Normalized convolution of defocused contributors reduces foreground color bleed.
No semantic mask, contour dilation or hand-edited silhouette is used by this pass.

## Results

Initial narrow symmetric focus band rejected: it blurred hands, belt and hair,
especially on close-ups and back views. Widened near focus tolerance and selected
farther body reference using multiple interior rectangles instead of one torso box.
Final MP4 fully decoded:314 frames,24fps,1500×2048,13.083s. Sampled overview reviewed throughout each segment; native comparison patches33/194/277 and12 encoded samples checked. No gross added silhouette loss in inspected samples. This is not every-frame native review or playback. Background blur varies by estimated depth; floor artifacts from upstream synthesis remain visible near focus.

Video: `/fsx/tmp/luster_gt_flythrough_20261022_v11_dof/luster_hybrid_depth_dof.mp4`. Original version untouched.

## Insights / limits

Depth-based blur naturally leaves the floor near the girl sharper than the far
background. Same-depth objects cannot be excluded semantically. Monocular depth
can make edge errors; this avoids old mask errors but does not guarantee perfect
hair strands. Cached depth and inputs remain available for reproducible changes.


ProRes master: `luster_hybrid_depth_dof_ProRes4444XQ.mov` in the same output directory. Encoded directly from8-bit PNGs, prores_ks profile5,10-bit4:4:4 encoder input, decoded as12-bit4:4:4, SDR BT.709.1500x2048/24fps/314frames,692355865bytes. Higher-depth container does not create extra source dynamic range. Full decode passed. Explicit BT.709 roundtrip frame194: {'psnr': 57.01538257209373, 'mae': 0.09294986979166667}.

## Frozen reproduction settings (2026-10-22)

This is the user-approved **Luster presentation recipe**, not a new training
baseline or a measured improvement in NeRF reconstruction metrics.
The script is scene-specific: `R`, `RAW`, `H`, `B`, the 314-frame count and focus
rectangles are fixed to this sequence; adapt them and visually validate focus
before applying it to another dataset. It does not generate the upstream inputs.

| Setting | Accepted value |
|---|---|
| Depth model | Depth Anything V2 Large, `vitl`, features256, channels256/512/1024/1024 |
| Upstream code revision | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| Weight SHA256 | `a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345` |
| Inference | CUDA, float32, input size756, cached float16 disparity |
| Focus samples | x48–53% of width; y25/32/50/60/75% of height, each ±20px |
| Focus reference | Minimum of the five median disparities (farthest sampled depth) |
| Depth scale | Whole-frame disparity P95−P5 |
| Temporal focus/scale | Gaussian sigma2 frames |
| Temporal depth | DIS medium at375×512; adjacent frames warped to current; weight0.3×clip(1−depth disagreement/0.10,0,1) |
| Normalized disparity | `q = (disparity − focus) / scale` |
| In-focus interval | `−0.12 ≤ q ≤ 0.65` |
| Blur sigma per pixel | `24 × clip(max(−q−0.12, q−0.65)/0.45, 0, 1)` |
| Blur pyramid | Gaussian sigma2/4/8/12/18/24px, continuous interpolation |
| Defocused contributors | `clip(max(−q−0.07, q−0.60)/0.16,0,1)`; normalized weighted convolution |
| Frame output | RGB PNG,1500×2048,24fps |

### Inputs and execution

Existing input directories, preserved read-only by this pass:

- `RAW`: `/fsx/tmp/luster_gt_flythrough_20261021_v4_no_blur` (plan and unblurred frames).
- `H`: `/fsx/tmp/luster_gt_flythrough_20261021_v8_transfer` (completed hybrid frames).
- `B`: `/fsx/tmp/luster_gt_flythrough_20261021_v9_hair` (matching pre-transfer frames).
- In flights, source RGB is `RAW + (H − B)`; GT stops use RAW directly.
- Model checkout: `R/runtime/Depth-Anything-V2`; weight file: `R/runtime/depth_anything_v2_vitl.pth`.
- `R`: `/fsx/tmp/luster_gt_flythrough_20261022_v11_dof`.

Model source and weights are available from the upstream link above and
`https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth`.
The verified local Python environment provides PyTorch2.7.1+cu128, torchvision,
OpenCV, NumPy, SciPy and Pillow; the model code is imported from the checkout.

From `LookCloser/` on olive-lark:

```bash
/home/pavlo/.conda/envs/nerfstudio/bin/python -u scripts/render_luster_depth_dof.py \
  > /fsx/tmp/luster_gt_flythrough_20261022_v11_dof/run.log 2>&1
```

This reuses cached depth but **rewrites rendered frames and MP4**; copy/change `R`
for a new variant and do not run concurrently against the same output directory.
`--pilot` renders 11 selected frames without temporal stabilization, so use it only
for initial focus checks; final acceptance must use the full run.

### Professional master export

Encode directly from PNGs, never from the delivery MP4:

```bash
R=/fsx/tmp/luster_gt_flythrough_20261022_v11_dof
FF=/fsx/tmp/luster_gt_flythrough_20261021_v4_no_blur/runtime/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
"$FF" -hide_banner -n -framerate 24 -start_number 0 -i "$R/frames/%05d.png" \
  -frames:v 314 \
  -vf 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv444p10le' \
  -c:v prores_ks -profile:v 5 -vendor apl0 -tag:v ap4x -alpha_bits 0 -threads 8 \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
  -movflags +faststart -an "$R/luster_hybrid_depth_dof_ProRes4444XQ.mov"
```

`-n` preserves an existing master. Validate full decoding and inspect final encoded
samples, particularly hands, hair, jewelry and transitions between GT and flights.
For numeric roundtrip checks, explicitly decode the BT.709 matrix/range: the local
OpenCV decoder gave a different RGB conversion and misleadingly lower PSNR.
The ProRes container does not add HDR or recover precision absent from8-bit PNGs.

Evidence: `R/validation.json`, `R/prores_validation.json`, `R/review/encoded_samples.jpg`,
`R/prores_decode.log`; all314 master frames decoded successfully.
