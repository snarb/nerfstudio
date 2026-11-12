# Native-camera detail transfer for the accepted Marshawn flight

## What was tested

The accepted union-interior video gains a separate 3840×2160 hybrid version.
The 70 neural instants and camera trajectory are unchanged. Field RGB and
conditional termination q10/q50/q90 are evaluated at actual 4K camera rays.
Source RGB comes from same-instant, original-pixel-size PNG crops retained for
the selected training recipe. These are not enlarged HD textures. Original
PQ16 was converted once by the campaign's fixed exposure/tone curve; this
adapter does not claim unquantized RAW donor colors.

The existing qualified implementation is reused from
`marshawn_video_1660_2015_24fps_20261107/video/reference/legacy/campaign/misha50_hybrid.py`
with its unchanged `transfer_luster_hybrid` visibility and detail helpers.
Recipe: 16 nearest TRAIN cameras, one winning donor per pixel, detail sigma 4,
gain 0.75, target pixel scale 2 (effective sigma 8 at 4K). Conditional depth
intervals, source alpha, depth agreement, roundtrip, angular resolution and
photometric confidence retain neural fallback when no donor qualifies.

Adapters bind current frame/checkpoint, normalized calibrated camera poses,
and current native PNG crops. Donor sampling uses four original-resolution
pixel taps in display-linear light. Geometry can use a smaller source raster;
RGB never passes through that raster. Hashes verify RGB and semantic inputs.
Current-sequence source eligibility uses reviewed production subject opaque
cores, valid pixels and any explicit veto arrays. These masks are not the old
1660 take's independently annotated UNKNOWN maps; that difference is explicit.

## Silhouette and photographic tail

The previous accepted contour suppression is lifted to 4K and applied to native
4K field RGB. Hybrid details are restricted to the unchanged opaque interior;
nonqualifying pixels remain exactly the 4K neural baseline. Thus detail transfer
does not reintroduce the destructive depth gate at bat/hat/body overlaps.
The contour guide is still HD-derived; native 4K matting is not claimed.

The last 14 photographic frames retain their original masks, crops and motion.
Their finished HD masters are untouched. Only a 2× Lanczos4 presentation resize
is applied for the common 4K video canvas; no additional photographic detail
or reframing is claimed for that tail. Both original and resized hashes are
recorded in `tail_identity.json`.

## Validation and delivery

Native sampling tests check exact pixel-center mapping through a reduced
geometry raster, invalid tap rejection, and display-linear interpolation.
Per-frame assertions enforce unchanged exterior and exact neural fallback.
The detached eight-GPU controller is restartable, validates finished PNG hashes,
retries failed frames and encodes only after all frames exist.

Pending: 4K pilot visual review and full sequence delivery. No training or model
weights are modified by these inference jobs.

### Pilot gate

Both native-sampling tests passed. Actual 4K output indices 5 and 50 completed
in 120.4s and 112.9s respectively, reusing source geometry from the preceding HD
pilot. Qualified transfer covered 64.3% and 45.0% of target-valid foreground.
Native-pixel comparisons show subtle added cloth/face detail and no new bat or
hat/body holes in the inspected patches. Pilots passed visual review; the full
sequence is running under the restartable controller. These are coverage values,
not reference-image quality metrics or a claim of universal artifact removal.
