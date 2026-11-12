# Marshawn sparse24: inference-only depth and opacity cleanup

## What was tested

The user requested a new GC video with black outside the actors, including
hair, without restarting training. The original72 selected checkpoints
(908,913,…,1263), repaired923/988 included, remain unchanged.

Pilots:908,918,923,928,988,1098,1188,1263. Compared a depth-only band,
loose/balanced/strict depth-plus-opacity support, and original native RGB crops
for hair/prop review. No new SAM segmentation or SAM silhouette is applied to
the rendered image. Existing training-time semantic admission stays active.

The renderer first saves raw RGB, expected ray depth and accumulated opacity.
Connected regions with opacity>0.98,area>=10000 full-HD pixels and vertical
extent>=120px provide
surface anchors. Nearest-anchor depth defines a per-ray band:15cm on either
side with an additional10cm smooth falloff. A field forward hook attenuates
sample density **before** recomputing transmittance and integrating RGB.

Depth alone leaves considerable contour debris. The combined method also
uses distance from the opaque field anchors:soft=8px plus6px fade;
strict=3px plus5px fade. It is therefore a **view-conditioned depth and opacity
heuristic**, not a pure depth threshold or a recovered semantic3D surface.
It should not be reused for another camera without recomputing its guides.
GC-specific border handling retains components>=200px touching the right
edge whose entire bounding box is in the lower30% of the image, preserving
the fourth actor as he leaves the canvas. The original2000px cutoff admitted
detached overhead islands in1118–1183; the complete sequence review caught
this and v2 uses the larger threshold. Three larger flat islands in1163/1168/
1173 required the additional vertical-extent check in final v3. This explicit
GC-specific heuristic distinguishes the observed flat islands from the
much taller actor components; it is not a general semantic classifier.
Versions are retained on disk, with separate support guides and receipts.

For temporal stability, forward/backward checked optical flow aligns the
field-derived signed support distance from adjacent frames. A three-frame
median changes that distance by at most2px; flow consistency must be<2px and
depth disagreement<20cm. RGB is never temporally averaged. This does not
guarantee perfectly stable hair from independently trained frame models.

## Results

Pilot inspection:depth-only leaves halos; combined cleanup removes detached
patches and substantially reduces contour haze. The bat in923 remains
continuous. Reviewed hair detail in1188 against the native RGB crop. The
strict option trades some faint edge detail for a cleaner silhouette; soft
is retained as an alternative. Final v3 delivers72 frames,24fps,3seconds;
both individual videos are1920×1080 and the side-by-side comparison1920×540.
All three MP4s decoded through all72 frames. All raw/clean tensors are finite.
The72 raw rerenders match the prior delivery within at most one8-bit code
value, confirming reload parity including the restored1108 archive.
Reviewed the final contact sheet and consecutive crops918–943,1168–1193,
1183–1208. The detached overhead islands are removed in the reviewed final
sequence; fine warm contour rims and some edge noise remain. This is a
substantial cleanup, not a perfect foreground matte.

Final links, uploaded and metadata-verified on the user's existing private Drive:

- [Strict cleanup](https://drive.google.com/file/d/15V-WUiZyW894oisYI2v-f03ciHZ1sxo7/view)
- [Softer cleanup](https://drive.google.com/file/d/1VI8j_nOWsuOQUhqZYMorix0-qeH1gfzt/view)
- [Before/after](https://drive.google.com/file/d/1OBMX7olpLLZh6J2KTumL9fuY0ul_OxLz/view)

Evidence:`marshawn_908_1268_diagnostics/depth_cleanup_20261112/` contains
`delivery_receipt_v3.json`,`validation_v3.json`, support/reuse receipts and
review images. V3 reused69 renders whose guide arrays exactly matched v2;
only1163/1168/1173 required another render. Source checkpoints remain intact.

Four CPU unit tests cover:calibrated front/back/out-of-frame sample gating and
reversible hook removal with unchanged sample RGB; isolated opaque-component
rejection with depth inherited from a connected body anchor; preserving the
partial lower-right actor while rejecting an overhead island and an unrelated
upper-right border component; rejecting a large flat island that passes the
area threshold.

Silver output directory:
`/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/depth_cleanup_20261112/`

Reproduction (inference only,8GPU, restartable):

```bash
ROOT=/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109
/home/ubuntu/lookcloser/envs/marshawn_leader/bin/python \
  "$ROOT/tools/run_depth_cleanup_delivery.py" --root "$ROOT" \
  --output "$ROOT/operations/depth_cleanup_20261112"
```

## Insights and limitations

Expected depth can mix actor and background contributions. A depth band
cannot identify background density that is at the actor's distance; connected
opacity support is necessary for the visible improvement here. Very faint
or detached hair can be suppressed, especially in the strict result. The
method is useful cleanup, not ground-truth alpha matting or a guarantee that
all remaining edge colour is foreground.

Frame1108 archive repair:its native archive contained zero-length placeholders
where symbolic links should have been, plus a truncated confidence semantic
file`EA.npz`(5,085,184 bytes instead of6,086,953). Restored that file from WS07
after matching its SHA to the original admission manifest; reconstructed
links from the original local frame tree, retaining placeholders as backups.
All72 confidence-semantic hashes checked before the repair was applied.
Receipt:`external_models/ws07/001108/render_support_repair_20261112.json`.
Checkpoint weights and admission grids were not changed.
