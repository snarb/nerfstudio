# Previous Marshawn flight: lying-actor spots and body halos

## What was tested

The user rejected the wider orbit and requested the previous70-degree low flight,
with the lying actor's attached background spots removed and safer body cleanup.
The prior14-frame native BC/MatAnyone2 photographic tail stays unchanged.

The original depth/opacity cleanup accepts large opaque connected regions. Some
floor/background residue is connected to a body and therefore passes that test.
It cannot reliably distinguish real clothing from an attached reconstructed spot.

An independent agent tested temporal MatAnyone2 on70raw neural views, then
bounded its reductions on strict RGB. Unrestricted use removes the bat and can
reintroduce background/hair from raw RGB. The selected body variant therefore
uses only `strictRGB * factor<=1`:4–8px upper-body boundary,8–12px lower-body
boundary, explicit temporally interpolated head/hair/prop protection, and
unchanged deep body interiors. See `marshawn_body_matte_cleanup.md`.

The parent tested masks from12nearby and24surrounding cameras on native8K
silhouettes at frame1083. Near-camera voting scarcely removed the leg ghosts;
surrounding-camera voting also rejected true head/prop pixels. Those prototypes
are rejected and NOT applied to the delivered video. Expected-depth projection
and imperfect silhouettes are insufficient evidence for destructive exclusion.
`lying_multiview_probe.py` records the nearby-camera diagnostic; the ring variant
and evidence remain in Silver `operations/lying_cleanup_20261112/`.

For the lying actor, a feathered time-varying region instead uses temporal Mat2
alpha intersected with existing strict opacity:
`strictRGB * min(1,Mat2alpha/max(strictOpacity,epsilon))`.
This removes attached ghosts without bringing raw background back between legs.
The tested sequence preserves major limb/hand/boot anatomy in reviewed native
patches; subtle edge changes cannot be equated with verified ground-truth alpha.
The region fades outside the frame as the shot approaches the standing actors.

## Results and validation

The combined factor is the minimum of the independently reviewed body and lying
factors. Explicit head/hair/prop guard rectangles override both. Every output
pixel is <=its previous strict value; every pixel in the explicit guards is
identical. These invariants are checked on all70frames. No model is retrained.
The camera manifest is byte-identical to the previous low flight. The14GTtail
PNGs are reused unchanged with checksums. Encoder verifies84decoded frames,
1920×1080,24fps,3.5seconds.

Reviewed lying patches: indices000,010,020,030,035,040,045; upper-body native
patches035and060. Ghosts around the lying shoulder/knee/boots are markedly
reduced. Body contour cleanup is strongest on legs and more conservative on
sleeves/torso. Thin ambiguous illuminated edges are deliberately not globally
eroded. This is not a claim of perfect semantic alpha or recovery of missing
geometry. No PSNR/SSIM/LPIPS is claimed without a trusted alpha/reference image.

## Reproduction and artifacts

Silver campaign root:
`/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109`.

- `operations/body_matte_cleanup_20261112/hybrid_body/factor/`:70uint16 body factors.
- `operations/lying_cleanup_20261112/matte/`:70lying factors, previews, receipts.
- `operations/flying_previous_bodyclean_20261112/`:combined PNGs, contacts,
  `cleanup_receipt.json`, `delivery_receipt.json`, and full video.
- `scripts/marshawn_sequence/lying_matte_cleanup.py`:lying factor generation.
- `scripts/marshawn_sequence/assemble_body_cleanup.py`:guarded combination,
  invariant checks and encoding.

Source renders, raw images, models and training STOP markers are unchanged.

Delivered video: https://drive.google.com/file/d/17AgEpQ2mYPudfecFmHJBS8OuIdqd4Gqt/view .
Before/after1083: https://drive.google.com/file/d/1T7fZR3IdkKf3sgboT7JsZIivzoG5vPrT/view .
SHA256:`e35a7a290b9f1a3475f3a0b823c184c217e4210523610c723e1f2cf6c135b219`.
All84frames were inspected in contact sheets; native body/lying patches reviewed
separately. No assertion of pixel-perfect semantic segmentation is made.
