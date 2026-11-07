# Earlier DE source coverage and actual image effect

What was tested: a read-only native-photo audit of DE at1665,1670,1675,1680 and
1685, prompted by the later1690 disconnected-fragment omission. Current native
actor A, hair H/ROI proofs and frozen source photographs were inspected. No
source, mask, model, AABB or training parameters were changed.

| Source frame | Manual native-photo finding |
|---|---|
|1665|Rear black boot substantially present in A; lower grey face/hand mostly hidden, so no missing-face claim.|
|1670|Visible grey hip/trouser patches omitted from A; rear black boot substantially present.|
|1675|Exposed curled hand/fist and adjacent dim actor fragment patchy in A. A separate grey face is not clearly visible.|
|1680|Upper dim fragment partly omitted; lower extended hand mostly present; rear black boot largely present.|
|1685|Exposed upper head/face region and parts of lower hand omitted; rear boot toe/edge also partly absent.|

The1675/1680/1685 current DE H crops target upper black hair and do not reach
these lower fragments. Some A-missing skin/finger pixels are UNKNOWN; therefore
A incompleteness alone is not evidence that they voted geometry clear. Contour
checks against the same incomplete A also cannot establish complete source
coverage. The1660 canonical native DE input was not available in this namespace;
it was not reconstructed for this finite audit.

[Five-frame native detail contact](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/DE_previous_source_readonly/DE1665_1685_native_detail.jpg).
The source-bound corrected wording proof is
`operations/DE_previous_source_readonly/proof_v2.json` under that campaign,
SHA `40e283de3298c0751d275188578b53a625a98707b3878a07a0c72bb04bb4b261`.
It binds32 immutable source inputs. The earlier proof is preserved; its1675
face wording was corrected after closer photographic inspection.

One bounded actual-image test then checked selected1675 checkpoint
`c4aee93b5c460990a9b31257e55e57256f99444d10500a7590bd85a15e0b4318`,
step139999 /140000 updates. The private SG copy and1334 scientific data files
were byte-verified against actual production C inputs. The original loader's
FR was0.3; HC archive parity passed with maximum1 RGB code difference. Pipeline
tensor SHA was identical before and after HC plus actual TRAIN-camera DE query.
A natural SG7 lease was released immediately afterward; olive/dev3 jobs were
not interrupted.

The actual curled hand remains visibly reconstructed and opaque despite small
A gaps. Those gaps retain positive RGB supervision through W64: photographic
hand pixel(940,822) has A=false, W64=0.96076 and rendered alpha0.99999934;
(945,817) has A=false, W64=0.87066 and alpha0.99999970. GT/neural skin RGB is
close at these examples. This establishes that missing native A is not the
same as missing RGB supervision. The test does not establish a source-caused
hand geometry hole, and does not justify retraining the earlier models.
Known floor fragments and background shell remain visible.

[Actual GT, neural, alpha and frozen-A contact](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/DE1675_image_effect_readonly/v2/DE1675_GT_neural_alpha_frozenA_difference_HD.png)
and [hand pixels, nearest4x preview](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/DE1675_image_effect_readonly/v2/DE1675_curled_hand_HD_nearest4x.png).
These are diagnostic crops from the single original HD-camera render, not new
camera queries or high-resolution reconstructions. Exact source/checkpoint,
GPU identity, parity, tensor hashes and pixel measurements are preserved in
`operations/DE1675_image_effect_readonly/v2/{complete,hand_effect_review}.json`.
The current production R0/hybrid profile and72-frame workflow are unchanged.

A final CPU point check read actual frozen C1680/C1685 W32/W64. Native points
were mapped with the frozen OPENCV calibration and independently roundtripped
(maximum2.1e-8 native pixels), then the nearest actual HD pixel was read:

| Frame / visible part | Native RAW x,y | HD x,y | W32 | W64 | Semantic / source UNKNOWN |
|---|---|---|---:|---:|---|
|1680 upper dim fragment, anatomy uncertain|4030,3530|944,828|0.18110|0.46808|0 /0|
|1685 upper head/face region|4050,3540|949,831|0.73723|0.86013|1 /1|
|1685 lower-hand finger|3980,3740|932,878|0.79234|0.88999|1 /1|
|1685 black rear-boot toe|3990,3970|935,933|0|0.05679|0 /0|

All four examples are A-missing in the native archive and actual HD core, but
retain positive W64. The1685 toe has zero W32, so supervision differs across
these two weight maps; it does not have zero W64. Semantic/source UNKNOWN and
RGB weight are separate fields: UNKNOWN1 does not imply W64=0. The table reads
base semantic UNKNOWN, not the extra MAT/hair geometry layers; UNKNOWN0 alone
is not proof a cell was cleared. These finite points do not establish complete
actor-mask recall or quality at other pixels.

[Bound native/HD photographs and actual W32/W64 fields](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/DE_previous_source_readonly/1680_1685_actual_C_W32_W64_native_point_contact.png).
The exact hashes, coordinates, source flags and values are saved in
`operations/DE_previous_source_readonly/actual_C_weight_points.json`.
No additional GPU query, early-source correction or model redo followed this
source audit. The72-frame production workflow remains the priority.
