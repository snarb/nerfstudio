# Marshawn1660 camera-adjacent density

## What was tested

The canonical linear AP1 HDR4k model had a broad brown veil in held-out DC. Three pixels were declared from its render before querying native packed density/RGB. Same-sample partial-cell integration removed near prefixes analytically; no color was fitted. A separate full-HD control changed only the inference near plane from0.01 to0.052world (14 native0.003 steps).

## Results

At4k, the three DC rays terminate primarily at0.0115world. Prefix mass before0.05 is90.67–99.43%. At pixel1700,450, removing the prefix changes physical AP1 RGB from[.02588,.02116,.01396] to[.01157,.01054,.00918], versus source[.01173,.01085,.00841]. At2k, the same rays had much less prefix mass; this camera-adjacent opacity explains part of the worsening evaluation despite stable train-witness geometry.

| Same4kcheckpoint / inference | DC SDR PSNR | Mean SDR PSNR | Mean SDR SSIM | Mean SDR LPIPS |
| --- | ---: | ---: | ---: | ---: |
| near0.01 |22.5463|26.0606|.763444|.352099|
| near0.052 |27.5348|27.7135|.775720|.321434|

Both rows use the same canonical GT and fixed display transform. Root visually inspected the full DC image and three-view contact: the large brown veil disappears; background tile holes and floor mottling remain. This is not an accepted final reconstruction.

All44938 actual TRAIN observation pairs are farther than0.42585856world; none is closer than0.1. No9022-point witness projects inside a DC/HC/KA front frustum within0.1world. These are known-point checks, not a certificate of unseen empty space. Four TRAIN source photos place all three near-density samples at the black camera/rig assembly around the DC optical center; closest witness8063 is tracked by LC/MB/MC. Exact physical self-occlusion cannot be certified by projection alone.

[Native ray evidence](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/geometry4000_BODY_DC/DC_three_native_prefix.json), [source-photo audit](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/DC_near_source_geometry/audit.json), [root image review](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/root_near0052_visual_review.json), [near-control renders](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/gate4000_near0052_HD_root/all3_common_GT_HDR_safe13k.jpg).

## Insights

The earlier native-PNG KA prefix test was secondary and does not generalize to this DC/HDR failure. Camera/rig geometry can be legitimate scene content while its learned density corrupts rays starting at that camera. A grid guard cannot by itself fix this visibility problem. A bounded continuation with corrected train/eval near distance is being prepared from the complete4k state; include the old4041-attempt upper bound in its22000-attempt family cap. Close camera paths require their own clearance check.
