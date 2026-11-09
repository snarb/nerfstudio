# Finite floor-prior 12k BODY preservation gate

What was tested: exact checkpoint `1aaa4bf246c39a9c83bec08cb8802fc2ef53f21a3dd172153fb85d31c63f0295` after the mandatory12k pause, before12001. Reused the original128/112point/63train-camera BODY selection SHA `06e131767fb9ecd09b15eb28664a1917b8b1c21d4b976dd288599d5953628cab`. No training, optimizer, objective, callback or production resume occurred in this audit.

Results:

| Checkpoint | Native core cells | Known point misses /9022 | BODY misses /495 | Dense prefix mass >10% /128 | Dense band mass ≥50% /128 | Dense median band mass |
|---|---:|---:|---:|---:|---:|---:|
| Cold8k | 171709 | 70 | 8 | 0 | 126 | .999641 |
| RandomBG10k parent | 262882 | 55 | 11 | 0 | 126 | .999700 |
| Finite floor-prior12k | 283268 | 55 | 7 | 0 | 126 | .999607 |
| Safe uniform13k | 162070 | 60 | 6 | 0 | 126 | .999957 |

Dense denotes16384 uniform samples. Actual native, temporary all-cells, and8192sample profiles also retain0/128 prefix cases over10% and126/128 band cases over50%. Same historical point5512 HA/IB band exceptions remain. New dense maximum prefix mass is .0003503 versus parent .0003562. No new strong BODY foreground termination was measured.

Exactly one diagnostic3×3×3 maximum filter of the authoritative12k GPU core admits982972 cells. It leaves1/9022 known miss, point2813, and0/495 BODY misses. This mask was never installed. No EMA-threshold reconstruction or old-checkpoint mask was used. Native7 BODY misses are a subset of parent11:5921,5922,7635,8496 were resolved; no new BODY skipped-point IDs appeared. Total55-point counts can contain different identities across checkpoints.

The exact selection, XYZ/camera/UV geometry, observation/point source hashes, normalization, live bounds and native model/field/sampler source hashes agree with all three references. Frozen model/EMA/mask hashes are unchanged after actual packed-weight tracing. The reader exited0.

All12 BODY GT/native crop pairs and12 selected high-sigma missed-point crop pairs were actually viewed. People remain coherent; face/hair/text/clothing/rig detail is soft. No broad foreground veil or obvious BODY destruction was seen in these selected patches. That does not certify mat/floor cleanliness, hidden geometry or the assumed planar floor band's physical depth.

Evidence:

- [Matched comparison and source consistency](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_BODY128_comparison.json).
- [Actual BODY128 traces and source receipt](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_body128/audit.json).
- [Native and one-cell membership diagnostic](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_grid_membership_diagnostic.json).
- [BODY GT/native contact](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_body128/body12_contact.jpg).
- [Known-miss GT/native contact](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_body128/active_miss_contact.jpg).
- [Actual manual review receipt](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/gate12000_body128/manual_review_occupancy.json).

Insight: the fixed BODY probe shows preservation under this small floor-prior trial. It is scoped fitted train-ray evidence, not independent depth GT or a whole-voxel occupancy certificate. Independent nine-floor-ray geometry and actual HD floor crops remain necessary before choosing a winner. No final candidate mask or atlas was generated; accepted artifacts remain unchanged.


### Normal-RGB-head auxiliary16k gate

Exact16k checkpoint `bc20f3158409ec711c08aff2e5afa5f871757f781d1a5a20fa4637fb0fc53d27`: SAME128 visible-body rays, native/allcells/uniform8192/uniform16384 all0/128 prefix mass>10%,126/128 surface-band mass≥50%. Native known cell misses65/9022 (parent14k68), body11/495 unchanged. All12 body GT/pred panels were manually inspected: coherent bodies with expected fine-detail blur, no new gross foreground fog on this train-witness subset. Sidecars were byteexact aliases of the authoritative export written under the parent directory by the inherited artifact-path configuration; source policy and alias receipt preserve that history. No running config was changed. Evidence: `native_png_floor_normalRGB_HEADONLY_STAGED/gate16000_body128/{audit.json,comparison14k.json,manual_review_occupancy.json}`. Known points/rays do not establish hidden geometry or floor cleanliness; no guard certificate transfers from old weights.
