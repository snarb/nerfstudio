# Marshawn random-background 10k BODY geometry gate

What was tested: frozen random-background continuation checkpoint `5bfc68fe6d7d4be819df4643b55dc291da75d8fa7dc354427ad627cbf9fde907`, saved at the mandatory 10k pause before step10001. Reused exactly the original BODY128 selection SHA `06e131767fb9ecd09b15eb28664a1917b8b1c21d4b976dd288599d5953628cab`: 128 observed rays, 112 unique points, 63 train cameras. Compared the actual native sampler, temporary all-cells sampler, and uniform8192/16384 integrations. No objectives, training callbacks, optimizer or resume were executed.

Results:

| Checkpoint | Native active cells | Known point misses /9022 | BODY misses /495 | Uniform16384 prefix mass >10% /128 | Band mass ≥50% /128 | Median band mass |
|---|---:|---:|---:|---:|---:|---:|
| Cold8k parent | 171709 | 70 | 8 | 0 | 126 | .999641 |
| RandomBG10k | 262882 | 55 | 11 | 0 | 126 | .999700 |
| Safe uniform13k | 162070 | 60 | 6 | 0 | 126 | .999957 |

All native/all-cells profiles also have zero prefix-mass cases above10% and126/128 surface-band cases above50%. The same two preexisting point5512 rays in HA/IB remain outside the narrow surface-band gate across all three checkpoints. Free intervals end before `max(.01,.05*witnessDepth)`; surface bands are `±max(.005,.01*witnessDepth)`. These are prepared-world distances, not declared physical meters.

RandomBG10k dense maximum prefix mass is .0003562, versus .0005547 in cold8k and .0046199 in safe13k. No newly strong BODY foreground termination is measured along this fixed set. Three additional BODY points are skipped relative to cold8k:6583,6963,7956. The native mask remains incomplete for known witness membership, despite coherent rendered crop appearance.

Actual native packed weights were checked against recomputed integration. Original model/EMA/mask hashes were restored and verified. All12 BODY GT/native pairs and12 selected high-sigma missed-point crop pairs were manually inspected. People remain coherent; face/hair/clothing/rig detail is blurred. This is not a clean-floor or complete-geometry acceptance.

Evidence:

- [Matched comparison and checkpoint/source hashes](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/gate10000_BODY128_comparison.json).
- [Actual 10k BODY audit](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/gate10000_body128/audit.json).
- [BODY crop pairs](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/gate10000_body128/body12_contact.jpg).
- [Known skipped-point crop pairs](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/gate10000_body128/active_miss_contact.jpg).
- [Actual manual review receipt](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/gate10000_body128/manual_review_occupancy.json).
- [New matched cold8k reference, written only under the new stage](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_resume8k_randomBG_ALL3_STAGED/cold8000_body128_reference/audit.json).

Insight: BODY128 reveals no new strong foreground mass, but does not cover all mat/floor or hidden volume. The independent nine-ray KA audit finds increased opacity with nearly unchanged deep conditional median on the darkest pixel; therefore opacity repair is not verified geometry repair. The 15-point approximate cloth plane is extrapolated at all nine floor queries. Do not hard-carve or transfer an older grid certificate. No candidate mask or final atlas was generated before winner selection. The accepted ZIP and all old checkpoints remain unchanged.

The observation graph contains real SfM track-camera IDs; cached UV are calibrated projections of triangulated XYZ, not independent raw feature coordinates. Visibility, fitted depth and sparse support remain limitations.
