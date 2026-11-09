# Radial-joint R0 step2000 occupancy diagnosis

What was tested: immutable checkpoint `f2765413898ecbf25cc4b40b7338a99ebfa8f39d7dbf8af6a2a241d27e510e87`; actual exportedGPUcore, exact one-cell halo, known SfM/body/MVS point membership and selected actual renderer ray profiles. No worker mask or NN was changed.

| Evidence | Default core | Exactly ONE halo |
|---|---:|---:|
| Admitted cells |979978|1958140|
|9022 known-point cell misses |24|1(p2813)|
|495 unique BODY cell misses |0|0|
|52 photometric MVS candidate cell misses |0|0|

Same BODY128 selection: native/allcells/uniform8192/uniform16384 all0/128 prefix termination mass>10%,126/128 surface-band mass≥50%. Actual12train GT/pred contact was viewed; bodies coherent with expected2k blur.43 native-admitted cells are touched by the128 known visible-ray free prefixes; median/max integrated optical thickness3.37e−10/.000485. These are traversal-overhead candidates, not proof that whole voxels are physically empty: a cell may contain another surface off the sampled line.

Actual frozen floor rays KAq0=(205,735),q7=(242,747) are opaque≈1. Default→onehalo maximum RGB changes1.81e−5/2.06e−5, less than.006 of one8bit code. Median depths remain1.08250/1.06150world. Two actor rays change≤.000167RGB. Dense16k/32k references, rawσ/T/weights/RGB and actual cell IDs are preserved. No physical plane depth is assigned to these floor eval rays.

All24XYZ slice pages per policy were exported with authoritativecore/guard labels;8 target-region pages were manually inspected (Z16–47,X48–63,Y80–95 for each policy). Core has broad slabs/scattered admission at2k; onehalo fills most of the box. Unobserved labels remain unknown, not physical occupied/free truth. This targeted2k review is not a full manual release review of every atlas page.

Point2813 is visible in actual TRAIN DE/EE/FE/GE source photos as a thin vertical metal-pole silhouette at a horizontal cardboard-seam occlusion junction. All4context/detail panels were manually inspected. Nearest cached detected SIFT positions lie.625/.245/.132/.199px from the fittedXYZ reprojections, but original feature IDs were discarded; this is not independent reprojection/depth validation. Real hardware is supported, while exact3D boundary localization remains uncertain. A protection candidate may be tested conservatively later, without declaring the point physical ground truth. No point protection was applied.

Artifacts under `/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/`: `gate2000_grid_CPU/` with source-bound membership/atlas/free-prefix reports; `gate2000_BODY128/`; `gate2000_grid_target_rays/`; `p2813_train_source_review/`. Aggregate provenance `gate2000_grid_CPU/review_bindings.json`. These2k measurements do not certify later/live weights, clean floor rendering, or ideal hidden geometry.
