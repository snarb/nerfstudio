# Marshawn1660 versus frozen static fight-scene recipe

The old source of truth is [static leader reproduction recipe](static_leader_reproduction_recipe.md), dataset007740. This comparison was requested on2026-11-09. The new reference is the strict63/native-crop P/SAM15/core/hash25 model at140000 updates; tuning remains active until16:13UTC. This is not a claim that the new settings improve007740.

| Parameter | Frozen007740 recipe | CurrentMarshawn reference |
|---|---|---|
| log2_hashmap_size |23|25|
| Training rays/update |4096|8192|
| Coarse traversal step |.00625|.0015625|
| RGB objective |Full-scene Charbonnier|Actor softW64, analytic proposal-confidence normalization|
| Sampling |OriginalFAS from0, strength1/no ramp|body80/protective-band/whole-valid; balancedSAM3head15%; ramp4096–8192|
| Adam beta2 |.999|.99|
| Distortion multiplier |.01|.01→.1, ramp4096–12288|
| Opaque-core deficit prior |None|.01 on32-native-pixel eroded actor cores|
| ARM capped-interval allocator |Historical|Corrected|
| Density activation |Historical softplus|trunc_exp|
| SH direction mapping |Historical uncorrected|Corrected|
| 2D frequency inputs |Frozen dataset maps|Uniform16 constant proxy|
| Planned complete ancestry |106317updates|140000updates|
| Split |66train+3eval images|63train+3heldout DC/HC/KA|

Both use16levels×2features, maximum hash resolution8192, seed42, LR.01→.0001/horizon200000, FR1→.3 at75941, warmup4096, adaptive maximum1024samples/ray, occupancy resolution128/stable occupancy. A log2table limit25 versus23 means four times the maximum entries at a hashed level; lower dense levels and total model memory do not simply multiply byfour. Dataset/calibration/AABB/photometry remain scene-specific.

The new proposal concentrates effort on the two actors and heads. Beta2/coreprior address observed partial-opacity defects. The capacity/traversal changes were retained through measured Marshawn image gates. ARM/SH mapping changes are implementation changes; frozen007740 has a reproducibility contract and must be separately retrained/revalidated before adopting them. Oldhash25 superiority is untested here.

The matched current140k hash25-versus24 contrast gives4.604% lower actor LPIPS at26.552% higher Silver model-loop time; it is a quality/cost tradeoff, not a universal speedup. Raw exterior floor sheets/halos remain. The separate cap10 inference policy reaches heldoutLPIPS.09514483 versus default.09884040 with unchanged140k weights, and is provisional pending novel/temporal inspection; it is not a capped-training trajectory. See [campaign report](marshawn_native63_24h.md).

The current HC source GT and render share1464×2360 original-pixel crop/calibration. Earlier native-versus-HD A/I32 results used66TRAIN; their19.5463% benefit is explicitly in-sample. A fresh63train-HD/3native-heldout control is prepared to isolate input resolution on the current recipe.
