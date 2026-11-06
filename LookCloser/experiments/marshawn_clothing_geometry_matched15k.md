# Marshawn A/B/C: matched15188 clothing geometry

## What was tested

All103 predeclared TRAIN clothing rays were replayed at the first common15188 gate: original SH reference; A with18-ray prefix0.003; B with84-ray prefix0.003; and C with the same84-ray prefix plus native-softplus positive band0.001. The same31 supervised source points and whole-patch reserves remain sealed. Each checkpoint was protected before pruning, and all three full validation views/RGB/alpha/metrics were copied for review.

The durable watcher and CPU reporter ran automatically with60-second polling and exited after the report. GPU replays ran sequentially under explicit read-only overlap authorization, a10GiB allocator cap, actual core snapshots, fixed source queries and producer identity checks. No optimizer update, training callback or ongoing-arm modification occurred.

## Results

**C improves actual geometry on both newly supervised clothing and previously reserved garment patches.** Source expansion alone is mixed: B improves the new fitted region but loses some of A's nearby-shirt generalization. These source diagnostics do not certify an ideal grid or replace the held-out images.

Native unconditional mass inside the diagnostic witness±.02 band:

| Predeclared subset | Rays | SH | A | B | C |
|---|---:|---:|---:|---:|---:|
| Original fitted shirt subset | 18 | 65.26% | 82.33% | 86.02% | **99.60%** |
| New fitted clothing subset | 66 | 82.98% | 80.04% | 84.59% | **99.48%** |
| Original nearby reserves | 8 | 64.32% | **86.02%** | 65.03% | 83.82% |
| New whole-patch reserves | 9 | 75.96% | 81.62% | 75.58% | **87.60%** |

The words fitted/reserved describe the experiment design: A fits only the original18; B/C fit84. All observations remain TRAIN RGB evidence, and the original four reserves are spatially correlated with the shirt training subset. The new reserves cover an entire black sleeve patch and two grey trouser patches, with all their auxiliary observations excluded.

C improves each new reserved patch versus B under native traversal:

| Reserved patch | B band mass | C band mass |
|---|---:|---:|
| Black sleeve5568 | 63.99% | 83.69% |
| Grey knee2219 | 94.60% | 99.56% |
| Grey hip trousers5556 | 68.13% | 79.56% |

New-reserve early mass falls18.84%→9.47%. The effect also appears when occupancy is bypassed: dense positive-band unconditional mass rises0.5611→0.6385 for the whole-patch reserves and0.4506→0.6787 for old nearby reserves. It is not uniformly better for every point/profile: dense±.02 mass at the grey knee is86.61% in B and86.31% in C.

The actual positive-band policy is narrower than the diagnostic±.02: halfwidth`max(.005,.01*depth)`. Exact partial-interval integration shows the intended mechanism. On new fitted rays, mean native band optical thickness rises1.305→5.420 and local band opacity0.7117→0.9934. On new reserved patches, local opacity rises0.7906→0.8765. Local opacity is conditional on reaching the interval; unconditional mass is also reported to avoid hiding foreground absorption. The diagnostic uses observed pixel rays, not an exact replay of the floating-point training prior ray.

All103 rays for all four models pass the same8192→16384 RGB/opacity criterion: opacity difference≤.005 and maximum foreground RGB difference≤1/255. Per-component mass changes and raw convergence values are retained. Largest before/band/after/transmission change is about0.00146, smaller than the main group differences. Transmission is never renormalized away.

- [Native mass by fitted/reserved subset](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/clothing103_matched15k/native_mass.png)
- [GT, SH, A, B, C actor crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/clothing103_matched15k_review/actors_GT_SH_A_B_C.jpg)
- [All three full views](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/clothing103_matched15k_review/fullviews_GT_SH_A_B_C.jpg)

All montages and the mass plot were visually inspected. C has the best aggregate held-out metrics among these four at15188, but conspicuous fog/sheets remain around the actors.

| Actor-mask held-out metrics | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| SH | 26.4216 | 0.78052 | 0.069681 |
| A | 25.9117 | 0.77545 | 0.070676 |
| B | 26.1116 | 0.78403 | 0.066107 |
| C | **26.8296** | **0.78962** | **0.064969** |

C versus B improves black-actor LPIPS0.067999→0.063875 and PSNR25.3370→26.0434. Grey-actor PSNR improves26.8862→27.6157, while grey LPIPS worsens0.064214→0.066063. C therefore does not dominate every actor/metric. Root reviewed the visual outputs; this report makes no automatic quality promotion.

## Evidence and insights

Results: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/clothing103_matched15k`. `comparison.json` includes every per-ray native/halo/all-cell/dense result, local-band integrals, SHA256 bindings, actual SH/softplus/FR1 flags and producer health. `reserved_patch_breakdown.json` prevents the nine-ray aggregate from concealing a failed actor or patch. The complete traces/checkpoints remain protected on silver in corresponding `independent_occupancy` directories; figures, statistics, full images and metrics are copied locally.

The original28 pixel rays were verified unchanged. New queries use75 actual reviewed feature pixels; maximum witness-to-pixel-ray distance is0.000921 world units. Two focused analytic tests verify partial-band integration, foreground transmittance and sample gaps. No density is invented in skipped intervals.

All producers remained alive and advanced during each replay: reference overlap on A17741→19321, A19321→20877, B21823→23380, C20962→22340. The audit processes finished; production training continues under the existing controller/plateau policy. Parent supervision remains active.

Positive-band support addresses the prefix-only tendency to move termination behind the observed cloth. It generalizes to the predeclared withheld garment patches at this gate. It does not resolve the full-scene fog or prove that every occupancy cell is correct. A separate native-exponent geometry experiment can test whether this mechanism combines with the exponent's improved native traversal; that is a new controlled branch, not an edit to these arms.
