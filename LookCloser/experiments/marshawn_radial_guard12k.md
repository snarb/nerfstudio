# Marshawn1660: selected12k sparse traversal guards

## What was tested

Read-only inference masks on the immutable selected radial-K4 R0 checkpoint, step12000, SHA `5f0300871fa9db215236c270992ba9ace8f20a631479fa36b3228506e1ffd2aa`. The authoritative GPU core and EMA came from the checkpoint-bound export. The 9022 TRAIN SfM witnesses determine seed cells; neither eval RGB nor the uncertain MVS52 depths enter these masks. All tests leave the original field, core and EMA unchanged.

Policies use one clipped 3×3×3 dilation at most. Local-NEW expands only witness cells missing from core; local-ALL expands all witness-cell neighborhoods. Global expands the complete union. These labels describe traversal admission, not physical occupancy.

## Results

| Policy | Admitted cells | Added cells | Known9022 misses | BODY495 misses |
|---|---:|---:|---:|---:|
| Native core | 1,363,068 | 0 | 59 | 2 |
| Core union witness cells; no halo | 1,363,103 | **35** | **0** | **0** |
| Local-NEW halo | 1,363,588 | 520 | 0 | 0 |
| Local-ALL witness halo | 1,369,939 | 6,871 | 0 | 0 |
| Global halo after witness union | 2,042,994 | 679,926 | 0 | 0 |

Ordinary global core-only halo admits2,042,945 cells and still misses p2813. The 35-cell union covers it directly. All52 MVS candidates remain covered, but were not added as seeds.

Frozen native traces used the exact same48 edge rays previously recorded at8k, the original BODY128 selection (112 points,63 cameras, SHA `06e131767fb9ecd09b15eb28664a1917b8b1c21d4b976dd288599d5953628cab`), and **all238 actual TRAIN observations of all59 currently missed12k points** (57 cameras). The latter check covers the new12k additions independently of the older edge selection.

| Ray set | Native band mass ≥50% | Minimal35-cell union | All-cells / global halo / uniform16384 |
|---|---:|---:|---:|
| Same48 edge rays | 44/48 | **48/48** | 48/48 |
| Actual12k missed-point observations | 204/238 | **238/238** | 238/238 |
| Matched BODY128 | 126/128 | **126/128** | 126/128 |

No tested ray has over10% termination mass in the conservative prefix ending before `depth−max(.01,.05depth)`, under any policy. Surface bands are ±`max(.005,.01depth)`. Actual packed weights were captured from the renderer and checked against density integration. Native RGB was checked against weighted field RGB. Uniform8192/16384 integrations use the same frozen field; maximum surface-band mass discrepancy on BODY128 is8.45e−5.

The two BODY band exceptions are the same point5512 from HA and IB. They persist with all-cells and dense integration, with a conditional median displacement of about−2.1%; they are not occupancy loss. Fitted SfM visibility/depth remains a conditional reference, not independent depth truth.

p2813 has zero native surface-band mass from GE/DE. Minimal union raises it to .999851/.999986. The prior8k neighbor-sensitive case p6643 now has .968360 native band mass and .968347 with global halo, so this12k field no longer needs the8k neighborhood repair.

## Insights and next steps

The **35-cell union is the smallest passing candidate** on current12k known-point membership, all current missed-point source rays, the matched48 boundaries and BODY128. Larger halos provide no improvement to these measured coverage criteria. The field/core/EMA digest was restored exactly after every reader; no callback, optimizer or training worker was used. Eleven CPU tests passed, including noncumulative/clipped dilation, unrelated core preservation, ordered masks and archive immutability.

FullHD mask-quality comparison and final whole-atlas manual inspection are still pending authorization/selection. These ray results do not certify hidden empty space, surface details away from tested rays, or the absence of gray floor mottling. No cleanliness or ideal-grid claim follows from zero known cell misses.

Evidence:

- [CPU four-policy comparison](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/guard12000_CPU_v2/four_policy_comparison.json)
- [Same48 + BODY128 actual ray audit](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/guard12000_ray_comparison/audit.json)
- [All238 current-miss source rays](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/guard12000_new_miss_ray_comparison/audit.json)
- [Frozen CPU helpers and11-test receipt](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/local_guard_tools12k_v2/tests_receipt.json)

The prior8k evidence remains separately preserved; neither its masks nor its checkpoint certificate were transferred to12k. The final R0 training-attempt upper bound is12261, per the root terminal receipt.
