# Radial R0 step8000 inference protection

What was tested: the same frozen8k field with actual native core, core∪9022 observed witness cells without halo, and exactly ONE halo after that union. Separate readers temporarily changed admission masks and restored the original field/core/EMA. Workers were not edited.

| Policy | Admitted cells | Known9022 FN | BODY495 FN | Own PSNR/SSIM/LPIPS | Common-old PSNR/SSIM/LPIPS |
|---|---:|---:|---:|---|---|
| Native |1300826|64|8|30.238811/.812888/.267723|30.595448/.822404/.261532|
| Minimal, NO halo |1300867(+41)|0|0|30.247694/.813047/.267399|30.605027/.822564/.261211|
| Union then ONE halo |2039270|0|0|30.348099/.813787/.267001|30.711426/.823295/.260808|

All52 photometric MVS candidate cells are admitted by every policy; MVS points were not seeds. Actual `nerfacc.grid._query` on GPU float32 confirms64→0→0 known misses. Five CPU tests cover exact union/one-halo, immutable source/core/EMA, outside-point rejection, storedR1 rejection and explicit selected step/no fallback.

All three native HD own/common contacts and three native floor crops were manually viewed. Neither derived policy creates broad haze in these views. Coherent opaque people remain, but KA gray/brown waves and HC/DC mat mottling persist. Coverage and metric gains do not establish floor cleanliness.

48 actual TRAIN ray traces cover the41 added cells, all8 missed BODY points and p2813 GE/DE controls. Twelve source/native/minimal/halo boundary crops were manually viewed. All8 BODY rays reach surface-band mass≥.99965 after minimal protection, without strong free-prefix termination. For p4020/GD the band mass repairs0→.999972 (minimal) or.999955 (halo). Across48 rays, band≥.5 improves42→47→48. The remaining minimal case p6643/KD lies close to a voxel face: band0→.16524→.87435. Thus zero point-cell FN does not certify complete neighboring-surface admission.

24 XYZ atlas pages per policy were prepared with actual-core/observed-seed/guard/unknown labels. Full final-policy manual review awaits root quality/policy selection. No physical occupied/free certificate follows from these traversal masks.

Actual readers loaded pre-eval container SHA `d0bca1932628874ffa310b4eb08e3ce9bc364eb50a6296cfb5e8e2f8aa3c9e7e`. Native post-eval save after CONT changed the container to `1b01f7f70be929a22190805306736434650a9207ee690cffa41ee3df75c6a2bd`. All38 model-state tensors hash identically to both HD readers and the edge reader: `ca77e5732e7f7f0f4378bf9cde747e8ef13d2cf4845aeab92760e705fd78154b`. Native core/EMA and every derived-grid array also match bitwise. This is inference tensor equivalence, not optimizer/RNG container equality. Old receipts retain their original SHA.

Artifacts under `/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/`: `minimal_guard_tools/`, `minimal_guard8k_CPU/`, `guard8000_HD_comparison/`, `guard8000_edge_audit/`, `immutable_8k_posteval_source/`, and `minimal_guard8k_posteval_CPU/`. The last two preserve a separate-inode read-only source and explicit array/container equivalence for reproducible current-container loading.
