# Clever-shadow frequency-worker feasibility

A finite read-only inspection found `ubuntu@clever-shadow` reachable. It has no
FSX mount and approximately136GB free root storage. No packages, files, jobs,
GPU allocations or fleet/broker configuration were changed.

The closest existing environment is `/home/ubuntu/sam3env/bin/python`:

| ABI property | Clever | Qualified SG reference |
|---|---|---|
| Python |3.11.17|3.11.16|
| Architecture / SOABI |x86_64 /cpython311|same|
| glibc |2.39|2.39|
| Torch |2.7.1+cu128|2.7.1+cu128|
| CXX11 ABI |true|true|
| Compiled CUDA targets |includes sm120/compute120|same|

Clever has no installed tinycudann. Its libstdc++ provides GLIBCXX3.4.33,
while the existing SG SM120 TCNN extension requires at most3.4.32 and glibc2.33.
A private copied binary overlay is therefore ABI-plausible without compilation,
but was not copied or runtime-qualified. The frozen source is14.84MB, the SG
TCNN extension53.72MB and one HD input approximately3MB: the minimal base is
about71.6MB before exact dependency closure. Whole RAW/video/model staging is
unnecessary for independent2D frequency fitting.

The candidate also lacks tyro, rich, pytorch_msssim, scipy and jaxtyping.
The frozen fitter imports the nerfstudio dataparser registry and its parser
chain, so the full exact import closure must be qualified; silently selecting
the alternative SSIM fallback would change the recipe. The existing CA4D Docker
image and host CA4D virtual environments also lack TCNN/nerfstudio/tyro, giving
no ready-to-run container shortcut.

Conclusion: not ready as-is; a small private overlay plus actual source-bound
one-image canary is a plausible separate operational step. This inspection did
not establish GPU execution compatibility or add an eleventh worker. The
existing ten-resource campaign was unchanged. Evidence is saved at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/clever_readonly_feasibility_20261108/report.json`.
