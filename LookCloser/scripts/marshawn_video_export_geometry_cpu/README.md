# Marshawn capped-video export

These are byte-identical sources of the qualified external helpers deployed in
`C/operations/video_geometry_cpu_params_v2`. They are deliberately outside the
sealed training-code snapshot. This package records the current Marshawn
campaign; its model roots and seal are campaign-specific.

`route.py` gathers bound floating-point geometry from each model's original
training host. `worker.py` keeps native saved R0 geometry, verifies the selected
HC render and preserves checkpoint/YAML paths. `worker_params.py` performs CPU
pixel transfer from the original shared-FSX TIFFs, with the qualified sigma4,
gain.75, 16-source and smooth-focal parameters. The unchanged `worker.py`
also retains the archived sigma8 CPU implementation; production invokes its
GPU branch and the separate parameter worker's CPU branch.

Deployment requires the campaign-owned `root_release.json`, which binds these
files and `hybrid_params.json`; approval is not copied into this source package.
The current authority and artifact hashes are recorded in
`experiments/marshawn_video_path_1660_24fps.md`. The weak-grid guard and direct
IC-pose blend remain disabled. `render_gate_loader.py` is imported only for
its tensor-tree hash; production loading uses `cap_loader.py`.

The fleet starts export after all72 models complete. GPU leases use the same
physical-UUID locks as training. CPU workers are adopted by PID/start/boot,
verify geometry and semantic hashes, then the frozen held2020 crop renderer
and encoder create both120-frame movies. Full-video visual review is required.
