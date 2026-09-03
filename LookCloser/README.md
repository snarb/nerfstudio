# LookCloser

This directory extends Nerfstudio with the LookCloser model and its measured preprocessing and
rendering experiments. The paper implementation remains available as `lookcloser`; optional
geometry/rendering tools do not change that model unless explicitly selected.

## Selected geometry backend for the DEC5 rig

The selected no-training actor geometry path is **fixed-pose COLMAP PatchMatch -> continuous
TSDF**. Feed-forward DA3, MapAnything, MonoMVSNet and MVSMamba did not meet the visual or
actor-only metric bar. Splatfacto depth remains a diagnostic reference, not a per-frame
requirement.

Use `scripts/run_colmap_patchmatch_tsdf.py` on one display-referred JPG frame dataset. The input
must contain calibrated `frames`, explicit filename train/eval splits and at least two train
cameras. The runner:

1. exports the supplied calibration without feature matching or bundle adjustment;
2. runs full-resolution photometric and then geometric PatchMatch on train cameras only;
3. fuses the geometric depths into one strict CUDA TSDF;
4. builds a geometry-only 16-camera angular texture subset;
5. raycasts the mesh and renders by hard nearest-source selection, using later cameras only for
   visibility holes.

No image/person mask is accepted. RGB sources are never averaged, and eval RGB is unavailable to
geometry and prediction construction. By default the command uses the measured full-resolution
recipe (`1920`, three iterations per pass, geometric gates `6/2`, TSDF voxel/truncation
`0.0005/0.004`, tensor extraction weight `2`).
After cropping, it also removes disconnected islands smaller than the larger of 100 triangles or
`0.2%` of the dominant component. This scale-aware rule removed the residual ear/hair fragment in
the validation frame without using image coordinates and remains proportional when later frames
produce a denser or sparser mesh.

```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python LookCloser/scripts/run_colmap_patchmatch_tsdf.py \
  --data /path/to/one_frame_jpeg_dataset \
  --output-dir /path/to/output \
  --colmap-bin /usr/local/bin/colmap
```

The validated binary is CUDA COLMAP `3.13.0.dev0`, commit `5509fffe`. The runner fails closed on
another build because COLMAP 4.1.1 and a packaged 3.13 binary produced sparse, zero, or duplicated
depth maps with identical inputs. `--allow-unverified-colmap-build` is only for a build that has
passed a depth-map canary.

The final image is
`OUTPUT/render/nearest_fill16/eval_pred_0000.png`; the mesh is
`OUTPUT/colmap_patchmatch_tsdf.ply`. `pipeline_request.json` prevents `--resume` from silently
mixing artifacts made with different inputs, binaries or parameters, while
`pipeline_manifest.json` records the completed recipe and hashes.

Metrics are optional and must use an independent actor surface plus a region JSON; room/full-frame
metrics are intentionally not part of the DEC5 decision:

```bash
python LookCloser/scripts/run_colmap_patchmatch_tsdf.py \
  --data /path/to/one_frame_jpeg_dataset \
  --output-dir /path/to/output \
  --colmap-bin /usr/local/bin/colmap \
  --score-metrics \
  --metric-surface-depth-manifest /path/to/fixed_surface/mesh_depth_manifest.json \
  --roi-boxes-json /path/to/actor_roi.json
```

See `experiments/dec5_000899_offtheshelf_geometry.md` for the measured comparison and ear-artifact
ablation.

## Fifty-frame PatchMatch-TSDF campaign

`scripts/run_colmap_patchmatch_tsdf_campaign.py` is the opt-in, resumable controller for the
first 50 numeric DEC5 5A-3 frames. It stages one temporary JPEG dataset at a time, transfers the
fixed calibration by unique `physical_camera`, runs the pinned single-frame recipe on `dev3`,
verifies 62 full-resolution geometric maps and retained hashes, and publishes a frame only after
manual-GT-only face metrics and a visual verdict exist. It never creates a permanent 50-by-65
JPEG copy and never sends a face ROI to the geometry or rendering host.

Initialize and verify the pinned remote environment before reconstruction:

```bash
python LookCloser/scripts/run_colmap_patchmatch_tsdf_campaign.py init --preflight
python LookCloser/scripts/run_colmap_patchmatch_tsdf_campaign.py reconstruct \
  --frames 000899 000901 000903
```

Face polygons live in the output root under `config/face_polygons/FRAME.json`. Each file is
bound to the held-out display GT hash and must declare that it was drawn manually on GT without
using the prediction. `score`, `review`, and `finalize` are explicit states; thus reconstructed
or scored scratch cannot appear as a completed CSV row. The independent final checker is
`scripts/audit_colmap_patchmatch_tsdf_campaign.py`. The durable 3D output is the extracted TSDF
mesh plus its manifests—not a serialized raw Open3D TSDF volume.
