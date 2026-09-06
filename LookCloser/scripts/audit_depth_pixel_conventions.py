#!/usr/bin/env python3
"""Installed Open3D synthetic canary for integer-depth nearest lookup.

No DEC5 RGB/depth is used. A tilted plane is rendered analytically using the
pinned COLMAP MVS integer-coordinate convention, then fused with the actual
installed CPU/CUDA VoxelBlockGrid. Mesh residuals are measured against the plane.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from colmap_integer_depth_fusion import IntegerCenteredDepthVolume
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256


def analytic_plane_depth(intrinsic, width, height, normal, distance):
    yy, xx = np.indices((height, width), dtype=np.float64)
    direction = np.stack(((xx-intrinsic[0, 2])/intrinsic[0, 0],
                          (yy-intrinsic[1, 2])/intrinsic[1, 1], np.ones_like(xx)), -1)
    return (distance/(direction@np.asarray(normal))).astype(np.float32)


def synthetic_plane(device, nearest):
    import open3d as o3d
    intrinsic = np.array([[90., 0, 32], [0, 90., 32], [0, 0, 1]])
    normal = np.array([.25, .15, 1.]); distance = .7
    depth = analytic_plane_depth(intrinsic, 64, 64, normal, distance)
    volume = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32), attr_channels=((1,), (1,)),
        voxel_size=.002, block_resolution=8, block_count=20000, device=o3d.core.Device(device))
    if nearest:
        volume = IntegerCenteredDepthVolume(volume)
    image = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=o3d.core.Device(device)))
    k = o3d.core.Tensor(intrinsic); e = o3d.core.Tensor(np.eye(4))
    blocks = volume.compute_unique_block_coordinates(image, k, e, 1., 2., 10.)
    for _ in range(3):
        volume.integrate(blocks, image, k, e, 1., 2., 10.)
    mesh = volume.extract_triangle_mesh(weight_threshold=2.).cpu().to_legacy()
    vertices = np.asarray(mesh.vertices)
    admitted = vertices[(np.abs(vertices[:, :2]) < .12).all(1)]
    residual = (admitted@normal-distance)/np.linalg.norm(normal)
    if len(admitted)<100 or not np.isfinite(residual).all():
        raise RuntimeError('Invalid synthetic plane mesh')
    return mesh, dict(device=device, sampling='nearest_integer' if nearest else 'legacy_floor',
                      vertices=len(vertices), admitted_vertices=len(admitted),
                      mean_plane_residual=float(residual.mean()),
                      median_plane_residual=float(np.median(residual)),
                      plane_residual_rmse=float(np.sqrt(np.mean(residual**2))),
                      plane_residual_p95_abs=float(np.quantile(np.abs(residual), .95)))


def main():
    import open3d as o3d
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['CPU:0', 'CUDA:0'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing canary')
    args.output.mkdir(parents=True)
    scripts = [Path(__file__), Path(__file__).with_name('colmap_integer_depth_fusion.py')]
    atomic_json(args.output/'request.json', dict(open3d_version=o3d.__version__, devices=args.devices,
        scripts={str(p):sha256(p) for p in scripts}, no_scene_or_held_rgb=True,
        plane_normal=[.25, .15, 1.], plane_distance=.7, depth_pixel_offset=0.,
        voxel_size=.002, sdf_trunc=.02,
        sources=['https://raw.githubusercontent.com/colmap/colmap/5509fffe/src/colmap/mvs/model.cc',
                 'https://raw.githubusercontent.com/colmap/colmap/5509fffe/src/colmap/mvs/patch_match_cuda.cu',
                 'https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h']))
    rows=[]; outputs={}
    for device in args.devices:
        for nearest in [False, True]:
            mesh,row=synthetic_plane(device, nearest)
            path=args.output/f"{device.replace(':','_')}_{row['sampling']}.ply"
            if not o3d.io.write_triangle_mesh(str(path), mesh):
                raise RuntimeError('Cannot retain synthetic mesh')
            outputs[str(path)]=sha256(path);rows.append(row)
            print(row, flush=True)
    atomic_json(args.output/'findings.json', dict(rows=rows, outputs=outputs,
        request_sha256=sha256(args.output/'request.json'),
        scope='Synthetic integer-centered plane only; no claim of real-skin repair or physical RGB-center correctness'))


if __name__ == '__main__':
    main()
