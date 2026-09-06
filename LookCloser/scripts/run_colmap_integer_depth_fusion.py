#!/usr/bin/env python3
"""Run the existing TSDF CLI with an isolated integer-depth lookup adapter.

All ordinary fusion arguments are passed unchanged. This opt-in entry point
does not modify the existing runner, dataset calibration, block discovery or
renderer. It writes an explicit companion manifest for the adapted lookup.
"""
from __future__ import annotations

from pathlib import Path
import sys

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from colmap_integer_depth_fusion import IntegerCenteredDepthVolume


def main(argv=None):
    import open3d as o3d
    import fuse_depth_tsdf_mesh as fusion
    argv = sys.argv[1:] if argv is None else list(argv)
    args = fusion.parse_args(argv)
    if args.backend != 'tensor' or not args.tensor_full_block_integration or args.tensor_free_space_min_views:
        raise ValueError('Integer-depth control requires tensor full-block fusion without free-space veto')
    manifest_path = args.output.with_suffix('.integer_depth_request.json')
    if manifest_path.exists():
        raise ValueError('Preserve existing integer-depth request')
    scripts = [Path(__file__), Path(__file__).with_name('colmap_integer_depth_fusion.py'), Path(fusion.__file__)]
    request = dict(method='COLMAP_integer_depth_Open3D_nearest_lookup',
                   arguments=argv, open3d_version=o3d.__version__,
                   integration_principal_point_delta=[.5, .5],
                   block_discovery_principal_point_delta=[0., 0.],
                   input_calibration_changed=False, source_depth_values_changed=False,
                   uses_rgb=False, uses_semantic_masks=False,
                   scripts={str(p): sha256(p) for p in scripts},
                   source_transforms_sha256=sha256(args.data/'transforms.json'))
    atomic_json(manifest_path, request)
    original = o3d.t.geometry.VoxelBlockGrid
    try:
        o3d.t.geometry.VoxelBlockGrid = lambda *a, **kw: IntegerCenteredDepthVolume(original(*a, **kw))
        status = fusion.main(argv)
    finally:
        o3d.t.geometry.VoxelBlockGrid = original
    if status != 0:
        raise RuntimeError('Underlying fusion did not complete')
    atomic_json(args.output.with_suffix('.integer_depth_manifest.json'),
                dict(state='complete', request_sha256=sha256(manifest_path),
                     mesh=str(args.output), mesh_sha256=sha256(args.output),
                     mesh_metadata_sha256=sha256(args.output.with_suffix('.json')),
                     raw_volume_serialized=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
