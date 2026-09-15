"""Bounded 000995 TSDF control: remove unobserved negative field, protect any near depth.

Opt-in only. Integration and extraction remain in the existing fuser. The
control records the same pre-extraction evidence without modifying the field.
Not a raw-volume serializer or a claim of correct inferred hidden geometry.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from joint_temporal_texture import read, sha, atomic_json
from tsdf_free_space_veto import free_space_evidence

ROOT = Path('/mnt/data/dec5_protected_field_veto/000995')


def near_native(depth, u, v, z, tolerance=.0015):
    finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z) & (z > 0)
    x = torch.round(torch.where(finite, u, 0)).long()
    y = torch.round(torch.where(finite, v, 0)).long()
    inside = finite & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    obs = depth[y.clamp(0, depth.shape[0]-1), x.clamp(0, depth.shape[1]-1)]
    return inside & torch.isfinite(obs) & (obs > 0) & ((obs-z).abs() <= tolerance)


def eligible(tsdf, weight, near, far, minimum_views=6):
    if minimum_views < 2 or any(x.shape != tsdf.shape for x in (weight, near, far)):
        raise ValueError('Mismatched field/evidence arrays or invalid threshold')
    return torch.isfinite(tsdf) & (tsdf < 0) & (weight > 0) & (near == 0) & (far >= minimum_views)


def constrain(volume, observations, *, output, enabled, minimum_views=6):
    import open3d as o3d
    xyz_owner, index_owner = volume.voxel_coordinates_and_flattened_indices()
    tsdf_owner, weight_owner = volume.attribute('tsdf'), volume.attribute('weight')
    o3d.core.cuda.synchronize()
    xyz_all = torch.utils.dlpack.from_dlpack(xyz_owner.to_dlpack())
    index_all = torch.utils.dlpack.from_dlpack(index_owner.to_dlpack()).long()
    tsdf = torch.utils.dlpack.from_dlpack(tsdf_owner.to_dlpack()).reshape(-1)
    weight = torch.utils.dlpack.from_dlpack(weight_owner.to_dlpack()).reshape(-1)
    selected = (weight[index_all] > 0) & (tsdf[index_all] < 0)
    indices = index_all[selected]; xyz = xyz_all[selected]
    before = tsdf[indices].clone(); weights = weight[indices].clone()
    near = torch.zeros(len(xyz), dtype=torch.int32, device=xyz.device)
    far = torch.zeros_like(near); rows = []; near_views = []; far_views = []
    for ci, (depth, intrinsic, extrinsic, audit) in enumerate(observations):
        d = torch.as_tensor(depth, device=xyz.device)
        K = torch.as_tensor(intrinsic, dtype=xyz.dtype, device=xyz.device)
        E = torch.as_tensor(extrinsic, dtype=xyz.dtype, device=xyz.device)
        nv = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device); fv = torch.zeros_like(nv)
        for start in range(0, len(xyz), 131072):
            end = min(start+131072, len(xyz)); p = xyz[start:end] @ E[:3, :3].T + E[:3, 3]
            z = p[:, 2]; safe = torch.where(z > 0, z, 1.)
            # Native COLMAP integer centers, NOT RGB-array coordinates.
            u = K[0, 0]*p[:, 0]/safe+K[0, 2]; v = K[1, 1]*p[:, 1]/safe+K[1, 2]
            nv[start:end] = near_native(d, u, v, z)
            fv[start:end] = free_space_evidence(d, u, v, z)
        near += nv.int(); far += fv.int()
        near_views.append(nv.cpu().numpy()); far_views.append(fv.cpu().numpy())
        rows.append(dict(audit, intrinsic=np.asarray(intrinsic).tolist(), extrinsic=np.asarray(extrinsic).tolist()))
        print(f'protected_field_view={ci+1} negative_queries={len(xyz)} near={int(nv.sum())} far={int(fv.sum())}', flush=True)
    if len(rows) != 62: raise ValueError('Exactly 62 train observations required')
    change = eligible(before, weights, near, far, minimum_views)
    if enabled: tsdf[indices[change]] = 1.
    torch.testing.assert_close(weight[indices], weights, rtol=0, atol=0)
    after = tsdf[indices].clone()
    torch.testing.assert_close(after[~change], before[~change], rtol=0, atol=0)
    torch.cuda.synchronize()
    np.savez_compressed(output/'field_evidence.npz', points=xyz.cpu().numpy(),
        flattened_indices=indices.cpu().numpy(), tsdf_before=before.cpu().numpy(),
        tsdf_after=after.cpu().numpy(), weights=weights.cpu().numpy(),
        near_by_camera=np.stack(near_views), far_by_camera=np.stack(far_views),
        eligible=change.cpu().numpy())
    result = dict(enabled=enabled, active_voxels=len(xyz_all), negative_observed_queries=len(xyz),
        eligible_negative_voxels=int(change.sum()), changed_negative_voxels=int(change.sum()) if enabled else 0,
        protected_negative_voxels=int((near > 0).sum()), minimum_far_views=minimum_views,
        near_tolerance=.0015, native_depth_principal_point_delta=0., far_radius=2,
        far_gap=.005, far_relative_gap=.01, far_tap_fraction=.8, far_middle_spread=.005,
        unknown_weights_unchanged=True, positive_field_unchanged=True, all_weights_unchanged=True,
        observations=rows, raw_volume_serialized=False, field_evidence_sha256=sha(output/'field_evidence.npz'))
    atomic_json(output/'field_result.json', result)
    return result


def run(arm):
    import fuse_depth_tsdf_mesh as fusion
    import tsdf_free_space_veto as veto
    out = ROOT/arm; out.mkdir(exist_ok=False)
    command = next(c for n, c in read(Path('/mnt/data/dec5_full_block_transfer/000995/commands.json')) if n == 'fuse-tsdf')
    args = command[2:]
    args[args.index('--data')+1] = str(ROOT/'depth_dataset')
    args[args.index('--output')+1] = str(out/'mesh.ply')
    args += ['--tensor-full-block-integration', '--tensor-free-space-min-views', '6']
    request = dict(arm=arm, frame='000995', arguments=args, enabled=arm=='protected',
        geometry_inputs={str(p.relative_to(ROOT/'depth_dataset')): sha(p)
                         for p in (ROOT/'depth_dataset').rglob('*') if p.is_file()},
        scripts={str(p): sha(p) for p in [Path(__file__), Path(fusion.__file__), Path(veto.__file__)]},
        masks_used=False, rgb_used=False, heldout_used=False, production_changed=False)
    atomic_json(out/'request.json', request)
    original = veto.constrain_volume
    try:
        veto.constrain_volume = lambda volume, obs, **kw: constrain(volume, obs, output=out,
            enabled=arm=='protected', minimum_views=kw['minimum_views'])
        if fusion.main(args) != 0: raise RuntimeError('Fusion failed')
    finally:
        veto.constrain_volume = original
    atomic_json(out/'complete.json', dict(request_sha256=sha(out/'request.json'),
        hashes={n: sha(out/n) for n in ('mesh.ply', 'mesh.json', 'field_evidence.npz', 'field_result.json')},
        visual_status='pending', production_changed=False, raw_volume_serialized=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('arm', choices=['control', 'protected'])
    run(p.parse_args().arm)
