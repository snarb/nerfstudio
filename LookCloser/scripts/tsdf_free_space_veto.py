"""Train-depth free-space evidence for opt-in pre-extraction TSDF carving.

No RGB is read. Unknown depth, an occluding nearer layer and a mixed depth
footprint cannot vote to remove a voxel. This is an experimental geometric
constraint, not proof that the individual stereo measurements are correct.
"""
from __future__ import annotations

import math
import torch


def free_space_evidence(depth, u, v, z, *, minimum_gap=.005, radius=2):
    """Torch equivalent of the audited native 5x5 farther-layer criterion."""
    if depth.ndim != 2 or u.shape != v.shape or u.shape != z.shape:
        raise ValueError("Invalid depth/projection shapes")
    if not math.isfinite(minimum_gap) or minimum_gap <= 0 or radius < 0:
        raise ValueError("Invalid free-space evidence thresholds")
    height, width = depth.shape
    finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z) & (z > 0)
    x = torch.round(torch.where(finite, u, 0)).long()
    y = torch.round(torch.where(finite, v, 0)).long()
    inside = finite & (x >= radius) & (x < width-radius) & (y >= radius) & (y < height-radius)
    selected = torch.nonzero(inside, as_tuple=True)[0]
    result = torch.zeros_like(z, dtype=torch.bool)
    xx, yy, zz = x[selected], y[selected], z[selected]
    values = torch.stack([depth[yy+dy, xx+dx] for dy in range(-radius, radius+1)
                          for dx in range(-radius, radius+1)], dim=1)
    positive = torch.isfinite(values) & (values > 0)
    gap = torch.maximum(torch.full_like(zz, minimum_gap), .01*zz)
    farther = positive & (values > (zz+gap)[:, None])
    ordered = torch.sort(torch.where(positive, values, float('inf')), dim=1).values
    lo, hi = ordered[:, int(.2*(values.shape[1]-1))], ordered[:, int(.8*(values.shape[1]-1))]
    stable = torch.isfinite(hi) & (lo > 0) & ((hi-lo) <= .005*lo)
    result[selected] = (farther.sum(1) >= math.ceil(.8*values.shape[1])) & stable
    return result


def apply_veto(tsdf, weight, indices, votes, minimum_views):
    """Force contradicted, already observed voxels positive; preserve unknowns."""
    if minimum_views < 2 or indices.shape != votes.shape:
        raise ValueError("Invalid veto vote threshold or inventory")
    selected = indices[(votes >= minimum_views) & (weight[indices] > 0)]
    negative_before = int((tsdf[selected] < 0).sum().item())
    # Do not manufacture weight/support. Positive values are only empty-space
    # constraints; marching cubes still requires the original integration weight.
    tsdf[selected] = 1.
    return {'constrained_observed_voxels': len(selected),
            'negative_voxels_changed': negative_before,
            'unknown_voxels_promoted': 0}


def constrain_volume(volume, observations, *, minimum_views=3, minimum_gap=.005,
                     batch_size=262144):
    """Apply robust train-camera votes to the VBG before marching cubes.

    observations yields (HW normalized depth, OpenCV K, world-to-camera, audit).
    Open3D owners remain live for the duration of all shared DLPack tensors.
    """
    import open3d as o3d
    xyz_owner, index_owner = volume.voxel_coordinates_and_flattened_indices()
    tsdf_owner, weight_owner = volume.attribute('tsdf'), volume.attribute('weight')
    if o3d.core.cuda.is_available():
        o3d.core.cuda.synchronize()
    xyz = torch.utils.dlpack.from_dlpack(xyz_owner.to_dlpack())
    indices = torch.utils.dlpack.from_dlpack(index_owner.to_dlpack()).long()
    tsdf = torch.utils.dlpack.from_dlpack(tsdf_owner.to_dlpack()).reshape(-1)
    weight = torch.utils.dlpack.from_dlpack(weight_owner.to_dlpack()).reshape(-1)
    votes = torch.zeros(len(indices), dtype=torch.int32, device=xyz.device)
    rows = []
    for depth, intrinsic, extrinsic, audit in observations:
        depth = torch.as_tensor(depth, device=xyz.device)
        intrinsic = torch.as_tensor(intrinsic, dtype=xyz.dtype, device=xyz.device)
        extrinsic = torch.as_tensor(extrinsic, dtype=xyz.dtype, device=xyz.device)
        count = 0
        for start in range(0, len(xyz), batch_size):
            stop = min(start+batch_size, len(xyz))
            q = xyz[start:stop] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            z = q[:, 2]
            safe_z = torch.where(z > 0, z, 1.)
            u = intrinsic[0, 0]*q[:, 0]/safe_z + intrinsic[0, 2] - .5
            v = intrinsic[1, 1]*q[:, 1]/safe_z + intrinsic[1, 2] - .5
            free = free_space_evidence(depth, u, v, z, minimum_gap=minimum_gap)
            votes[start:stop] += free.int()
            count += int(free.sum().item())
        rows.append(dict(audit, free_space_voxel_votes=count))
        print(f'free_space_view={len(rows)} votes={count}', flush=True)
    result = apply_veto(tsdf, weight, indices, votes, minimum_views)
    if xyz.is_cuda:
        torch.cuda.synchronize(xyz.device)
    result.update(active_voxel_count=len(xyz), minimum_views=minimum_views,
                  minimum_gap_normalized=minimum_gap, relative_minimum_gap=.01,
                  radius=2, minimum_native_tap_fraction=.8,
                  maximum_middle_depth_spread_fraction=.005,
                  vote_histogram=torch.bincount(votes.long()).cpu().tolist(),
                  train_views=rows, uses_rgb=False, uses_semantic_masks=False,
                  modifies_volume_before_extraction=True, raw_volume_serialized=False)
    return result
