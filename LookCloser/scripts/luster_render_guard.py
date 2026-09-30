"""Conservative train-silhouette envelope for frozen occupancy traversal only."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from blur_runtime import sha


def build_guard(pipe,data,destination,margin_voxels=3.):
    data=Path(data);destination=Path(destination)
    if not np.isfinite(margin_voxels) or margin_voxels<1:raise ValueError('Guard needs at least one hull-voxel margin')
    audit=json.loads((data/'bounds_audit.json').read_text())
    points=np.load(data/'hull.npz')['points']
    pitch=float(audit['pitch_world'])*float(audit['scale'])
    grid=pipe.model.occupancy_grid
    aabbs=grid.aabbs.detach().cpu().numpy();resolution=grid.resolution.detach().cpu().numpy()
    coords=grid.grid_coords.detach().cpu().numpy();tree=cKDTree(points);masks=[]
    for bounds in aabbs:
        cell=(bounds[3:]-bounds[:3])/resolution
        centers=bounds[:3]+(coords+.5)*cell
        distance=tree.query(centers,k=1,workers=2)[0]
        # Include any grid cell intersecting a hull-center ball with a generous
        # margin. The extra half cell diagonal makes this conservative on the
        # differently shaped occupancy grid.
        allowed=distance<=margin_voxels*pitch+.5*np.linalg.norm(cell)
        masks.append(allowed.reshape(tuple(resolution)))
    np.savez_compressed(destination,allowed=np.stack(masks),aabbs=aabbs,resolution=resolution)
    record=dict(path=str(destination),sha256=sha(destination),margin_voxels=margin_voxels,
                margin_normalized=margin_voxels*pitch,source_hull_sha256=sha(data/'hull.npz'),
                source='Train silhouettes only; field parameters and original checkpoint unchanged')
    record.update(apply_guard(pipe,record))
    return record


def apply_guard(pipe,record):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('Occupancy guard hash differs')
    with np.load(path) as values:
        allowed=values['allowed'];aabbs=values['aabbs'];resolution=values['resolution']
    grid=pipe.model.occupancy_grid
    if allowed.dtype!=np.bool_ or tuple(allowed.shape)!=tuple(grid.binaries.shape):raise ValueError('Guard shape/type differs from occupancy grid')
    if not np.allclose(aabbs,grid.aabbs.detach().cpu().numpy(),atol=1e-7,rtol=0) or not np.array_equal(resolution,grid.resolution.cpu().numpy()):raise ValueError('Guard coordinates differ from scene')
    before=int(grid.binaries.sum())
    grid.binaries &= torch.from_numpy(allowed).to(grid.binaries.device)
    return dict(occupied_before=before,occupied_after=int(grid.binaries.sum()),allowed_cells=int(allowed.sum()))
