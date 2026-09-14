"""Read-only observed-stereo audit of frozen 3D jaw-notch proposals.

Missing depth is unknown, never evidence of free space. A farther observation
only contradicts a proposal after agreement with three OTHER train depth maps.
This diagnostic does not accept a repair or change any published geometry.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from study_confidence_depth_prior import load_real, support, project_integer, unproject
from study_jaw_boundary_notches import PARENT
from diffusion_mesh_repair import scene_for

RAW = Path('/mnt/data/dec5_jaw_3d_boundary_notches')
ATTRIBUTION = Path('/mnt/data/dec5_jaw_3d_boundary_guarded_attribution')
TOLERANCE = .001


def barycentric_samples(vertices):
    """Vertices, edge thirds and centroid: ten fixed samples per triangle."""
    weights = np.array([[1,0,0],[0,1,0],[0,0,1],
                        [2,1,0],[1,2,0],[0,2,1],[0,1,2],[1,0,2],[2,0,1],[1,1,1]], float)
    weights /= weights.sum(1, keepdims=True)
    return np.einsum('sk,tkd->tsd', weights, vertices)


def observed_at(points, camera, depth):
    uv, z = project_integer(camera, points)
    finite = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
    xy = np.rint(np.where(np.isfinite(uv), uv, -9999)).astype(np.int64)
    available = finite & (xy[:,0] >= 0) & (xy[:,0] < depth.shape[1]) & (xy[:,1] >= 0) & (xy[:,1] < depth.shape[0])
    observed = np.zeros(len(points), float)
    ids = np.flatnonzero(available)
    observed[ids] = depth[xy[ids,1], xy[ids,0]]
    available &= np.isfinite(observed) & (observed > 0)
    return xy, z, observed, available


def run(root, frame):
    control = root/'controls'/frame
    complete = read(control/'complete.json')
    if complete['request_sha256'] != sha(control/'request.json'):
        raise ValueError('Changed reconstruction request')
    for name, digest in complete['hashes'].items():
        if sha(control/name) != digest: raise ValueError(f'Changed control {name}')
    qc = read(control/'depth_qc.json')
    if qc['maps'] != 62 or qc['shape'] != [1080,1920]: raise ValueError('Native depth QC failed')
    out = root/'analysis'/frame
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(transforms=str(control/'staged63/transforms.json'), dense=str(control/'pipeline/dense'))
    atomic_json(out/'real_depth_input.json', spec)
    rows, depths, receipt = load_real(root/'analysis', frame)
    if len(rows) != 62 or len({r['physical_camera'] for r in rows}) != 62: raise ValueError('Train inventory mismatch')
    if any(r['physical_camera'] == 'F004_B005_1210O9' for r in rows): raise ValueError('Heldout leakage')
    # Verify individual observed maps against the completed stereo stage receipt.
    stage = read(control/'stages/patchmatch-geometric.json')
    if stage['request_sha256'] != sha(control/'request.json'): raise ValueError('Changed stereo request')
    for path, digest in stage['retained_hashes'].items():
        if sha(path) != digest: raise ValueError('Changed observed depth')
    source = next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id'] == frame)
    raw_result = read(RAW/frame/'result.json')
    if sha(source['mesh']) != raw_result['source_mesh_sha256'] or sha(RAW/frame/'candidate.ply') != raw_result['candidate_sha256']:
        raise ValueError('Changed frozen geometry')
    request = dict(frame=frame, script_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('study_confidence_depth_prior.py')),
        control_complete_sha256=sha(control/'complete.json'), raw_result_sha256=sha(RAW/frame/'result.json'),
        parent_request_sha256=sha(PARENT/'request.json'),
        source_transform_sha256=sha(spec['transforms']), geometric_stage_sha256=sha(control/'stages/patchmatch-geometric.json'),
        spot_attribution_sha256=sha(ATTRIBUTION/frame/'spot_admission.json'),
        real_depth_receipt=receipt, samples_per_triangle=10, agreement_tolerance=TOLERANCE,
        trusted_free_separation=3*TOLERANCE, trusted_observation_other_views=3,
        reference_roundtrip_px=1.5, minimum_parallax_degrees=1, missing_depth_is_unknown=True,
        heldout_used=False, production_changed=False, classification_only_not_an_admission_rule=True)
    if (out/'request.json').exists() and read(out/'request.json') != request:
        raise ValueError('Immutable diagnostic request mismatch')
    atomic_json(out/'request.json', request)
    if (out/'result.json').exists():
        result = read(out/'result.json')
        if result['request_sha256'] != sha(out/'request.json') or result['evidence_sha256'] != sha(out/'evidence.npz'):
            raise ValueError('Changed completed diagnostic')
        print(frame, 'verified completed diagnostic', flush=True)
        return
    old = o3d.io.read_triangle_mesh(source['mesh'])
    new = o3d.io.read_triangle_mesh(str(RAW/frame/'candidate.ply'))
    v, t, nt = np.asarray(old.vertices), np.asarray(old.triangles), np.asarray(new.triangles)
    if not np.array_equal(v, np.asarray(new.vertices)) or not np.array_equal(t, nt[:len(t)]): raise ValueError('Prefix changed')
    proposals = nt[len(t):]
    samples = barycentric_samples(v[proposals]); points = samples.reshape(-1,3)
    ref = dict(source['camera'], physical_camera='diagnostic_virtual_not_a_train_camera')
    votes, raw_free = support(points, ref, rows, depths)
    free = np.zeros((62, len(points)), np.uint8)
    agreement = np.zeros_like(free); unavailable = np.zeros_like(free)
    old_occlusion = np.zeros_like(free); trusted_old = np.zeros_like(free)
    query_confirmed_old = np.zeros_like(free)
    oldscene = scene_for(v,t)
    for index, (camera, depth) in enumerate(zip(rows, depths)):
        xy, z, observed, valid = observed_at(points, camera, depth)
        agreement[index] = valid & (np.abs(observed-z) <= TOLERANCE)
        unavailable[index] = ~valid
        far = np.flatnonzero(valid & (observed > z + 3*TOLERANCE))
        if len(far):
            observed_points = unproject(camera, xy[far,0], xy[far,1], observed[far])
            corroboration, _ = support(observed_points, camera, rows, depths)
            free[index,far] = corroboration >= 3
        # Old-mesh hits at exactly the proposed sample direction; unlike the
        # previous blanket guard, distinguish actual measured support of old hit.
        center = np.asarray(camera['transform_matrix'])[:3,3]
        directions = (points-center)/z[:,None]
        rays = np.column_stack((np.broadcast_to(center, points.shape), directions)).astype(np.float32)
        hit = oldscene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        behind = np.flatnonzero(np.isfinite(hit) & (hit > z+TOLERANCE))
        old_occlusion[index,behind] = 1
        if len(behind):
            oldpoints = center + hit[behind,None]*directions[behind]
            corroboration, _ = support(oldpoints, camera, rows, depths)
            trusted_old[index,behind] = corroboration >= 3
            # A real farther surface supported in OTHER views can legitimately
            # become occluded here. Also require this query camera to observe it.
            query_confirmed_old[index,behind] = ((corroboration >= 3) & valid[behind]
                & (np.abs(observed[behind]-hit[behind]) <= TOLERANCE))
        if (index+1)%10 == 0: print(frame, 'measured_camera', index+1, '/62', flush=True)
    np.savez_compressed(out/'evidence.npz', samples=samples, votes=votes.reshape(-1,10), raw_free=raw_free.reshape(-1,10),
                        trusted_free=free.reshape(62,-1,10), raw_agreement=agreement.reshape(62,-1,10),
                        unavailable=unavailable.reshape(62,-1,10), old_occlusion=old_occlusion.reshape(62,-1,10),
                        trusted_old_occlusion=trusted_old.reshape(62,-1,10),
                        query_confirmed_old_occlusion=query_confirmed_old.reshape(62,-1,10))
    spot_ids = [r['proposal'] for r in read(ATTRIBUTION/frame/'spot_admission.json')['triangles']]
    details = []
    for k in range(len(proposals)):
        ss = slice(k*10,(k+1)*10)
        details.append(dict(proposal=k, in_selected_spot=k in spot_ids,
            measured_votes=votes[ss].tolist(), trusted_free_votes=free[:,ss].sum(0).tolist(),
            old_occlusion_views=old_occlusion[:,ss].sum(0).tolist(),
            trusted_old_occlusion_views=trusted_old[:,ss].sum(0).tolist(),
            query_confirmed_old_occlusion_views=query_confirmed_old[:,ss].sum(0).tolist(),
            free_veto_cameras=[r['physical_camera'] for r,a in zip(rows,free[:,ss]) if a.any()],
            trusted_old_cameras=[r['physical_camera'] for r,a in zip(rows,trusted_old[:,ss]) if a.any()]))
    atomic_json(out/'result.json', dict(request_sha256=sha(out/'request.json'), evidence_sha256=sha(out/'evidence.npz'),
        depth_qc=qc, triangles=details, selected_spot=[r for r in details if r['in_selected_spot']],
        interpretation='Finite sampling diagnostic, not a complete surface or visibility certificate', production_accepted=False))
    print(frame, 'complete selected_spot=', [r for r in details if r['in_selected_spot']], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/data/dec5_jaw_measured_depth'))
    p.add_argument('--frame', required=True, choices=['001193','001195'])
    args = p.parse_args(); run(args.root, args.frame)
