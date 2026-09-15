"""Expand four-view-consensus additions into holes visible in >=2 train cameras.

Fixes reference-only eligibility, not stereo depth or a final PM confidence rule.
Compared with the same no-final-veto raw consensus control, not production approval.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json, cameras
from plane_patch_evidence import project_rgb
from review_foundation_hand_geometry import grid_triangles
from diffusion_mesh_repair import scene_for
from build_independent_plane_patch_guard import ROOT as PREVIOUS, LOWER, FRAME

ROOT = Path('/mnt/data/dec5_multiview_forearm_admission')


def missing_in_train_views(points, rows, scene, minimum_separation=.003):
    votes = np.zeros(len(points), np.uint8);available = np.zeros(len(points), np.uint8)
    for row in rows:
        uv, z = project_rgb(points, row)
        inside = (z > 0) & (uv[:, 0] > 3) & (uv[:, 0] < row['w']-4) & (uv[:, 1] > 3) & (uv[:, 1] < row['h']-4)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        center = np.asarray(row['transform_matrix'])[:3, 3]
        direction = points[ids]-center
        rays = np.column_stack([np.broadcast_to(center, direction.shape), direction]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        # Unnormalized direction makes the candidate occur at ray parameter1.
        # Convert the distance along this ray to that camera's camera-z units.
        behind = ~np.isfinite(hit) | ((hit-1)*z[ids] > minimum_separation)
        votes[ids] += behind;available[ids] += 1
    return votes, available


def run():
    ROOT.mkdir(exist_ok=False)
    control = read(LOWER / 'empty_ray/request.json');q = read(LOWER / 'foreground/request.json')
    original = o3d.io.read_triangle_mesh(q['source_mesh']);ov, ot = np.asarray(original.vertices), np.asarray(original.triangles)
    assert sha(q['source_mesh']) == q['source_mesh_sha256']
    fields = np.load(LOWER / 'bias/point_fields.npz');prefix = control['reference']+'_offset_diagnostic_'
    xyz = fields[prefix+'xyz'];agreement = np.load(LOWER / 'empty_ray/proposal.npz')['agreement']
    previous = np.load(LOWER / 'foreground/proposal.npz')['eligible']
    assert not (previous & ~agreement).any()
    rows, _, _ = cameras(FRAME)
    request = dict(frame=FRAME, source_mesh=q['source_mesh'], source_mesh_sha256=q['source_mesh_sha256'],
        source_request_sha256=sha(LOWER / 'foreground/request.json'),
        raw_control_request_sha256=sha(PREVIOUS / 'unguarded_diagnostic/qualification_request.json'),
        point_fields_sha256=sha(LOWER / 'bias/point_fields.npz'), agreement_sha256=sha(LOWER / 'empty_ray/proposal.npz'),
        original_eligibility_sha256=sha(LOWER / 'foreground/proposal.npz'), cameras=rows,
        rule=dict(previous_eligibility_preserved=True, expansion_min_missing_train_views=2,
                  camera_z_separation=.003, photographed_margin=3, retain_exact_cross_pair_agreement=True,
                  maximum_edge=.002, final_pm_veto_enabled=False),
        geometry_inferred=True, production_updated=False, full_video_approved=False, heldout_used=False,
        final_pm_veto_disabled_same_as_raw_control=True,
        scripts={str(Path(__file__).with_name(n).resolve()): sha(Path(__file__).with_name(n)) for n in
                 [Path(__file__).name, 'plane_patch_evidence.py', 'review_foundation_hand_geometry.py']})
    atomic_json(ROOT / 'request.json', request)
    votes, available = missing_in_train_views(xyz[agreement], rows, scene_for(ov, ot))
    vp = np.zeros(agreement.shape, np.uint8);vp[agreement] = votes
    ap = np.zeros(agreement.shape, np.uint8);ap[agreement] = available
    eligible = agreement & (previous | (vp >= 2))
    v, t = grid_triangles(xyz, eligible, maximum_edge=.002)
    vv, tt = np.concatenate([ov, v]), np.concatenate([ot, t+len(ov)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv), o3d.utility.Vector3iVector(tt))
    mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(ROOT / 'mesh.ply'), mesh)
    saved = o3d.io.read_triangle_mesh(str(ROOT / 'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices)[:len(ov)], ov)
    np.testing.assert_array_equal(np.asarray(saved.triangles)[:len(ot)], ot)
    np.savez_compressed(ROOT / 'admission.npz', old=previous, agreement=agreement, new=eligible, votes=vp, available=ap)
    result = dict(request_sha256=sha(ROOT / 'request.json'), mesh_sha256=sha(ROOT / 'mesh.ply'),
        admission_sha256=sha(ROOT / 'admission.npz'), old_eligible=int(previous.sum()), new_eligible=int(eligible.sum()),
        added_triangles=len(t), original_arrays_preserved=True, new_region_votes_min=int(vp[eligible & ~previous].min()),
        final_pm_guard_applied=False, production_updated=False, visual_status='pending')
    atomic_json(ROOT / 'result.json', result);print(result, flush=True)


if __name__ == '__main__':
    run()
