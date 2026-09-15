"""Diagnostic upper bound: same four-view proposal WITHOUT final PM depth veto.

Not an accepted mesh, no depth-confidence claim, no production/default changes.
This control tests whether the proposed surface itself can resolve the defect.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from build_independent_plane_patch_guard import ROOT as BASE, LOWER, FRAME


def run():
    root = BASE / 'unguarded_diagnostic';root.mkdir(exist_ok=False)
    q = read(LOWER / 'foreground/request.json');stage = read(LOWER / FRAME / 'request.json')
    proposal = np.load(LOWER / 'foreground/proposal.npz')
    assert sha(q['source_mesh']) == q['source_mesh_sha256']
    original = o3d.io.read_triangle_mesh(q['source_mesh'])
    vertices = np.concatenate([np.asarray(original.vertices), proposal['added_vertices']])
    triangles = np.concatenate([np.asarray(original.triangles), proposal['added_triangles']+len(original.vertices)])
    request = dict(frame=FRAME, source_mesh=q['source_mesh'], source_mesh_sha256=q['source_mesh_sha256'],
        proposal_sha256=sha(LOWER / 'foreground/proposal.npz'), lower_request_sha256=sha(LOWER / 'foreground/request.json'),
        source_rgb_receipt=stage['rgb_receipt'], guards_disabled_explicit=True,
        disabled_guards=['final corroborated-PatchMatch free-space veto'],
        retained_gates=['two disjoint calibrated neural stereo pairs', 'corrected disparity consistency',
                        'four train masks', 'maximum triangle edge .002', 'missing foreground layer eligibility'],
        measured_depth_contradictions_expected=True, geometry_inferred=True, heldout_used=False,
        production_updated=False, full_video_approved=False, diagnostic_only=True,
        scripts={str(Path(__file__).resolve()): sha(__file__)})
    atomic_json(root / 'qualification_request.json', request)
    dest = root / 'geometry';dest.mkdir()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest / 'mesh.ply'), mesh)
    saved = o3d.io.read_triangle_mesh(str(dest / 'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices), vertices)
    np.testing.assert_array_equal(np.asarray(saved.triangles), triangles)
    atomic_json(dest / 'result.json', dict(request_sha256=sha(root / 'qualification_request.json'),
        mesh_sha256=sha(dest / 'mesh.ply'), guard_passed=False, guard_applied=False, diagnostic_only=True,
        retained_triangles=len(proposal['added_triangles']), old_mesh_arrays_preserved=True,
        production_updated=False, visual_status='pending'))
    atomic_json(root / 'qualification_result.json', dict(request_sha256=sha(root / 'qualification_request.json'),
        geometry_result_sha256=sha(dest / 'result.json'), guard_applied=False, production_updated=False))
    print('UNGUARDED DIAGNOSTIC', len(proposal['added_triangles']), sha(dest / 'mesh.ply'), flush=True)


if __name__ == '__main__':
    run()
