"""Prune inferred inset shell by corroborated measured free space in 62 views.

Unlike the earlier local interpolation arm, this diagnostic uses a silhouette-
bounded Poisson prior, NOT observed-neighborhood certificates/direct depth.
Passing this guard is a safety screen, not a geometry/texture quality verdict.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from probe_inset_head_completion import ROOT, SOURCE, FRAMES
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run(frame):
    source = ROOT / frame / 'inset_001000'; folder = ROOT / frame / 'guarded'
    folder.mkdir(parents=True, exist_ok=False)
    result = read(source / 'result.json')
    assert sha(source / 'mesh.ply') == result['hashes']['mesh.ply']
    base = read(SOURCE / frame / 'request.json')
    rows, depths, receipt = load_real(Path(base['depth_root']), frame)
    assert receipt == base['depth_receipt']
    old = o3d.io.read_triangle_mesh(base['source_mesh'])
    mesh = o3d.io.read_triangle_mesh(str(source / 'mesh.ply'))
    v = np.asarray(mesh.vertices); t = np.asarray(mesh.triangles).copy(); nt = len(old.triangles)
    np.testing.assert_array_equal(v[:len(old.vertices)], np.asarray(old.vertices))
    np.testing.assert_array_equal(t[:nt], np.asarray(old.triangles))
    atomic_json(folder / 'request.json', dict(frame=frame, source_result_sha256=sha(source / 'result.json'),
        source_mesh_sha256=sha(source / 'mesh.ply'), depth_receipt=receipt,
        prior='mask-constrained radially inset Poisson shell', inset=.001,
        observed_neighborhood_certificates=False, direct_depth_admission=False,
        native_free_space_guard=dict(offsets=[0,.5], free_depth_separation=.003, other_views=3, maximum_rounds=8),
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
                 [Path(__file__).name,'guard_jaw_measured_depth.py','study_confidence_depth_prior.py']},
        production_updated=False, heldout_used=False, inferred_not_measured=True))
    retained = np.arange(len(t) - nt); initial = len(retained); rounds = []
    for iteration in range(8):
        scene = scene_for(v, t); remove = set(); checks = []
        for ci, (row, depth) in enumerate(zip(rows, depths)):
            for offset in [0,.5]:
                implicated, count, raw_count = measured_pixel_veto(scene,row,depth,rows,depths,nt,len(t),offset)
                remove.update(implicated.tolist())
                checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%10 == 0:
                atomic_json(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed_triangles=len(remove),checks=checks))
        print(frame,'guard',iteration,'remove',len(remove),flush=True)
        if not remove:
            break
        take = np.ones(len(t),bool); take[list(remove)] = False
        assert take[:nt].all(); retained = retained[take[nt:]]; t = t[take]
    passed = not rounds[-1]['removed_triangles']
    candidate = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    candidate.compute_vertex_normals(); o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),candidate)
    np.savez_compressed(folder/'evidence.npz',retained_candidate_triangle_ids=retained)
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),initial_added=initial,
        added=len(retained),rounds=rounds,native_free_space_guard_passed=passed,original_prefix_exact=True,
        observed_neighborhood_certificates=False,hashes={n:sha(folder/n) for n in ['mesh.ply','evidence.npz']},
        production_updated=False,visual_status='pending'))
    assert passed, 'Native guard did not converge; candidate is not accepted'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--frame',required=True,choices=FRAMES)
    run(parser.parse_args().frame)
