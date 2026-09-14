"""Verify that the annotation-only control adds faces without moving old geometry."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json


def audit(root, frame):
    previous = Path('/mnt/data/dec5_forearm_color_qualified_curved') / frame
    current = root / frame
    request = read(current / 'request.json')
    if request['curvature_policy']['known_annotation_margin'] != 3:
        raise ValueError('Wrong control')
    if request['annotation_domain_helper_sha256'] != sha(Path(__file__).with_name('annotation_mask_domain.py')):
        raise ValueError('Changed semantic helper')
    source = next(r for r in read('/mnt/data/dec5_phase30_dynamic_150/request.json')['inventory'] if r['frame_id'] == frame)
    base = o3d.io.read_triangle_mesh(source['mesh'])
    nv, nt = len(base.vertices), len(base.triangles)
    records = []
    for variant in ['transferred', 'guarded']:
        meshes = []
        for folder in [previous, current]:
            result = read(folder / 'geometry_result.json')
            if result['request_sha256'] != sha(folder / 'request.json'):
                raise ValueError('Changed request')
            filename = variant + '.ply'
            if sha(folder / filename) != result['hashes'][filename]:
                raise ValueError('Changed geometry')
            mesh = o3d.io.read_triangle_mesh(str(folder / filename))
            v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if not np.array_equal(v[:nv], np.asarray(base.vertices)) or not np.array_equal(t[:nt], np.asarray(base.triangles)):
                raise ValueError('Original production prefix changed')
            meshes.append((v, t))
        (ov, ot), (v, t) = meshes
        if not set(map(tuple, ov[nv:])).issubset(set(map(tuple, v[nv:]))):
            raise ValueError('Previously retained proposal vertex moved or disappeared')
        def geometry_faces(vertices, faces):
            return {tuple(sorted(map(tuple, tri))) for tri in vertices[faces[nt:]]}
        before, after = geometry_faces(ov, ot), geometry_faces(v, t)
        lost, added = len(before - after), len(after - before)
        if variant == 'transferred' and lost:
            raise ValueError('Annotation-only admission removed old faces')
        records.append(dict(variant=variant, old_faces=len(before), new_faces=len(after),
                            added_faces=added, removed_faces=lost, old_vertices_exact=True))
    fresh = root / 'fresh_audit' / (frame + '.json')
    if read(fresh)['geometry_result_sha256'] != sha(current / 'geometry_result.json'):
        raise ValueError('Fresh ray audit not for this mesh')
    result = dict(frame=frame, controls=records, original_production_prefix_exact=True,
                  fresh_audit_sha256=sha(fresh), script_sha256=sha(__file__),
                  geometry_result_sha256=sha(current / 'geometry_result.json'),
                  artifact_free=False, production_accepted=False)
    atomic_json(current / 'independent_control_audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/data/dec5_forearm_annotation_domain_only'))
    p.add_argument('--frames', nargs='+', default=['001029', '001033', '001037'])
    a = p.parse_args()
    for frame in a.frames:
        audit(a.root, frame)
