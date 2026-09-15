"""Verify matched extraction controls, retained bindings and actual ray misses."""
from pathlib import Path
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
from study_jaw_tsdf_extraction_weight import ROOT, FRAMES, ARMS


def check_command(reference, actual, output, weight):
    expected = list(reference)
    for flag, value in [('--output', str(output)), ('--tensor-weight-threshold', str(weight))]:
        assert expected.count(flag) == actual.count(flag) == 1
        expected[expected.index(flag) + 1] = value
    assert actual == expected, 'Control changed an undeclared fusion parameter'


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    request = read(ROOT/'request.json')
    result = read(ROOT/'result.json')
    bindings = {}
    def bind(path, digest):
        assert sha(path) == digest, str(path)
        bindings[str(path)] = digest
    bind(ROOT/'request.json', result['request_sha256'])
    for name, digest in request['scripts'].items():
        bind(Path(__file__).with_name(name), digest)
    assert request['frames'] == FRAMES and request['arms'] == ARMS
    assert len(result['records']) == 6
    records = []
    for frame in FRAMES:
        folder = ROOT/frame
        q = read(folder/'request.json')
        bind(ROOT/'request.json', q['parent_request_sha256'])
        reference = Path(request['source_root'])/frame/'stages/fuse-original.json'
        bind(reference, q['reference_receipt_sha256'])
        assert q['reference_command'] == read(reference)['command']
        data = Path(q['reference_command'][q['reference_command'].index('--data')+1])
        bind(data/'transforms.json', q['transforms_sha256'])
        transforms = read(data/'transforms.json')
        rows = [r for r in transforms['frames'] if r['file_path'] in transforms['train_filenames']]
        assert len(rows) == len({r['physical_camera'] for r in rows}) == 62
        assert q['train_cameras'] == [r['physical_camera'] for r in rows]
        assert set(q['depth_hashes']) == {str(data/r['depth_file_path']) for r in rows}
        for p, h in {**q['depth_hashes'], **q['source_mesh_hashes']}.items(): bind(p, h)
        review = read(folder/'review/result.json')
        bind(Path(__file__).with_name('review_jaw_tsdf_extraction_weight.py'), review['script_sha256'])
        for p,h in review['input_hashes'].items(): bind(p,h)
        for item in review['files']: bind(item['path'], item['sha256'])
        camera = next(r['camera'] for r in read('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')['inventory'] if r['frame_id']==frame)
        box = next(r['bbox_inclusive'] for r in read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components'] if r['frame_id']==frame)
        x0,y0,x1,y1 = box
        for arm, weight in ARMS.items():
            stage = read(folder/f'stages/fuse-{arm}.json')
            check_command(q['reference_command'], stage['command'], folder/arm/'mesh.ply', weight)
            bind(folder/'request.json', stage['request_sha256'])
            for p,h in stage['retained_hashes'].items(): bind(p,h)
            record = next(r for r in result['records'] if r['frame']==frame and r['arm']==arm)
            bind(folder/arm/'mesh.ply', record['mesh_sha256'])
            bind(folder/arm/'mesh.json', record['metadata_sha256'])
            mesh = o3d.io.read_triangle_mesh(str(folder/arm/'mesh.ply'))
            meta = read(folder/arm/'mesh.json')
            assert len(mesh.vertices)==meta['vertices'] and len(mesh.triangles)==meta['triangles']
            assert np.isfinite(np.asarray(mesh.vertices)).all() and len(mesh.triangles)>0
            _, counts, _ = mesh.cluster_connected_triangles()
            assert len(counts)==meta['connected_components']
            assert sorted(counts)==sorted(meta['component_triangles'])
            mesh.transform(np.asarray(review['render_only_normalization'][arm]))
            depth, _, _ = camera_depth(scene_for(np.asarray(mesh.vertices), np.asarray(mesh.triangles)), camera)
            misses = int((~np.isfinite(np.rot90(depth)[y0:y1+1,x0:x1+1])).sum())
            expected = next(r for r in review['records'] if r['camera']=='old_moving' and r['arm']==arm)
            assert misses==expected['selected_spot_misses']
            records.append(dict(frame=frame, arm=arm, misses=misses, components=len(counts)))
    atomic_json(ROOT/'audit.json', dict(records=records, verified_input_hashes=bindings,
        script_sha256=sha(__file__), production_accepted=False, ray_misses_replayed=True))
    print({'bindings':len(bindings), 'records':records}, flush=True)


if __name__=='__main__': main()
