from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import render_midsequence_jaw_completion as study
from joint_temporal_texture import sha


def fixture_inputs(tmp_path, monkeypatch, corrupt=False):
    video=tmp_path/'video'; video.mkdir(); output=tmp_path/'output'
    monkeypatch.setattr(study,'VIDEO',video); monkeypatch.setattr(study,'ROOT',output)
    original=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[0,1,0]]),
                                      o3d.utility.Vector3iVector([[0,1,2]]))
    old=tmp_path/'old.ply'; assert o3d.io.write_triangle_mesh(str(old),original)
    rows=[dict(physical_camera=n,transform_matrix=np.eye(4).tolist(),fl_x=1000) for n in study.VIEWS[1:]]
    monkeypatch.setattr(study,'cameras',lambda frame:(deepcopy(rows),None,None))
    inventory=[]; roots={}
    for frame in study.FRAMES:
        root=tmp_path/'geometry'/frame; root.mkdir(parents=True); roots[frame]=root
        p=np.asarray(original.vertices).copy()
        if corrupt and frame==study.FRAMES[0]:p[0,0]=.1
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p),o3d.utility.Vector3iVector([[0,1,2]]))
        assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),mesh)
        (root/'result.json').write_text(json.dumps(dict(observed_guard_passed=True,original_prefix_exact=True,
            hashes={'mesh.ply':sha(root/'mesh.ply')})))
        (root/'request.json').write_text(json.dumps(dict(source_mesh_sha256=sha(old))))
        (root/'audit.json').write_text('{}')
        inventory.append(dict(frame_id=frame,mesh=str(old),mesh_sha256=sha(old),camera={'physical_camera':'moving'},
            source_masks={'root':'unchanged'},metadata='unchanged_metadata'))
    monkeypatch.setattr(study,'geometry',lambda frame:roots[frame])
    parent=dict(inventory=inventory,ordered_frame_ids=study.FRAMES,
        source_rows=[{'source_dataset':str(tmp_path/frame)} for frame in study.FRAMES],script_hashes={})
    (video/'request.json').write_text(json.dumps(parent))
    return output,parent,rows


def test_preparation_preserves_sources_times_and_unmasks_only_native_query(tmp_path,monkeypatch):
    out,parent,rows=fixture_inputs(tmp_path,monkeypatch)
    study.prepare()
    files=list(out.glob('*/rgb/*/*/request.json')); assert len(files)==13
    for path in files:
        q=json.loads(path.read_text()); frame=path.parents[3].name; view=path.parents[1].name
        assert q['ordered_frame_ids']==[frame]
        assert len(q['inventory'])==1 and q['inventory'][0]['frame_id']==frame
        assert len(q['source_rows'])==1 and Path(q['source_rows'][0]['source_dataset']).name==frame
        assert q['inventory'][0]['source_masks']=={'root':'unchanged'}
        if view!='moving':
            target=q['inventory'][0]['camera']
            assert target['physical_camera']=='diagnostic_unmasked_target_'+view
            assert target['reference_physical_camera']==view
        assert q['full_video_candidate'] is False
    assert json.loads((study.VIDEO/'request.json').read_text())==parent
    assert [r['physical_camera'] for r in rows]==study.VIEWS[1:]
    with pytest.raises(FileExistsError):study.prepare()


def test_unverified_original_vertex_movement_is_rejected(tmp_path,monkeypatch):
    out,_,_=fixture_inputs(tmp_path,monkeypatch,corrupt=True)
    with pytest.raises(AssertionError):study.prepare()
    assert not list(out.glob('*/rgb/*/*/request.json'))
