"""The adapter must bind the actual refined mesh without changing its parent."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import study_subface_instance_qualification as adapter
from joint_temporal_texture import atomic_json, sha, read


def configure_fixture(tmp_path,monkeypatch):
    parent=tmp_path/'parent';root=tmp_path/'out';control=parent/'refined/000995'
    control.mkdir(parents=True);(parent/'000995').mkdir();root.mkdir()
    mesh=control/'mesh.ply';mesh.write_bytes(b'bound refined mesh')
    request=parent/'000995/request.json'
    original={'mesh':'original.ply','mesh_sha256':'original-digest','scripts':{}}
    atomic_json(request,original)
    atomic_json(control/'request.json',{'study_request_sha256':sha(request)})
    atomic_json(control/'result.json',{'control':'refined',
        'request_sha256':sha(control/'request.json'),'hashes':{'mesh.ply':sha(mesh)}})
    atomic_json(root/'adapter.json',{'test':True})
    monkeypatch.setattr(adapter,'PARENT',parent);monkeypatch.setattr(adapter,'ROOT',root)
    # configure mutates module globals; make pytest restore those as well.
    for key in ['ROOT','PARENT','read']:
        monkeypatch.setattr(adapter.worker,key,getattr(adapter.worker,key))
    adapter.configure()
    return request,mesh,original


def test_rebinds_refined_mesh_without_mutating_request(tmp_path,monkeypatch):
    request,mesh,original=configure_fixture(tmp_path,monkeypatch)
    value=adapter.worker.read(request)
    assert value['mesh']==str(mesh)
    assert value['mesh_sha256']==sha(mesh)
    assert value['original_production_mesh']==original['mesh']
    assert read(request)==original


def test_rejects_changed_refined_mesh(tmp_path,monkeypatch):
    request,mesh,_=configure_fixture(tmp_path,monkeypatch)
    mesh.write_bytes(b'changed mesh')
    with pytest.raises(AssertionError):adapter.worker.read(request)
