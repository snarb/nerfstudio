import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from study_foundation_lower_forearm import PAIRS


def test_lower_pairs_are_disjoint_train_views():
    from joint_temporal_texture import HELD_CAMERAS
    names=[n for pair in PAIRS for n in pair]
    assert len(names)==len(set(names))==4
    assert not set(names)&HELD_CAMERAS
    assert {n[5] for n in names}=={'D','E'}


def test_geometry_adapter_uses_same_producers_in_isolated_roots(monkeypatch,tmp_path):
    import build_foundation_lower_forearm as adapter
    import build_foundation_consensus_patch as empty
    import build_foundation_foreground_patch as front
    seen=[]
    for module,keys in [(empty,['ROOT','BIAS','SOURCES']),(front,['ROOT','CONTROL','BIAS'])]:
        for key in keys:monkeypatch.setattr(module,key,getattr(module,key))
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    monkeypatch.setattr(adapter,'sha',lambda p:'digest')
    monkeypatch.setattr(adapter,'atomic_json',lambda *args:None)
    monkeypatch.setattr(empty,'run',lambda:seen.append((empty.ROOT,empty.BIAS,empty.SOURCES)))
    monkeypatch.setattr(front,'run',lambda:seen.append((front.ROOT,front.CONTROL,front.BIAS)))
    adapter.geometry()
    assert seen==[(tmp_path/'empty_ray',tmp_path/'bias',[tmp_path/'001037']),
                  (tmp_path/'foreground',tmp_path/'empty_ray',tmp_path/'bias')]
