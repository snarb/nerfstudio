"""Single-source texture contracts: adjacency, visibility, no camera averaging."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from hard_surface_texture import face_adjacency,select_surface_sources,gather_hard_rgb


def test_adjacency_uses_geometric_edges_and_excludes_disconnected_faces():
    triangles=np.array([[0,1,2],[2,1,3],[4,5,6]])
    np.testing.assert_array_equal(np.sort(face_adjacency(triangles),axis=1),[[0,1]])
    assert face_adjacency(np.empty((0,3),int)).shape==(0,2)
    # Non-manifold junctions are deliberately not treated as a two-face edge.
    assert not len(face_adjacency(np.array([[0,1,2],[1,0,3],[0,1,4]])))


def test_surface_smoothing_reduces_seams_but_never_selects_invisible_camera():
    triangles=np.array([[0,1,2],[2,1,3],[2,3,4],[4,3,5]])
    quality=np.array([[1,.99,1,0],[.99,1,.99,1]],np.float32)
    colors=np.zeros((2,4,3),np.float32);colors[1]=.5
    labels,report=select_surface_sources(colors,quality,triangles,smoothness=.08)
    np.testing.assert_array_equal(labels,[1,1,1,1])
    assert all(a>b for a,b in zip(report['energy'],report['energy'][1:]))
    assert not report['averages_rgb']


def test_zero_smoothing_is_best_quality_and_missing_faces_are_explicit():
    triangles=np.array([[0,1,2],[2,1,3],[2,3,4]])
    quality=np.array([[1,.1,0],[.1,1,0]],np.float32)
    labels,report=select_surface_sources(np.zeros((2,3,3)),quality,triangles,smoothness=0)
    np.testing.assert_array_equal(labels,[0,1,-1])
    assert report['unsupported_faces']==1


def test_color_consistency_rejects_highlight_without_averaging_output():
    triangles=np.array([[0,1,2],[2,1,3]])
    colors=np.full((3,2,3),.2);colors[0]=.9
    quality=np.array([[1,1],[.9,.9],[.8,.8]])
    labels,report=select_surface_sources(colors,quality,triangles,color_weight=.5)
    assert (labels==1).all() and report['color_weight']==.5
    raw=torch.tensor(colors.transpose(0,2,1))
    rgb,_,_=gather_hard_rgb(raw,torch.tensor(quality),torch.tensor(labels))
    torch.testing.assert_close(rgb,raw[1])


def test_two_label_graph_matches_exhaustive_global_minimum():
    import itertools
    triangles=np.array([[0,1,2],[2,1,3],[2,3,4],[4,3,5]])
    rng=np.random.default_rng(72)
    for _ in range(5):
        quality=rng.uniform(.1,1,(2,4));colors=rng.uniform(0,1,(2,4,3))
        labels,report=select_surface_sources(colors,quality,triangles,smoothness=.017)
        def energy(state):
            unary=-.025*np.log(quality/quality.max(0))
            value=sum(unary[label,i] for i,label in enumerate(state))
            for i,j in face_adjacency(triangles):
                a,b=state[i],state[j]
                value+=.017*((a!=b)+2.5*(np.abs(colors[a,i]-colors[b,i]).mean()+np.abs(colors[a,j]-colors[b,j]).mean()))
            return value
        optimum=min(map(energy,itertools.product(range(2),repeat=4)))
        assert energy(labels)==pytest.approx(optimum,abs=1e-6)
        assert report['energy'][-1]==pytest.approx(optimum,abs=1e-6)


def test_hard_gather_uses_exactly_one_camera_with_visibility_only_fallback():
    colors=torch.arange(2*3*4,dtype=torch.float32).reshape(2,3,4)
    weights=torch.tensor([[.1,0.,1.,0.],[1.,1.,1.,0.]])
    result,chosen,fallback=gather_hard_rgb(colors,weights,torch.tensor([0,0,1,1]))
    torch.testing.assert_close(result,torch.stack([colors[0,:,0],colors[1,:,1],colors[1,:,2],torch.zeros(3)],1))
    assert chosen.tolist()==[0,1,1,-1]
    assert fallback.tolist()==[False,True,False,False]


def test_hard_gather_invalid_labels_fall_back_without_invalid_index():
    colors=torch.ones((2,3,2));weights=torch.ones((2,2))
    _,chosen,fallback=gather_hard_rgb(colors,weights,torch.tensor([-1,99]))
    assert chosen.tolist()==[0,0] and fallback.all()


def test_hard_bake_requires_separate_output_before_any_writes(tmp_path):
    import bake_joint_temporal_mesh as baker
    with pytest.raises(ValueError,match='separate output'):
        baker.bake(tmp_path/'old','000973',128,hard_source=True)
    with pytest.raises(ValueError,match='separate output'):
        baker.bake(tmp_path/'old','000973',128,hard_source=True,output_root=tmp_path/'old')
    assert not list(tmp_path.iterdir())


def test_hard_bake_refuses_changed_recipe_before_loading_images(tmp_path,monkeypatch):
    import bake_joint_temporal_mesh as baker
    root=tmp_path/'calibration';out=tmp_path/'new';mesh=tmp_path/'mesh.ply'
    monkeypatch.setattr(baker,'geometry_paths',lambda frame:(mesh,tmp_path/'meta.json'))
    monkeypatch.setattr(baker,'sha',lambda path:'same_hash')
    baker.atomic_json(root/'cache'/'000973'/'request.json',{'mesh_sha256':'same_hash'})
    baker.atomic_json(out/'frames'/'000973'/'hard_texture_request.json',{'old_recipe':True})
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before loading images or building atlas')
    monkeypatch.setattr(baker,'atlas_geometry',forbidden)
    monkeypatch.setattr(baker,'load_frame',forbidden)
    with pytest.raises(ValueError,match='request mismatch'):
        baker.bake(root,'000973',128,hard_source=True,output_root=out)
    assert baker.read(out/'frames'/'000973'/'hard_texture_request.json')=={'old_recipe':True}


@pytest.mark.parametrize('smoothness',[-1,float('nan')])
def test_invalid_graph_options(smoothness):
    with pytest.raises(ValueError,match='graph options'):
        select_surface_sources(np.zeros((1,1,3)),np.ones((1,1)),np.array([[0,1,2]]),smoothness=smoothness)
