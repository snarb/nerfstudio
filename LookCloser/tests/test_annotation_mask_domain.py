import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from annotation_mask_domain import known_domain

def test_half_pixel_unannotated_sliver_is_unknown_not_skin_veto():
    uv=np.array([[2.1,40],[2.49,40],[2.51,40],[3.,40],[10,10]],float)
    assert known_domain(uv,np.ones(5),1920,1080).tolist()==[False,False,True,True,True]

def test_behind_camera_and_renderer_border_remain_unavailable():
    uv=np.array([[10,10],[2.,40],[1918,40],[40,1078]],float)
    assert not known_domain(uv,np.array([-1,1,1,1]),1920,1080).any()

def test_known_negative_veto_and_two_known_sources_required(monkeypatch):
    import annotation_mask_domain as module
    rows=[dict(physical_camera=str(i),w=20,h=20) for i in range(3)]
    masks={str(i):np.ones((20,20),bool) for i in range(3)};masks['2'][:]=False
    positions={'0':5.,'1':5.,'2':5.}
    def fake_project(points,selected):
        u=positions[selected[0]['physical_camera']]
        return np.tile([u,5.],(1,len(points),1)),np.ones((1,len(points)))
    monkeypatch.setattr(module,'project',fake_project)
    vertices=np.array([[0,0,0],[.001,0,0],[0,.001,0]]);faces=np.array([[0,1,2]])
    assert len(module.semantic_faces(vertices,faces,rows,masks)[0])==0
    # An inset positive ROI is not an exhaustive silhouette: an unlabelled
    # third view is unknown in the explicit partial-annotation experiment.
    assert len(module.semantic_faces(vertices,faces,rows,masks,positive_only_annotations=True)[0])==1
    masks['1'][:]=False
    assert len(module.semantic_faces(vertices,faces,rows,masks,positive_only_annotations=True)[0])==0
    masks['1'][:]=True
    positions['2']=2.2
    assert len(module.semantic_faces(vertices,faces,rows,masks)[0])==1
    positions['1']=2.2
    assert len(module.semantic_faces(vertices,faces,rows,masks)[0])==0


def test_opt_in_domain_requires_boundary_conditioned_comparison(tmp_path):
    import pytest
    from study_forearm_production_delta import prepare
    output=tmp_path/'not_created'
    with pytest.raises(ValueError,match='Known annotation domain requires'):
        prepare(output,'001037',known_annotation_domain=True)
    assert not output.exists()
