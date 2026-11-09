import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_marshawn_release_grid import witness_guard
from marshawn_occupancy_audit import dilate_one
from trace_marshawn_novel_rays import project,component_support
from compare_marshawn_release_grids import applied_mask,frozen_without_binary


def test_union_precedes_exactly_one_halo_and_preserves_core():
    core=np.zeros((1,9,9,9),bool);core[0,2,2,2]=True;original=core.copy()
    boxes=np.array([[0,0,0,9,9,9]],float)
    seed,protected,active,ordinary=witness_guard(core,boxes,[[6.5,6.5,6.5]])
    assert np.array_equal(core,original)
    assert np.array_equal(active,dilate_one(original|seed))
    assert active.sum()==54 and ordinary.sum()==27
    assert not active[0,0,2,2] # Would be admitted by an accidental second halo.
    assert active[0,6,6,6] and not ordinary[0,6,6,6]
    with pytest.raises(ValueError,match='outside'):witness_guard(core,boxes,[[9,0,0]])


def test_frustum_projection_uses_nerfstudio_gl_axes_and_support_is_not_visibility():
    frame=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=120,cx=50,cy=60,w=100,h=120)
    uv,depth,inside=project(frame,[[0,0,-1],[.1,.1,-1],[0,0,1],[2,0,-1]])
    assert np.allclose(uv[:2],[[50,60],[60,48]])
    assert inside.tolist()==[True,True,False,False]
    result=component_support(np.array([.2,.3,.5]),np.array([True,True,False]),np.array([0,2,0]))
    assert result['termination_mass']==pytest.approx(.5)
    assert result['fraction_outside_all_train_frusta']==pytest.approx(.4)


def test_temporary_candidate_mask_restores_all_state_even_on_render_failure():
    import torch
    from torch import nn
    from probe_marshawn_occupancy import digest_model
    class Grid(nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('binaries',torch.tensor([[[[False,True]]]]));self.register_buffer('occs',torch.tensor([.01,.5]))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.tensor([1.]));self.occupancy_grid=Grid();self.device=torch.device('cpu')
    model=Model();before=digest_model(model);fixed=frozen_without_binary(model)
    for fail in [False,True]:
        try:
            with applied_mask(model,np.ones((1,1,1,2),bool)):
                assert model.occupancy_grid.binaries.all()
                assert frozen_without_binary(model)==fixed
                if fail:raise RuntimeError('render failure')
        except RuntimeError as error:assert str(error)=='render failure'
        assert digest_model(model)==before
