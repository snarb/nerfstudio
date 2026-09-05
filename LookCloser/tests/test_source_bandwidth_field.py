from pathlib import Path
import sys
from itertools import combinations
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from source_bandwidth_field import fit_camera_graph, quality_field_from_graphs, local_bandwidth_source_costs


def observations(values):
    return [{'primary_rank':i,'source_rank':j,'relative_blur_variance':values[j]-values[i]}
            for i,j in combinations(range(len(values)),2) for _ in range(4)]


def test_exact_graph_predicts_withheld_pairs_and_recovers_relative_quality():
    result=fit_camera_graph(observations([2.,0.,1.,3.]))
    assert result['qualified'] and result['held_edges']==6
    assert result['p90_absolute_error']<1e-12
    v=result['relative_variances']
    assert v[0]-v[1]==pytest.approx(2.)


def test_withheld_value_cannot_influence_its_own_prediction():
    rows=observations([2.,0.,1.,3.]);before=fit_camera_graph(rows)
    for r in rows:
        if (r['primary_rank'],r['source_rank'])==(0,1):r['relative_blur_variance']=10.
    after=fit_camera_graph(rows)
    get=lambda result:next(r['predicted'] for r in result['checks'] if r['pair']==[0,1])
    assert get(before)==pytest.approx(get(after))
    assert not after['qualified']


def test_sparse_or_disconnected_graph_is_not_qualified():
    rows=observations([0.,1.,2.,3.])
    assert not fit_camera_graph(rows[:4])['qualified']
    rows=[r for r in rows if r['source_rank']<3]
    rows += [{'primary_rank':3,'source_rank':4,'relative_blur_variance':1.}]*4
    assert not fit_camera_graph(rows)['qualified']


def test_no_models_exactly_preserve_global_relative_prior():
    baseline=np.array([0.,-1.,2.],np.float32)
    field,stats=quality_field_from_graphs([], (3,32,40),np.ones((32,40),np.float32),baseline)
    np.testing.assert_array_equal(field,np.broadcast_to(baseline[:,None,None],field.shape))
    assert not stats['enabled']


def test_quality_smoothing_does_not_cross_depth_discontinuity():
    depth=np.ones((32,256),np.float32);depth[:,128:]=2.
    model={'cell':[0,0,0],'qualified':True,'nodes':[0,1,2],
           'relative_variances':{0:0.,1:-1.,2:1.}}
    field,stats=quality_field_from_graphs([model],(3,32,256),depth,np.zeros(3,np.float32))
    assert stats['enabled'] and field[1,:,:100].mean()<-.99
    assert np.abs(field[:,:,128:]).max()<1e-6


def test_unrepresented_camera_keeps_its_existing_prior():
    model={'cell':[0,0,0],'qualified':True,'nodes':[0,1,2],
           'relative_variances':{0:0.,1:-1.,2:1.}}
    baseline=np.array([0.,0.,0.,2.],np.float32)
    field,_=quality_field_from_graphs([model],(4,24,32),np.ones((24,32),np.float32),baseline)
    assert np.max(np.abs(field[3]-2.))<1e-4


def test_nonfinite_pair_fails():
    with pytest.raises(ValueError):fit_camera_graph([{'primary_rank':0,'source_rank':1,'relative_blur_variance':float('nan')}])


def test_local_costs_do_not_modify_rgb_visibility_or_geometry():
    rgb=np.random.default_rng(3).random((3,80,96,3),dtype=np.float32)
    valid=np.zeros(rgb.shape[:-1],bool);depth=np.ones(rgb.shape[1:3],np.float32)
    copies=[a.copy() for a in [rgb,valid,depth]]
    costs,audit=local_bandwidth_source_costs(rgb,valid,depth,.01)
    assert not costs.any() and audit['qualified_cells']==0 and not audit['modifies_rgb']
    for value,original in zip([rgb,valid,depth],copies):np.testing.assert_array_equal(value,original)
