import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_marshawn_local_guard import local_guard,build_three,all_witness_guard,build_four
from test_marshawn_minimal_guard import fixture_args


def test_only_new_seed_neighbors_expand_existing_core_is_not_dilated():
    core=np.zeros((1,11,11,11),bool);core[0,2,2,2]=True
    seed=np.zeros_like(core);seed[0,2,2,2]=True;seed[0,8,8,8]=True
    actual,new=local_guard(core,seed)
    expected=core.copy()
    for x in range(7,10):
        for y in range(7,10):
            for z in range(7,10):expected[0,x,y,z]=True
    assert np.array_equal(actual,expected)
    assert new.sum()==1 and actual.sum()==28
    assert not actual[0,3,2,2] # Neighbor of old core remains skipped.
    assert not actual[0,6,8,8] # Two-away from new seed remains skipped.
    assert np.array_equal(local_guard(core,seed)[0],actual)


def test_no_missing_seed_is_exact_noop():
    core=np.zeros((1,5,5,5),bool);core[0,2,2,2]=True
    actual,new=local_guard(core,core.copy())
    assert not new.any() and np.array_equal(actual,core)


def test_boundary_halo_is_clipped_not_wrapped():
    core=np.zeros((1,5,5,5),bool);seed=core.copy();seed[0,0,0,0]=True
    actual,_=local_guard(core,seed)
    assert actual.sum()==8 and actual[0,1,1,1] and not actual[0,4,0,0]


def test_full_three_policy_source_immutable_and_ordered(tmp_path):
    args=fixture_args(tmp_path);before=args.checkpoint.read_bytes()
    r=build_three(args)
    assert r['counts']['minimal_guard']['active_cells']==2
    assert r['counts']['local_new_seed_halo1']['active_cells']==9
    assert r['counts']['witness_union_halo1']['active_cells']==35
    assert r['counts']['local_new_seed_halo1']['all_point_misses']==0
    assert r['counts']['local_new_seed_halo1']['subject_point_misses']==0
    z=np.load(args.output/'local_new_seed_halo1/grid.npz');source=np.load(args.grid)
    assert z['ema'].tobytes()==source['ema'].tobytes()
    assert z['core'].tobytes()==source['core'].tobytes()
    assert args.checkpoint.read_bytes()==before and not r['field_instantiated']


def test_all_witness_neighbors_expand_without_dilating_unwitnessed_core():
    core=np.zeros((1,11,11,11),bool)
    core[0,2,2,2]=True;core[0,8,8,8]=True
    seed=np.zeros_like(core);seed[0,2,2,2]=True
    assert np.array_equal(local_guard(core,seed)[0],core)
    actual=all_witness_guard(core,seed)
    expected=core.copy();expected[0,1:4,1:4,1:4]=True
    assert np.array_equal(actual,expected)
    assert actual.sum()==28 and not actual[0,4,2,2] and not actual[0,7,8,8]
    assert np.array_equal(all_witness_guard(core,seed),actual)


def test_four_policy_archive_immutable_and_subset(tmp_path):
    args=fixture_args(tmp_path);before=args.checkpoint.read_bytes()
    r=build_four(args)
    local=np.load(args.output/'local_new_seed_halo1/grid.npz')['active']
    with np.load(args.output/'local_all_witness_halo1/grid.npz') as z:
        active=z['active'];source=np.load(args.grid)
        assert not (local & ~active).any()
        assert not (active & ~z['witness_union_halo1']).any()
        assert z['core'].tobytes()==source['core'].tobytes()
        assert z['ema'].tobytes()==source['ema'].tobytes()
        assert z['ema'].dtype==source['ema'].dtype and z['ema'].shape==source['ema'].shape
    assert r['counts']['local_all_witness_halo1']['all_point_misses']==0
    assert r['counts']['local_all_witness_halo1']['subject_point_misses']==0
    assert args.checkpoint.read_bytes()==before
