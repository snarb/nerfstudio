import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_marshawn_minimal_guard import derive,build

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_minimal_union_is_not_halo_and_onehalo_is_noncumulative():
    core=np.zeros((1,7,7,7),bool);core[0,5,5,5]=True;before=core.copy()
    masks=derive(core,np.array([[0,0,0,7,7,7]]),np.array([[1.2,1.2,1.2],[1.3,1.3,1.3]]))
    assert np.array_equal(core,before)
    assert masks['minimal_guard'].sum()==2  # Duplicate witnesses share one cell.
    assert not masks['minimal_guard'][0,2,1,1]
    assert masks['witness_union_halo1'][0,2,1,1]
    assert not masks['witness_union_halo1'][0,3,1,1] # Would be admitted by halo2.
    assert not masks['ordinary_halo1'][0,1,1,1]

def fixture_args(tmp_path,radius=0):
    core=np.zeros((1,5,5,5),bool);core[0,3,3,3]=True
    active=core.copy()
    if radius:
        active=derive(core,np.array([[0,0,0,1,1,1]]),np.array([[.7,.7,.7]]))['ordinary_halo1']
    ema=np.arange(125,dtype=np.float32);boxes=np.array([[0,0,0,1,1,1]],np.float32)
    grid=tmp_path/'source.npz';np.savez_compressed(grid,core=core,active=active,ema=ema,
        resolution=np.array([5]*3),world_aabb=boxes.reshape(2,3),cascade_aabbs=boxes)
    checkpoint=tmp_path/'step-000008000.ckpt';checkpoint.write_bytes(b'IMMUTABLE SYNTHETIC CHECKPOINT; no field')
    receipt=tmp_path/'source.json';receipt.write_text(json.dumps(dict(radius=radius,
        grid_archive_sha256=sha(grid),checkpoint_sha256=sha(checkpoint),
        core_cells=int(core.sum()),active_cells=int(active.sum()),checkpoint_step=8000,
        core_update_step=8000,native_period=16)))
    points=np.random.default_rng(42).uniform(.05,.1,(9022,3));w=tmp_path/'w.npz';np.savez(w,xyz=points)
    subject=tmp_path/'s.npz';np.savez(subject,xyz=np.r_[points[:495],points[:6]])
    return SimpleNamespace(grid=grid,grid_receipt=receipt,checkpoint=checkpoint,
        witnesses=w,subject=subject,output=tmp_path/'derived',expected_step=8000,mvs=None)

def test_roundtrip_covers_all_and_preserves_checkpoint_core_ema(tmp_path):
    a=fixture_args(tmp_path);before={p:sha(p)for p in [a.grid,a.grid_receipt,a.checkpoint,a.witnesses,a.subject]}
    r=build(a);assert r['no_halo_minimal_added_cells']==1
    assert r['counts']['minimal_guard']['all_point_misses']==0
    assert r['counts']['minimal_guard']['subject_points']==495
    assert all(sha(p)==h for p,h in before.items())
    source=np.load(a.grid)
    for name in ['minimal_guard','witness_union_halo1']:
        z=np.load(a.output/name/'grid.npz')
        assert z['ema'].shape==source['ema'].shape
        assert z['ema'].tobytes()==source['ema'].tobytes()
        assert z['core'].tobytes()==source['core'].tobytes()
    assert not r['field_instantiated'] and not r['worker_mask_changed']

def test_rejects_stored_r1_instead_of_redilating(tmp_path):
    a=fixture_args(tmp_path,radius=1)
    with pytest.raises(ValueError,match='R0 core'):build(a)
    assert not a.output.exists()

def test_rejects_wrong_selected_step_without_fallback(tmp_path):
    a=fixture_args(tmp_path);a.expected_step=6000
    with pytest.raises(ValueError,match='step mismatch'):build(a)
    assert not a.output.exists()

def test_rejects_outside_witness_without_clamping():
    with pytest.raises(ValueError,match='outside'):
        derive(np.zeros((1,5,5,5),bool),np.array([[0,0,0,1,1,1]]),np.array([[1.001,.5,.5]]))
