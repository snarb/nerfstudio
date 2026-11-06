import json,sys,hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_semantic_admission import AdmissionLookup,install_density_admission,load_support,sha
from marshawn_semantic_probe_checks import gradient_max_difference
from marshawn_semantic_cell_admission import WholeCellAdmission
from marshawn_semantic_occupancy import validate_policy

class Field:
    def __init__(self):self.weight=torch.tensor(1.,requires_grad=True)
    def _normalize_positions(self,p):
        selector=((p>0)&(p<1)).all(-1)
        return (p*selector[...,None]).reshape(-1,3),selector,p.shape[:-1],p.reshape(-1,3)
    def get_density(self,p):
        _,selector,_,_=self._normalize_positions(p)
        return torch.exp(p.sum(-1)*self.weight)*selector
    def query_points(self,p):return self.get_density(p)
    def density_fn(self,p):return self.get_density(p)


def test_all_ones_is_bitwise_density_gradient_and_rng_identity():
    p=torch.tensor([[.2,.3,.4],[.8,.9,.7],[0.,.4,.4]])
    a,b=Field(),Field();rng=torch.random.get_rng_state().clone()
    install_density_admission(b,torch.ones(4,4,4,dtype=torch.bool),torch.tensor([[0.,0.,0.],[1.,1.,1.]]))
    for method in ['get_density','query_points','density_fn']:
        x,y=getattr(a,method)(p),getattr(b,method)(p)
        assert torch.equal(x,y)
        assert torch.equal(torch.autograd.grad(x.sum(),a.weight)[0],torch.autograd.grad(y.sum(),b.weight)[0])
    assert torch.equal(rng,torch.random.get_rng_state())


def test_all_paths_reject_sigma_and_gradient_outside_but_preserve_inside():
    mask=torch.zeros(4,4,4,dtype=torch.bool);mask[1,2,3]=True
    a,b=Field(),Field();install_density_admission(b,mask,torch.tensor([[0.,0.,0.],[1.,1.,1.]]))
    p=torch.tensor([[.3,.6,.9],[.1,.2,.3]])
    for method in ['get_density','query_points','density_fn']:
        x,y=getattr(a,method)(p),getattr(b,method)(p)
        assert y[0]==x[0]and y[1]==0
        assert torch.autograd.grad(y[1],b.weight)[0]==0


def test_lookup_xyz_floor_edges_and_outside_unknown():
    mask=torch.zeros(2,3,4,dtype=torch.bool);mask[1,2,3]=True
    lookup=AdmissionLookup(mask,torch.tensor([[0.,0.,0.],[2.,6.,12.]]))
    p=torch.tensor([[1.,4.,9.],[.999,4.,9.],[2.,6.,12.],[-.1,1.,1.]])
    assert lookup(p).tolist()==[True,False,True,True]
    with pytest.raises(ValueError):AdmissionLookup(mask,torch.zeros(2,3))


def test_halo_fresh_and_final_intersection_preserves_core():
    from marshawn_occupancy_audit import dilate_one
    core=np.zeros((1,5,5,5),bool);core[0,2,2,2]=True;before=core.copy()
    support=np.ones_like(core);support[:,:2]=False
    for warmup in (False,True):
        for r in (0,1):
            active=(np.ones_like(core)if warmup else core if r==0 else dilate_one(core))&support
            validate_policy(core,active,r,warmup,support)
            assert np.array_equal(core,before)
            wrong=active.copy();wrong[0,0,0,0]=True
            with pytest.raises(ValueError):validate_policy(core,wrong,r,warmup,support)


def fixture_source(tmp):
    source=tmp/'source';data=tmp/'data';source.mkdir();data.mkdir()
    cameras=[f'{i:02}'for i in range(63)];bounds=np.array([[0.,0.,0.],[1.,1.,1.]])
    ready=dict(aabb=bounds.tolist(),world_to_box=np.eye(4).tolist(),records=[dict(camera=c,image=f'images/train_{c}.png')for c in cameras])
    (data/'ready.json').write_text(json.dumps(ready));(data/'transforms.json').write_text('{}')
    base=np.zeros((128,)*3,bool);base[40:44,40:44,40:44]=True
    from scipy.ndimage import maximum_filter
    pad=maximum_filter(base,size=3,mode='constant',cval=0)
    np.savez(source/'semantic_grid.npz',semantic_gate0=base,semantic_gate_pad1=pad,aabb=bounds,world_to_box=np.eye(4))
    receipt=dict(no_eval_masks_or_RGB_read=True,native_pixel_support_width=64,train_cameras=cameras,grid_sha256=sha(source/'semantic_grid.npz'),transforms_sha256=sha(data/'transforms.json'),source_ready_sha256=sha(data/'ready.json'))
    (source/'grid_receipt.json').write_text(json.dumps(receipt));mapped=WholeCellAdmission(pad,bounds).map_target_grid([128]*3,bounds)
    meta=dict(kind='reviewed_TRAIN_W64_pad1_whole_LC_cells',train_camera_count=63,files_sha256={p.name:sha(p)for p in source.iterdir()},mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),all103_witnesses_retained=True,qualified_shoe_anchors_retained=2)
    manifest=source/'manifest.json';manifest.write_text(json.dumps(meta));return manifest,data,bounds


def test_source_hash_frame_and_eval_membership_tamper(tmp_path):
    manifest,data,bounds=fixture_source(tmp_path);digest=sha(manifest)
    mask,_=load_support(manifest,digest,data,bounds,[128]*3)
    assert mask.any()and not mask.all()
    with pytest.raises(AssertionError):load_support(manifest,digest,data,bounds+.01,[128]*3)
    (manifest.parent/'grid_receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):load_support(manifest,digest,data,bounds,[128]*3)
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='manifest changed'):load_support(manifest,digest,data,bounds,[128]*3)


def test_native_identity_probe_handles_empty_parameter_gradient():
    # Tiny-CUDA-NN exposes zero-length parameter tensors for some components.
    a,b=torch.empty(0),torch.empty(0)
    torch.testing.assert_close(a,b,rtol=1e-3,atol=3e-6)
    assert gradient_max_difference(a,b)==0.
    assert gradient_max_difference(torch.tensor([1.,3.]),torch.tensor([2.,4.]))==1.
    with pytest.raises(ValueError):gradient_max_difference(a,torch.zeros(1))

    with pytest.raises(ValueError):gradient_max_difference(a,a.double())
    with pytest.raises(ValueError):gradient_max_difference(None,None)
