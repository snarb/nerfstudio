import copy,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_bag_loss_correction import apply_correction,corrected_confidence,proposal_means,sha
from marshawn_tuning_sampling import weighted_charbonnier

def row(shape=(2,2000)):
    n=int(np.prod(shape));w=np.ones(shape,np.float32);core=np.ones(shape,bool)
    return dict(weight=w,W64=w,grey=core.copy(),black=np.zeros(shape,bool),core=core,band=np.zeros(shape,bool),valid=core.copy(),pools=dict(core=np.arange(n),band=np.array([],dtype=int),valid=np.arange(n)),fas=dict(probabilities=np.array([1.]+[0.]*15),cells=[[np.arange(n)]]+[[]for _ in range(15)]),proposal_means=np.ones(4))
def pool(r,k):return r['pools'][k]if len(r['pools'][k])else r['pools']['core']

def test_only_intersection_changes_and_raw_alias_remains():
    r=row((2,3));r['grey'][0,1]=False;e=np.zeros((2,3),bool);e[0,:2]=True;u=np.zeros_like(e);u[1,2]=True
    w,chosen=corrected_confidence(r,e,u,1)
    assert w[0,0]==0 and w[0,1]==1 and w[1,2]==1
    np.testing.assert_array_equal(r['weight'],np.ones((2,3)))
    assert chosen.sum()==1

@pytest.mark.parametrize('kind',['black','shell','count','dtype'])
def test_unsafe_source_rejected(kind):
    r=row((2,3));e=np.zeros((2,3),bool);e[0,0]=True;u=np.zeros_like(e);count=1
    if kind=='black':r['black'][0,0]=True
    if kind=='shell':u[0,0]=True
    if kind=='count':count=2
    if kind=='dtype':e=e.astype(np.uint8)
    with pytest.raises(ValueError):corrected_confidence(r,e,u,count)

def test_exact_proposal_mean_and_zero_weight_gradient():
    r=row((2,3));r['weight']=np.array([[0.,0.,.25],[.5,.75,1.]],np.float32)
    np.testing.assert_allclose(proposal_means(SimpleNamespace(_pool=pool),r),np.full(4,r['weight'].mean(dtype=np.float64)),rtol=0,atol=0)
    pred=torch.full((6,3),.5,requires_grad=True);target=torch.zeros_like(pred);w=torch.from_numpy(r['weight'].reshape(-1))
    loss=weighted_charbonnier(pred,target,w,float(w.mean()));loss.backward()
    assert torch.equal(pred.grad[:2],torch.zeros_like(pred.grad[:2])) and bool((pred.grad[2:]>0).all()) and bool(torch.isfinite(pred.grad).all())

def test_binding_rng_and_resume_tamper(tmp_path):
    cameras=['MA','MB','MC','NA','NB','ND'];counts=[1388,1229,1057,930,1141,777];rows=[row()for _ in range(63)]
    source=[dict(camera=c,mask_sha256='a'*64)for c in cameras+[f'other{i}'for i in range(57)]];records=[]
    for c,n in zip(cameras,counts):
        e=np.zeros((2,2000),bool);e.flat[:n]=True;u=np.zeros_like(e);u[-1,-1]=True
        np.save(tmp_path/f'{c}_bag_exclusion_HD.npy',e);np.save(tmp_path/f'{c}_bag_unknown_HD.npy',u)
        records.append(dict(camera=c,exclusion=f'{c}_bag_exclusion_HD.npy',unknown=f'{c}_bag_unknown_HD.npy',false_grey_pixels=n,original_semantic_sha256='a'*64))
    meta=dict(kind='QUALIFIED_TRAIN_FALSE_GREY_RGB_WEIGHT_ONLY',expected_false_grey_pixels=6522,uses_EVAL=False,fixed_admission_changed=False,proposal_pools_changed=False,records=records,files_sha256={p.name:sha(p)for p in tmp_path.iterdir()})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(meta));loaded=[]
    adapter=SimpleNamespace(objective='actor',sampling='body80',width=64,rows=rows,source_records=source,_pool=pool,state_dict=lambda:dict(step=17),load_state_dict=lambda s:loaded.append(copy.deepcopy(s)))
    rng=pickle.dumps(np.random.get_state());original=[r['W64'].copy()for r in rows];receipt=apply_correction(adapter,path,sha(path))
    assert pickle.dumps(np.random.get_state())==rng and receipt['changed_RGB_confidence_pixels']==6522
    for r,w in zip(rows,original):np.testing.assert_array_equal(r['W64'],w)
    state=adapter.state_dict();adapter.load_state_dict(state);assert loaded[0]['step']==17
    state['bag_loss_correction']['manifest_sha256']='tampered'
    with pytest.raises(ValueError,match='Resume'):adapter.load_state_dict(state)
    with pytest.raises(ValueError,match='already'):apply_correction(adapter,path,sha(path))
    assert adapter.state_dict()['bag_loss_correction']['manifest_sha256']==sha(path)
