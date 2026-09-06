from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hard_source_gradient_leveling import level_source_gradients


def fixture():
    torch.set_num_threads(4)
    source=torch.full((3,32,40),.4);source[:,::2]=.45
    shifted=source+.08
    selection=torch.zeros((32,40),dtype=torch.long);selection[:,20:]=1
    prediction=torch.where((selection==1)[None],shifted,source)
    return source,shifted,selection,prediction,[torch.ones_like(selection,dtype=torch.bool)]*2,torch.ones_like(selection,dtype=torch.float32)


def test_additive_seam_disappears_without_averaging_source_detail():
    source,shifted,labels,pred,valid,depth=fixture()
    out,offset,stats=level_source_gradients(pred,labels,[source,shifted],valid,depth)
    # Half of the original image is shifted by .08: the global gauge retains .04.
    torch.testing.assert_close(out,source+.04,atol=1e-5,rtol=0)
    assert abs(float(offset.mean()))<1e-8
    assert stats['source_labels_unchanged'] and not stats['source_rgb_averaging']
    assert stats['solver']['max_relative_residual']<5e-9


def test_true_depth_edge_and_no_common_source_are_not_color_leveled():
    source,shifted,labels,pred,valid,depth=fixture();depth[:,20:]=2
    out,_,_=level_source_gradients(pred,labels,[source,shifted],valid,depth)
    assert torch.equal(out,pred)
    valid=[labels==0,labels==1]
    out,_,stats=level_source_gradients(pred,labels,[source,shifted],valid,torch.ones_like(depth))
    assert torch.equal(out,pred) and stats['edges'][0]['unknown_seam_edges_disconnected']==32


def test_guidance_uses_one_visible_camera_and_ignores_invalid_rgb():
    source,shifted,labels,pred,valid,depth=fixture()
    valid=[torch.ones_like(labels,dtype=torch.bool),torch.ones_like(labels,dtype=torch.bool)]
    valid[0][:,20:]=False
    a,_,_=level_source_gradients(pred,labels,[source,shifted],valid,depth)
    poisoned=source.clone();poisoned[:,~valid[0]]=100
    b,_,stats=level_source_gradients(pred,labels,[poisoned,shifted],valid,depth)
    assert torch.equal(a,b) and stats['edges'][0]['guidance_source_edges']==[0,32]


def test_no_seam_and_empty_support_are_exact_identity():
    source,_,labels,_,valid,depth=fixture();labels[:]=0
    out,_,_=level_source_gradients(source,labels,[source],valid[:1],depth)
    assert torch.equal(out,source)
    labels[:]=-1
    out,_,_=level_source_gradients(source,labels,[source],valid[:1],depth*0)
    assert torch.equal(out,source)


def test_nonfinite_rgb_fails_closed():
    source,shifted,labels,pred,valid,depth=fixture();pred[0,0,0]=float('nan')
    with pytest.raises(ValueError):level_source_gradients(pred,labels,[source,shifted],valid,depth)
