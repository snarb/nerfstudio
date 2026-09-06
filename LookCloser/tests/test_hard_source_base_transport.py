from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hard_source_base_transport import anchored_region,transport_source_bases


def test_exposed_chroma_is_gain_invariant_and_clipped_rgb_unreliable():
    from hard_source_base_transport import exposed_chromaticity
    from patchmatch_color_calibration import apply_camera_gain
    rgb=torch.tensor([.5,.35,.2])[:,None,None].expand(3,3,4)
    first,valid=exposed_chromaticity(rgb)
    second,valid2=exposed_chromaticity(apply_camera_gain(rgb,[1.7]*3))
    torch.testing.assert_close(first,second,atol=2e-7,rtol=0)
    assert valid.all() and valid2.all()
    assert not exposed_chromaticity(torch.ones_like(rgb))[1].any()


def test_incompatible_same_point_color_is_not_a_transport_seed():
    a,b,labels,pred,valid,depth=fixture()
    b=torch.tensor([.6,.4,.2])[:,None,None].expand_as(b).clone()
    pred=torch.where((labels==1)[None],b,a)
    out,_,stats=transport_source_bases(pred,labels,[a,b],valid,depth,
        max_color_jump=.04,max_seed_chroma_difference=.04)
    assert torch.equal(out,pred) and stats['patches'][0]['seeds']==0
    assert stats['patches'][0]['rejected_chroma_seed_edges']>0


def test_luminance_transport_preserves_exposed_chromaticity_and_primary():
    from hard_source_base_transport import exposed_chromaticity
    a,b,labels,pred,valid,depth=fixture()
    b=torch.tensor([.6,.4,.2])[:,None,None].expand_as(b).clone()
    pred=torch.where((labels==1)[None],b,a)
    out,_,stats=transport_source_bases(pred,labels,[a,b],valid,depth,luminance_only=True)
    old,_=exposed_chromaticity(pred);new,_=exposed_chromaticity(out)
    torch.testing.assert_close(old,new,atol=3e-7,rtol=0)
    assert torch.equal(out[:,labels==0],pred[:,labels==0])
    assert stats['exposed_chromaticity_preserved']
    assert stats['patches'][0]['gain_min']>=.5 and stats['patches'][0]['gain_max']<=2.


def test_bounded_response_enforces_gain_and_chroma_limits():
    from hard_source_base_transport import bound_rgb_response,exposed_chromaticity
    torch.manual_seed(42)
    original=torch.rand((3,12,14))*.7+.1;requested=torch.rand_like(original)*2-.5
    out=bound_rgb_response(original,requested)
    old_chroma,_=exposed_chromaticity(original);new_chroma,_=exposed_chromaticity(out)
    assert float((old_chroma-new_chroma).square().mean(0).sqrt().max())<.025001
    def inverse(rgb):
        linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
        return linear/(1-linear)
    gain=inverse(out)/inverse(original)
    assert float(gain.min())>=.499998 and float(gain.max())<=2.000005
    assert torch.equal(bound_rgb_response(torch.zeros_like(original),requested),torch.zeros_like(original))


def test_bounded_rgb_keeps_primary_and_corrects_constant_secondary():
    a,b,labels,pred,valid,depth=fixture()
    out,_,stats=transport_source_bases(pred,labels,[a,b],valid,depth,bounded_rgb=True)
    torch.testing.assert_close(out,a,atol=2e-5,rtol=0)
    assert torch.equal(out[:,labels==0],pred[:,labels==0])
    assert stats['maximum_exposed_chroma_rms_shift']==.025


def fixture():
    torch.set_num_threads(4)
    source=torch.full((3,32,40),.4);shifted=source+.1
    labels=torch.zeros((32,40),dtype=torch.long);labels[8:24,10:30]=1
    prediction=torch.where((labels==1)[None],shifted,source)
    valid=[torch.ones_like(labels,dtype=torch.bool)]*2
    depth=torch.ones_like(labels,dtype=torch.float32)
    return source,shifted,labels,prediction,valid,depth


def test_constant_camera_patch_is_corrected_and_primary_unchanged():
    a,b,labels,pred,valid,depth=fixture()
    out,offset,stats=transport_source_bases(pred,labels,[a,b],valid,depth)
    torch.testing.assert_close(out,a,atol=2e-5,rtol=0)
    assert torch.equal(out[:,labels==0],pred[:,labels==0])
    assert not stats['source_rgb_averaging'] and stats['source_labels_unchanged']
    assert torch.equal(offset[:,labels==0],torch.zeros_like(offset[:,labels==0]))


def test_broad_interior_shading_is_replaced_not_merely_offset_at_seam():
    a,b,labels,pred,valid,depth=fixture()
    yy,xx=torch.meshgrid(torch.arange(32),torch.arange(40),indexing='ij')
    hump=.1*torch.exp(-((xx-20)**2+(yy-16)**2)/30)
    b=a+hump;pred=torch.where((labels==1)[None],b,a)
    out,_,_=transport_source_bases(pred,labels,[a,b],valid,depth,base_smoothness=4.)
    assert float((out-a).abs()[:,labels==1].mean()) < .45*float((pred-a).abs()[:,labels==1].mean())


def test_true_depth_boundary_and_unanchored_patch_unchanged():
    a,b,labels,pred,valid,depth=fixture();depth[labels==1]=2.
    out,_,stats=transport_source_bases(pred,labels,[a,b],valid,depth)
    assert torch.equal(out,pred) and stats['patches'][0]['anchored_pixels']==0


def test_invalid_source_pixels_do_not_enter_lowpass_or_boundary_seeds():
    a,b,labels,pred,valid,depth=fixture()
    valid=[valid[0].clone(),valid[1].clone()];valid[1][:3]=False
    out,_,_=transport_source_bases(pred,labels,[a,b],valid,depth)
    b=b.clone();b[:,~valid[1]]=100
    changed,_,_=transport_source_bases(pred,labels,[a,b],valid,depth)
    assert torch.equal(out,changed)


def test_zero_strength_and_one_source_are_exact_identity():
    a,b,labels,pred,valid,depth=fixture()
    out,_,_=transport_source_bases(pred,labels,[a,b],valid,depth,strength=0)
    assert torch.equal(out,pred)
    labels[:]=0
    out,_,_=transport_source_bases(a,labels,[a],valid[:1],depth)
    assert torch.equal(out,a)


def test_secondary_fine_detail_is_retained_not_replaced_by_primary_detail():
    a,b,labels,pred,valid,depth=fixture()
    yy,xx=torch.meshgrid(torch.arange(32),torch.arange(40),indexing='ij')
    checker=((yy+xx)%2)*.04-.02
    b=b+checker;pred=torch.where((labels==1)[None],b,a)
    out,_,_=transport_source_bases(pred,labels,[a,b],valid,depth,base_smoothness=16.)
    # Primary has no checker detail. A shared RGB average would reduce this
    # secondary detail much more than its explicitly defined low-pass residual.
    assert .018<float(out[:,12:20,14:26].std())<.021


@pytest.mark.parametrize('value',[0.,-1.,np.nan,np.inf])
def test_invalid_base_scale_fails_closed(value):
    a,b,labels,pred,valid,depth=fixture()
    with pytest.raises(ValueError):
        transport_source_bases(pred,labels,[a,b],valid,depth,base_smoothness=value)


def test_anchoring_respects_depth_graph_components_and_missing_depth():
    region=np.ones((3,6),bool);depth=np.ones((3,6));depth[:,3:]=2.
    seeds=np.zeros_like(region);seeds[1,0]=True
    anchored=anchored_region(region,depth,seeds)
    assert anchored[:,:3].all() and not anchored[:,3:].any()
    depth[:,1]=0
    anchored=anchored_region(region,depth,seeds)
    assert anchored[:,0].all() and not anchored[:,1:].any()


def test_color_edge_keeps_unanchored_material_base_instead_of_gray_transport():
    a,b,labels,pred,valid,depth=fixture()
    b=b.clone();b[:,8:24,10:30]=torch.tensor([.65,.4,.2])[:,None,None]
    pred=torch.where((labels==1)[None],b,a)
    out,_,stats=transport_source_bases(pred,labels,[a,b],valid,depth,max_color_jump=.04)
    assert torch.equal(out,pred) and stats['patches'][0]['anchored_pixels']==0


def test_color_anchoring_cuts_the_same_graph_as_solver():
    region=np.ones((3,6),bool);depth=np.ones((3,6));seeds=np.zeros_like(region);seeds[1,0]=True
    guide=np.full((3,6,3),.4);guide[:,3:]=.8
    anchored=anchored_region(region,depth,seeds,guide=guide,max_color_jump=.04)
    assert anchored[:,:3].all() and not anchored[:,3:].any()


def test_invisible_selected_source_fails_closed():
    a,b,labels,pred,valid,depth=fixture();valid[1]=torch.zeros_like(labels,dtype=torch.bool)
    with pytest.raises(ValueError,match='invisible'):
        transport_source_bases(pred,labels,[a,b],valid,depth)
