"""CPU-only audit of saved actual patch RGB/order and SSIM pixel derivatives."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def manual_map(pred,target,padded=True):
    x=torch.arange(-5,6,dtype=pred.dtype);g=torch.exp(-x.square()/(2*1.5**2));g/=g.sum()
    kernel=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
    if padded:pred=F.pad(pred,(5,5,5,5),mode='reflect');target=F.pad(target,(5,5,5,5),mode='reflect')
    moments=F.conv2d(torch.cat([pred,target,pred*pred,target*target,pred*target]),kernel,groups=3)
    mp,mt,pp,tt,pt=moments.chunk(5)
    vp=(pp-mp.square()).clamp_min(0);vt=(tt-mt.square()).clamp_min(0);cov=pt-mp*mt
    luminance=(2*mp*mt+.01**2)/(mp.square()+mt.square()+.01**2)
    structure=(2*cov+.03**2)/(vp+vt+.03**2)
    return luminance*structure,luminance,structure


def effects(grad,pred):
    centered=pred-pred.mean((2,3),keepdim=True)
    return dict(pixel_gradient_l2=float(grad.square().sum().sqrt()),
        brightness_derivative_RGB=grad.sum((0,2,3)).tolist(),
        brightness_derivative_all=float(grad.sum()),
        contrast_derivative=float((grad*centered).sum()),
        convention='Positive derivative means gradient descent lowers additive brightness or multiplicative contrast; pixel-space only, not a prediction of coupled field updates')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--packet',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2);a.output.mkdir(exist_ok=False)
    saved=np.load(a.packet);probe=json.loads(a.probe.read_text());index=saved['indices'];rgb=saved['prediction'];gt=saved['target']
    assert rgb.shape==gt.shape==(len(index),3)and len(index)==968
    pred=torch.tensor(rgb.reshape(-1,11,11,3).transpose(0,3,1,2),requires_grad=True)
    target=torch.tensor(gt.reshape(-1,11,11,3).transpose(0,3,1,2))
    actual=ssim(pred,target,data_range=1.,gaussian_kernel=True,sigma=1.5,kernel_size=11,k1=.01,k2=.03,reduction='none')
    values,luminance,structure=manual_map(pred,target);manual=values.mean((1,2,3))
    assert float((manual-actual).abs().max())<.001
    global_score=ssim(pred,target,data_range=1.,gaussian_kernel=True,sigma=1.5,kernel_size=11,k1=.01,k2=.03)
    assert abs(float(global_score-actual.mean()))<1e-6
    gd=torch.autograd.grad(.01*(1-global_score),pred,retain_graph=True)[0]
    gm=torch.autograd.grad(.01*(1-manual.mean()),pred,retain_graph=True)[0]
    valid,_,_=manual_map(pred,target,padded=False)
    gv=torch.autograd.grad(.01*(1-valid.mean()),pred,retain_graph=True)[0]
    mse=(pred-target).square().mean();gr=torch.autograd.grad(mse,pred,retain_graph=True)[0]
    frames={f['physical_camera']:f for f in json.loads((a.data/'transforms.json').read_text())['frames']}
    rows=[]
    for i,row in enumerate(probe['rows']):
        ids=index[i*121:(i+1)*121];assert len(set(ids[:,0]))==1
        yy,xx=np.meshgrid(np.arange(11),np.arange(11),indexing='ij');assert np.array_equal(ids[:,1].reshape(11,11),yy+ids[0,1]);assert np.array_equal(ids[:,2].reshape(11,11),xx+ids[0,2])
        image=np.asarray(Image.open(a.data/frames[row['camera']]['file_path']).convert('RGB'))
        independent=image[ids[:,1],ids[:,2]].astype(np.float32)/255
        assert np.array_equal(independent,gt[i*121:(i+1)*121])
        gd_i=gd[i:i+1];gv_i=gv[i:i+1]
        cosine=float((gd_i*gv_i).sum()/(gd_i.norm()*gv_i.norm()).clamp_min(1e-20))
        rows.append(dict(kind=row['kind'],camera=row['camera'],x=row['x'],y=row['y'],actual_group_camera_id=int(ids[0,0]),
            target_RGB_equals_independent_PNG=True,pred_mean_RGB=pred[i].detach().mean((1,2)).tolist(),target_mean_RGB=target[i].mean((1,2)).tolist(),
            pred_std_RGB=pred[i].detach().std((1,2)).tolist(),target_std_RGB=target[i].std((1,2)).tolist(),
            ssim=float(actual[i]),manual_ssim=float(manual[i]),valid_center_ssim=float(valid[i].mean()),
            mean_luminance_term=float(luminance[i].mean()),mean_structure_term=float(structure[i].mean()),
            MSE=effects(gr[i:i+1],pred[i:i+1]),DSSIM001=effects(gd_i,pred[i:i+1]),
            reflected_vs_valid_gradient_cosine=cosine))
    report=dict(passed=True,torchmetrics_version=torchmetrics.__version__,CPU_only=True,no_field_or_optimizer_loaded=True,
        packet_sha256=hashlib.sha256(a.packet.read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),
        all_patch_cameras_and_rowmajor_HWC_verified=True,all_targets_bitexact_RGB_uint8_div255=True,
        no_gamma_PQ_channel_swap_or_extra_normalization=True,functional_reduction='mean over11x11 positions and3channels, then mean over8patches',
        explicit_data_range=1.,autocast=False,gaussian_kernel_size=11,gaussian_sigma=1.5,
        reflected_windows_per_patch=120,valid_nonreflected_windows_per_patch=1,
        manual_score_max_error=float((manual-actual).abs().max()),
        manual_gradient_relative_L2_error=float((gd-gm).norm()/gd.norm()),rows=rows,
        limitations=['Pixel derivative signs do not prove the sign of learned RGB after Adam/shared-field coupling.','Reflecting an11px patch for an11px window differs from full-image SSIM using actual neighboring pixels.','Patch sampler distribution and dynamic initialization changed alongside the loss; no matched patch-only control was trained.'])
    (a.output/'pixel_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,score_error=report['manual_score_max_error'],gradient_error=report['manual_gradient_relative_L2_error'])))


if __name__=='__main__':main()
