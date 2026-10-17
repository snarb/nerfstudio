"""Reuse the frozen evaluation LPIPS metric for TRAIN-only patch confidence.

This module has no image paths, camera IDs, target GT, masks or reconstruction
model. Inputs are geometrically supported TRAIN comparisons supplied by caller.
"""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import torch
from luster_hybrid_common import read, sha, write
from transfer_luster_hybrid import gaussian


def metric_state_sha(metric):
    digest=hashlib.sha256()
    for key,tensor in sorted(metric.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def load_metric(root,device='cuda'):
    import lpips
    import torchvision
    metric=lpips.LPIPS(net='alex',spatial=True).eval().to(device)
    metric.requires_grad_(False)
    code=Path(inspect.getfile(lpips.LPIPS))
    pretrained=Path(lpips.__file__).parent/'pretrained_networks.py'
    linear_weights=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    trunk_weights=Path(torch.hub.get_dir())/'checkpoints'/Path(torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url).name
    assert linear_weights.exists() and trunk_weights.exists()
    receipt=dict(metric='existing evaluation LPIPS AlexNet spatial v0.1',
        metric_state_sha256=metric_state_sha(metric),torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        device=device,training=False,parameters_frozen=True,
        inputs={str(p):sha(p) for p in [code,pretrained,Path(lpips.__file__),linear_weights,trunk_weights,Path(__file__)]},
        uses_target_gt=False,uses_evaluation_masks=False,new_reconstruction_model=False)
    path=root/'train_perceptual_utility/metric_receipt.json'
    if path.exists():assert read(path)==receipt
    else:write(path,receipt)
    return metric,path


def utility_from_errors(before,after,support,floor=.001):
    before=np.asarray(before,np.float32);after=np.asarray(after,np.float32)
    assert before.shape==after.shape==support.shape
    assert np.isfinite(before[support]).all() and np.isfinite(after[support]).all()
    before=np.where(support,np.maximum(before,0),0)
    after=np.where(support,np.maximum(after,0),0)
    mass=gaussian(support.astype(np.float32),16.)
    total=before+after
    known=support&(mass>=.8)&(total>=floor)
    utility=(before-after)/(total+floor)
    return np.where(known,np.clip(utility,-1,1),0).astype(np.float32),known


def perceptual_utility(candidate,witness,baseline,support,metric):
    """Same neutral exterior for all three native TRAIN-frame RGB arrays."""
    assert candidate.shape==witness.shape==baseline.shape==(*support.shape,3)
    if not support.any():return np.zeros(support.shape,np.float32),support.copy()
    assert not metric.training and not any(p.requires_grad for p in metric.parameters())
    device=next(metric.parameters()).device
    def tensor(rgb):
        assert np.isfinite(rgb[support]).all()
        value=np.where(support[...,None],np.clip(rgb,0,1),.5).astype(np.float32)
        return torch.from_numpy(value).permute(2,0,1)[None].to(device)*2-1
    with torch.inference_mode():
        truth=tensor(witness)
        before=metric(tensor(baseline),truth)[0,0].cpu().numpy()
        after=metric(tensor(candidate),truth)[0,0].cpu().numpy()
    return utility_from_errors(before,after,support)
