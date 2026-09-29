"""Train-only native pixel patches inside verified opaque foreground.

Patch objectives are optional additions to the LookCloser ray objective. They
use no anatomy, evaluation images, or silhouette replacement.
"""
import math

import torch
import torch.nn.functional as F


class NativeOpaquePatches:
    def __init__(self, targets, size=16, seed=1447, confidence=None):
        self.targets = targets
        self.size = size
        height, width = targets.core.shape[1:]
        rh, rw = targets.images.shape[1]/height, targets.images.shape[2]/width
        if rh != rw or not rh.is_integer() or size < 11:
            raise ValueError('Native patches require equal integer scale and size >= 11')
        self.ratio = int(rh)
        if confidence is not None and confidence.shape != targets.core.shape:
            raise ValueError('Confidence must match the base-image core masks')
        footprint = math.ceil(size/self.ratio)
        self.anchors = []
        for i,core in enumerate(targets.core):
            if confidence is not None: core=core & (confidence[i].to(core.device)>=1.)
            valid = F.max_pool2d((~core).float()[None,None], footprint, stride=1)[0,0] == 0
            self.anchors.append(torch.nonzero(valid).to(torch.int32))
        self.cameras = [i for i,a in enumerate(self.anchors) if len(a)]
        if not self.cameras: raise ValueError('No fully opaque native training patches')
        self.generator = torch.Generator(device=targets.images.device).manual_seed(seed)

    def sample(self, count, cameras):
        if count < 1: raise ValueError('Positive patch count required')
        device = self.targets.images.device
        selected = torch.randint(len(self.cameras), (count,), generator=self.generator, device=device).tolist()
        views = torch.tensor([self.cameras[i] for i in selected], device=device)
        starts = []
        for view in views.tolist():
            pool = self.anchors[view]
            index = torch.randint(len(pool), (), generator=self.generator, device=device)
            starts.append(pool[index].long()*self.ratio)
        starts = torch.stack(starts)
        yy,xx = torch.meshgrid(torch.arange(self.size,device=device),torch.arange(self.size,device=device),indexing='ij')
        yx = starts[:,None,None,:]+torch.stack([yy,xx],-1)[None]
        indices = views[:,None,None].expand(-1,self.size,self.size)
        rgb = self.targets.images[indices,yx[...,0],yx[...,1]].float()/255
        coords = (yx.float()+.5)/self.ratio
        rays = cameras.to(device).generate_rays(camera_indices=indices.reshape(-1,1),coords=coords.reshape(-1,2))
        return rays, rgb, dict(indices=indices, native_yx=yx, base_coords=coords)


def patch_objective(prediction, target, ssim_weight=0.):
    """Paper Charbonnier epsilon; optional spatial SSIM on native RGB patches."""
    from nerfstudio.model_components.mesh_distillation import weighted_charbonnier
    from pytorch_msssim import ssim
    prediction = prediction.reshape_as(target).float()
    objective = weighted_charbonnier(prediction,target,torch.ones_like(target[...,:1]))
    if ssim_weight:
        with torch.autocast(device_type=prediction.device.type,enabled=False):
            structural = ssim(prediction.permute(0,3,1,2),target.float().permute(0,3,1,2),
                              data_range=1.,size_average=True,win_size=11)
        objective = objective + ssim_weight*(1-structural)
    return objective
