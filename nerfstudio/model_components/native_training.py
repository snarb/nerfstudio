"""Center-aligned high-resolution color sampling for opaque training rays."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def sample_native_rgb(images, camera, yx, base_height, base_width):
    """Bilinear sample uint8 NHWC images at base-image pixel-center coordinates.

    Coordinates use camera convention (.5 is the first center). Different native
    width/height scale factors are supported. No rounding of subpixel rays.
    """
    height, width = images.shape[1:3]
    xy = yx[:, [1, 0]]*yx.new_tensor([width/base_width, height/base_height])-.5
    xy = torch.minimum(torch.maximum(xy, torch.zeros_like(xy)), xy.new_tensor([width-1, height-1]))
    low = xy.floor().long(); fraction = xy-low
    high = torch.minimum(low+1, low.new_tensor([width-1, height-1]))
    result = torch.zeros((len(camera), 3), device=images.device, dtype=torch.float32)
    for dx in [0, 1]:
        for dy in [0, 1]:
            x = high[:, 0] if dx else low[:, 0]; y = high[:, 1] if dy else low[:, 1]
            weight = (fraction[:, 0] if dx else 1-fraction[:, 0])*(fraction[:, 1] if dy else 1-fraction[:, 1])
            result += images[camera, y, x].float()*weight[:, None]/255
    return result


class NativeTrainingTargets:
    """Train-only native targets, gated by a conservative opaque interior.

    Uncertain boundaries keep their original targets/rays. This avoids treating
    photographed background color as isolated foreground. No anatomical labels,
    query-view RGB, source-camera switching, or post-render patches are used.
    """
    def __init__(self, manifest, dataset, device, exclude=None):
        manifest = Path(manifest)
        metadata = json.loads(manifest.read_text())
        if not metadata['train_only']:
            raise ValueError('Native targets must be training-only')
        by_name = {r['stem']: r for r in metadata['records']}
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.images = []; cores = []
        root = Path(dataset.metadata['distillation_root'])
        if exclude is not None and len(exclude)!=len(dataset.image_filenames):raise ValueError('Native exclusion camera count mismatch')
        for index,(image_path, row) in enumerate(zip(dataset.image_filenames, dataset.metadata['distillation_rows'])):
            record = by_name[image_path.stem]; path = manifest.parent/record['file']
            if digest(image_path) != record['hd_sha256'] or digest(path) != record['native_sha256']:
                raise ValueError('Native training image identity mismatch')
            rgb = np.load(path, allow_pickle=False)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError('Expected uint8 native RGB')
            self.images.append(torch.from_numpy(rgb).to(device))
            mask = np.array(Image.open(root/row['mask_path'])) > 0
            if 'alpha_file_path' in row:
                mask &= np.array(Image.open(root/row['alpha_file_path'])) == 255
            if exclude is not None:
                excluded=exclude[index].detach().cpu().numpy()
                if excluded.shape!=mask.shape or excluded.dtype!=np.bool_:raise ValueError('Invalid native exclusion mask')
                mask &= ~excluded
            core = torch.from_numpy(mask).to(device).float()[None, None]
            # A full one-pixel neighborhood must be opaque before jittering.
            core = 1-F.max_pool2d(F.pad(1-core, (1, 1, 1, 1), value=1), 3, stride=1)
            cores.append(core[0, 0].bool())
        self.images = torch.stack(self.images)
        self.core = torch.stack(cores)
        self.height, self.width = self.core.shape[1:]

    def apply(self, cameras, batch, jitter=True, observed=None):
        device = self.images.device
        camera, y, x = batch['indices'].to(device).T
        eligible = self.core[camera, y, x] & (batch['mask'].to(device).reshape(-1) > 0)
        center = torch.stack([y, x], -1).float()+.5
        offset=torch.rand_like(center)-.5 if jitter else torch.zeros_like(center)
        extra=torch.zeros_like(eligible)
        if observed is not None:
            active=batch['observed_background_valid'].to(device).reshape(-1)>0
            if (active&eligible).any():raise ValueError('Native opaque and observed composite targets overlap')
            background,safe=observed.sample_subpixel(camera,center+offset)
            extra=active&safe
        coords=center+offset*(eligible|extra)[:,None]
        color = sample_native_rgb(self.images, camera, coords, self.height, self.width)
        # Native photograph is a composite on measured-background rays. Never
        # reinterpret it as isolated foreground; the composite objective owns it.
        batch = dict(batch)
        for key in ['image', 'foreground_target']:
            old = batch[key].to(device)
            select=eligible|extra if key=='image' else eligible
            batch[key] = torch.where(select[:, None], color, old)
        if observed is not None:
            old=batch['observed_background_rgb'].to(device)
            batch['observed_background_rgb']=torch.where(extra[:,None],background,old)
            batch['native_observed_valid']=extra[:,None]
            batch['native_sample_coords']=coords
            if not hasattr(self,'observed_sampling_summary'):
                alpha=batch['alpha_target'].to(device).reshape(-1)
                self.observed_sampling_summary=dict(rays=len(camera),observed=int(active.sum()),
                    native_observed=int(extra.sum()),native_positive_partial=int((extra&(alpha>.02)&(alpha<.999)).sum()),
                    fallback_incomplete_footprint=int((active&~safe).sum()),
                    protocol='Native photo and bilinear measured HD background at identical subpixel coordinates; opaque native patches unchanged')
        rays = cameras.to(device).generate_rays(camera_indices=camera[:, None], coords=coords)
        return rays, batch
