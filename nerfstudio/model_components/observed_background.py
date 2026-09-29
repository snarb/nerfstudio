"""Optional train-only compositing evidence from measured background plates.

No alpha target is inferred. Unobserved pixels and opaque native-color rays keep
their original supervision. Temporal stability is fallible; uncertainty enters
the photometric residual instead of a hard opacity constraint.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def observed_composite_objective(actor_rgb,alpha,background,photo,valid,error):
    """Display-space robust interval residual; no gradient rewards opacity zero.

    The background error interval is independent of predicted alpha. Letting the
    model scale its own tolerance by transparency would introduce an escape.
    """
    selected=valid.reshape(-1)>0
    if not selected.any():return actor_rgb.sum()*0
    rgb=actor_rgb[selected].float()+(1-alpha[selected].float())*background[selected].float()
    excess=((rgb-photo[selected].float()).abs()-error[selected].float()).clamp_min(0)
    return (torch.sqrt(excess.square()+1e-4)-.01).mean()


class ObservedBackgroundTargets:
    def __init__(self,path,dataset,trimap_path=None):
        self.path=Path(path);receipt=json.loads((self.path/'receipt.json').read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if dataset.metadata['distillation_split']!='train' or receipt['uses_eval_cameras'] or receipt['uses_target_frame_for_background'] or not receipt['target_conversion_byte_exact']:
            raise ValueError('Observed backgrounds require independent train-only observations')
        source=Path(receipt['arguments']['data']);manifest=source/'transforms.json'
        if digest(manifest)!=receipt['source_manifest_sha256']:raise ValueError('Background source manifest changed')
        meta=json.loads(manifest.read_text());by_name={r['physical_camera']:r for r in meta['frames'] if r['file_path'] in meta['train_filenames']}
        target_meta=json.loads((Path(dataset.metadata['distillation_root'])/'transforms.json').read_text())
        records={r['physical_camera']:r for r in receipt['records']}
        if len(records)!=len(receipt['records']):raise ValueError('Duplicate plate cameras')
        self.rgb=[];self.valid=[];self.error=[];identity=[];overrides=[]
        calibration=['camera_model','transform_matrix','fl_x','fl_y','cx','cy','w','h','k1','k2','k3','k4','p1','p2']
        for image,row in zip(dataset.image_filenames,dataset.metadata['distillation_rows']):
            camera=row['physical_camera'];original=by_name[camera];record=records[camera]
            for key in calibration:
                # Each manifest owns its global defaults; absent distortion means zero.
                if not np.array_equal(row.get(key,target_meta.get(key,0)),original.get(key,meta.get(key,0))):
                    raise ValueError('Observed background calibration changed: '+key)
            if digest(image)!=digest(source/original['file_path']):raise ValueError('Observed background photograph changed')
            folder=self.path/Path(original['file_path']).stem
            for filename,key in [('background.png','background_sha256'),('valid.png','valid_sha256')]:
                if digest(folder/filename)!=record[key]:raise ValueError('Observed plate identity mismatch')
            rgb=np.array(Image.open(folder/'background.png').convert('RGB'))
            valid=np.array(Image.open(folder/'valid.png'))>0
            if rgb.shape!=(row['h'],row['w'],3) or valid.shape!=rgb.shape[:2]:raise ValueError('Plate dimensions mismatch')
            trimap_identity={}
            if trimap_path is not None:
                matte=Path(trimap_path)/image.stem
                proof=json.loads((matte/'receipt.json').read_text())
                if proof['actual_eval_used'] or proof['source_sha256']!=digest(image) or proof['physical_camera']!=camera:
                    raise ValueError('Trimap requires matching train-only photograph and camera')
                alpha=Path(dataset.metadata['distillation_root'])/row['alpha_file_path']
                if proof['alpha_sha256']!=digest(alpha) or proof['trimap_sha256']!=digest(matte/'trimap.png'):
                    raise ValueError('Trimap/alpha identity mismatch')
                tri=np.array(Image.open(matte/'trimap.png'))
                if tri.shape!=valid.shape or not np.isin(tri,[0,128,255]).all():raise ValueError('Invalid trimap')
                # Saturation of an estimated alpha is not evidence of an opaque
                # interior. Only independently observed background can replace it.
                overrides.append(torch.from_numpy(valid&(tri==128)))
                trimap_identity=dict(trimap_sha256=proof['trimap_sha256'],alpha_sha256=proof['alpha_sha256'],
                                     observed_unknown_pixels=int(overrides[-1].sum()))
            # A camera-specific discrepancy measured on current known background.
            # This quantile is an uncertainty scale, not a certified confidence bound.
            # Coverage-only extensions retain the established uncertainty scale.
            error=max(float(record.get('supervision_error',record['background_residual_p90'])),1/255)
            if not np.isfinite(error) or not 0<error<1:raise ValueError('Invalid observed-background uncertainty')
            self.rgb.append(torch.from_numpy(rgb));self.valid.append(torch.from_numpy(valid));self.error.append(error)
            identity.append(dict(camera=camera,rgb_sha256=digest(image),background_sha256=record['background_sha256'],valid_sha256=record['valid_sha256'],error=error,**trimap_identity))
        self.rgb=torch.stack(self.rgb);self.valid=torch.stack(self.valid);self.error=torch.tensor(self.error)
        self.opaque_override=torch.stack(overrides) if overrides else None
        self.receipt=dict(plate_receipt_sha256=digest(self.path/'receipt.json'),records=identity,
                          convention='display RGB',scope='non-opaque train rays plus observed trimap unknowns' if trimap_path else 'non-opaque train rays; fixed calibration',uses_eval=False)

    def apply(self,batch):
        camera,y,x=batch['indices'].cpu().T
        uncertain=batch['alpha_target'].detach().cpu().reshape(-1)<.999
        if self.opaque_override is not None:uncertain=uncertain|self.opaque_override[camera,y,x]
        valid=self.valid[camera,y,x]&uncertain
        result=dict(batch)
        result['observed_background_rgb']=self.rgb[camera,y,x].float()/255
        result['observed_background_valid']=valid[:,None].float()
        result['observed_background_error']=self.error[camera,None]
        return result

    def sample_subpixel(self,camera,coords):
        """Sample measured HD RGB at base-camera pixel-center coordinates.

        Reject incomplete interpolation footprints and image-edge extrapolation.
        Unknown plate values never become native training colors. Uncertainty is
        camera-specific and is kept independent of the model's opacity.
        """
        from nerfstudio.model_components.native_training import sample_native_rgb
        device=coords.device
        if not hasattr(self,'_subpixel') or self._subpixel[0].device!=device:
            self._subpixel=(self.rgb.to(device),self.valid.to(device))
        rgb,valid=self._subpixel;h,w=valid.shape[1:]
        xy=coords[:,[1,0]]-.5
        inside=(xy>=0).all(-1)&(xy<=xy.new_tensor([w-1,h-1])).all(-1)
        xy=xy.clamp_min(0).minimum(xy.new_tensor([w-1,h-1]))
        low=xy.floor().long();high=(low+1).minimum(low.new_tensor([w-1,h-1]));fraction=xy-low
        safe=inside.clone()
        for dx in [0,1]:
            for dy in [0,1]:
                x=high[:,0] if dx else low[:,0];y=high[:,1] if dy else low[:,1]
                weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
                safe &= (weight==0)|valid[camera,y,x]
        sampled=sample_native_rgb(rgb,camera,coords,h,w)
        return torch.where(safe[:,None],sampled,0.),safe
