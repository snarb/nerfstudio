"""Canonical AP1 HDR control: frozen existing LC field, explicit signed transfer."""
from pathlib import Path
import json,math
import torch
from nerfstudio.models.lookcloser import LookCloserModel
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline
from nerfstudio.utils import colormaps
from marshawn_canonical_hdr_preflight import canonical_pq_from_unbaked_ap1,FrozenLookCloserDensityProxy
from marshawn_los_objective_frozen import FreeSpaceObjective
from marshawn_surface_band_objective import SurfaceBandObjective


def common_sdr(rgb,matrix,ev=0.):
    with torch.autocast(device_type=rgb.device.type,enabled=False):
        mat=torch.as_tensor(matrix,device=rgb.device,dtype=torch.float32)
        x=(rgb.float()@mat.T).clamp_min(0)*(4.*2.**float(ev));x=x/(1+x)
        return torch.where(x<=.0031308,12.92*x,1.055*x.clamp_min(0).pow(1/2.4)-.055).clamp(0,1)


class CanonicalAP1HDRModel(LookCloserModel):
    """No field/module replacement; loss, metrics and display contract only."""
    def get_metrics_dict(self,outputs,batch):
        # Preserve native sampling/occupancy diagnostics; replace legacy PQ metric.
        result=super().get_metrics_dict(outputs,batch)
        pred=canonical_pq_from_unbaked_ap1(outputs['rgb']);target=canonical_pq_from_unbaked_ap1(batch['image'].to(self.device))
        mse=(pred-target).square().mean();result['psnr']=-10*torch.log10(mse.clamp_min(1e-12));result['pq_psnr']=result['psnr']
        result['pq_upper_clip_rate']=(outputs['rgb'].float()*52.13360318135631>4000).float().mean()
        result['canonical_prediction_floor_clamp_fraction']=(outputs['rgb'].float()*52.13360318135631<.005).float().mean()
        return result

    def get_loss_dict(self,outputs,batch,metrics_dict=None):
        if self.config.reconstruction_loss_type!='pq_mse':raise ValueError('Only reviewed canonical PQ-MSE')
        rgb=outputs['rgb'];target=batch['image'].to(self.device)
        if not torch.isfinite(rgb).all()or not torch.isfinite(target).all()or(target<0).any():raise ValueError('Physical finite positive AP1 input required')
        predpq=canonical_pq_from_unbaked_ap1(rgb);gtpq=canonical_pq_from_unbaked_ap1(target)
        loss={'rgb_loss':(predpq-gtpq).square().mean()}
        if self.config.distortion_loss_mult:
            distortion=self._packed_distortion_loss(outputs['packed_spacing_starts'],outputs['packed_spacing_ends'],outputs['packed_weights'],outputs['packed_ray_indices'],len(rgb)).mean()
            loss['distortion_loss']=self.config.distortion_loss_mult*distortion
        if self.config.depth_loss_mult:raise ValueError('No extra depth/floor objective in this control')
        return loss

    def get_image_metrics_and_images(self,outputs,batch):
        gt=batch['image'].to(self.device);pred=outputs['rgb'];gpq=canonical_pq_from_unbaked_ap1(gt);ppq=canonical_pq_from_unbaked_ap1(pred)
        def metrics(a,b):
            a=a.permute(2,0,1)[None];b=b.permute(2,0,1)[None]
            return dict(psnr=float(self.psnr(a,b)),ssim=float(self.ssim(a,b,data_range=1.)),lpips=float(self.lpips(a,b)))
        values=metrics(gpq,ppq);images={'canonical_PQ':torch.cat([gpq,ppq],1),'accumulation':colormaps.apply_colormap(outputs['accumulation']),'depth':colormaps.apply_depth_colormap(outputs['depth'],accumulation=outputs['accumulation'])}
        idx=int(batch['image_idx']);name=self._canonical_eval_names[idx];ev=self.config.canonical_old_scalar_EV[name]
        for prefix,exposure in [('unscalar',0.),('common_scalar',ev)]:
            a=common_sdr(gt,self.config.canonical_ap1_to_709,exposure);b=common_sdr(pred,self.config.canonical_ap1_to_709,exposure)
            values.update({prefix+'_'+k:v for k,v in metrics(a,b).items()});images[prefix+'_SDR']=torch.cat([a,b],1)
        images['img']=images['common_scalar_SDR']
        return values,images


class CanonicalAP1HDRPipeline(DynamicBatchPipeline):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        m=self.model
        for d in [self.datamanager.train_dataset,self.datamanager.eval_dataset]:
            d.metadata['image_color_space']='linear-ACES-AP1-D60' # correct legacy genericEXR label; pixelbytesunchanged
        if m.field.appearance_embedding_dim or m.field.enable_feature_reweighting or m.freq_grid.enabled:raise ValueError('No appearance/frequency features')
        names=[Path(p).stem.split('_')[-1]for p in self.datamanager.eval_dataset.image_filenames]
        object.__setattr__(m,'_canonical_eval_names',names)
        args=(Path(self.config.canonical_geometry_manifest),'scene_loose',m.field.aabb.detach().cpu().numpy(),Path(self.config.datamanager.dataparser.data)/'transforms.json',Path(self.config.canonical_preparation))
        free=FreeSpaceObjective(*args,weight=.01,rays=128,samples=64,seed=42,device=m.device)
        surface=SurfaceBandObjective(*args,weight=.01,rays=128,samples=16,seed=42,device=m.device)
        object.__setattr__(self,'_canonical_free',free);object.__setattr__(self,'_canonical_surface',surface);object.__setattr__(self,'_canonical_density_proxy',FrozenLookCloserDensityProxy(m.field))
    def get_train_loss_dict(self,step):
        outputs,losses,metrics=super().get_train_loss_dict(step)
        losses['observed_sfm_free_space'],_=self._canonical_free(self.model.field,step)
        ids=self._canonical_free.sample_batch(step)['indices']
        losses['observed_sfm_surface_band'],_=self._canonical_surface(self._canonical_density_proxy,step,ids)
        return outputs,losses,metrics

from run_marshawn_hd_campaign import MarshawnDataParser
class CanonicalHDRDataParser(MarshawnDataParser):
    def _generate_dataparser_outputs(self,split='train'):
        outputs=super()._generate_dataparser_outputs(split)
        outputs.metadata['hdr_calibration']=dict(linear_scale=self.config.canonical_linear_scale,initial_radiance=self.config.canonical_initial_radiance,nits_per_scene_unit=52.13360318135631,black_nits=.005,peak_nits=4000.,source='Explicit_train63_AP1Y_statistics_no_eval_RGB')
        outputs.metadata['image_color_space']='linear-ACES-AP1-D60'
        return outputs
