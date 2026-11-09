"""Opt-in matched linear-709 control; native neural parameters remain unchanged."""
import torch
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colormaps
from run_marshawn_hd_campaign import MarshawnDataParser
from marshawn_radial_native_candidate import NativeRadialColdPipeline

MAX_LINEAR = 10000.0 / 203.0


def bounded_linear(p, median_y):
    """FP32 rational shifted sigmoid: 0->0, .5->TRAIN median, 1->MAX."""
    if not 0 < float(median_y) < MAX_LINEAR:
        raise ValueError('TRAIN-only initialization median must lie strictly inside output range')
    with torch.autocast(device_type=p.device.type, enabled=False):
        p = p.float()
        a = float(median_y) / (MAX_LINEAR-float(median_y))
        return MAX_LINEAR * (a*p) / (1-p+a*p)


def display_sdr(linear):
    with torch.autocast(device_type=linear.device.type, enabled=False):
        x = linear.float().clamp_min(0)
        x = x/(1+x)
        return torch.where(x <= .0031308, 12.92*x, 1.055*x.pow(1/2.4)-.055)


def pq203(linear):
    """ST2084, 203nits per baked linear709 unit; floor only in objective."""
    with torch.autocast(device_type=linear.device.type, enabled=False):
        l = (linear.float()*203.0).clamp(.005,10000.0)/10000.0
        m1,m2,c1,c2,c3 = 2610/16384,2523/32,3424/4096,2413/128,2392/128
        t = l.pow(m1)
        return ((c1+c2*t)/(1+c3*t)).pow(m2)


class Linear709DataParser(MarshawnDataParser):
    def _generate_dataparser_outputs(self, split='train'):
        out = super()._generate_dataparser_outputs(split)
        # Bypass generic EXR auto-calibration. These metadata never scale pixels.
        out.metadata['hdr_calibration'] = dict(linear_scale=self.config.linear709_median_y,
            initial_radiance=self.config.linear709_median_y, nits_per_scene_unit=203.,
            black_nits=.005, peak_nits=10000., source='EXPLICIT_train63_strided_Y_initialization_only_no_pixel_scaling')
        out.metadata['image_color_space'] = 'linear-srgb'
        return out


class BoundedLinear709Model(NGPModel):
    """Same native field, instance-local physical RGB interpretation/loss/display."""
    def populate_modules(self):
        super().populate_modules()
        self.renderer_rgb.clamp_output = False
        # Forward hook is instance-local, has no trainable state and no RNG use.
        self.field.register_forward_hook(self._physical_field_output)

    def _physical_field_output(self, field, args, output):
        output[FieldHeadNames.RGB] = bounded_linear(output[FieldHeadNames.RGB], self.config.linear709_median_y)
        return output

    def get_metrics_dict(self, outputs, batch):
        pred,gt = outputs['rgb'],batch['image'].to(self.device)
        if not torch.isfinite(pred).all() or not torch.isfinite(gt).all() or (gt < 0).any():
            raise FloatingPointError('Finite nonnegative physical linear709 RGB required')
        return dict(psnr=self.psnr(display_sdr(pred),display_sdr(gt)),
            pq_psnr=self.psnr(pq203(pred),pq203(gt)),
            num_samples_per_batch=outputs['num_samples_per_ray'].sum(),
            physical_prediction_floor_fraction=(pred*203 < .005).float().mean())

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        gt = batch['image'].to(self.device)
        if gt.shape[-1] != 3:
            raise ValueError('Opaque three-channel EXR contract required')
        # Native random BG is Uniform[0,1] LINEAR units, only for training loss.
        # RGB3 GT is returned unchanged; inference remains premultiplied black.
        pred, target = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs['rgb'], pred_accumulation=outputs['accumulation'], gt_image=gt)
        # Explicit loss-unit normalization, not a pixel/exposure transformation.
        return {'rgb_loss':float(self.config.linear709_pq_mse_multiplier)*
                (pq203(pred)-pq203(target)).square().mean()}

    def get_image_metrics_and_images(self, outputs, batch):
        gt,pred = batch['image'].to(self.device),outputs['rgb']
        a,b = display_sdr(gt),display_sdr(pred)
        a4,b4 = a.permute(2,0,1)[None],b.permute(2,0,1)[None]
        values = dict(psnr=float(self.psnr(a4,b4)),ssim=float(self.ssim(a4,b4,data_range=1.)),
            lpips=float(self.lpips(a4,b4)),pq_psnr=float(self.psnr(pq203(pred),pq203(gt))))
        return values,dict(img=torch.cat([a,b],1),
            accumulation=colormaps.apply_colormap(outputs['accumulation']),
            depth=colormaps.apply_depth_colormap(outputs['depth'],accumulation=outputs['accumulation']))


class BoundedLinear709Pipeline(NativeRadialColdPipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.model.field.appearance_embedding_dim != 0 or self.model.renderer_rgb.clamp_output:
            raise ValueError('No appearance dimensions and no physical-render clipping required')
        if self.datamanager.config.cache_images_type != 'float32':
            raise ValueError('EXR pixels require float32 cache')
