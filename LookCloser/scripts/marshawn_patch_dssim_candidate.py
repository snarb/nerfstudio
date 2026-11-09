"""Optional MSE + (1-SSIM) patch auxiliary; no neural parameters or buffers.

This preserves the frozen native distortion/LOS/surface pipeline. Its standard
PatchPixelSampler changes the ray distribution explicitly; it is not a pure
loss-only causal comparison with the original independent-pixel sampler.
"""
from __future__ import annotations

import math
import torch


def validate_patch_indices(indices, patch_size=11):
    """Require camera-constant, contiguous row-major native sampler patches."""
    area = patch_size ** 2
    if patch_size != 11 or indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("Reviewed 11x11 patch indices [N, camera/y/x] required")
    if len(indices) == 0 or len(indices) % area:
        raise ValueError("Nonempty whole patches required; dynamic initial rays must be >=121")
    grouped = indices.reshape(-1, patch_size, patch_size, 3)
    anchor = grouped[:, :1, :1]
    offset = torch.zeros((1, patch_size, patch_size, 3), device=indices.device, dtype=indices.dtype)
    offset[..., 1] = torch.arange(patch_size, device=indices.device)[:, None]
    offset[..., 2] = torch.arange(patch_size, device=indices.device)[None, :]
    if not torch.equal(grouped, anchor + offset):
        raise ValueError("Patch camera/row/column order mismatch")
    return grouped


def patch_dssim(prediction, target, indices, patch_size=11):
    """Float32 functional SSIM, data_range=1; DSSIM convention is 1-SSIM."""
    from torchmetrics.functional.image import structural_similarity_index_measure
    validate_patch_indices(indices, patch_size)
    if prediction.shape != target.shape or prediction.shape != (len(indices), 3):
        raise ValueError("Matching native RGB [N,3] arrays required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Finite RGB required")
    if (target < 0).any() or (target > 1).any():
        raise ValueError("Target must be normalized scalar SDR RGB")
    # Disable autocast for local mean/variance subtraction, retaining gradients
    # through the native prediction cast. No clipping or display transform.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.float().reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
        truth = target.float().reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
        return 1.0 - structural_similarity_index_measure(
            pred, truth, data_range=1.0, gaussian_kernel=True,
            sigma=1.5, kernel_size=11, k1=.01, k2=.03)


class TrainBatchCapture:
    """Temporary non-neural instrumentation of the one actual next_train call."""
    def __init__(self, datamanager):
        self.datamanager, self.batch, self.calls = datamanager, None, 0

    def __enter__(self):
        self.had_instance_value = 'next_train' in self.datamanager.__dict__
        self.original = self.datamanager.next_train
        def captured(*args, **kwargs):
            ray, batch = self.original(*args, **kwargs)
            self.batch = batch
            self.calls += 1
            return ray, batch
        self.datamanager.next_train = captured
        return self

    def __exit__(self, *exception):
        if self.had_instance_value:
            self.datamanager.next_train = self.original
        else:
            del self.datamanager.next_train


def __getattr__(name):
    if name != 'NativePatchDSSIMPipeline':
        raise AttributeError(name)
    from marshawn_packed_distortion_candidate import NativePackedDistortionPipeline
    class NativePatchDSSIMPipeline(NativePackedDistortionPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = self.config
            if cfg.patch_dssim_size != 11 or cfg.datamanager.patch_size != 11:
                raise ValueError('Reviewed standard 11px patch sampler required')
            if cfg.datamanager.pixel_sampler.patch_size != 11 or not cfg.datamanager.pixel_sampler.ignore_mask:
                raise ValueError('Explicit PatchPixelSampler11/ignore_mask=True required after all-mask QA')
            if not math.isfinite(cfg.patch_dssim_weight) or cfg.patch_dssim_weight < 0:
                raise ValueError('Finite nonnegative auxiliary weight required')
            if cfg.max_num_samples_per_ray != 2048 or cfg.target_num_samples != 262144:
                raise ValueError('Explicit dynamic initial estimate 2048/target262144 required')
            if self.datamanager.train_pixel_sampler.num_rays_per_batch < 121:
                raise ValueError('Zero initial patch batch; do not launch native CUDA sampler')

        def get_train_loss_dict(self, step):
            with TrainBatchCapture(self.datamanager) as capture:
                outputs, losses, metrics = super().get_train_loss_dict(step)
            if capture.calls != 1 or capture.batch is None:
                raise ValueError('Exactly one actual training batch required')
            batch = capture.batch
            pred, truth = self.model.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs['rgb'], pred_accumulation=outputs['accumulation'],
                gt_image=batch['image'].to(self.device))
            losses['patch_dssim'] = self.config.patch_dssim_weight * patch_dssim(
                pred, truth, batch['indices'], self.config.patch_dssim_size)
            return outputs, losses, metrics
    NativePatchDSSIMPipeline.__name__ = name
    NativePatchDSSIMPipeline.__qualname__ = name
    NativePatchDSSIMPipeline.__module__ = __name__
    globals()[name] = NativePatchDSSIMPipeline
    return NativePatchDSSIMPipeline
