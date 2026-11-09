"""Experimental front-quantile projection ONLY during frequency-grid updates.

The RGB renderer, training depth objectives and geometry outputs keep their
usual definitions. Existing packed marching intervals are reused, with no
extra density queries. The opaque-support gate is deliberately unchanged.
"""
import numpy as np
import torch
from dec5_hair import packed_depth_quantile_vectorized


def install(pipe,quantile=None,minimum_opacity=None):
    if quantile is not None and not 0<quantile<1:raise ValueError('Quantile must lie strictly inside (0,1)')
    if minimum_opacity is not None:
        if not 0<=minimum_opacity<=1:raise ValueError('Invalid update opacity gate')
        if pipe.config.frequency_projection_mode!='normalized_jacobian' or not hasattr(pipe,'dec5_frequency_sampler'):
            raise ValueError('Opacity probe requires person-supported Jacobian updates')
    if quantile is None and minimum_opacity is None:raise ValueError('No frequency-update intervention requested')
    original=pipe._update_frequency_grid
    counters=dict(quantile=quantile,updates=0,valid_rays=0,absolute_shift_sum=0.,
                  forward_shift_sum=0.,additional_density_queries=0,minimum_opacity=minimum_opacity,
                  opaque_rays=0,relaxed_gate_rays=0)
    def update(step):
        wrapped=pipe.model.get_outputs
        def outputs(rays):
            result=wrapped(rays)
            result=dict(result)
            if minimum_opacity is not None:
                opacity=result['accumulation']
                counters['opaque_rays']+=int((opacity>=.9).sum())
                counters['relaxed_gate_rays']+=int((opacity>=minimum_opacity).sum())
                # The unchanged supported-update implementation tests >=.9.
                # This override exists ONLY inside that frequency-update call.
                result['accumulation']=(opacity>=minimum_opacity).to(opacity)
            if quantile is None or 'packed_ray_indices' not in result:return result
            values=[result[k].detach().float().cpu().numpy() for k in
                    ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
            distance=packed_depth_quantile_vectorized(*values,
                result['packed_ray_indices'].detach().cpu().numpy(),
                rays.nears.detach().cpu().numpy(),rays.fars.detach().cpu().numpy(),
                quantile=quantile,minimum_mass=.001)
            expected=result['depth'];front=torch.as_tensor(distance,device=expected.device,dtype=expected.dtype).reshape_as(expected)
            valid=torch.isfinite(front)&(front>0)&torch.isfinite(expected)&(expected>0)
            delta=(expected-front)[valid].detach().cpu().numpy()
            counters['valid_rays']+=int(valid.sum());counters['absolute_shift_sum']+=float(np.abs(delta).sum())
            counters['forward_shift_sum']+=float(delta.sum())
            return dict(result,depth=torch.where(valid,front,expected))
        pipe.model.get_outputs=outputs
        try:
            original(step);counters['updates']+=1
        finally:pipe.model.get_outputs=wrapped
    pipe._update_frequency_grid=update
    return counters
