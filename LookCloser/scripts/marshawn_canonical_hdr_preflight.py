"""Uninstalled canonical HDR helpers for read-only proposal/gradient preflight.

No accepted checkpoint, field implementation, dataset or runner is patched.
"""
import math
import torch

SIGNED_GAIN = 52.13360318135631
BLACK_FLOOR_NITS = .005
CONTENT_PEAK_NITS = 4000.


def canonical_pq_from_unbaked_ap1(rgb, *, gain=SIGNED_GAIN,
        black_floor=BLACK_FLOOR_NITS, content_peak=CONTENT_PEAK_NITS):
    """Vendor encode_tier: clip(gain*linear, floor, peak), ST2084 ref10000.

AP1 channels remain AP1; no Rec709/Rec2020 matrix is silently introduced.
Float64 transfer math follows the vendor reference, then returns FP32 loss codes.
"""
    if not all(math.isfinite(float(v)) for v in [gain,black_floor,content_peak]) or not 0<gain or not 0<black_floor<content_peak<=10000:
        raise ValueError('Invalid signed radiometric contract')
    with torch.autocast(device_type=rgb.device.type,enabled=False):
        nits=(rgb.double()*float(gain)).clamp(float(black_floor),float(content_peak))
        powered=(nits/10000.).pow(2610./16384.)
        code=((3424./4096.+(2413./4096.*32.)*powered)/(1.+(2392./4096.*32.)*powered)).pow(2523./4096.*128.)
    return code.float()


class FrozenLookCloserDensityProxy:
    """Expose actual LC h0 and average_init_density=e to stable surface objective.

This plain accessor owns no modules or trainable parameters. Its temporary hook
captures the same differentiable geometry output used by density_fn, and is
removed even if that call fails. The underlying field cache is never overwritten.
"""
    def __init__(self,field):
        if getattr(field,'density_activation',None)!='trunc_exp':
            raise ValueError('Only frozen trunc_exp(h0+1) is reviewed')
        if getattr(field,'spatial_distortion',None) is not None:
            raise ValueError('Bounded field without spatial distortion required')
        if getattr(field,'enable_feature_reweighting',True) or getattr(field.freq_grid,'enabled',True):
            raise ValueError('Frequency grid and feature reweighting must both be disabled')
        if getattr(field,'density_geometry_encoding',None) is not None or getattr(field,'density_geometry_mlp',None) is not None:
            raise ValueError('Independent density geometry branch is unsupported')
        self.field=field
        self._density_before_activation=None
        self.average_init_density=math.e
        self.spatial_distortion=None

    @property
    def aabb(self):
        return self.field.aabb

    def density_fn(self,points):
        captured=[]
        def hook(_module,_args,output):captured.append(output[..., :1])
        handle=self.field.mlp_geo.register_forward_hook(hook)
        try:density=self.field.density_fn(points)
        finally:handle.remove()
        if len(captured)!=1 or captured[0].numel()!=density.numel():
            raise ValueError('Expected exactly one matching frozen LC geometry call')
        self._density_before_activation=captured[0]
        # SurfaceBandObjective itself enforces strict interior coordinates.
        with torch.no_grad():
            expected=torch.exp(captured[0].float()+1.)
            if not torch.allclose(density.float().reshape_as(expected),expected,rtol=2e-5,atol=1e-44):
                raise ValueError('Frozen field density no longer equals exp(h0+1)')
        return density
