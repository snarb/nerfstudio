"""Separate reviewed geometry objectives using the actual exp(h+1) field path.

Sampling/geometry contracts match the softplus arms. No activation mutation,
logit-derived replacement or alternative backward function is installed.
"""
import json
from pathlib import Path
import numpy as np
import torch
from marshawn_sh_geometry import SHGeometry,SHGeometryAdapter,native_opacity_nll
from marshawn_softplus_prefix_candidate import load_reviewed_artifact,sha256


class ExpGeometry(SHGeometry):
    def __init__(self,*args,band_weight=0.,**kwargs):
        super().__init__(*args,band_weight=0.,**kwargs)
        if band_weight not in (0.,.0003,.001,.003):raise ValueError('Declared native-exp band coefficient required')
        self.band_weight=float(band_weight)

    def check_native(self,field):
        if getattr(field,'density_activation',None)!='trunc_exp' or getattr(field,'correct_sh_directions',None)is not True:
            raise ValueError('Actual SH-corrected native exp(h+1) field required')
        if not torch.allclose(field.aabb.to(self.bounds),self.bounds,rtol=0,atol=1e-6):raise ValueError('Native-exp geometry bounds changed')

    def integral(self,field,batch,samples):
        self.check_native(field);xyz=batch['points'].reshape(-1,3)
        if not bool(((xyz>self.bounds[0])&(xyz<self.bounds[1])).all()):raise ValueError('Native-exp sample outside strict AABB')
        native=field.density_fn(xyz)
        with torch.autocast(device_type=self.device.type,enabled=False):
            sigma=native.reshape(self.rays,samples).float()
            if not bool(torch.isfinite(sigma).all())or bool((sigma<0).any()):raise FloatingPointError('Nonfinite/negative native-exp density')
            tau=sigma.mean(1)*batch['length'].float()
        return tau

    def __call__(self,field,step):
        tau=self.integral(field,self.sample(step),self.samples)
        return self.weight*tau.mean(),dict(prefix_tau_mean=tau.detach().mean(),prefix_tau_max=tau.detach().max(),prefix_ray_count=self.rays)

    def band(self,field,step,*,weight=None):
        tau=self.integral(field,self.sample_band(step),self.band_samples)
        return (self.band_weight if weight is None else weight)*native_opacity_nll(tau).mean(),dict(band_tau_mean=tau.detach().mean(),band_opacity_mean=(-torch.expm1(-tau.detach())).mean())


class ExpGeometryAdapter(SHGeometryAdapter):
    def state_dict(self):
        state=super().state_dict();state['native_density']='trunc_exp_hplus1';return state


def install_exp_geometry(pipeline,sampler,spec,ready):
    model=pipeline.model
    if (spec.get('correct_sh_directions')is not True or spec.get('density_activation')!='trunc_exp'
            or model.config.correct_sh_directions is not True or model.field.correct_sh_directions is not True
            or model.config.density_activation!='trunc_exp' or model.field.density_activation!='trunc_exp'
            or pipeline.config.frequency_projection_mode!='legacy'):
        raise ValueError('Exact corrected-SH/native-exp/legacy-projection parent required')
    # Verify the native field formula; never replace its output or derivative.
    h=torch.tensor([-4.,0.,2.],device=model.device)
    with torch.inference_mode():torch.testing.assert_close(model.field.activate_density(h),torch.exp(h+1.),rtol=1e-6,atol=1e-7)
    prefix=spec['prefix_candidate'];path=Path(prefix['manifest']);meta=json.loads(path.read_text());data=Path(spec['data'])
    if sha256(path)!=prefix['manifest_sha256']:raise ValueError('Sealed expanded geometry manifest changed')
    if meta.get('supervised_rays')!=84 or meta.get('supervised_points')!=31:raise ValueError('The sealed expanded84 source required')
    for name,key in [('transforms.json','transforms_sha256'),('box.json','box_sha256')]:
        if sha256(data/name)!=meta['bindings'][key]:raise ValueError('Geometry coordinate source changed')
    if sha256(path.parent/'visibility_review.json')!=meta['bindings']['visibility_review_sha256']:raise ValueError('Visibility review changed')
    for camera,digest in meta['source_semantic_hashes'].items():
        if sha256(data/'semantics'/f'{camera}.npz')!=digest:raise ValueError('Geometry source semantic changed')
    train=sorted(r['camera']for r in ready['records']if Path(r['image']).name.startswith('train_'))
    arrays,receipt=load_reviewed_artifact(path,expected_bindings=meta['bindings'],bounds=ready['aabb'],train_cameras=train)
    review=json.loads((path.parent/'visibility_review.json').read_text())
    for point in review['points']:
        ids=np.flatnonzero(arrays['point_idx']==point['point'])
        if not len(ids):continue
        half=np.maximum(.005,.01*arrays['witness_depth'][ids]).min()
        for check in point['original_feature_graph']['leave_one_source_view_out']:
            if not check['positive_heldout_depth']or check['xyz_shift_world']>=half:raise ValueError('Measured source shift exceeds declared band')
    objective=ExpGeometry(arrays,ready['aabb'],train,weight=prefix['weight'],band_weight=spec['band_weight'],rays=128,samples=64,seed=spec['seed'],device=model.device)
    return ExpGeometryAdapter(pipeline,sampler,objective,receipt)
