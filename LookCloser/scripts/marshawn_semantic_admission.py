"""Fixed TRAIN semantic admission for density and traversal, separate from learned EMA.

This opt-in module adds no learned parameters, loss, RNG draw or density activation.
The registered model subclass also installs the gate on inference-only reload.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from marshawn_semantic_cell_admission import WholeCellAdmission


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def rng_signature(state):
    """Canonical RNG-state digest independent of Tensor storage identities."""
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().numpy();h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(state);return h.hexdigest()


def load_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity=False):
    """Validate sealed TRAIN provenance before constructing the conservative cover."""
    path=Path(manifest_path);data=Path(data_dir)
    if sha(path)!=expected_sha:raise ValueError('Semantic manifest changed')
    meta=json.loads(path.read_text())
    if meta['kind']!='reviewed_TRAIN_W64_pad1_whole_LC_cells' or meta['train_camera_count']!=63:raise ValueError('Wrong semantic source contract')
    for name,digest in meta['files_sha256'].items():
        file=path.parent/name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(file)!=digest:raise ValueError('Semantic artifact changed: '+name)
    record=json.loads((path.parent/'grid_receipt.json').read_text())
    if record.get('no_eval_masks_or_RGB_read') is not True or record['native_pixel_support_width']!=64 or len(set(record['train_cameras']))!=63 or set(record['train_cameras'])&{'DC','HC','KA'}:raise ValueError('Admission source is not exactly TRAIN W64')
    if record['grid_sha256']!=sha(path.parent/'semantic_grid.npz'):raise ValueError('Source grid receipt mismatch')
    if record['transforms_sha256']!=sha(data/'transforms.json') or record['source_ready_sha256']!=sha(data/'ready.json'):raise ValueError('Dataset/source frame changed')
    ready=json.loads((data/'ready.json').read_text())
    if sorted(record['train_cameras'])!=sorted(r['camera'] for r in ready['records'] if Path(r['image']).name.startswith('train_')):raise ValueError('Actual TRAIN membership changed')
    with np.load(path.parent/'semantic_grid.npz',allow_pickle=False) as z:
        source=z['semantic_gate_pad1'].copy();base=z['semantic_gate0'].copy();box=z['aabb'].copy();transform=z['world_to_box'].copy()
    from scipy.ndimage import maximum_filter
    if source.dtype!=np.bool_ or source.shape!=(128,128,128) or not np.array_equal(source,maximum_filter(base,size=3,mode='constant',cval=0)):raise ValueError('Source halo must be exactly one noncumulative26-neighbor dilation')
    np.testing.assert_allclose(box,ready['aabb'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(transform,ready['world_to_box'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(bounds,box,rtol=0,atol=2e-7)
    if tuple(resolution)!=(128,128,128):raise ValueError('Reviewed single128grid required')
    mapped=WholeCellAdmission(source,box).map_target_grid(resolution,bounds)
    if hashlib.sha256(mapped.tobytes()).hexdigest()!=meta['mapped_bool_sha256']:raise ValueError('Whole-cell cover changed')
    if not meta['all103_witnesses_retained'] or meta['qualified_shoe_anchors_retained']!=2:raise ValueError('Missing positive support validation')
    if identity:mapped=np.ones_like(mapped)
    return mapped,dict(manifest_sha256=expected_sha,source_grid_sha256=record['grid_sha256'],mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),identity_control=identity)


class AdmissionLookup(torch.nn.Module):
    """Floor-frame XYZ lookup in the already conservatively covered LC cells."""
    def __init__(self,mask,bounds):
        if mask.dtype!=torch.bool or mask.ndim!=3:raise ValueError('Boolean XYZ mask required')
        super().__init__()
        self.register_buffer('mask',mask,persistent=False)
        self.register_buffer('bounds',bounds.to(device=mask.device,dtype=torch.float32),persistent=False)
        if self.bounds.shape!=(2,3) or not bool(torch.isfinite(self.bounds).all()) or not bool((self.bounds[1]>self.bounds[0]).all()):raise ValueError('Increasing finite bounds required')
        self.register_buffer('resolution',torch.tensor(mask.shape,device=mask.device,dtype=torch.long),persistent=False)

    def forward(self,positions):
        shape=positions.shape[:-1];p=positions.reshape(-1,3).float()
        # Original field AABB selector still excludes outside points. Returning
        # True here labels outside-domain semantic coverage UNKNOWN, never clear.
        normalized=(p-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        inside=((normalized>=0)&(normalized<=1)).all(-1)
        indices=torch.floor(normalized*self.resolution).long()
        indices=torch.maximum(torch.zeros_like(indices),torch.minimum(indices,self.resolution-1))
        return ((~inside)|self.mask[indices[:,0],indices[:,1],indices[:,2]]).reshape(shape)


def install_density_admission(field,mask,bounds):
    """Gate the shared native selector used by get_density and query_points.

    Native density_fn delegates to get_density. Fixed warmup, ARM and inference
    therefore use the same exact density support; encoding/activation/RGB and
    admitted-point derivatives remain unchanged.
    """
    if getattr(field,'_semantic_admission_installed',False):raise ValueError('Density gate already installed')
    import types
    lookup=AdmissionLookup(mask,bounds);original=field._normalize_positions
    def normalized(self,positions):
        flat,selector,shape,world=original(positions)
        return flat,selector&lookup(positions),shape,world
    field._semantic_original_normalize=original
    field._normalize_positions=types.MethodType(normalized,field)
    field._semantic_admission_installed=True
    return lookup


def __getattr__(name):
    if name not in ('SemanticAdmissionModel','SemanticAdmissionModelConfig'):raise AttributeError(name)
    from dataclasses import dataclass,field as dataclass_field
    from typing import Type
    from nerfstudio.models.lookcloser import LookCloserModel,LookCloserModelConfig

    @dataclass
    class SemanticAdmissionModelConfig(LookCloserModelConfig):
        _target: Type=dataclass_field(default_factory=lambda:SemanticAdmissionModel)
        semantic_manifest: str=''
        semantic_manifest_sha256: str=''
        semantic_data: str=''
        semantic_identity_control: bool=False

    class SemanticAdmissionModel(LookCloserModel):
        def populate_modules(self):
            super().populate_modules()
            c=self.config
            if (c.density_activation!='trunc_exp' or c.correct_sh_directions is not True or c.occupancy_dilation_radius not in (0,1)
                or c.occupancy_thre_clamp_mult!=1. or getattr(c,'geometry_support_enabled',False) or c.occupancy_prior_grid_path is not None
                or len(self.occupancy_grid.binaries)!=1):raise ValueError('Unsupported semantic candidate parent')
            bounds=self.occupancy_grid.aabbs[0].reshape(2,3)
            mask,receipt=load_support(c.semantic_manifest,c.semantic_manifest_sha256,c.semantic_data,bounds.detach().cpu().numpy(),self.occupancy_grid.binaries.shape[1:],c.semantic_identity_control)
            self.register_buffer('semantic_admission',torch.from_numpy(mask).to(bounds.device),persistent=False)
            self.register_buffer('semantic_learned_core',self.occupancy_grid.binaries.detach().clone(),persistent=True)
            self.semantic_receipt=receipt
            self.semantic_lookup=install_density_admission(self.field,self.semantic_admission,bounds)
            # No RNG and no rewrite of occupancy EMA/core; cold binaries are zero.
            self.occupancy_grid.binaries.logical_and_(self.semantic_admission[None])
            self.register_load_state_dict_post_hook(self._check_restored_admission)

        def _check_restored_admission(self,module,incompatible_keys):
            if bool((self.occupancy_grid.binaries&~self.semantic_admission[None]).any()):raise ValueError('Saved active grid violates source admission')

        def _postprocess_occupancy_grid(self,step=None):
            if step is None:raise ValueError('Explicit update step required for admission provenance')
            self.semantic_learned_core.copy_(self.occupancy_grid.binaries)
            super()._postprocess_occupancy_grid(step)
            self.occupancy_grid.binaries.logical_and_(self.semantic_admission[None])
            self._tuning_occupancy_capture=dict(core=self.semantic_learned_core.detach().clone(),active=self.occupancy_grid.binaries.detach().clone(),step=int(step),radius=int(self.config.occupancy_dilation_radius),binary_warmup=int(step)<self.config.occupancy_binary_warmup_steps,semantic_receipt=self.semantic_receipt)
            self._last_occupancy_binaries=self.occupancy_grid.binaries.detach().clone()
            self._last_occupancy_stats['semantic_active_ratio']=float(self.occupancy_grid.binaries.float().mean())

        def _ensure_eval_occupancy_dilation(self):
            super()._ensure_eval_occupancy_dilation()
            self.occupancy_grid.binaries.logical_and_(self.semantic_admission[None])

        def _admit_depth_occupancy(self):
            if self.config.occupancy_prior_grid_path is not None:raise ValueError('Separate depth admission is not authorized')
            self.occupancy_grid.binaries.logical_and_(self.semantic_admission[None])

        def load_state_dict(self,*args,**kwargs):
            result=super().load_state_dict(*args,**kwargs)
            if bool((self.occupancy_grid.binaries&~self.semantic_admission[None]).any()):raise ValueError('Saved active grid violates source admission')
            return result

    for cls in (SemanticAdmissionModel,SemanticAdmissionModelConfig):
        cls.__module__=__name__;cls.__qualname__=cls.__name__;globals()[cls.__name__]=cls
    return globals()[name]
