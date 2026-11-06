"""Separate SH-corrected native-softplus prefix and optional opaque-band arms."""
import json
from pathlib import Path

import numpy as np
import torch

from marshawn_softplus_prefix_candidate import (
    SoftplusPrefixCandidate, PrefixSamplingAdapter, load_reviewed_artifact, sha256,
)


def native_opacity_nll(tau):
    """Stable -log(1-exp(-tau)) with the actual density integral gradient."""
    x = tau.float()
    if not bool(torch.isfinite(x).all()) or bool((x <= 0).any()):
        raise FloatingPointError('Positive finite native band optical thickness required')
    return -torch.log(-torch.expm1(-x))


class SHGeometry(SoftplusPrefixCandidate):
    def __init__(self, arrays, bounds, train_cameras, *, band_weight=0., **kwargs):
        super().__init__(arrays, bounds, train_cameras, **kwargs)
        if band_weight not in (0., .001, .003):
            raise ValueError('Only declared band coefficients accepted')
        self.band_weight, self.band_samples = float(band_weight), 16
        o, d = np.asarray(arrays['origins']), np.asarray(arrays['directions'])
        depth, b = np.asarray(arrays['witness_depth']), np.asarray(bounds)
        half = np.maximum(.005, .01*depth)
        parallel = np.abs(d) < 1e-12
        inv = 1./np.where(parallel, 1., d)
        left, right = (b[0]-o)*inv, (b[1]-o)*inv
        enter = np.where(parallel, -np.inf, np.minimum(left, right)).max(1)
        leave = np.where(parallel, np.inf, np.maximum(left, right)).min(1)
        start = np.maximum(np.maximum(depth-half, enter+1e-5), .01)
        end = np.minimum(depth+half, leave-1e-5)
        if (start >= end).any() or (depth < start).any() or (depth > end).any():
            raise ValueError('Current bounds exclude opaque witness band')
        self.band_start = torch.tensor(start, dtype=torch.float32, device=self.device)
        self.band_end = torch.tensor(end, dtype=torch.float32, device=self.device)

    def sample_band(self, step):
        # Source identities match the prefix packet, but the band substream is
        # independent and never consumes/changes prefix or global RNG draws.
        ids = self.sample(step)['indices']
        gen = torch.Generator(device=self.device)
        gen.manual_seed((self.seed+step*1000003+0x53555246) % (2**63-1))
        jitter = torch.rand((self.rays, self.band_samples), generator=gen, device=self.device)
        jitter = jitter.clamp(torch.finfo(torch.float32).eps, 1.-torch.finfo(torch.float32).eps)
        unit = (torch.arange(self.band_samples, device=self.device)[None]+jitter)/self.band_samples
        length = self.band_end[ids]-self.band_start[ids]
        distance = self.band_start[ids, None]+unit*length[:, None]
        points = self.values['origins'][ids, None]+self.values['directions'][ids, None]*distance[..., None]
        return dict(indices=ids, points=points, length=length)

    def band(self, field, step, *, weight=None):
        if getattr(field, 'density_activation', None) != 'softplus':
            raise ValueError('Actual native softplus required')
        if getattr(field, 'correct_sh_directions', None) is not True:
            raise ValueError('Actual corrected SH directions required')
        if not torch.allclose(field.aabb.to(self.bounds), self.bounds, rtol=0, atol=1e-6):
            raise ValueError('Band field bounds changed')
        batch = self.sample_band(step)
        xyz = batch['points'].reshape(-1, 3)
        if not bool(((xyz > self.bounds[0]) & (xyz < self.bounds[1])).all()):
            raise ValueError('Band sample outside strict field bounds')
        native = field.density_fn(xyz)
        with torch.autocast(device_type=self.device.type, enabled=False):
            sigma = native.reshape(self.rays, self.band_samples).float()
            tau = sigma.mean(1)*batch['length'].float()
            loss = (self.band_weight if weight is None else weight)*native_opacity_nll(tau).mean()
        return loss, dict(band_tau_mean=tau.detach().mean(),
                          band_opacity_mean=(-torch.expm1(-tau.detach())).mean())

    def __call__(self, field, step):
        if getattr(field, 'correct_sh_directions', None) is not True:
            raise ValueError('Actual corrected SH directions required')
        return super().__call__(field, step)


class SHGeometryAdapter(PrefixSamplingAdapter):
    def __init__(self, pipeline, sampler, objective, receipt):
        self.sampler, self.objective, self.receipt, self.step = sampler, objective, receipt, 0
        original = pipeline.get_train_loss_dict
        def get_losses(step):
            outputs, losses, metrics = original(step)
            if 'train_clothing_prefix_tau' in losses or 'train_clothing_band_nll' in losses:
                raise ValueError('Geometry objective already installed')
            value, diagnostics = objective(pipeline.model.field, int(step))
            losses['train_clothing_prefix_tau'] = value; metrics.update(diagnostics)
            if objective.band_weight:
                band, diagnostics = objective.band(pipeline.model.field, int(step))
                losses['train_clothing_band_nll'] = band; metrics.update(diagnostics)
            return outputs, losses, metrics
        pipeline.get_train_loss_dict = get_losses

    def state_dict(self):
        return dict(**super().state_dict(), band_weight=self.objective.band_weight,
                    band_samples=self.objective.band_samples, band_seed_offset=0x53555246,
                    actual_SH_required=True, native_density='softplus')

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('band_weight', 'band_samples', 'band_seed_offset', 'actual_SH_required', 'native_density'):
            if state[key] != expected[key]:
                raise ValueError('Geometry resume policy changed: '+key)
        super().load_state_dict(state)


def install_sh_geometry(pipeline, sampler, spec, ready):
    if (spec.get('correct_sh_directions') is not True
            or pipeline.model.config.correct_sh_directions is not True
            or pipeline.model.field.correct_sh_directions is not True
            or pipeline.model.field.density_activation != 'softplus'):
        raise ValueError('SH-corrected native-softplus config and actual field required')
    prefix = spec['prefix_candidate']; path = Path(prefix['manifest'])
    if sha256(path) != prefix['manifest_sha256']:
        raise ValueError('Geometry manifest changed')
    meta = json.loads(path.read_text()); data = Path(spec['data'])
    for name, bound in [('transforms.json', 'transforms_sha256'), ('box.json', 'box_sha256')]:
        if sha256(data/name) != meta['bindings'][bound]:
            raise ValueError('Geometry data coordinates changed')
    if sha256(path.parent/'visibility_review.json') != meta['bindings']['visibility_review_sha256']:
        raise ValueError('Geometry review changed')
    for camera, digest in meta['source_semantic_hashes'].items():
        if sha256(data/'semantics'/f'{camera}.npz') != digest:
            raise ValueError('Geometry source semantics changed')
    train = sorted(r['camera'] for r in ready['records'] if Path(r['image']).name.startswith('train_'))
    arrays, receipt = load_reviewed_artifact(path, expected_bindings=meta['bindings'],
                                            bounds=ready['aabb'], train_cameras=train)
    # Measured leave-one-source-view motion must fit the declared soft band.
    review = json.loads((path.parent/'visibility_review.json').read_text())
    for point in review.get('points', []):
        ids = np.flatnonzero(arrays['point_idx'] == point['point'])
        if not len(ids):
            continue
        half = np.maximum(.005, .01*arrays['witness_depth'][ids]).min()
        for check in point['original_feature_graph']['leave_one_source_view_out']:
            if not check['positive_heldout_depth'] or check['xyz_shift_world'] >= half:
                raise ValueError('Available source uncertainty exceeds declared band')
    objective = SHGeometry(arrays, ready['aabb'], train, weight=prefix['weight'],
        band_weight=spec['band_weight'], rays=128, samples=64, seed=spec['seed'], device=pipeline.model.device)
    return SHGeometryAdapter(pipeline, sampler, objective, receipt)
