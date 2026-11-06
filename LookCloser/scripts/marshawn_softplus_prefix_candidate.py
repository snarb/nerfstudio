"""Opt-in TRAIN witness-prefix penalty using the field's native softplus path.

No monkeypatch, training hook, occupancy mutation, positive-band target or
exponential-logit adapter is installed. Artifact preparation/visibility review
and a real mixed-precision zero-update gradient gate must precede training.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_geometry(arrays, bounds, train_cameras):
    """Reject wrong coordinates, unqualified observations and unsafe prefixes."""
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (2, 3) or not np.isfinite(b).all() or not (b[1] > b[0]).all():
        raise ValueError('Invalid current field bounds')
    names = np.asarray(arrays['camera_names']).astype(str).tolist()
    if len(names) != len(set(names)) or names != list(train_cameras):
        raise ValueError('Exact ordered TRAIN camera domain required')
    o, d, xyz = [np.asarray(arrays[k], dtype=np.float64)
                 for k in ('origins', 'directions', 'witness_xyz')]
    start, end, depth = [np.asarray(arrays[k], dtype=np.float64)
                         for k in ('t_start', 't_end', 'witness_depth')]
    cameras, points = [np.asarray(arrays[k]) for k in ('camera_idx', 'point_idx')]
    n = len(start)
    if n < 1 or any(v.shape != (n, 3) for v in (o, d, xyz)) or any(
            v.shape != (n,) for v in (start, end, depth, cameras, points)):
        raise ValueError('Invalid prefix array shapes')
    if not all(np.isfinite(v).all() for v in (o, d, xyz, start, end, depth)):
        raise ValueError('Nonfinite geometry')
    if any(not np.issubdtype(v.dtype, np.integer) for v in (cameras, points)):
        raise ValueError('Integer camera/point provenance required')
    if (cameras < 0).any() or (cameras >= len(names)).any() or (points < 0).any():
        raise ValueError('Invalid original observation identity')
    pairs = np.stack([points, cameras], axis=1)
    if len(np.unique(pairs, axis=0)) != n:
        raise ValueError('Duplicate original observation')
    _, counts = np.unique(points, return_counts=True)
    # A point is triangulated from >=3 original views, while only explicitly
    # reviewed source rays need enter this auxiliary. Do not synthesize a third
    # supervised observation merely to pass a view-count guard.
    source_counts = np.asarray(arrays.get('source_track_view_count', []))
    if (source_counts.shape != (n,) or not np.issubdtype(source_counts.dtype, np.integer)
            or (source_counts < 3).any() or counts.min() < 2):
        raise ValueError('Each point needs three original TRAIN track views and two reviewed rays')
    if not np.allclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError('Distances require unit directions')
    if not np.allclose(o + d * depth[:, None], xyz, rtol=0, atol=1e-6):
        raise ValueError('Witness coordinate chain mismatch')
    margin = np.maximum(.01, .05 * depth)
    if (start < 0).any() or (end <= start).any() or (end > depth-margin+1e-7).any():
        raise ValueError('Prefix crosses conservative witness uncertainty margin')
    for t in (start, end):
        p = o + d * t[:, None]
        if (p < b[0]-1e-7).any() or (p > b[1]+1e-7).any():
            raise ValueError('Prefix outside current field AABB')
    if (xyz < b[0]-1e-7).any() or (xyz > b[1]+1e-7).any():
        raise ValueError('Reviewed actor witness outside current field AABB')
    return n


def load_reviewed_artifact(manifest, *, expected_bindings, bounds, train_cameras):
    """Load only a separately sealed accepted-floor clothing-prefix artifact."""
    path = Path(manifest)
    meta = json.loads(path.read_text())
    if (meta.get('kind') != 'reviewed_train_clothing_visible_prefixes'
            or meta.get('coordinate_system') != 'accepted_floor_world'
            or meta.get('uses_eval_supervision') is not False
            or meta.get('original_observed_track_graph_only') is not True
            or meta.get('manual_clothing_visibility_review') is not True
            or meta.get('distance_scale') != 1.):
        raise ValueError('Reviewed TRAIN-only rigid-floor geometry contract required')
    required = {'transforms_sha256', 'box_sha256', 'tracks_sha256',
                'witnesses_sha256', 'semantic_source_sha256', 'visibility_review_sha256'}
    if not required.issubset(expected_bindings) or meta.get('bindings') != expected_bindings:
        raise ValueError('Current data/geometry/review bindings mismatch')
    if not np.allclose(meta['aabb'], bounds, rtol=0, atol=1e-7):
        raise ValueError('Manifest uses another AABB')
    archive = (path.parent / meta['archive']).resolve()
    if archive.parent != path.parent.resolve() or sha256(archive) != meta['archive_sha256']:
        raise ValueError('Artifact path or SHA256 mismatch')
    with np.load(archive, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    validate_geometry(arrays, bounds, train_cameras)
    return arrays, dict(manifest_sha256=sha256(path), archive_sha256=sha256(archive),
                        bindings=expected_bindings, observations=len(arrays['t_start']))


class SoftplusPrefixCandidate:
    """Optional differentiable geometry scalar; caller composes existing losses."""
    def __init__(self, arrays, bounds, train_cameras, *, weight=.01,
                 rays=128, samples=64, seed=42, device='cpu'):
        validate_geometry(arrays, bounds, train_cameras)
        if not np.isfinite(weight) or weight < 0 or rays < 1 or samples < 1:
            raise ValueError('Finite nonnegative weight and positive sample counts required')
        self.weight, self.rays, self.samples = float(weight), int(rays), int(samples)
        self.seed, self.device = int(seed), torch.device(device)
        self.bounds = torch.as_tensor(np.asarray(bounds), dtype=torch.float32, device=device)
        self.values = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
                       for k in ('origins', 'directions', 't_start', 't_end')}
        cameras = np.asarray(arrays['camera_idx'])
        order = np.argsort(cameras, kind='stable')
        _, counts = np.unique(cameras, return_counts=True)
        self.order = torch.as_tensor(order, device=device)
        self.offsets = torch.as_tensor(np.r_[0, np.cumsum(counts)], device=device)

    def sample(self, step):
        if not isinstance(step, int) or step < 0:
            raise ValueError('Nonnegative integer optimizer step required')
        gen = torch.Generator(device=self.device)
        gen.manual_seed((self.seed + step*1000003) % (2**63-1))
        ncam = len(self.offsets)-1
        base, extra = divmod(self.rays, ncam)
        order = torch.randperm(ncam, generator=gen, device=self.device)
        camera = torch.cat([order.repeat(base), order[:extra]])
        counts = self.offsets[1:]-self.offsets[:-1]
        local = (torch.rand(self.rays, generator=gen, device=self.device)*counts[camera]).long()
        ids = self.order[self.offsets[camera]+local]
        # Open strata avoid sampling exactly on the field's excluded AABB faces.
        jitter = torch.rand((self.rays, self.samples), generator=gen, device=self.device)
        jitter = jitter.clamp(torch.finfo(torch.float32).eps, 1.-torch.finfo(torch.float32).eps)
        unit = (torch.arange(self.samples, device=self.device)[None]+jitter)/self.samples
        start, end = self.values['t_start'][ids], self.values['t_end'][ids]
        length = end-start
        distance = start[:, None]+unit*length[:, None]
        xyz = self.values['origins'][ids, None]+self.values['directions'][ids, None]*distance[..., None]
        return dict(indices=ids, points=xyz, length=length, camera_groups=camera)

    def __call__(self, field, step):
        if getattr(field, 'density_activation', None) != 'softplus':
            raise ValueError('Candidate is explicitly qualified for native softplus only')
        if not torch.allclose(field.aabb.to(self.bounds), self.bounds, rtol=0, atol=1e-6):
            raise ValueError('Runtime field AABB differs from qualified ray coordinates')
        batch = self.sample(step)
        xyz = batch['points'].reshape(-1, 3)
        if not bool(((xyz > self.bounds[0]) & (xyz < self.bounds[1])).all()):
            raise ValueError('Float32 samples touch/exceed field boundary; inset artifact endpoints')
        # No raw-logit assumptions: preserve actual density activation and its gradient.
        native = field.density_fn(xyz)
        if native.numel() != self.rays*self.samples:
            raise ValueError('Expected one native density per prefix sample')
        with torch.autocast(device_type=self.device.type, enabled=False):
            sigma = native.reshape(self.rays, self.samples).float()
            if not bool(torch.isfinite(sigma).all()) or bool((sigma < 0).any()):
                raise FloatingPointError('Nonfinite or negative native prefix density')
            tau = sigma.mean(1)*batch['length'].float()
            loss = self.weight*tau.mean()
        return loss, dict(prefix_tau_mean=tau.detach().mean(),
                          prefix_tau_max=tau.detach().max(),
                          prefix_ray_count=self.rays)


class PrefixSamplingAdapter:
    """Compose one explicit auxiliary while retaining the base sampler state."""
    def __init__(self, pipeline, sampler, objective, receipt):
        self.sampler, self.objective, self.receipt = sampler, objective, receipt
        self.step = 0
        original = pipeline.get_train_loss_dict

        def get_losses(step):
            outputs, losses, metrics = original(step)
            if 'train_clothing_prefix_tau' in losses:
                raise ValueError('Prefix objective already installed')
            value, diagnostic = objective(pipeline.model.field, int(step))
            losses['train_clothing_prefix_tau'] = value
            metrics.update(diagnostic)
            return outputs, losses, metrics
        pipeline.get_train_loss_dict = get_losses

    def set_step(self, step):
        self.step = step
        self.sampler.set_step(step)

    def state_dict(self):
        return dict(schema='prefix_candidate_v1', sampler=self.sampler.state_dict(),
            step=self.step, receipt=self.receipt, seed=self.objective.seed,
            weight=self.objective.weight, rays=self.objective.rays, samples=self.objective.samples,
            rng_policy='stateless seed + optimizer_step*1000003; independent of global RNG')

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('schema', 'receipt', 'seed', 'weight', 'rays', 'samples', 'rng_policy'):
            if state[key] != expected[key]:
                raise ValueError('Prefix resume provenance/policy changed: '+key)
        self.sampler.load_state_dict(state['sampler'])
        self.set_step(state['step'])

    def audit_state(self):
        return dict(**self.state_dict(), sampler_audit=self.sampler.audit_state())


def install_prefix(pipeline, sampler, spec, ready):
    """Explicit opt-in only, called by the separate candidate worker."""
    prefix = spec['prefix_candidate']
    path = Path(prefix['manifest'])
    if sha256(path) != prefix['manifest_sha256']:
        raise ValueError('Frozen candidate manifest changed')
    meta = json.loads(path.read_text())
    data = Path(spec['data'])
    if sha256(path.parent/'visibility_review.json') != meta['bindings']['visibility_review_sha256']:
        raise ValueError('Candidate visibility review changed')
    if (meta['bindings']['transforms_sha256'] != sha256(data/'transforms.json')
            or meta['bindings']['box_sha256'] != sha256(data/'box.json')):
        raise ValueError('Candidate field coordinates/data differ from sealed artifact')
    for cam, digest in meta['source_semantic_hashes'].items():
        if sha256(data/'semantics'/f'{cam}.npz') != digest:
            raise ValueError('Candidate actor semantic source changed')
    train = sorted(r['camera'] for r in ready['records'] if Path(r['image']).name.startswith('train_'))
    arrays, receipt = load_reviewed_artifact(path, expected_bindings=meta['bindings'],
                                            bounds=ready['aabb'], train_cameras=train)
    objective = SoftplusPrefixCandidate(arrays, ready['aabb'], train,
        weight=prefix['weight'], rays=128, samples=64, seed=spec['seed'], device=pipeline.model.device)
    return PrefixSamplingAdapter(pipeline, sampler, objective, receipt)
