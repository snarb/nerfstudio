"""Finite positive floor-band SCENE PRIOR, never measured depth or free space.

Plain external data/objective: no NN parameters, registered buffers, RGB targets,
occupancy edits or negative/carving constraints. Approval is a controller gate.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def opacity_nll_from_log_tau(log_tau):
    """Same stable log-domain formula as the frozen native SfM band auxiliary."""
    log_tau = log_tau.float()
    small = -log_tau + .5 * torch.exp(log_tau.clamp(max=-10))
    regular_tau = torch.exp(log_tau.clamp(min=-10, max=15))
    regular = -torch.log(-torch.expm1(-regular_tau))
    return torch.where(log_tau < -10, small, regular)


def validate_floor_domain(triangles, areas, normal, offset, halfwidth, bounds):
    triangles, areas, normal, bounds = [np.asarray(x, dtype=np.float64)
                                       for x in (triangles, areas, normal, bounds)]
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError('Nonempty explicit world triangles [T,3,3] required')
    if areas.shape != (len(triangles),) or normal.shape != (3,) or bounds.shape != (2, 3):
        raise ValueError('Invalid finite floor geometry shapes')
    if not all(np.isfinite(x).all() for x in [triangles, areas, normal, bounds]) or not math.isfinite(offset):
        raise ValueError('Nonfinite floor geometry')
    if not np.isclose(np.linalg.norm(normal), 1, rtol=0, atol=1e-9):
        raise ValueError('Unit world normal required for physical band length')
    if abs(float(halfwidth) - .02) > 1e-12 or np.any(bounds[1] <= bounds[0]):
        raise ValueError('Reviewed halfwidth .02 world and positive AABB required')
    actual_area = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1)
    if (actual_area <= 0).any() or not np.allclose(areas, actual_area, rtol=1e-8, atol=1e-14):
        raise ValueError('Positive area weights must match explicit triangles')
    if np.max(np.abs(triangles @ normal + offset)) > 1e-7:
        raise ValueError('Domain vertices must lie on the declared prior plane')
    # Convex prism extrema occur at these endpoints; no silent clipping allowed.
    endpoints = triangles[:, :, None, :] + np.array([-halfwidth, halfwidth])[None, None, :, None] * normal
    if not ((endpoints > bounds[0] + 1e-6) & (endpoints < bounds[1] - 1e-6)).all():
        raise ValueError('Every complete normal band must be strictly inside AABB')
    return triangles, actual_area, normal, bounds


class FloorBandScenePrior:
    """Positive opaque-band likelihood on the approved finite triangle union."""
    def __init__(self, manifest, expected_sha256, current_bounds, transforms, preparation,
                 *, weight=.001, columns=64, samples=32, seed=42, device='cpu'):
        path = Path(manifest)
        if sha(path) != expected_sha256:
            raise ValueError('Floor-prior manifest SHA changed')
        payload = json.loads(path.read_text())
        if payload['kind'] != 'FINITE_SOFT_GEOMETRIC_HYPOTHESIS_NOT_MEASURED_DEPTH':
            raise ValueError('Explicit finite SCENE PRIOR, not measured depth, required')
        for flag in ['no_eval_RGB_or_gains_fitted', 'no_los_negative_constraints', 'no_carving']:
            if payload.get(flag) is not True:
                raise ValueError('Missing declared prior limitation: ' + flag)
        if not math.isfinite(weight) or weight < 0 or columns != 64 or samples != 32:
            raise ValueError('Reviewed nonnegative weight,64columns,32normal samples required')
        for source in [transforms, preparation]:
            source = str(Path(source).resolve())
            if payload['bindings'].get(source) != sha(source):
                raise ValueError('Current transforms/preparation binding mismatch')
        archive = Path(payload['domain_triangles_file'])
        if sha(archive) != payload['domain_triangles_sha256']:
            raise ValueError('Finite domain archive SHA changed')
        with np.load(archive, allow_pickle=False) as data:
            triangles, areas, normal, bounds = validate_floor_domain(
                data['triangles_xyz'], data['triangle_area'], data['normal'],
                float(data['offset']), float(data['halfwidth']), current_bounds)
            if not np.allclose(data['aabb'], bounds, rtol=0, atol=1e-6):
                raise ValueError('Archive/current model AABB mismatch')
            votes = data['triangle_common_clear_train_votes']
            cameras = data['source_cameras'].tolist()
            if votes.shape != (len(triangles), len(cameras)) or votes.dtype != np.bool_ or (votes.sum(axis=1) < 3).any():
                raise ValueError('Each finite triangle needs>=3 common clear TRAIN votes')
            if not np.allclose(normal, payload['fit_normal'], rtol=0, atol=1e-10) or abs(float(data['offset']) - payload['fit_offset']) > 1e-10:
                raise ValueError('Plane/archive contract mismatch')
        metadata = json.loads(Path(transforms).read_text())
        train = {Path(name).stem.rsplit('_', 1)[-1] for name in metadata['train_filenames']}
        if len(set(cameras)) != len(cameras) or not set(cameras) <= train:
            raise ValueError('All domain cameras must belong to the actual TRAIN split')
        if not np.isclose(areas.sum(), payload['surface_area_preparedworld'], rtol=1e-8, atol=1e-12):
            raise ValueError('Declared finite domain area mismatch')
        self.weight, self.columns, self.samples, self.seed = float(weight), columns, samples, int(seed)
        self.halfwidth, self.device = .02, torch.device(device)
        self.triangles = torch.as_tensor(triangles.copy(), dtype=torch.float32, device=device)
        self.areas = torch.as_tensor(areas.copy(), dtype=torch.float32, device=device)
        self.normal = torch.as_tensor(normal.copy(), dtype=torch.float32, device=device)
        self.bounds = torch.as_tensor(bounds.copy(), dtype=torch.float32, device=device)
        self.receipt = dict(manifest=str(path), manifest_sha256=expected_sha256,
            domain_archive=str(archive), domain_archive_sha256=payload['domain_triangles_sha256'],
            target='positive opaque probability within finite normal band', weight=weight,
            band_halfwidth_world=.02, columns=columns, normal_samples=samples,
            finite_surface_area=float(areas.sum()), prior_not_ground_truth=True,
            approved_for_training=payload.get('approved_for_training', False),
            no_negative_or_carving_constraints=True, no_RGB_or_eval_targets=True,
            gradient_path='native cached raw logdensity +log(average_init_density); bypass native trunc_exp STE for this auxiliary only',
            assumptions=payload.get('limitations', []))

    def sample_batch(self, step):
        if not isinstance(step, int) or step < 0:
            raise ValueError('Nonnegative integer global step required')
        generator = torch.Generator(device=self.device)
        local_seed = int(np.random.SeedSequence([self.seed, step, 0x464C4F4F]).generate_state(1, dtype=np.uint64)[0])
        generator.manual_seed(local_seed % (2**63 - 1))
        indices = torch.multinomial(self.areas, self.columns, replacement=True, generator=generator)
        uv = torch.rand((self.columns, 2), generator=generator, device=self.device)
        root = torch.sqrt(uv[:, 0])
        bary = torch.stack([1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]], -1)
        centers = (self.triangles[indices] * bary[..., None]).sum(1)
        jitter = torch.rand((self.columns, self.samples), generator=generator, device=self.device)
        offsets = -self.halfwidth + 2 * self.halfwidth * (torch.arange(self.samples, device=self.device)[None] + jitter) / self.samples
        points = centers[:, None] + offsets[..., None] * self.normal
        if not bool(((points > self.bounds[0]) & (points < self.bounds[1])).all()):
            raise ValueError('FP32 prior sample outside native AABB; never clip silently')
        return dict(triangle_indices=indices, centers=centers, offsets=offsets, points=points)

    def __call__(self, field, step):
        batch = self.sample_batch(step);points = batch['points'].reshape(-1, 3)
        if not hasattr(field, 'average_init_density') or getattr(field, 'spatial_distortion', None) is not None:
            raise ValueError('Reviewed bounded native density field required')
        if not bool(((points > field.aabb[0]) & (points < field.aabb[1])).all()):
            raise ValueError('Every band point must be strictly inside actual field AABB')
        density = field.density_fn(points)
        raw = field._density_before_activation  # capture immediately after this query
        scale = float(field.average_init_density)
        if raw.numel() != self.columns * self.samples or density.numel() != raw.numel() or not math.isfinite(scale) or scale <= 0 or not bool(torch.isfinite(raw).all()):
            raise ValueError('Invalid native density/preactivation shape or scale')
        with torch.autocast(device_type=self.device.type, enabled=False):
            log_sigma = raw.reshape(self.columns, self.samples).float() + math.log(scale)
            log_tau = torch.logsumexp(log_sigma, 1) - math.log(self.samples) + math.log(2 * self.halfwidth)
            loss = self.weight * opacity_nll_from_log_tau(log_tau).mean()
            alpha = -torch.expm1(-torch.exp(log_tau.detach().clamp(max=15)))
        return loss, dict(floor_scene_prior_alpha_mean=alpha.mean(), floor_scene_prior_alpha_min=alpha.min(),
                          floor_scene_prior_alpha_p10=torch.quantile(alpha, .1), floor_scene_prior_logtau_min=log_tau.detach().min())


def __getattr__(name):
    if name != 'NativeFloorBandPriorPipeline':
        raise AttributeError(name)
    from marshawn_packed_distortion_candidate import NativePackedDistortionPipeline
    class NativeFloorBandPriorPipeline(NativePackedDistortionPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = self.config;manifest = getattr(cfg, 'floor_band_prior_manifest', None)
            weight = float(getattr(cfg, 'floor_band_prior_weight', 0))
            prior = None
            if manifest is not None:
                data = Path(cfg.datamanager.dataparser.data)
                prior = FloorBandScenePrior(manifest, cfg.floor_band_prior_manifest_sha256,
                    self.model.scene_aabb.reshape(2, 3).detach().cpu().numpy(), data / 'transforms.json',
                    cfg.final_geometry_preparation, weight=weight, device=self.device)
            elif weight != 0:
                raise ValueError('Nonzero finite floor prior requires explicit bound manifest')
            object.__setattr__(self, '_floor_band_scene_prior', prior)

        def get_train_loss_dict(self, step):
            outputs, losses, metrics = super().get_train_loss_dict(step)
            if self._floor_band_scene_prior is not None:
                loss, diagnostic = self._floor_band_scene_prior(self.model.field, step)
                losses['finite_floor_scene_prior'] = loss;metrics.update(diagnostic)
            return outputs, losses, metrics
    NativeFloorBandPriorPipeline.__name__ = name
    NativeFloorBandPriorPipeline.__qualname__ = name
    NativeFloorBandPriorPipeline.__module__ = __name__
    globals()[name] = NativeFloorBandPriorPipeline
    return NativeFloorBandPriorPipeline
