"""Isolated camera-balanced projected-source geometry prior; no optimizer ownership."""
import hashlib
import random
import struct

ARMS = {'control': (0., 0., .05), 'old_prefix': (.003, 0., .05),
        'closer_prefix': (.003, 0., .02), 'closer_prefix_band': (.003, .001, .02)}
RESERVED = {63, 64, 71, 72, 2219, 5556, 5568}


class CameraBalancedStream:
    """Independent Python RNG; one uniform camera then one uniform observation."""
    def __init__(self, camera_ids, seed=4202084, batch_size=128):
        if seed != 4202084 or batch_size != 128 or len(camera_ids) != 84:
            raise ValueError('Only sealed84 / private4202084 / packet128 is proposed')
        self.groups = {}
        for index, camera in enumerate(camera_ids):
            self.groups.setdefault(int(camera), []).append(index)
        self.cameras = sorted(self.groups)
        self.rng = random.Random(seed)
        self.batch_size = batch_size
        self.draws = 0
        self.chain = hashlib.sha256(b'focused-clothing84-v1').hexdigest()
        self.last = None

    def draw(self):
        indices = []
        for _ in range(self.batch_size):
            group = self.groups[self.cameras[self.rng.randrange(len(self.cameras))]]
            indices.append(group[self.rng.randrange(len(group))])
        raw = struct.pack('<' + 'I' * len(indices), *indices)
        self.last = hashlib.sha256(raw).hexdigest()
        self.chain = hashlib.sha256(bytes.fromhex(self.chain) + raw).hexdigest()
        self.draws += 1
        return indices

    def state_dict(self):
        version, internal, gaussian = self.rng.getstate()
        return dict(schema=1, groups={str(k): v for k, v in self.groups.items()},
                    batch_size=self.batch_size, random_state=[version, list(internal), gaussian],
                    draws=self.draws, chain=self.chain, last=self.last)

    def load_state_dict(self, state):
        if state['schema'] != 1 or state['groups'] != {str(k): v for k, v in self.groups.items()} or state['batch_size'] != self.batch_size:
            raise ValueError('Geometry camera membership or packet policy changed')
        version, internal, gaussian = state['random_state']
        self.rng.setstate((version, tuple(internal), gaussian))
        self.draws, self.chain, self.last = state['draws'], state['chain'], state['last']


class FocusedClothingGeometry:
    def __init__(self, model, arrays, source_rows, archive_sha256, arm):
        import torch
        if arm not in ARMS or len(archive_sha256) != 64:
            raise ValueError('Unknown source-bound geometry arm')
        ids = list(map(int, arrays['point_idx']))
        if len(ids) != 84 or len(set(ids)) != 31 or set(ids) & RESERVED:
            raise ValueError('Qualified31points84observations excluding all reserves required')
        if len(source_rows) != 84 or any(row['index'] != i or row['point'] != ids[i] for i, row in enumerate(source_rows)):
            raise ValueError('Clipped-band source row identity/order differs')
        self.model, self.arm, self.archive_sha256 = model, arm, archive_sha256
        self.prefix_coefficient, self.band_coefficient, self.gap = ARMS[arm]
        self.stream = CameraBalancedStream(arrays['camera_idx'])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=model.device)
        self.origins, self.directions = tensor(arrays['origins']), tensor(arrays['directions'])
        self.near, self.depth = tensor(arrays['t_start']), tensor(arrays['witness_depth'])
        torch.testing.assert_close(self.depth, tensor([row['depth'] for row in source_rows]), rtol=0, atol=1e-6)
        old_far = self.depth - torch.maximum(torch.full_like(self.depth, .01), .05 * self.depth)
        torch.testing.assert_close(old_far, tensor(arrays['t_end']), rtol=0, atol=1e-6)
        torch.testing.assert_close(old_far, tensor([row['prefix_end'] for row in source_rows]), rtol=0, atol=1e-6)
        self.far = self.depth - torch.maximum(torch.full_like(self.depth, .01), self.gap * self.depth)
        self.band_near = tensor([row['band_start'] for row in source_rows])
        self.band_far = tensor([row['band_end'] for row in source_rows])
        if not bool(((self.near < self.far) & (self.far < self.band_near) & (self.band_near < self.depth) & (self.depth < self.band_far)).all()):
            raise ValueError('Prefix/band intervals overlap or exclude source witness')
        if not bool(model.semantic_lookup(tensor(arrays['witness_xyz'])).all()):
            raise ValueError('Source witness outside unchanged semantic admission')
        self.last = None

    def state_dict(self):
        return dict(schema=1, arm=self.arm, archive_sha256=self.archive_sha256,
                    prefix_samples=512, band_samples=256, stream=self.stream.state_dict())

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(k) != expected[k] for k in ('schema', 'arm', 'archive_sha256', 'prefix_samples', 'band_samples')):
            raise ValueError('Geometry source or quadrature policy changed')
        self.stream.load_state_dict(state['stream'])

    def _tau(self, indices, left, right, samples):
        import torch
        from nerfstudio.cameras.rays import RaySamples, Frustums
        left, right = left[indices], right[indices]
        t = left[:, None] + (right-left)[:, None] * (torch.arange(samples, device=left.device, dtype=left.dtype) + .5) / samples
        xyz = (self.origins[indices, None, :] + self.directions[indices, None, :] * t[..., None]).reshape(-1, 3)
        zero = torch.zeros((len(xyz), 1), device=xyz.device)
        ray_samples = RaySamples(frustums=Frustums(origins=xyz, directions=torch.zeros_like(xyz), starts=zero, ends=zero, pixel_area=torch.ones_like(zero)))
        sigma, _ = self.model.field.get_density(ray_samples)
        sigma = sigma.float().reshape(len(indices), samples)
        admitted = self.model.semantic_lookup(xyz).reshape(len(indices), samples)
        if not bool((sigma[~admitted] == 0).all()):
            raise ValueError('Auxiliary query bypassed shared fixed semantic density selector')
        tau = sigma.mean(1) * (right-left)
        if not bool(torch.isfinite(tau).all()) or bool((tau < 0).any()):
            raise FloatingPointError('Nonfinite/negative native optical depth')
        return tau

    def loss(self):
        import torch
        # Every arm, including control, consumes exactly one independent packet per attempt.
        packet = self.stream.draw()
        indices = torch.tensor(packet, device=self.model.device, dtype=torch.long)
        value = torch.zeros((), device=self.model.device)
        diagnostic = dict(packet_sha256=self.stream.last, draws=self.stream.draws, arm=self.arm)
        if self.prefix_coefficient:
            tau = self._tau(indices, self.near, self.far, 512)
            value = value + self.prefix_coefficient * tau.mean()
            diagnostic['prefix_tau'] = float(tau.detach().mean())
        if self.band_coefficient:
            tau = self._tau(indices, self.band_near, self.band_far, 256)
            if not bool((tau > 0).all()):
                raise FloatingPointError('Positive band has zero optical depth; do not hide unsupported witness')
            band = -torch.log(-torch.expm1(-tau))
            value = value + self.band_coefficient * band.mean()
            diagnostic['band_tau'] = float(tau.detach().mean())
        self.last = diagnostic
        return value
