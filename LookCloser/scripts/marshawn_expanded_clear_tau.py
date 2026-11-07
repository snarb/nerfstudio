"""Explicitly reviewed AA/BA/HA clear-target population; frozen original helper remains intact."""
import hashlib
import json
from pathlib import Path

from marshawn_clear_tau import QualifiedClearTau as OriginalClearTau
from marshawn_clear_tau import digest, validate_targets as validate_small_targets

PROPOSAL_SHA = 'b2dc3905295399307b6143960e453b91044d3a6339c91b0aadfbe78cace0a2cb'
REVIEW_SHA = 'fe417ec5d971234ae368362370d02882476a731bf05df649ae2b5ae9537b871b'
SMALL_SHA = '7459a0bf273cd8b90504fa84834fbfb2bbd2d3b092ff9eac8812d21b3963bcf1'
COUNTS = {'AA': 1872, 'BA': 2689, 'HA': 2808}


def ray_geometry(frame, box, x, y):
    """Same pixel-center convention and strict floor predicates as the reviewed proposal."""
    import numpy as np
    m = np.asarray(frame['transform_matrix']); origin = m[:3, 3]
    direction = np.array([(x + .5 - frame['cx']) / frame['fl_x'],
                          -(y + .5 - frame['cy']) / frame['fl_y'], -1.]) @ m[:3, :3].T
    direction /= np.linalg.norm(direction)
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = (box[0] - origin) / direction, (box[1] - origin) / direction
        near = max(np.minimum(a, b).max(), .01)
        far = np.maximum(a, b).min(); floor = -origin[2] / direction[2]
    if not (np.isfinite(np.r_[origin, direction, near, far, floor]).all()
            and far > near and min(origin[2] + near*direction[2], origin[2] + far*direction[2]) > .10
            and floor > far):
        raise ValueError('Target violates reviewed AABB/floor predicates')
    return origin, direction, near, far, floor


def validate_targets(path, expected_sha, data, support_manifest, population):
    import numpy as np
    path, data = Path(path), Path(data)
    if population == 'small243':
        if expected_sha != SMALL_SHA:
            raise ValueError('Original243 target artifact must be byte-identical')
        return validate_small_targets(path, expected_sha, data, support_manifest)
    if population != 'expanded7369' or digest(path) != expected_sha:
        raise ValueError('Unreviewed target population or changed artifact')
    target = json.loads(path.read_text()); contract = target['polygon_review']
    proposal_path = path.parent / contract['proposal']
    review_path = path.parent / contract['review']
    if digest(proposal_path) != PROPOSAL_SHA or digest(review_path) != REVIEW_SHA:
        raise ValueError('Source polygon approval changed')
    proposal = json.loads(proposal_path.read_text()); review = json.loads(review_path.read_text())
    if (review.get('approved_for_experimental_targets') is not True
            or review.get('proposal_sha256') != PROPOSAL_SHA or review.get('total_pixels') != 7369
            or {r['camera']: r['pixels'] for r in review['rows']} != COUNTS
            or proposal['original_243_targets_sha256'] != SMALL_SHA):
        raise ValueError('Root review does not approve this exact target set')
    small_path = path.parent / contract['original_targets']
    small = validate_small_targets(small_path, SMALL_SHA, data, support_manifest)
    if digest(data/'transforms.json') != proposal['transforms_sha256'] or digest(data/'ready.json') != proposal['ready_sha256']:
        raise ValueError('Reviewed source camera or box changed')
    tf = json.loads((data/'transforms.json').read_text())
    frames = {r['physical_camera']: r for r in tf['frames'] if r['file_path'] in tf['train_filenames']}
    box = np.asarray(json.loads((data/'ready.json').read_text())['aabb'])
    np.testing.assert_array_equal(box, target['AABB']); np.testing.assert_array_equal(box, proposal['AABB'])
    if len(target['targets']) != 7369 or {r['camera'] for r in target['targets']} != set(COUNTS):
        raise ValueError('Only the7369 approved filtered pixels are eligible')
    source = {r['camera']: r for r in json.loads(Path(support_manifest).read_text())['records']}
    for record in proposal['records']:
        ca = record['camera']; frame = frames[ca]
        archive = Path(source[ca]['archive'])
        if not archive.exists(): archive = path.parent/'source_archives'/f'{ca}.npz'
        semantic = data/'semantics'/f'{ca}.npz'
        for p, expected in [(archive, record['source_archive_sha256']), (semantic, record['source_semantic_sha256']),
                            (data/frame['file_path'], record['source_RGB_sha256'])]:
            if digest(p) != expected: raise ValueError('Reviewed pixel source changed')
        if source[ca]['archive_sha256'] != record['source_archive_sha256']:
            raise ValueError('Source archive does not match semantic geometry manifest')
        approved_path = proposal_path.parent/f'{ca}_proposed_pixels.npz'
        if digest(approved_path) != record['proposal_archive_sha256']:
            raise ValueError('Filtered polygon pixel artifact changed')
        with np.load(approved_path) as z:
            accepted = set(zip(z['x'].tolist(), z['y'].tolist()))
            if accepted != set(zip(*np.where(z['proposed_pixels'])[::-1])):
                raise ValueError('Filtered polygon raster differs from accepted coordinates')
        rows = [r for r in target['targets'] if r['camera'] == ca]
        pixels = [(r['x'], r['y']) for r in rows]
        if len(pixels) != COUNTS[ca] or len(set(pixels)) != len(pixels) or set(pixels) != accepted:
            raise ValueError('Missing, duplicate, or unapproved target pixel')
        original = {(r['x'], r['y']) for r in small['targets'] if r['camera'] == ca}
        if not original <= accepted: raise ValueError('Expanded target set lost original reviewed patch')
        with np.load(archive) as z: support, unknown, valid = z['support'], z['unknown'], z['valid']
        with np.load(semantic) as z: train_valid, confidence = z['valid'], z['W64']
        for row in rows:
            x, y = row['x'], row['y']
            if not (valid[y, x] and train_valid[y, x] and not support[y, x] and not unknown[y, x] and confidence[y, x] == 0):
                raise ValueError('Target overlaps source actor support, UNKNOWN, confidence or invalid raster')
            o, d, near, far, floor = ray_geometry(frame, box, x, y)
            for a, b in [(o, row['origin']), (d, row['direction']), ([near, far, floor], [row['near'], row['far'], row['floor_t']])]:
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    return target


class QualifiedClearTau(OriginalClearTau):
    """Same objective and quadrature, only explicitly bound target population varies."""
    def __init__(self, model, targets, sha256, coefficient):
        import torch
        if type(coefficient) is not float or coefficient not in (0., .001):
            raise ValueError('Population contrast permits only control0 or tau.001')
        count = len(targets['targets'])
        if count not in (243, 7369): raise ValueError('Unreviewed population size')
        if count == 7369 and coefficient != .001: raise ValueError('Expanded arm must use tau.001')
        # Original initialization is unchanged, then arrays are extended to the validated population.
        super().__init__(model, {'targets': targets['targets'][:243]}, sha256, coefficient)
        self.population = count
        self.o = torch.tensor([r['origin'] for r in targets['targets']], device=model.device, dtype=torch.float32)
        self.d = torch.tensor([r['direction'] for r in targets['targets']], device=model.device, dtype=torch.float32)
        self.near = torch.tensor([r['near'] for r in targets['targets']], device=model.device, dtype=torch.float32)
        self.far = torch.tensor([r['far'] for r in targets['targets']], device=model.device, dtype=torch.float32)

    def state_dict(self):
        return dict(super().state_dict(), population=self.population)

    def load_state_dict(self, state):
        if state.get('population') != self.population: raise ValueError('Auxiliary population changed across resume')
        super().load_state_dict(state)

    def draw(self):
        import torch
        ids = torch.randint(self.population, (64,), generator=self.generator)
        jitter = torch.rand((64, 512), generator=self.generator)
        raw = ids.numpy().tobytes() + jitter.numpy().tobytes(); h = hashlib.sha256(raw).hexdigest()
        self.first_hash = self.first_hash or h
        self.chain = hashlib.sha256(bytes.fromhex(self.chain) + raw).hexdigest(); self.draws += 1
        return ids.to(self.o.device), jitter.to(self.o.device), h

    def evaluate(self, pipe, targets, output, step):
        import torch
        from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
        rng = _capture_rng_state(); training = self.model.training; self.model.eval(); rows = []
        order = [Path(p).stem.split('_')[-1] for p in pipe.datamanager.train_dataset.image_filenames]
        cameras = pipe.datamanager.train_dataset.cameras.to(self.model.device)
        try:
            with torch.no_grad():
                for ca in ('AA', 'BA', 'HA'):
                    ids = [i for i, r in enumerate(targets['targets']) if r['camera'] == ca]
                    alphas, taus = [], []
                    for start in range(0, len(ids), 128):
                        chunk = ids[start:start+128]; q = [targets['targets'][i] for i in chunk]
                        coords = torch.tensor([[r['y']+.5, r['x']+.5] for r in q], device=self.model.device)
                        rays = cameras[order.index(ca):order.index(ca)+1].generate_rays(0, coords=coords, keep_shape=False)
                        alphas.append(self.model(rays)['accumulation'].flatten().cpu())
                        taus.append(self.tau(torch.tensor(chunk, device=self.o.device)).cpu())
                    alpha, tau = torch.cat(alphas), torch.cat(taus)
                    cx, cy = {'AA': (961, 239), 'BA': (954, 175), 'HA': (1067, 113)}[ca]
                    original = torch.tensor([abs(targets['targets'][i]['x']-cx) <= 4 and
                                             abs(targets['targets'][i]['y']-cy) <= 4 for i in ids])
                    if int(original.sum()) != 81: raise ValueError('Common original9x9 diagnostic subset missing')
                    rows.append(dict(camera=ca, targets=len(ids), actual_native_alpha=alpha.tolist(),
                                     tau512_fixed_midpoint=tau.tolist(), mean_native_alpha=float(alpha.mean()),
                                     mean_tau512=float(tau.mean()), max_native_alpha=float(alpha.max()),
                                     common_original9x9_mean_native_alpha=float(alpha[original].mean()),
                                     common_original9x9_mean_tau512=float(tau[original].mean())))
        finally:
            self.model.train(training); _restore_rng_state(rng)
        result = dict(step=step, source_sha256=self.target_sha256, rows=rows, auxiliary_generator_advanced=False,
                      scope=f'Only{self.population} reviewed filtered TRAIN pixels; no global free-space claim',
                      evaluation_quadrature='fixed midpoint512 in chunks128; training uses private per-step jitter')
        path = Path(output)/'clear_targets'/f'step-{step:09d}.json'; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2)+'\n'); return result
