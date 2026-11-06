"""Read-only CPU audit of all released experimental frequency map outputs."""
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(payload):
    import numpy as np
    import torch
    from PIL import Image
    root = Path(payload['root']); generation = root/payload['generation']
    source = root/'resolution_4k_train63'
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    if actual != payload['source_inventory']: raise ValueError('Full source generation inventory changed')
    rows = []
    for job in payload['jobs']:
        camera = job.rsplit('_', 1)[-1]
        if camera in ('DC', 'HC', 'KA'): raise ValueError('EVAL camera cannot be fitted')
        spec = json.loads((generation/camera/'spec_silver.json').read_text())
        output = root/'jobs'/job/'run'
        complete = json.loads((output/'complete.json').read_text())
        if complete['recipe'] != spec['recipe']: raise ValueError('Recipe mismatch')
        for source, expected in spec['inputs'].items():
            if sha(source) != expected: raise ValueError('Source changed: '+source)
        if complete['source_image_sha256'] != sha(spec['image']): raise ValueError('Wrong source image')
        if complete['source_complete_sha256'] != sha(spec['source_complete']): raise ValueError('Wrong source generation')
        paths = {'map': output/'maps'/('train_'+camera+'.pt'), 'metadata': output/'maps'/('train_'+camera+'.json'), 'censor': output/'unresolved_patch_mask.png'}
        for key, field in [('map', 'map_sha256'), ('metadata', 'metadata_sha256'), ('censor', 'censor_mask_sha256')]:
            if sha(paths[key]) != complete[field]: raise ValueError('Output changed: '+key)
        frequency = torch.load(paths['map'], map_location='cpu', weights_only=True)
        if tuple(frequency.shape) != (253, 480) or not torch.isfinite(frequency).all() or not ((frequency >= 16) & (frequency <= 8192)).all():
            raise ValueError('Invalid scalar frequency map')
        levels = np.rint(np.log(frequency.numpy()/16)/np.log(8192/16)*15).astype(int)
        histogram = np.bincount(levels.ravel(), minlength=16).tolist()
        if histogram != complete['level_histogram']: raise ValueError('Histogram mismatch')
        remaining = 253*480
        for level in range(16):
            row = complete['counts_by_level'].get(str(level))
            if row is None:
                if remaining: raise ValueError('Missing unresolved-level evidence')
                break
            if row['evaluated'] != remaining or not 0 <= row['passed'] <= remaining: raise ValueError('Inconsistent progressive counts')
            remaining -= row['passed']
            if histogram[level] != row['passed']+(remaining if level == 15 else 0): raise ValueError('Assignment mismatch')
        mask = np.asarray(Image.open(paths['censor'])) > 0
        if mask.shape != (253, 480) or int(mask.sum()) != remaining or remaining != complete['exact_unresolved_patches']:
            raise ValueError('Censor mask/count mismatch')
        rows.append({'camera': camera, 'job': job, 'source_image_sha256': complete['source_image_sha256'], 'complete_sha256': sha(output/'complete.json'), 'output_hashes': {k:sha(p) for k,p in paths.items()}, 'total_patches': 253*480, 'unresolved_patches': remaining, 'unresolved_fraction': remaining/(253*480), 'maxlevel_fraction': float((levels==15).mean()), 'elapsed_seconds': complete['elapsed_seconds'], 'gpu_peak_bytes': complete['gpu_peak_memory_bytes']})
    if len(rows) != 63 or len({r['camera'] for r in rows}) != 63: raise ValueError('Incomplete camera set')
    return {'status': 'EXPERIMENTAL_FREQUENCY_CANDIDATE_COMPLETE', 'source_generation': str(root/'resolution_4k_train63'), 'full_source_files_hash_verified': len(actual), 'cameras': rows, 'total_unresolved_fraction': sum(r['unresolved_patches'] for r in rows)/sum(r['total_patches'] for r in rows), 'maxlevel_interpretation': 'Includes unresolved fallback, not a detail certificate', 'geometry_or_detail_quality_certified': False, 'default_promoted': False, 'NeRF_training_authorized': False, 'READY_created': False}


if 'PAYLOAD' in globals():
    print(json.dumps(audit(PAYLOAD), allow_nan=False))
