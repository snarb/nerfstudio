"""One source-bound progressive frequency fit for an independent video frame.

The frozen fitter owns the mathematics. This wrapper supplies per-image locking,
strict reuse checks and atomic publication, including after controller restarts.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time


RECIPE = dict(ssim_threshold=.95, train_steps_per_level=1000,
              eval_patch_batch_size=1024, patch_size=8, stride=8,
              n_levels=16, min_res=16, max_res=8192)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.%d.tmp' % os.getpid())
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def validate_spec(spec):
    required = {'frame', 'camera', 'image', 'image_sha256', 'output',
                'source_code', 'preprocess_source_sha256'}
    if not required <= set(spec):
        raise ValueError('Missing frequency spec fields: '+str(sorted(required-set(spec))))
    if not isinstance(spec['frame'], int) or not isinstance(spec['camera'], str):
        raise ValueError('Original integer frame ID and physical camera required')
    for name in ['image', 'output', 'source_code']:
        if not Path(spec[name]).is_absolute(): raise ValueError('Absolute '+name+' required')
    if Path(spec['output']).suffix != '.pt': raise ValueError('Output must be a .pt map')
    if spec.get('recipe', RECIPE) != RECIPE: raise ValueError('Frozen winning map recipe required')
    if not 0 < float(spec.get('allocator_GiB', 4)) <= 8: raise ValueError('Invalid allocator cap')
    source = Path(spec['source_code'])/'nerfstudio/scripts/lookcloser_preprocess.py'
    if sha(source) != spec['preprocess_source_sha256']: raise ValueError('Frozen fitter changed')
    if sha(spec['image']) != spec['image_sha256']: raise ValueError('Actual image bytes changed')
    return source


def validate_existing(spec):
    import torch
    path = Path(spec['output']); metadata = path.with_suffix('.json')
    if not path.exists() or not metadata.exists(): return False
    document = json.loads(metadata.read_text())
    # Legacy maps are permitted only for the exact tuned first-frame INPUTS.
    # Equal image bytes alone do not authorize transferring a neighboring fit.
    if document.get('frame', 1660) != spec['frame'] or (spec['frame'] != 1660 and 'frame' not in document):
        raise ValueError('Existing map belongs to another frame or lacks frame provenance')
    if 'camera' in document and document['camera'] != spec['camera']:
        raise ValueError('Existing map belongs to another physical camera')
    if document.get('source_image_sha256') != spec['image_sha256']:
        raise ValueError('Existing map belongs to another image/frame')
    for key, value in RECIPE.items():
        if document.get(key) != value: raise ValueError('Existing map recipe differs: '+key)
    if document.get('value_type') != 'scalar_resolution': raise ValueError('Wrong map representation')
    if document.get('n_features') != 2 or document.get('log2_hashmap_size') != 23:
        raise ValueError('Frequency model capacity differs from the frozen recipe')
    from PIL import Image
    width, height = Image.open(spec['image']).size
    values = torch.load(path, map_location='cpu', weights_only=True)
    if tuple(values.shape) != (height//8, width//8): raise ValueError('Wrong map lattice')
    if not bool(torch.isfinite(values).all()) or not bool(((values >= 16)&(values <= 8192)).all()):
        raise ValueError('Invalid frequency values')
    return True


def execute(spec):
    source = validate_spec(spec)
    sys.path[:0] = [spec['source_code'], str(Path(spec['source_code'])/'LookCloser/scripts')]
    destination = Path(spec['output']); destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix('.fit.lock')
    with lock_path.open('a+') as held:
        try: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return dict(status='busy', frame=spec['frame'], camera=spec['camera'])
        import torch
        if validate_existing(spec):
            return dict(status='complete', reused_same_image=True, output=str(destination), sha256=sha(destination))
        # No Torch CUDA operation precedes the physical-device binding check.
        expected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not expected.startswith('GPU-') or ',' in expected: raise ValueError('One physical GPU UUID required')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
        if expected not in inventory.splitlines(): raise ValueError('Leased physical UUID is absent')
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise ValueError('Expected one visible device mapped to CUDA0')
        actual = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
        if actual is not None and str(actual).lower().replace('gpu-', '') != expected.lower().replace('gpu-', ''):
            raise ValueError('Torch physical UUID differs')
        torch.cuda.set_per_process_memory_fraction(float(spec.get('allocator_GiB', 4))*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(2); random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        import numpy as np
        np.random.seed(42)
        from nerfstudio.scripts.lookcloser_preprocess import LookCloserPreprocessConfig, load_image_as_tensor
        import nerfstudio.scripts.lookcloser_preprocess as fitter
        if Path(fitter.__file__).resolve() != source.resolve(): raise ValueError('Fitter escaped frozen source')
        # The canonical fitter fixes stride to patch_size; stride is metadata,
        # not a constructor argument.
        config = LookCloserPreprocessConfig(**{k: v for k, v in RECIPE.items() if k != 'stride'})
        temporary = destination.with_name(destination.stem+'.%d.partial.pt' % os.getpid())
        started = time.time()
        atomic_json(destination.with_suffix('.progress.json'), dict(status='fitting', frame=spec['frame'],
                    camera=spec['camera'], pid=os.getpid(), started=started, GPU_uuid=expected))
        image = load_image_as_tensor(Path(spec['image']), device=torch.device('cuda'))
        config._process_single_image(img_path=Path(spec['image']), image_tensor=image,
            save_freq_path=temporary, debug_out_dir=None, scene_size=2.)
        metadata = json.loads(temporary.with_suffix('.json').read_text())
        metadata.update(**RECIPE, frame=spec['frame'], camera=spec['camera'],
            source_image_sha256=spec['image_sha256'], preprocess_source_sha256=sha(source),
            fit_script_sha256=sha(__file__), host=socket.gethostname(), GPU_uuid=expected,
            torch_version=torch.__version__, seed=42, seconds=time.time()-started,
            progressive_fit_from_scratch=True, previous_frame_state_loaded=False)
        # Validate the produced tensors before publishing either final pathname.
        values = torch.load(temporary, map_location='cpu', weights_only=True)
        if not bool(torch.isfinite(values).all()) or tuple(values.shape) != (image.shape[0]//8, image.shape[1]//8):
            raise ValueError('Fresh fitter emitted an invalid map')
        if sha(spec['image']) != spec['image_sha256']: raise ValueError('Image changed during fit')
        atomic_json(temporary.with_suffix('.json'), metadata)
        # Paired completion receipt is committed last; incomplete publication is
        # validated/recovered under this same lock, never mistaken for another image.
        temporary.replace(destination); temporary.with_suffix('.json').replace(destination.with_suffix('.json'))
        if not validate_existing(spec): raise ValueError('Published map failed validation')
        receipt = dict(status='complete', frame=spec['frame'], camera=spec['camera'], output=str(destination),
            source_image_sha256=spec['image_sha256'], map_sha256=sha(destination),
            metadata_sha256=sha(destination.with_suffix('.json')), seconds=time.time()-started,
            GPU_uuid=expected, peak_allocated_bytes=torch.cuda.max_memory_allocated(), reused_same_image=False)
        atomic_json(destination.with_suffix('.complete.json'), receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); spec = json.loads(args.spec.read_text())
    if args.check_only:
        validate_spec(spec); print(json.dumps(dict(valid=True, CUDA_started=False, recipe=RECIPE))); return
    result = execute(spec); print(json.dumps(result), flush=True)
    if result['status'] == 'busy': raise SystemExit(75)


if __name__ == '__main__': main()
