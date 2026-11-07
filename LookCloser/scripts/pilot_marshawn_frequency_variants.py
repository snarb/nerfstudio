"""Source-bound HD TRAIN pilot for two frequency factors and a baseline repeat.

Outputs are diagnostic maps outside every training dataset. No NeRF is trained.
Run one exclusively leased GPU; redirect stdout/stderr to a campaign log.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def main(args):
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['inputs_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed pilot input: ' + path)
    if sha(__file__) != plan['worker_sha256'] or plan['pilot_only'] is not True:
        raise ValueError('Wrong worker/scope')
    output = Path(plan['output'])
    if output.exists():
        raise ValueError('Preserve existing generation')
    lease = open('/tmp/lookcloser_tuning_gpu_%d_%d.lock' % (os.getuid(), args.gpu), 'a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(['nvidia-smi', '--id=' + str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    spec = importlib.util.spec_from_file_location('isolated_frequency_pilot', plan['preprocess'])
    prep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prep
    spec.loader.exec_module(prep)
    if sha(prep.__file__) != plan['inputs_sha256'][plan['preprocess']]:
        raise ValueError('Wrong imported preprocessing module')
    output.mkdir(parents=True)
    write(output / 'launch.json', dict(pid=os.getpid(), gpu=args.gpu, plan_sha256=sha(args.plan),
        preprocessing_path=prep.__file__, training_authorized=False, start=time.time()))
    summaries = []
    for camera, source in plan['images'].items():
        if camera in ('DC', 'HC', 'KA'):
            raise ValueError('EVAL cannot enter pilot')
        for label, threshold, final_lr in [('baseline', .95, None), ('repeat', .95, None),
                                           ('lr_decay', .95, .001), ('threshold093', .93, None)]:
            folder = output / (camera + '_' + label)
            folder.mkdir()
            torch.manual_seed(42)
            np.random.seed(42)
            image = prep.load_image_as_tensor(Path(source), device=torch.device('cuda'))
            if list(image.shape) != [1012, 1920, 3]:
                raise ValueError('Expected frozen HD raster')
            counts, initial, samples, steps, rates = {}, [], [], [], []
            censored = np.zeros((126, 240), bool)
            old_ssim, old_adam, old_randint = prep.compute_ssim, torch.optim.Adam, torch.randint
            def tensor_hash(tensors):
                h = hashlib.sha256()
                for tensor in tensors:
                    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                return h.hexdigest()
            class ObservedAdam(old_adam):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    initial.append(tensor_hash([p for group in self.param_groups for p in group['params']]))
                def step(self, *a, **kw):
                    rates.append(self.param_groups[0]['lr'])
                    result = super().step(*a, **kw)
                    steps.append(len(steps) + 1)
                    return result
            def observed_randint(*a, **kw):
                result = old_randint(*a, **kw)
                if len(samples) < 8:
                    samples.append(tensor_hash([result]))
                return result
            def observed_ssim(*a, **kw):
                scores = old_ssim(*a, **kw)
                frame = sys._getframe(1)
                if frame.f_code.co_name != 'eval_and_assign':
                    raise ValueError('Unexpected SSIM observer caller')
                level = int(frame.f_locals['level'])
                row = counts.setdefault(str(level), dict(evaluated=0, passed=0))
                row['evaluated'] += scores.numel()
                row['passed'] += int((scores >= threshold).sum())
                if level == 15:
                    idx = frame.f_locals['idxs'].detach().cpu().numpy()
                    failed = (scores < threshold).detach().cpu().numpy()
                    censored[idx[failed, 0], idx[failed, 1]] = True
                write(output / 'progress.json', dict(camera=camera, variant=label, level=level,
                    time=time.time(), pid=os.getpid(), counts=counts))
                return scores
            torch.optim.Adam, torch.randint, prep.compute_ssim = ObservedAdam, observed_randint, observed_ssim
            start = time.time()
            try:
                cfg = prep.LookCloserPreprocessConfig(max_res=8192, log2_hashmap_size=23,
                    train_steps_per_level=1000, ssim_threshold=threshold, eval_patch_batch_size=1024,
                    lr_final_per_level=final_lr)
                cfg._process_single_image(Path(source), image, folder / 'frequency.pt', None, scene_size=2.)
            finally:
                torch.optim.Adam, torch.randint, prep.compute_ssim = old_adam, old_randint, old_ssim
            remaining = 126 * 240
            for level in sorted(map(int, counts)):
                if counts[str(level)]['evaluated'] != remaining:
                    raise ValueError('Incomplete SSIM coverage')
                remaining -= counts[str(level)]['passed']
            if remaining != int(censored.sum()) or len(initial) != 1:
                raise ValueError('Censor/optimizer mismatch')
            np.save(folder / 'censored.npy', censored)
            row = dict(camera=camera, variant=label, threshold=threshold, final_lr=final_lr,
                initial_parameters=initial[0], first8_random_draws=samples,
                final_rng=tensor_hash([torch.get_rng_state(), torch.cuda.get_rng_state()]),
                optimizer_instances=len(initial), updates=len(steps), lr_first=rates[0], lr_last=rates[-1],
                counts=counts, unresolved=remaining, unresolved_fraction=remaining/(126*240),
                seconds=time.time()-start, map_sha256=sha(folder/'frequency.pt'),
                source_sha256=sha(source), primary_quality_measured=False)
            write(folder / 'complete.json', row)
            summaries.append(row)
            del image
            torch.cuda.empty_cache()
    write(output / 'complete.json', dict(pilot_only=True, training_authorized=False,
        plan_sha256=sha(args.plan), rows=summaries, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitation='First-draw/RNG checks and repeated baseline expose variation; no bitwise map equivalence or 3D quality promotion.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    main(parser.parse_args())
