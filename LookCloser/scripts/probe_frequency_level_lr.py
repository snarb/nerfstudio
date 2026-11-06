"""Tiny synthetic CUDA probe of the actual progressive fitter, never scene maps."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise ValueError("Preserve the existing probe result")
    if sha(args.original) != args.original_sha or sha(args.candidate) != args.candidate_sha:
        raise ValueError("Preprocessing source changed")
    lock = open('/tmp/lookcloser_tuning_gpu_%d_%d.lock' % (os.getuid(), args.gpu), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def gpu_rows():
        return subprocess.check_output(['nvidia-smi', '--id='+str(args.gpu),
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
    if gpu_rows():
        raise RuntimeError("GPU is occupied; no probe started")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(2**30 / torch.cuda.get_device_properties(0).total_memory)
    start = time.monotonic()
    image = (torch.arange(64*64*3, device='cuda').reshape(64, 64, 3) % 251).float() / 250
    original_adam, original_randint = torch.optim.Adam, torch.randint
    results, parameters = {}, {}

    def digest(values):
        h = hashlib.sha256()
        for tensor in values:
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    for label, path, final in [('original', args.original, None), ('original_repeat', args.original, None),
                               ('default', args.candidate, None),
                               ('decay', args.candidate, .001)]:
        spec = importlib.util.spec_from_file_location('frequency_probe_'+label, path)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        rates, counters, samples, instances = [], [], [], []

        class ObservedAdam(original_adam):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                self.initial = digest([p for group in self.param_groups for p in group['params']])
                instances.append(self)

            def step(self, *positional, **keyword):
                rates.append([group['lr'] for group in self.param_groups])
                value = super().step(*positional, **keyword)
                counts = {int(state['step']) for state in self.state.values()}
                if len(counts) != 1:
                    raise ValueError('Optimizer state step mismatch')
                counters.append(counts.pop())
                return value

        def observed_randint(*positional, **keyword):
            value = original_randint(*positional, **keyword)
            samples.append(digest([value]))
            return value

        torch.manual_seed(42)
        torch.optim.Adam, torch.randint = ObservedAdam, observed_randint
        options = dict(image_tensor=image, steps=None, train_steps_per_level=4,
                       batch_size=128, lr=.01, ssim_threshold=1., patch_size=8,
                       eval_patch_batch_size=64, n_levels=3, n_features=2,
                       min_res=16, max_res=64, log2_hashmap_size=10,
                       ssim_window_size=7, debug_levels=[2], debug_patch_count=0)
        if label not in ('original', 'original_repeat'):
            options['lr_final_per_level'] = final
        try:
            model, frequency, levels, bundle = module.train_progressive_and_estimate_frequency_map(**options)
        finally:
            torch.optim.Adam, torch.randint = original_adam, original_randint
        assert len(instances) == 1 and counters == list(range(1, 13)), counters
        parameters[label] = torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])
        assert torch.isfinite(parameters[label]).all() and torch.isfinite(frequency).all()
        results[label] = dict(rates=rates, Adam_steps=counters, optimizer_instances=len(instances),
                              initial_parameters=instances[0].initial, samples=samples,
                              final_rng=digest([torch.get_rng_state(), torch.cuda.get_rng_state()]))
        del model, frequency, levels, bundle, instances
        torch.cuda.empty_cache()

    assert results['original'] == results['default'] == results['original_repeat'], 'Default path state/RNG/LR changed'
    difference = float((parameters['original']-parameters['default']).abs().max())
    self_difference = float((parameters['original']-parameters['original_repeat']).abs().max())
    for key in ['initial_parameters', 'samples', 'final_rng', 'Adam_steps', 'optimizer_instances']:
        assert results['default'][key] == results['decay'][key], key
    expected = [.01 * 10**(-j/3) for j in range(4)] * 3
    assert all(abs(actual[0]-wanted) < 1e-15 for actual, wanted in zip(results['decay']['rates'], expected))
    rows = gpu_rows()
    for row in rows.splitlines():
        pid, mib = map(int, row.split(','))
        assert pid == os.getpid() and mib <= 3072, rows
    assert time.monotonic()-start < 120
    receipt = dict(passed=True, input='synthetic64x64', scene_maps_changed=False,
                   original_sha256=sha(args.original), candidate_sha256=sha(args.candidate),
                   script_sha256=sha(__file__), results=results,
                   default_parameter_max_difference=difference,
                   original_self_repeat_parameter_max_difference=self_difference,
                   bitwise_parameter_parity=difference == 0,
                   numerical_default_equivalence_certified=False,
                   limitation='Synthetic optimizer/RNG contract only. Repeated original measures numerical variation; no scene-map or quality promotion.',
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_seconds=time.monotonic()-start, GPU=args.gpu)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key != 'results'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['original', 'candidate', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['original-sha', 'candidate-sha']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    main(parser.parse_args())
