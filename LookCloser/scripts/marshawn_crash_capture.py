"""Opt-in failure evidence, never a training checkpoint or a recovery policy.

Install after Trainer.setup and before trainer.train. Successful iterations keep
only detached references to their main packet, then release them. A failure is
saved before re-raising the original exception. No RNG draws or NN mutations.
"""
import dataclasses
import json
import os
from pathlib import Path
import random
import traceback
import numpy as np
import torch


def snapshot_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def detached(value, cpu=False):
    if torch.is_tensor(value):
        result = value.detach()
        return result.cpu().clone() if cpu else result
    if isinstance(value, dict):
        return {k: detached(v, cpu) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detached(v, cpu) for v in value)
    if isinstance(value, list):
        return [detached(v, cpu) for v in value]
    if dataclasses.is_dataclass(value):
        return {f.name: detached(getattr(value, f.name), cpu) for f in dataclasses.fields(value)}
    if value is None or isinstance(value, (str, int, float, bool, np.ndarray)):
        return value
    return repr(value)


def tensor_stats(value):
    if torch.is_tensor(value):
        x = value.detach()
        finite = torch.isfinite(x)
        good = x[finite]
        return dict(shape=list(x.shape), dtype=str(x.dtype), count=x.numel(),
                    finite=int(finite.sum()), nan=int(torch.isnan(x).sum()),
                    posinf=int(torch.isposinf(x).sum()), neginf=int(torch.isneginf(x).sum()),
                    finite_min=float(good.min()) if good.numel() else None,
                    finite_max=float(good.max()) if good.numel() else None)
    if isinstance(value, dict):
        return {str(k): tensor_stats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_stats(v) for v in value]
    return None


class TrainingCrashCapture:
    def __init__(self, trainer, directory, *, phase_start, prior_attempts=0):
        self.trainer = trainer
        self.directory = Path(directory)
        self.phase_start, self.prior_attempts = int(phase_start), int(prior_attempts)
        self.attempts = 0
        self.active = False
        self.main_open = False
        self.packet = {}
        self.handles = []
        self.original_iteration = trainer.train_iteration
        self.manager = trainer.pipeline.datamanager
        self.original_next = self.manager.next_train

    def install(self):
        if getattr(self.trainer, '_crash_capture_installed', False):
            raise ValueError('Crash capture is already installed')
        model = self.trainer.pipeline.model
        field = model.field

        def first(name, value):
            if self.active and self.main_open and name not in self.packet:
                self.packet[name] = detached(value)

        def begin(_module, args, kwargs):
            if self.active and 'main_ray_bundle' not in self.packet:
                self.main_open = True
                first('main_ray_bundle', dict(args=args, kwargs=kwargs))

        def end(_module, _args, output):
            first('main_outputs', output)
            self.main_open = False

        self.handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        self.handles.append(model.register_forward_hook(end))
        self.handles.append(field.register_forward_pre_hook(
            lambda _m, args, kwargs: first('main_field_inputs', dict(args=args, kwargs=kwargs)),
            with_kwargs=True))
        self.handles.append(field.register_forward_hook(lambda _m, _a, out: first('main_field_outputs', out)))
        for name in ['mlp_base', 'mlp_geo', 'mlp_head', 'mlp_color']:
            module = getattr(field, name, None)
            if module is not None:
                self.handles.append(module.register_forward_hook(
                    lambda _m, _a, out, key=name: first('main_raw_'+key, out)))
        depth = getattr(model, 'renderer_depth', None)
        if depth is not None:
            self.handles.append(depth.register_forward_pre_hook(
                lambda _m, args, kwargs: first('main_packed_depth_inputs', dict(args=args, kwargs=kwargs)),
                with_kwargs=True))

        def next_train(*args, **kwargs):
            result = self.original_next(*args, **kwargs)
            if self.active and 'actual_train_batch_and_rays' not in self.packet:
                self.packet['actual_train_batch_and_rays'] = detached(result)
            return result

        def iteration(step, *args, **kwargs):
            self.attempts += 1  # Includes this attempt even if forward/backward fails.
            self.active, self.main_open = True, False
            self.packet = {}
            before = snapshot_rng()
            try:
                return self.original_iteration(step, *args, **kwargs)
            except BaseException as error:
                try:
                    self.dump(int(step), error, before)
                except BaseException as capture_error:
                    # Preserve the training exception even if CUDA/OOM blocks capture.
                    self.directory.mkdir(parents=True, exist_ok=True)
                    (self.directory/'capture_failed.txt').write_text(
                        'Capture error: '+repr(capture_error)+'\nOriginal: '+repr(error)+'\n')
                raise
            finally:
                self.active, self.main_open = False, False
                self.packet = {}

        self.manager.next_train = next_train
        self.trainer.train_iteration = iteration
        self.trainer._crash_capture_installed = True
        return self

    def dump(self, step, error, before):
        self.directory.mkdir(parents=True, exist_ok=True)
        base = self.directory/('failure-step-%09d-attempt-%06d' % (step, self.attempts))
        metadata = dict(kind='failure_diagnostic_NOT_VALID_RESUME_CHECKPOINT',
            valid_resume_checkpoint=False, eligible_for_selection=False,
            global_step=step, phase_start=self.phase_start, failed_attempt_counted=True,
            phase_attempts_exact=self.attempts,
            ancestry_attempts_exact_plus_declared_prior=self.prior_attempts+self.attempts,
            expected_contiguous_attempts=step-self.phase_start+1,
            exception=repr(error), traceback=traceback.format_exc(),
            state_warning='May contain partial backward/Adam updates; never resume this file',
            errors={})
        # Metadata survives even if later CUDA state copies fail.
        base.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        payload = dict(metadata=metadata)
        def collect(name, fn):
            try:
                payload[name] = detached(fn(), cpu=True)
            except BaseException as copy_error:
                metadata['errors'][name] = repr(copy_error)
        collect('rng_before_iteration', lambda: before)
        collect('rng_at_exception', snapshot_rng)
        collect('batch_rays_main_field_packet', lambda: self.packet)
        collect('pipeline_state_diagnostic', self.trainer.pipeline.state_dict)
        collect('optimizers', lambda: {k:v.state_dict() for k,v in self.trainer.optimizers.optimizers.items()})
        collect('schedulers', lambda: {k:v.state_dict() for k,v in self.trainer.optimizers.schedulers.items()})
        collect('scaler', self.trainer.grad_scaler.state_dict)
        metadata['main_packet_stats'] = tensor_stats(payload.get('batch_rays_main_field_packet', {}))
        tmp = base.with_suffix('.pt.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, base.with_suffix('.pt'))
        base.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')

    def close(self):
        self.trainer.train_iteration = self.original_iteration
        self.manager.next_train = self.original_next
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.packet = {}
        self.trainer._crash_capture_installed = False
