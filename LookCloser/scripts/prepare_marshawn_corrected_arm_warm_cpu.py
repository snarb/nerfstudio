"""Derive review-only corrected ARM package from qualified matched warm framework."""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def derive_worker(source):
    changes = [
        ('Matched full-state .1/.3 distortion forks; original scientific loop and plateau criterion.',
         'Matched corrected ARM flag forks; fixed cap1024, original objective and plateau.'),
        ('marshawn_distortion_warm_factory', 'marshawn_corrected_arm_warm_factory'),
        ('Exact two-arm packed-distortion warm specification required', 'Exact two-arm corrected ARM warm specification required'),
        ("effective_distortion=ARMS[s['arm']]", "effective_distortion=.1\n    effective_sampler=dict(corrected_arm_allocator=ARMS[s['arm']],max_steps_per_ray=1024)\n    if child and lineage.get('effective_sampler')!=effective_sampler:raise ValueError('Child sampler factor changed')\n    baseline.update(source_sampler_verified_before_treatment=True,source_sampler=dict(corrected_arm_allocator=False,max_steps_per_ray=1024),requested_sampler=effective_sampler)"),
        ("tracker=PlateauTracker.from_dict(lineage['plateau'])", "m.config.corrected_arm_allocator=effective_sampler['corrected_arm_allocator'];cfg.save_config()\n    tracker=PlateauTracker.from_dict(lineage['plateau'])"),
        ("distortion_fork_runtime", "corrected_arm_fork_runtime"),
        ("dict(effective_distortion=effective_distortion,spec_sha256", "dict(effective_sampler=effective_sampler,effective_distortion=effective_distortion,spec_sha256"),
        ("trainer.step=step;pipe.train();sampler.set_step(step);", "if m.config.corrected_arm_allocator!=effective_sampler['corrected_arm_allocator'] or m.config.max_steps_per_ray!=1024:raise ValueError('Live sampler factor changed')\n        trainer.step=step;pipe.train();sampler.set_step(step);"),
        ("control_checks.append(dict(step=sampler.step,", "control_checks.append(dict(effective_sampler=effective_sampler,packed_samples=int(weights.numel()),packed_max_samples_per_ray=int(torch.bincount(outputs['packed_ray_indices'],minlength=outputs['rgb'].shape[0]).max()),step=sampler.step,"),
        ("source_sha256=SOURCE_SHA,effective_distortion=effective_distortion,visual_review_complete", "source_sha256=SOURCE_SHA,effective_sampler=effective_sampler,effective_distortion=effective_distortion,visual_review_complete"),
        ("trainer.shutdown();return", "verify_source_immutable();trainer.shutdown();return"),
        ("def actual_state():", "def verify_source_immutable():\n        if sha(s['parent_checkpoint'])!=SOURCE_SHA:raise ValueError('Immutable source checkpoint changed during fork')\n    def actual_state():"),
    ]
    for before, after in changes:
        if before not in source:
            raise ValueError('Qualified worker structure changed: ' + before)
        source = source.replace(before, after)
    ast.parse(source)
    return source


def prepare(campaign, output, remote):
    base = campaign / 'distortion03_warm_continuation_candidate_v1'
    original = campaign / 'semantic_quarter_distortion_candidate_20261107'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(base / 'code', output / 'code', copy_function=os.link,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('anchor', 'confidence'):
        shutil.copytree(base / name, output / name, ignore=shutil.ignore_patterns('checkpoint.ckpt'))
    scripts = output / 'code/LookCloser/scripts'
    for name in ('marshawn_corrected_arm_warm_factory.py', 'audit_marshawn_corrected_arm_warm_cpu.py',
                 'verify_marshawn_corrected_arm_warm_phase.py'):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    old_worker = (base / 'code/LookCloser/scripts/train_marshawn_distortion_warm.py').read_text()
    worker = scripts / 'train_marshawn_corrected_arm_warm.py'
    worker.write_text(derive_worker(old_worker))
    (output / 'worker.diff').write_text(''.join(difflib.unified_diff(old_worker.splitlines(True), worker.read_text().splitlines(True), fromfile='qualified_distortion_worker', tofile=worker.name)))
    old_remote = read(base / 'preparation.json')['remote_root']
    audit = json.loads(json.dumps(read(base / 'anchor_audit.json')).replace(old_remote, remote))
    audit['upstream_anchor_audit_sha256'] = sha(base / 'anchor_audit.json')
    write(output / 'anchor_audit.json', audit)
    inventory = {str(p.relative_to(output / 'code')): sha(p) for p in (output / 'code').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    originals = read(original / 'source_manifest.json')['files_sha256']
    assert len(originals) == 1495
    assert all(inventory[k] == digest for k, digest in originals.items())
    write(output / 'source_manifest.json', dict(original1495_source_manifest_sha256=sha(original / 'source_manifest.json'),
          unchanged_original_files=len(originals), changed_original_files=[], files_sha256=inventory,
          added_helpers=sorted(set(inventory) - set(originals))))
    remote_path = Path(remote)
    bound = {str(remote_path / 'code' / path): digest for path, digest in inventory.items()}
    renders = {p.name: sha(p) for p in (output / 'anchor/renders').glob('*.png')}
    stopping = read(base / 'specs/zero_control01/spec.json')['stopping']
    for phase in ('zero', 'technical', 'production'):
        jobs = []
        for arm in ('legacy1024', 'corrected1024'):
            spec = read(base / 'specs' / (phase + '_control01') / 'spec.json')
            spec = json.loads(json.dumps(spec).replace(old_remote, remote))
            spec.update(arm=arm, output=str(remote_path / 'runs' / (phase + '_' + arm)),
                        release=str(remote_path / 'specs' / (phase + '_' + arm) / 'root_release.json'),
                        anchor_audit_sha256=sha(output / 'anchor_audit.json'), frozen_source_hashes=bound,
                        worker_sha256=sha(worker), source_render_sha256=renders)
            target = output / 'specs' / (phase + '_' + arm) / 'spec.json'
            write(target, spec)
            write(target.with_name('root_release.template.json'), dict(reviewer='root', technical_authorized=False,
                  production_authorized=False, spec_sha256=sha(target), confidence_manifest_sha256=spec['confidence_manifest_sha256']))
            for segment in (('first32', 'resume32') if phase == 'technical' else ('',)):
                name = 'correctedARM126316_' + phase + '_' + arm + ('_' + segment if segment else '')
                command = [str(remote_path.parent / 'python'), str(remote_path / 'code/LookCloser/scripts' / worker.name),
                           '--spec', str(remote_path / 'specs' / (phase + '_' + arm) / 'spec.json')]
                if segment == 'first32':
                    command += ['--stop-after-attempts', '32']
                jobs.append(dict(name=name, host='silver', command=command, cwd=str(remote_path / 'code/LookCloser'),
                    env=dict(PYTHONPATH=str(remote_path / 'code') + ':' + str(remote_path / 'code/LookCloser/scripts'),
                        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                        LOOKCLOSER_GPU_LEASE_RECEIPT=str(remote_path / 'remote_queue/jobs' / name / 'receipt.json')),
                    dependencies=['correctedARM126316_' + phase + '_' + arm + '_first32'] if segment == 'resume32' else [],
                    progress_files=[str(remote_path / 'runs' / (phase + '_' + arm) / 'progress.json')],
                    metrics_file=str(remote_path / 'runs' / (phase + '_' + arm) / ('latest.json' if segment == 'first32' else 'complete.json'))))
        plan = read(base / (phase + '_plan.json'))
        plan['hosts']['silver'].update(root=str(remote_path / 'remote_queue'), gpus=[3, 4])
        plan['jobs'] = jobs
        write(output / (phase + '_plan.json'), plan)
    write(output / 'preparation.json', dict(CPU_only=True, READY=False, GPU_started=False, releases_created=False,
          local_root=str(output), remote_root=remote, source_checkpoint_sha256=read(base / 'preparation.json')['source_checkpoint_sha256'],
          source_checkpoint_local=str(campaign / 'frozen_LC_semantic_capture_plan_v2/local_source/checkpoint.ckpt'),
          source_checkpoint_remote=read(base / 'preparation.json')['source_checkpoint_remote'],
          source1495_unchanged=True, science_factors=dict(legacy1024=False, corrected1024=True),
          cap=1024, distortion=.1, FR=.3, stopping=stopping,
          zero_behavior='Both arms use OLD legacy1024 for source parity; treatment starts only after zero branch returns',
          resource_policy=dict(worker_allocator_GiB=10, proposed_process_cap_MiB=32768,
              zero_technical_deadline_seconds=1800, production_deadline_seconds=None, mandatory_health_interval_seconds=3600),
          launcher_status='PLANS_ONLY; external guardian/physical UUID binding/root reviews not prepared or authorized'))
    write(output / 'CURRENT_STATUS.json', dict(READY=False, CPU_preparation='STAGED_AUDIT_PENDING', zero='NOT_RUN',
          technical='NOT_RUN', production='NOT_RELEASED', GPU_started=False, releases_created=False))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    args = parser.parse_args()
    prepare(args.campaign, args.output, args.remote)
