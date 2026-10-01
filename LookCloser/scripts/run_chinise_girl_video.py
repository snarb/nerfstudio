"""Fresh-machine Luster workflow. Exit 2 means inspect the recorded visual gate."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from prepare_luster_video import CONFIG, REPO, SCRIPTS, environment, write
from archive_luster_checkpoint import sha
from run_luster_video_campaign import numeric_pass


def read(path):
    return json.loads(Path(path).read_text())


def frame_range(start, end):
    if not 0 <= start <= end <= 999999:
        raise ValueError('Expected an inclusive, increasing six-digit frame range')
    return [f'{number:06d}' for number in range(start, end + 1)]


@contextmanager
def exclusive(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'fresh_workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another workflow owns this campaign') from exc
        yield


def command(script, *args):
    return [sys.executable, str(SCRIPTS / script), *map(str, args)]


def run_logged(root, label, argv, progress=None, allowed=(0,)):
    """Check our controller, descendants, progress and GPU/OOM evidence every30s."""
    import psutil
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    log_path = logs / f'{label}_{time.time_ns()}.log'
    with log_path.open('w') as log, (root / 'supervision.jsonl').open('a') as journal:
        worker = subprocess.Popen(argv, cwd=REPO, env=environment(), stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                code = worker.poll()
                try:
                    descendants = [p.pid for p in psutil.Process(worker.pid).children(recursive=True)] if code is None else []
                except psutil.NoSuchProcess:
                    descendants = []
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                    gpu_info = dict(exit=gpu.returncode, text=(gpu.stdout + gpu.stderr).strip())
                except (OSError, subprocess.TimeoutExpired) as exc:
                    gpu_info = dict(error=str(exc))
                with log_path.open('rb') as stream:
                    stream.seek(max(0, log_path.stat().st_size - 16000))
                    tail = stream.read().decode(errors='replace')
                try:
                    state = read(progress) if progress and Path(progress).exists() else None
                except json.JSONDecodeError:
                    state = {'partial_write': True}
                record = dict(time=time.time(), controller_pid=os.getpid(), worker_pid=worker.pid,
                              descendants=descendants, worker_exit=code, label=label, log=str(log_path),
                              progress=state, gpu=gpu_info, oom='out of memory' in tail.lower(),
                              free_GiB=psutil.disk_usage(root).free / 2**30)
                journal.write(json.dumps(record) + '\n'); journal.flush()
                write(root / 'live_check.json', record)
                if code is not None:
                    if code not in allowed:
                        raise RuntimeError(f'{label} exited {code}; inspect {log_path}')
                    return code
                time.sleep(30)
        except BaseException:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL); worker.wait()
            raise


def initialize(root, args):
    requested = dict(schema=1, frames=frame_range(args.start, args.end), fps=30,
                     host=args.host, archive_root=args.archive_root,
                     recipe_sha256=sha(CONFIG / 'recipe.json'),
                     normalization_sha256=sha(CONFIG / 'normalization.json'),
                     polygon_sha256=sha(CONFIG / 'cam011_000475_gap.json'))
    destination = root / 'fresh_workflow.json'
    if destination.exists():
        existing = read(destination)
        if any(existing.get(k) != v for k, v in requested.items()):
            raise ValueError('Campaign range, source/recipe identity or archive destination changed')
        return existing
    if (root / 'manifest.json').exists():
        raise ValueError('Use a new root; historical campaigns require an explicit migration')
    requested['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    requested['created_at'] = time.time()
    write(destination, requested)
    return requested


def preflight(root, config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment; check GPU device access before training')
    import tinycudann as tcnn
    # Exercise the extension, rather than trusting import success alone.
    encoder = tcnn.Encoding(3, {'otype': 'SphericalHarmonics', 'degree': 4})
    encoded = encoder(torch.full((1, 3), .5, device='cuda'))
    if not torch.isfinite(encoded).all():
        raise RuntimeError('TCNN CUDA preflight produced nonfinite values')
    check = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); assert Path('/fsx/tmp/luster/root_8s/bounds_8s.json').is_file()"
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', config['host'],
                    'python3 -c ' + shlex.quote(check) + ' ' + shlex.quote(config['archive_root'])], check=True)
    subprocess.run(['rsync', '--version'], stdout=subprocess.DEVNULL, check=True)
    probe_env = os.environ.copy()
    if probe_env.get('LUSTER_FFPROBE_LIBRARY_PATH'):
        probe_env['LD_LIBRARY_PATH'] = probe_env['LUSTER_FFPROBE_LIBRARY_PATH']
        probe_env.pop('LD_PRELOAD', None)
    subprocess.run(['ffprobe', '-version'], env=probe_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    write(root / 'runtime.json', dict(python=sys.executable, torch=torch.__version__,
                                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                    git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()))


def prepare(root, config):
    frames = config['frames']
    run_logged(root, 'prepare', command('prepare_luster_video.py', root, '--start', int(frames[0]),
               '--end', int(frames[-1]), '--host', config['host'], '--workers', 1), root / 'progress.json')
    # Use only the reviewed frame/camera/source identity; never extend this polygon to other frames.
    if '000475' in frames:
        data = root / 'frames/000475/data'
        if not (data / 'revision.json').exists():
            run_logged(root, 'repair475', command('repair_luster_background_polygon.py', root,
                       CONFIG / 'cam011_000475_gap.json'))
        elif read(data / 'revision.json')['id'] != 'cam011_gap_v1':
            raise ValueError('Unexpected frame475 data revision')
    run_logged(root, 'rois', command('prepare_luster_video_rois.py', root))
    # Frequency fitting is just-in-time on the same local GPU; no remote code install
    # or competing field/frequency worker is needed on a fresh machine.


def ensure_frequencies(root, frame, workers):
    base = root / 'frames' / frame; data = base / 'data'
    if not (data / 'frequency_complete.json').exists():
        run_logged(root, f'frequency_{frame}', command('prepare_luster_frequencies.py', data,
                   '--workers', workers), data / 'frequency_progress.json')
    run_logged(root, f'audit_{frame}', command('audit_luster_data.py', base, '--require-frequencies'))


def checkpoint_digest(root, checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        return sha(checkpoint)
    if checkpoint.with_suffix('.archive.json').exists():
        return read(checkpoint.with_suffix('.archive.json'))['sha256']
    journal = root / 'checkpoint_pruning.jsonl'
    if journal.exists():
        for line in reversed(journal.read_text().splitlines()):
            record = json.loads(line)
            if record['path'] == str(checkpoint):
                return record['sha256']
    raise FileNotFoundError(checkpoint)


def gate_identity(root, frame, run, kind):
    selected = read(run / 'selection.json')
    if kind == 'final':
        snapshot = read(root / 'snapshots' / f'{frame}.json')
        final = read(snapshot['selection'])
        if snapshot['run'] != str(run) or final['checkpoint'] != selected['checkpoint'] or final['step'] != selected['step']:
            raise ValueError('Final snapshot does not match the current training selection')
        checkpoint = Path(selected['checkpoint'])
        digest = sha(checkpoint) if checkpoint.exists() else read(checkpoint.with_suffix('.archive.json'))['sha256']
        if digest != snapshot['archived_checkpoint']['sha256']:
            raise ValueError('Final snapshot checkpoint identity changed')
        return dict(frame=frame, kind=kind, run=str(run), step=selected['step'],
                    checkpoint_sha256=snapshot['archived_checkpoint']['sha256'],
                    selection=snapshot['selection'], render_receipt=snapshot['render_receipt'])
    return dict(frame=frame, kind=kind, run=str(run), step=selected['step'],
                checkpoint_sha256=checkpoint_digest(root, selected['checkpoint']), selection=str(run / 'selection.json'),
                selection_sha256=sha(run / 'selection.json'))


def review_path(root, gate):
    if gate['kind'] == 'final':
        return root / 'visual_reviews' / f"{gate['frame']}.json"
    return root / 'stage_reviews' / f"{gate['frame']}_{Path(gate['run']).name}.json"


def require_review(root, gate):
    path = review_path(root, gate)
    if path.exists():
        review = read(path)
        if review.get('checkpoint_sha256') != gate['checkpoint_sha256'] or review.get('run') != gate['run']:
            raise ValueError(f'Stale visual review: {path}')
        if review.get('decision') in ('continue', 'export', 'accept'):
            return review['decision']
    write(root / 'fresh_status.json', dict(phase='visual_review_required', **gate, time=time.time()))
    print(f"Review required: {gate['frame']} {gate['kind']} step{gate['step']}; see {root / 'fresh_status.json'}", flush=True)
    raise SystemExit(2)


def seed(root, frame):
    """Replay the measured cold schedule with explicit native-review decisions."""
    previous = None
    for endpoint in (8000, 16000, 24000, 28000, 32000):
        run = root / 'frames' / frame / 'runs' / f's{endpoint:06d}'
        if not (run / 'complete.json').exists():
            argv = command('launch_luster_video_stage.py', root, frame, endpoint,
                           '--reason', 'Fresh cold seed; native review before each longer stage')
            if previous:
                argv += ['--resume-run', str(previous)]
            argv += ['--fr', '1' if endpoint <= 16000 else '.3']
            if endpoint >= 28000:
                argv += ['--lr-base', '.002']
            run_logged(root, f'seed_{endpoint}', argv, run / 'progress.json')
        previous = run
        if endpoint == 32000 or require_review(root, gate_identity(root, frame, run, 'seed_stage')) == 'export':
            return run
    raise AssertionError('Unreachable seed schedule')


def finish(root, frame, run, config):
    receipt = root / 'frames' / frame / 'finish_complete.json'
    if not receipt.exists() or read(receipt)['run'] != str(run):
        run_logged(root, f'finish_{frame}', command('finish_luster_video_frame.py', root, frame, run,
                   '--remote-root', config['archive_root']))
    return gate_identity(root, frame, run, 'final')


def sync_metadata(root, config):
    run_logged(root, 'archive_metadata', ['rsync', '-a', '--exclude=frames/', '--exclude=logs/',
               '--exclude=fresh_workflow.lock', str(root) + '/',
               f"{config['host']}:{config['archive_root']}/artifacts/campaign/"])


def run(root, args):
    config = initialize(root, args)
    os.environ['LUSTER_HOST'] = config['host']
    preflight(root, config)
    prepare(root, config)
    for index, frame in enumerate(config['frames']):
        snapshot_path = root / 'snapshots' / f'{frame}.json'
        if snapshot_path.exists():
            selected_run = Path(read(snapshot_path)['run'])
        else:
            ensure_frequencies(root, frame, args.frequency_workers)
            if index == 0:
                selected_run = seed(root, frame)
            else:
                initial = root / 'frames' / frame / 'runs/s006000'
                if not (initial / 'complete.json').exists():
                    parent = read(root / 'snapshots' / f'{int(frame)-1:06d}.json')
                    run_logged(root, f'initial_{frame}', command('launch_luster_video_stage.py', root,
                               frame, 6000, '--warm-parent', parent['run'], '--eval-steps', 4096, 6000,
                               '--reason', 'Inspect train detail after the first adjacent-frame stage'), initial/'progress.json')
                decision = require_review(root, gate_identity(root, frame, initial, 'warm_stage'))
                if decision == 'export':
                    selected_run = initial
                    gate = finish(root, frame, selected_run, config)
                    require_review(root, gate)
                    sync_metadata(root, config)
                    continue
                run_logged(root, f'train_{frame}', command('run_luster_video_campaign.py', root,
                           '--start', int(frame), '--end', int(frame), '--max-step', 24000,
                           '--remote-root', config['archive_root']), root / 'campaign_status.json', allowed=(0, 2))
                if snapshot_path.exists():
                    selected_run = Path(read(snapshot_path)['run'])
                else:
                    state = read(root / 'campaign_status.json')
                    if state.get('frame') != frame or not state.get('run'):
                        raise RuntimeError('Controller stopped without a reviewable candidate')
                    selected_run = Path(state['run'])
            gate = finish(root, frame, selected_run, config)
            run_logged(root, f'review_sheets_{frame}', command('review_luster_video_batch.py', root,
                       '--start', max(int(config['frames'][0]), int(frame)-5), '--end', int(frame)))
        gate = finish(root, frame, selected_run, config)
        require_review(root, gate)
        sync_metadata(root, config)
    run_logged(root, 'assemble', command('assemble_luster_video.py', root))
    write(root / 'fresh_status.json', dict(phase='final_video_review_required',
          manifest=str(root / 'final_video/manifest.json'), frames=config['frames'], time=time.time()))
    sync_metadata(root, config)
    print(f"Encoded {len(config['frames']) / 30:g}s; inspect both videos before declaring completion.")


def record_review(root, args):
    state = read(root / 'fresh_status.json')
    if state.get('phase') != 'visual_review_required':
        raise ValueError('No pending frame/stage review')
    frame = state['frame']; run_path = Path(state['run'])
    current = gate_identity(root, frame, run_path, state['kind'])
    if current != {key: state[key] for key in current}:
        raise ValueError('Candidate changed after the gate was recorded')
    allowed = ('accept', 'reject') if state['kind'] == 'final' else ('continue', 'export', 'reject')
    if args.decision not in allowed:
        raise ValueError(f"This gate permits {allowed}")
    if not args.reason.strip():
        raise ValueError('Record the observed train/eval detail and artifact evidence')
    evidence = []
    for name in args.evidence:
        path = Path(name).resolve()
        if not path.is_file():
            raise ValueError(f'Missing reviewed evidence: {path}')
        evidence.append(dict(path=str(path), sha256=sha(path)))
    review = dict(**current, decision=args.decision, accepted=args.decision == 'accept',
                  reason=args.reason, evidence=evidence, reviewed_at=time.time())
    if args.decision == 'accept' and not numeric_pass(read(current['selection'])):
        if not args.allow_numeric_exception:
            raise ValueError('Numeric screen failed; inspect the cause and explicitly justify --allow-numeric-exception')
        review['numeric_gate_override'] = dict(reason=args.reason, checkpoint_sha256=current['checkpoint_sha256'])
    write(review_path(root, current), review)
    print(f"Recorded {args.decision} for {frame}; rerun the same run command to continue.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('run', 'plan'):
        p = commands.add_parser(name); p.add_argument('root', type=Path)
        p.add_argument('--start', type=int, default=470); p.add_argument('--end', type=int, default=529)
        p.add_argument('--host', default='ubuntu@dev3'); p.add_argument('--archive-root', required=True)
        p.add_argument('--frequency-workers', type=int, default=4)
    p = commands.add_parser('review'); p.add_argument('root', type=Path)
    p.add_argument('--decision', choices=['continue', 'export', 'accept', 'reject'], required=True)
    p.add_argument('--reason', required=True); p.add_argument('--evidence', nargs='+', required=True)
    p.add_argument('--allow-numeric-exception', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'plan':
        frames = frame_range(args.start, args.end)
        print(json.dumps(dict(root=str(root), frames=frames, seconds=len(frames)/30,
              source_host=args.host, archive=args.archive_root, recipe=read(CONFIG/'recipe.json'),
              stages=['download and audit', 'sequence bounds and ROIs', 'per-frame frequency fitting',
                      'cold seed with visual gates', 'adjacent model training with final visual gates',
                      'verified body/detail video'], gpu_training_validated_on_this_machine=False), indent=2))
        return
    with exclusive(root):
        if args.action == 'review':
            record_review(root, args)
        else:
            if args.frequency_workers < 1:
                raise ValueError('At least one frequency worker is required')
            run(root, args)


if __name__ == '__main__':
    main()
