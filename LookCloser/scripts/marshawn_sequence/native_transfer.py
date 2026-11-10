"""Verified native transfer, shared by foreground preparation and CPU prefetch."""
import argparse
import fcntl
import shlex
import subprocess
import time
from pathlib import Path
from common import atomic, read, sha


def verify(native, receipt, frame):
    if receipt['frame'] != frame or read(native / 'native_ready.json')['frame'] != frame:
        raise ValueError('Native transfer frame differs')
    if sha(native / 'native_ready.json') != receipt['native_ready_sha256'] or sha(native / 'points3D.txt') != receipt['seed_sha256']:
        raise ValueError('Transferred native manifest/seed checksum differs')


def ensure_native(root, frame, nonblocking=False):
    from campaign import Host
    root = Path(root)
    if not 908 <= frame <= 1268:
        raise ValueError('Frame outside accepted sequence')
    settings = read(root / 'settings.json')
    local = root / 'frames' / f'{frame:06d}'
    local.mkdir(parents=True, exist_ok=True)
    native = local / 'native'
    with (local / 'native_transfer.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError:
            return 'already_running'
        # An optional prefetch must never rebuild a started or archived model.
        if nonblocking and any(p.exists() for p in (local / 'train/spec.json', local / 'frame_complete.json', local / 'archive_receipt.json')):
            return 'already_started'
        owners = settings['source_owners']
        owner = Host(owners[0 if frame <= 1111 else 1])
        committed = native / 'transfer_complete.json'
        released = native / 'source_cache_released.json'
        if committed.exists():
            transfer_receipt = read(committed)
            verify(native, transfer_receipt, frame)
            if released.exists():
                return 'ready'
        source = owner.get(owner.root / 'settings.json')
        if not source:
            if committed.exists():
                atomic(native / 'source_cache_release_pending.json', dict(frame=frame, error='Source unavailable; verified native data retained'))
                return 'ready'
            raise ConnectionError('Native source settings unavailable')
        tools = Path(source.get('production_tools', source['tools']))
        if not committed.exists():
            receipt = owner.root / 'prepared' / f'{frame:06d}' / 'producer_complete.json'
            if not owner.get(receipt):
                args = [source['python'], str(tools / 'producer.py'), '--root', str(owner.root), '--frame', str(frame)]
                if read(root / 'selected_recipe.json')['parameters'].get('format') == 'exr':
                    args.append('--linear')
                log = owner.root / ('producer_%06d.log' % frame)
                owner.command('nohup env PYTHONPATH=' + shlex.quote(source['dependency_path']) + ' ' + shlex.join(args) + ' > ' + shlex.quote(str(log)) + ' 2>&1 < /dev/null &')
            deadline = time.time() + 1800
            transfer_receipt = owner.get(receipt)
            while not transfer_receipt:
                if time.time() > deadline:
                    raise TimeoutError('CPU source producer did not finalize')
                time.sleep(15)
                transfer_receipt = owner.get(receipt)
            if transfer_receipt['frame'] != frame:
                raise ValueError('Source receipt frame differs')
            native.mkdir(exist_ok=True)
            # The extractor retains this lease if its parent crashes.
            # A retry waits for orphan extraction before touching native files.
            inherited = (lease.fileno(),)
            process = subprocess.Popen(owner.argv('tar -C ' + shlex.quote(str(receipt.parent)) + ' -cf - .'), stdout=subprocess.PIPE, pass_fds=inherited)
            try:
                receiver = subprocess.run(['tar', '-C', str(native), '-xf', '-'], stdin=process.stdout, pass_fds=inherited)
            finally:
                process.stdout.close()
            code = process.wait()
            if receiver.returncode or code:
                raise RuntimeError('Native transfer failed; source receipt preserved')
            verify(native, transfer_receipt, frame)
            atomic(committed, transfer_receipt)
        # Only the source owner's derivative cache is released, after commit.
        if not released.exists():
            args = [source['python'], str(tools / 'producer.py'), '--root', str(owner.root), '--frame', str(frame), '--release', '--expected-ready-sha', transfer_receipt['native_ready_sha256']]
            try:
                owner.command(shlex.join(args), timeout=120)
                atomic(released, dict(frame=frame, native_ready_sha256=transfer_receipt['native_ready_sha256']))
                (native / 'source_cache_release_pending.json').unlink(missing_ok=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                atomic(native / 'source_cache_release_pending.json', dict(frame=frame, error=str(error)))
        return 'ready'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--frame', required=True, type=int)
    parser.add_argument('--nonblocking', action='store_true')
    args = parser.parse_args()
    print(ensure_native(args.root, args.frame, args.nonblocking), flush=True)
