"""Real tar/lease/crash canaries using temporary data, never campaign inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from common import atomic, read, sha
from native_transfer import ensure_native


class LocalHost:
    def __init__(self, row):
        self.root = Path(row['root'])

    def get(self, path, default=None):
        try:
            return read(path)
        except (OSError, ValueError):
            return default

    def argv(self, command):
        settings = read(self.root / 'settings.json')
        if settings.get('delay'):
            code = ('import os,time;from pathlib import Path;'
                    f"Path({settings['marker']!r}).write_text(str(os.getpid()));"
                    f"time.sleep({settings['delay']!r});"
                    f"os.execl('/bin/bash','bash','-c',{command!r})")
            return [sys.executable, '-c', code]
        return ['bash', '-c', command]

    def command(self, command, timeout=60):
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout


def fixture(directory, delay=0):
    root = Path(directory) / 'receiver'
    owner = Path(directory) / 'owner'
    root.mkdir()
    owner.mkdir()
    atomic(root / 'settings.json', dict(source_owners=[dict(root=str(owner))] * 2))
    atomic(root / 'selected_recipe.json', dict(parameters=dict(format='png')))
    atomic(owner / 'settings.json', dict(python=sys.executable, tools=str(Path(__file__).parent), dependency_path='', delay=delay, marker=str(Path(directory) / 'stream_started')))
    prepared = owner / 'prepared/000908'
    (prepared / 'images').mkdir(parents=True)
    (prepared / 'images/AA.png').write_bytes(b'unchanged native RGB bytes\x00\xff')
    (prepared / 'points3D.txt').write_bytes(b'original seed bytes\n')
    atomic(prepared / 'native_ready.json', dict(frame=908, native_no_resize=True))
    atomic(prepared / 'producer_complete.json', dict(frame=908, native_ready_sha256=sha(prepared / 'native_ready.json'), seed_sha256=sha(prepared / 'points3D.txt')))
    return root, owner, prepared


class NativeTransferTests(unittest.TestCase):
    def test_byte_identical_transfer_release_and_corruption_detection(self):
        with tempfile.TemporaryDirectory(prefix='marshawn_transfer_test_') as directory:
            root, owner, prepared = fixture(directory)
            expected = (prepared / 'images/AA.png').read_bytes()
            with mock.patch('campaign.Host', LocalHost):
                self.assertEqual(ensure_native(root, 908, True), 'ready')
                native = root / 'frames/000908/native'
                self.assertEqual((native / 'images/AA.png').read_bytes(), expected)
                self.assertFalse(prepared.exists())
                self.assertTrue((owner / 'prepared_released/000908.json').exists())
                self.assertFalse((root / 'frames/000908/train').exists())
                self.assertFalse((root / 'frames/000908/frame_complete.json').exists())
                self.assertEqual(ensure_native(root, 908), 'ready')
                # A complete receiving cache must not require the source online.
                (owner / 'settings.json').unlink()
                self.assertEqual(ensure_native(root, 908), 'ready')
                atomic(native / 'native_ready.json', dict(frame=909))
                with self.assertRaisesRegex(ValueError, 'frame differs'):
                    ensure_native(root, 908)

    def test_orphan_tar_children_keep_lease_after_parent_crash(self):
        with tempfile.TemporaryDirectory(prefix='marshawn_transfer_crash_') as directory:
            root, owner, prepared = fixture(directory, delay=2)
            marker = Path(directory) / 'stream_started'
            code = ('from unittest import mock;from test_native_transfer import LocalHost;'
                    'from native_transfer import ensure_native;'
                    f"patch=mock.patch('campaign.Host',LocalHost);patch.start();ensure_native({str(root)!r},908)")
            parent = subprocess.Popen([sys.executable, '-c', code], env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.time() + 15
                while not marker.exists() and time.time() < deadline:
                    time.sleep(.02)
                self.assertTrue(marker.exists())
                source_pid = int(marker.read_text())
                receiver_holds_lease = False
                deadline = time.time() + 1
                while time.time() < deadline and not receiver_holds_lease:
                    children = Path(f'/proc/{parent.pid}/task/{parent.pid}/children').read_text().split()
                    for child in children:
                        if int(child) == source_pid:
                            continue
                        for fd in Path(f'/proc/{child}/fd').glob('*'):
                            try:
                                receiver_holds_lease |= os.readlink(fd).endswith('/native_transfer.lock')
                            except OSError:
                                pass
                    if not receiver_holds_lease:
                        time.sleep(.01)
                self.assertTrue(receiver_holds_lease, 'Actual tar extractor must inherit the write lease')
                parent.kill()
                parent.wait(timeout=5)
                with mock.patch('campaign.Host', LocalHost):
                    self.assertEqual(ensure_native(root, 908, True), 'already_running')
                    start = time.monotonic()
                    self.assertEqual(ensure_native(root, 908), 'ready')
                    self.assertGreater(time.monotonic() - start, .5)
                native = root / 'frames/000908/native'
                self.assertEqual((native / 'images/AA.png').read_bytes(), b'unchanged native RGB bytes\x00\xff')
                self.assertTrue((native / 'transfer_complete.json').exists())
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)

    def test_prefetch_never_rebuilds_started_model(self):
        with tempfile.TemporaryDirectory(prefix='marshawn_transfer_started_') as directory:
            root, owner, prepared = fixture(directory)
            atomic(root / 'frames/000908/train/spec.json', dict(immutable=True))
            with mock.patch('campaign.Host', LocalHost):
                self.assertEqual(ensure_native(root, 908, True), 'already_started')
            self.assertTrue(prepared.exists())
            self.assertFalse((root / 'frames/000908/native').exists())

    def test_receiver_receipt_prevents_source_cache_regeneration(self):
        from preparation_prefetch import prefetch
        with tempfile.TemporaryDirectory(prefix='marshawn_prefetch_receiver_') as directory:
            root, owner, prepared = fixture(directory)
            atomic(root / 'frames/000909/native/transfer_complete.json', dict(frame=909))
            atomic(root / 'queue.json', dict(jobs=[dict(id='frame000908'), dict(id='frame000909')]))
            atomic(root / 'agent_state.json', dict(active='frame000908', jobs={'frame000908': {}}))
            host = LocalHost(dict(root=str(root)))
            host.agent_root = root
            host.row = dict(name='receiver')
            with mock.patch('campaign.Host', side_effect=AssertionError('Ready receiver must not contact producer')):
                prefetch(root, [host])
            self.assertEqual(read(root / 'prefetch_status.json')['checks'][0]['status'], 'native_ready')


if __name__ == '__main__':
    unittest.main()
