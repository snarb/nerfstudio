import tempfile
import hashlib
import unittest
from pathlib import Path

from common import atomic, read
from retiring_host import RetiringHost


class RetirementTests(unittest.TestCase):
    def test_only_backed_completed_jobs_survive_retirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / 'campaigns' / 'scene'
            host = RetiringHost(dict(name='ws07', root='/source', local=True), campaign)
            completed = dict(status='complete', job=dict(completion='/source/frames/000908/frame_complete.json'))
            unbacked = dict(status='complete', job=dict(completion='/source/frames/000913/frame_complete.json'))
            atomic(host.cached_path('/source/queue.json'), dict(jobs=[]))
            atomic(host.cached_path('/source/agent_state.json'), dict(active='frame000918', jobs={
                'frame000908': completed, 'frame000913': unbacked,
                'frame000918': dict(status='running')}))
            atomic(host.cached_path(completed['job']['completion']), dict(checkpoint_sha256='saved'))
            atomic(host.cached_path(unbacked['job']['completion']), dict(checkpoint_sha256='absent'))
            folder = host.backup / 'ws07'
            atomic(folder / 'source_inventory.json', dict(records=[dict(path='/source/frames/000908/model/inference.ckpt', sha256='saved')]))
            atomic(folder / 'checkpoint_verification.json', dict(verified=1, errors=[]))
            host.activate()
            state = host.get('/source/agent_state.json')
            self.assertEqual(set(state['jobs']), {'frame000908'})
            self.assertIsNone(state['active'])
            self.assertTrue(RetiringHost(dict(name='ws07', root='/source', local=True), campaign).offline)

    def test_completed_weights_are_usable_during_optimizer_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / 'campaigns' / 'scene'
            host = RetiringHost(dict(name='ws07', root='/source', local=True), campaign)
            payload = b'checkpoint fixture'
            digest = hashlib.sha256(payload).hexdigest()
            completion = '/source/frames/000908/frame_complete.json'
            atomic(host.cached_path('/source/queue.json'), dict(jobs=[]))
            atomic(host.cached_path('/source/agent_state.json'), dict(jobs={
                'frame000908': dict(status='complete', job=dict(completion=completion))}))
            atomic(host.cached_path(completion), dict(checkpoint_sha256=digest))
            path = '/source/frames/000908/model/inference.ckpt'
            folder = host.backup / 'ws07'
            checkpoint = folder / 'filesystem' / path.lstrip('/')
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(payload)
            atomic(folder / 'source_inventory.json', dict(records=[
                dict(path=path, size=len(payload), sha256=digest)]))
            host.activate()
            self.assertIn('frame000908', host.get('/source/agent_state.json')['jobs'])

    def test_export_created_after_snapshot_is_found_in_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / 'campaigns' / 'scene'
            host = RetiringHost(dict(name='ws08', root='/source', local=True), campaign)
            payload = b'new export'
            digest = hashlib.sha256(payload).hexdigest()
            completion = '/source/frames/001243/frame_complete.json'
            atomic(host.cached_path('/source/queue.json'), dict(jobs=[]))
            atomic(host.cached_path('/source/agent_state.json'), dict(jobs={
                'frame001243': dict(status='complete', job=dict(completion=completion))}))
            atomic(host.cached_path(completion), dict(checkpoint_sha256=digest))
            archive = host.archive / '001243'
            atomic(archive / 'model/manifest.json', dict(sha256=digest))
            atomic(archive / 'support_preserved.json', dict(weights_present=True))
            (archive / 'model/inference.ckpt').write_bytes(payload)
            host.activate()
            self.assertIn('frame001243', host.get('/source/agent_state.json')['jobs'])


if __name__ == '__main__':
    unittest.main()
