import tempfile
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


if __name__ == '__main__':
    unittest.main()
