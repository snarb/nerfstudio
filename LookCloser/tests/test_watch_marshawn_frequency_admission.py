import copy
import importlib.util
from pathlib import Path
import pytest


def module():
    p = Path(__file__).resolve().parents[1]/'scripts/watch_marshawn_frequency_admission.py'
    s = importlib.util.spec_from_file_location('admission', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def healthy():
    return {'status': 'running', 'worker': {'pid': 123}, 'supervisor': {'pid': 122}, 'worker_alive': True, 'supervisor_alive': True, 'spec_sha256': 'bound', 'started': 800, 'heartbeat': 990}


def test_fresh_running_and_completed_qualify():
    m = module(); j = healthy(); assert m.qualified_job(j, 1000)
    j.update(status='complete', finished=999, worker_alive=False, supervisor_alive=False)
    assert m.qualified_job(j, 20000)


@pytest.mark.parametrize('status', ['pending', 'dispatching', 'failed', 'unknown', 'orphaned', 'skipped'])
def test_non_evidenced_dispatch_never_qualifies(status):
    j = healthy(); j['status'] = status
    assert not module().qualified_job(j, 1000)


@pytest.mark.parametrize('change', [{'heartbeat': 700}, {'heartbeat': 1001}, {'worker_alive': False}, {'supervisor_alive': False}, {'worker': {}}, {'spec_sha256': None}])
def test_stale_or_malformed_health_blocks(change):
    j = healthy(); j.update(change)
    assert not module().qualified_job(j, 1000)
