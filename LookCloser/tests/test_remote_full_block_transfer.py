import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1]/'scripts/run_remote_full_block_transfer.py'
spec = importlib.util.spec_from_file_location('remote_full_block_transfer', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_captured_command_tuples_resume_as_json(tmp_path):
    path = tmp_path/'request.json'
    commands = [('stage', ['python', 'worker.py', '--fixed'])]
    module.immutable(path, commands)
    before = path.read_bytes()
    module.immutable(path, commands)
    assert path.read_bytes() == before
    with pytest.raises(AssertionError, match='Immutable'):
        module.immutable(path, [('stage', ['changed'])])
    assert path.read_bytes() == before


def test_request_refuses_nonfinite_or_changed_input(tmp_path):
    path = tmp_path/'request.json'
    module.immutable(path, {'hash':'abc'})
    with pytest.raises(AssertionError): module.immutable(path, {'hash':'def'})
    with pytest.raises(ValueError): module.immutable(tmp_path/'bad.json', {'value':float('nan')})
    assert not (tmp_path/'bad.json').exists()
