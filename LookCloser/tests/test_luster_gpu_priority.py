"""Process completion races must not kill the campaign GPU coordinator."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
import prioritize_luster_training as priority


@pytest.mark.parametrize('error',[priority.psutil.NoSuchProcess(123),priority.psutil.ZombieProcess(123)])
def test_finished_frequency_parent_is_ignored(monkeypatch,error):
    def gone():raise error
    monkeypatch.setattr(priority.psutil,'Process',lambda pid:SimpleNamespace(cmdline=gone))
    assert priority.owned_descendants(123,Path('/campaign'))=={}


@pytest.mark.parametrize('command',[
    ['python','another_job.py','/campaign'],
    ['python','prepare_luster_frequencies.py','/campaign_other/data'],
])
def test_reused_pid_with_unrelated_command_is_ignored(monkeypatch,command):
    monkeypatch.setattr(priority.psutil,'Process',lambda pid:SimpleNamespace(cmdline=lambda:command))
    assert priority.owned_descendants(123,Path('/campaign'))=={}


def test_only_owned_frequency_children_are_returned(monkeypatch):
    child=SimpleNamespace(pid=456)
    monkeypatch.setattr(priority.psutil,'Process',lambda pid:SimpleNamespace(
        cmdline=lambda:['python','/repo/prepare_luster_frequencies.py','/campaign/frames/000480/data'],
        children=lambda recursive:[child]))
    assert priority.owned_descendants(123,Path('/campaign'))=={456:child}
