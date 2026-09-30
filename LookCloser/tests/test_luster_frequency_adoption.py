"""Reconnect to an orphaned worker without stealing an active queue's claim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from run_luster_frequency_queue import validate_adoption


def claim(**overrides):
    return dict(frame='000491',worker='dev3',pid=12345,**overrides)


def test_exited_owner_can_be_adopted(monkeypatch):
    def missing(self):raise FileNotFoundError
    monkeypatch.setattr(Path,'read_text',missing)
    validate_adoption(claim(),'000491','dev3')


def test_live_owner_cannot_be_adopted(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda self:'12345 (python queue) S 1 2')
    with pytest.raises(ValueError,match='live owner'):
        validate_adoption(claim(),'000491','dev3')


def test_zombie_owner_can_be_adopted(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda self:'12345 (python queue) Z 1 2')
    validate_adoption(claim(),'000491','dev3')


@pytest.mark.parametrize('frame,worker,complete',[
    ('000490','dev3',False),('000491','local',False),('000491','dev3',True),
])
def test_completed_or_different_claim_cannot_be_adopted(frame,worker,complete):
    with pytest.raises(ValueError,match='completed or differently owned'):
        validate_adoption(claim(complete=complete),frame,worker)
