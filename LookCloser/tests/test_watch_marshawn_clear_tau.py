import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/watch_marshawn_clear_tau.py';s=importlib.util.spec_from_file_location('tau_watcher_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_partial_gate_is_not_archived_or_marked_complete(tmp_path):
 root=tmp_path;renders=root/'runs/diagnostic_control/renders/step-000111816';renders.mkdir(parents=True);(renders/'metrics.json').write_text('{}');result=m.archive(root);assert result['archives']==[]and not result['all_complete'];assert all(v['complete']is None for v in result['arms'].values())
def test_completed_json_does_not_automatically_approve_geometry(tmp_path):
 for arm in m.ARMS:
  run=tmp_path/'runs'/('diagnostic_'+arm);run.mkdir(parents=True);(run/'complete.json').write_text('{}')
 with pytest.raises(ValueError):m.archive(tmp_path)
