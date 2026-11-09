"""Budget and terminal-state checks for the standard-library fleet controller."""
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('fleet',Path(__file__).parents[1]/'scripts/gpu_experiment_fleet.py')
fleet=importlib.util.module_from_spec(spec);spec.loader.exec_module(fleet)

class FleetTests(unittest.TestCase):
 def test_budget_rejects_before_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);plan=p/'plan.json';plan.write_text(json.dumps(dict(max_gpu_hours=.1,jobs=[dict(name='a',gpu=0,max_seconds=600,command=['python','x.py'])])))
   a=SimpleNamespace(plan=plan,root=p/'run',python=Path('/usr/bin/python3'),supervisor=Path(__file__),dry_run=False,append=False)
   with patch.object(fleet,'gpu_snapshot',return_value=[dict(index=0,uuid='GPU-a',processes=[])]),self.assertRaisesRegex(ValueError,'aggregate'):
    fleet.launch(a)
   self.assertFalse(a.root.exists())
 def test_busy_gpu_rejects_before_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);plan=p/'plan.json';plan.write_text(json.dumps(dict(max_gpu_hours=1,jobs=[dict(name='a',gpu=0,max_seconds=60,command=['python','x.py'])])))
   a=SimpleNamespace(plan=plan,root=p/'run',append=False)
   with patch.object(fleet,'gpu_snapshot',return_value=[dict(index=0,uuid='GPU-a',processes=[dict(pid=999)])]),self.assertRaisesRegex(RuntimeError,'already has compute'):
    fleet.launch(a)
   self.assertFalse(a.root.exists())
 def test_terminal_receipt_and_setup_failure(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);c=p/'ok_control';c.mkdir();(c/'result.json').write_text(json.dumps(dict(returncode=0,elapsed_seconds=90,oom_evidence=False)))
   jobs=[dict(name='ok',gpu=0,control_root=str(c),launched_epoch=time.time()-100),dict(name='bad',gpu=1,control_root=str(p/'bad_control'),controller_pid=999,launched_epoch=time.time()-100)]
   (p/'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
   with patch.object(fleet,'gpu_snapshot',return_value=[]),patch.object(fleet,'live_controller',return_value=False):r=fleet.status(p)
   self.assertEqual([j['state'] for j in r['jobs']],['complete','setup-failed']);self.assertAlmostEqual(r['gpu_hours'],.025)

 def test_append_reserves_gpu_before_worker_allocates_cuda(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);root=p/'run';root.mkdir();plan=p/'plan.json';plan.write_text(json.dumps(dict(max_gpu_hours=1,jobs=[dict(name='next',gpu=0,max_seconds=60,command=['python','x.py'])])))
   fleet.atomic(root/'manifest.json',dict(max_gpu_hours=1,supervisor='supervisor.py',wall_deadline_epoch=time.time()+600,jobs=[dict(name='old',gpu=0,controller_pid=123,max_seconds=120,control_root=str(root/'old_control'))]))
   a=SimpleNamespace(plan=plan,root=root,append=True)
   with patch.object(fleet,'gpu_snapshot',return_value=[dict(index=0,uuid='GPU-a',processes=[])]),patch.object(fleet,'live_controller',return_value=True),self.assertRaisesRegex(ValueError,'One job'):
    fleet.launch(a)
 def test_shared_wall_deadline_is_enforced(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);root=p/'run';root.mkdir();plan=p/'plan.json';plan.write_text(json.dumps(dict(max_gpu_hours=1,jobs=[dict(name='next',gpu=0,max_seconds=60,command=['python','x.py'])])))
   fleet.atomic(root/'manifest.json',dict(max_gpu_hours=1,supervisor='supervisor.py',wall_deadline_epoch=time.time()+10,jobs=[]))
   a=SimpleNamespace(plan=plan,root=root,append=True)
   with patch.object(fleet,'gpu_snapshot',return_value=[dict(index=0,uuid='GPU-a',processes=[])]),self.assertRaisesRegex(ValueError,'wall deadline'):
    fleet.launch(a)

if __name__=='__main__':unittest.main()
