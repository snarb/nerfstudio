import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('health',Path(__file__).parents[1]/'scripts/watch_marshawn_4k_health.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_pid_reuse_and_zombie_are_not_alive():
 expected=dict(pid=3,start='8',boot='boot')
 assert m.matches(expected,dict(expected,state='S'))
 assert not m.matches(expected,dict(expected,state='Z'))
 assert not m.matches(expected,dict(expected,start='9',state='S'))
 assert not m.matches(expected,dict(expected,boot='another',state='S'))
 assert not m.matches(expected,None)
