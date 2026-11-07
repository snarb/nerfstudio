import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/marshawn_gpu_isolation.py';s=importlib.util.spec_from_file_location('gpu_isolation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
UUID='GPU-cb1a0b07-abd6-8e55-9669-ccbb2c45a560'
def test_missing_and_index_environment_fail_before_torch(monkeypatch):
 monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
 with pytest.raises(RuntimeError):m.require_environment(UUID)
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4')
 with pytest.raises(RuntimeError):m.require_environment(UUID)
def test_actual_physical_uuid_must_match(monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
 def fake(count=1,current=0,uuid=UUID):return SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count,current_device=lambda:current,get_device_properties=lambda n:SimpleNamespace(uuid=uuid,name='fake')))
 assert m.verify_cuda(UUID,fake())['actual_physical_uuid']==UUID
 for t in[fake(8),fake(current=1),fake(uuid=UUID[:-1]+'1')]:
  with pytest.raises(RuntimeError):m.verify_cuda(UUID,t)
def test_child_environment_is_exact_single_uuid(monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2')
 assert m.child_environment(UUID)['CUDA_VISIBLE_DEVICES']==UUID
 with pytest.raises(ValueError):m.child_environment('4')
