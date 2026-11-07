import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_margin_checkpoint import run,sha

def test_missing_exact_release_precedes_any_gpu_or_archive_access(tmp_path):
 release=tmp_path/'release.json';release.write_text(json.dumps(dict(reviewer='root',readonly_authorized=False,plan_sha256='bad')))
 plan=tmp_path/'policy.json';plan.write_text(json.dumps(dict(release=str(release))))
 with pytest.raises(ValueError,match='Exact read-only release'):run(plan,'source_margin002')

def test_source_drift_fails_before_lease_or_torch(tmp_path):
 source=tmp_path/'source';source.write_text('wrong');release=tmp_path/'release.json';plan=tmp_path/'policy.json'
 plan.write_text(json.dumps(dict(release=str(release),models={'source_margin002':dict(archive=str(tmp_path/'unopened'),output=str(tmp_path/'result'))},code=str(tmp_path/'code'),helpers=str(tmp_path/'helpers'),files_sha256={str(source):'invalid'})))
 release.write_text(json.dumps(dict(reviewer='root',readonly_authorized=True,plan_sha256=sha(plan))))
 with pytest.raises(ValueError,match='source drift'):run(plan,'source_margin002')
 assert not (tmp_path/'result/GPU_identity.json').exists()
