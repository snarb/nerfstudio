import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_marshawn_source_W32 import main,sha
@pytest.mark.parametrize('approved,digest',[(False,'match'),(True,'wrong')])
def test_exact_release_precedes_gpu_and_source_access(tmp_path,approved,digest):
 p=tmp_path/'policy.json';p.write_text('{}')
 (tmp_path/'release.json').write_text(json.dumps(dict(approved_readonly=approved,policy_sha256=sha(p)if digest=='match'else'wrong')))
 with pytest.raises(ValueError,match='Exact read-only release'):main(SimpleNamespace(root=tmp_path,arm='unused',gpu=0))
