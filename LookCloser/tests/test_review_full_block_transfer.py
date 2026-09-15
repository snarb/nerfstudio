import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_full_block_transfer import retained_path


def test_retained_paths_stay_below_explicit_root(tmp_path):
    assert retained_path(tmp_path,'fuse-original/mesh.ply')==tmp_path/'fuse-original/mesh.ply'
    for name in ('/tmp/unrelated.ply','../elsewhere','fuse-original/../../outside',''):
        with pytest.raises(ValueError): retained_path(tmp_path,name)
