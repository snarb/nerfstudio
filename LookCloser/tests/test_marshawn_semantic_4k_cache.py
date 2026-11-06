import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'scripts/probe_marshawn_semantic_4k_cache.py'
spec=importlib.util.spec_from_file_location('cache_probe',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_camera_permutation_allowed_not_duplication():
    m.check_cache_ids([2,0,1],3)
    for ids in ([0,0,2],[0,1],[0,1,3]):
        with pytest.raises(ValueError):m.check_cache_ids(ids,3)

def test_patch_grid_requires_actual_complete_4k_coverage():
    m.check_map_shape((253,480),(2024,3840))
    for shape,source in [((126,240),(2024,3840)),((253,480),(2028,3840))]:
        with pytest.raises(ValueError):m.check_map_shape(shape,source)

def test_source_hash_mutation_and_traversal_rejected(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'original');manifest={'a':m.sha(p)};m.check_source_inventory(tmp_path,manifest)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):m.check_source_inventory(tmp_path,manifest)
    with pytest.raises(ValueError):m.check_source_inventory(tmp_path,{'../a':'x'})
