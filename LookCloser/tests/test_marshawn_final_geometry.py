"""Retained choices use step-named schema2 sidecars, with exact hash checks."""
import hashlib,importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('finalgeom',Path(__file__).parents[1]/'scripts/watch_marshawn_final_geometry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def record(tmp_path):
 (tmp_path/'checkpoint.ckpt').write_bytes(b'checkpoint');(tmp_path/'step-000111316_grid.npz').write_bytes(b'actual grid')
 return dict(path=str(tmp_path),step=111316,checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(),grid_sha256=hashlib.sha256(b'actual grid').hexdigest())
def test_step_named_archive(tmp_path):
 r=record(tmp_path);assert m.verify_choice(r)==tmp_path/'checkpoint.ckpt'
def test_sidecar_tamper_rejected(tmp_path):
 r=record(tmp_path);(tmp_path/'step-000111316_grid.npz').write_bytes(b'changed')
 with pytest.raises(ValueError):m.verify_choice(r)
def test_alias_cannot_replace_missing_actual_sidecar(tmp_path):
 r=record(tmp_path);(tmp_path/'step-000111316_grid.npz').rename(tmp_path/'grid.npz')
 with pytest.raises(FileNotFoundError):m.verify_choice(r)
