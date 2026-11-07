"""Adversarial CPU regression for actual-import proof on virtual Torch modules."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_marshawn_lc_semantic_capture_v3 as worker


def policy(root,files=None):
    return dict(bound_inputs=files or {},code_sha256={},branch=str(root/'branch'),
        scientific_files_sha256={},helpers=str(root/'helpers'))


@pytest.mark.parametrize('name,file',[('torch.ops','_ops.py'),('torch.classes','_classes.py'),('other.virtual','virtual.py')])
def test_relative_virtual_nonproject_files_are_skipped(tmp_path,monkeypatch,name,file):
    monkeypatch.setattr(worker,'sys',SimpleNamespace(modules={name:SimpleNamespace(__file__=file)}))
    assert worker.project_import_bindings(policy(tmp_path),tmp_path)=={}


@pytest.mark.parametrize('name',['nerfstudio.fake','_lc_fake','run_marshawn_fake','launch_marshawn_fake'])
@pytest.mark.parametrize('kind',['relative','absolute_unbound','absolute_missing'])
def test_named_project_modules_require_real_absolute_bound_file(tmp_path,monkeypatch,name,kind):
    file=tmp_path/'unbound.py'
    if kind=='relative':file='unbound.py'
    elif kind=='absolute_unbound':file.write_text('unbound real bytes\n')
    monkeypatch.setattr(worker,'sys',SimpleNamespace(modules={name:SimpleNamespace(__file__=str(file))}))
    with pytest.raises(ValueError):worker.project_import_bindings(policy(tmp_path),tmp_path)


def test_real_absolute_torch_file_inside_package_is_not_exempt(tmp_path,monkeypatch):
    file=tmp_path/'code/_ops.py';file.parent.mkdir();file.write_text('unexpected physical file\n')
    monkeypatch.setattr(worker,'sys',SimpleNamespace(modules={'torch.ops':SimpleNamespace(__file__=str(file))}))
    with pytest.raises(ValueError,match='Unbound actual project/helper import'):
        worker.project_import_bindings(policy(tmp_path),tmp_path)


def test_real_bound_project_module_passes(tmp_path,monkeypatch):
    file=tmp_path/'helpers/known.py';file.parent.mkdir();file.write_text('bound real bytes\n')
    monkeypatch.setattr(worker,'sys',SimpleNamespace(modules={'_lc_known':SimpleNamespace(__file__=str(file))}))
    result=worker.project_import_bindings(policy(tmp_path,{str(file):worker.sha(file)}),tmp_path)
    assert result['_lc_known']['path']==str(file.resolve())
