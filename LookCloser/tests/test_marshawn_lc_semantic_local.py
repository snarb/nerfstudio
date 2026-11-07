"""CPU checks for local capture gating, supervision and source-only parity."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_marshawn_lc_semantic_local as preparation
import run_marshawn_lc_semantic_capture_local as worker


def test_path_materialization_preserves_alias_and_scientific_numbers(tmp_path,monkeypatch):
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    source=tmp_path/'source.yml';target=tmp_path/'local.yml'
    source.write_text('data: &data !!python/object/apply:pathlib.PosixPath\n- /\n- cache\n- ubuntu\n- lookcloser\n- campaigns\n- marshawn_tuning_20261106\n- data\nalias: *data\nFR: 0.3\ncoarse: 0.0015625\ncap: 1024\ncorrected: false\n')
    rows,node=preparation.materialize_yaml(source,target)
    text=target.read_text()
    assert preparation.SC not in text
    assert '0.0015625' in text and '0.3' in text and '1024' in text and 'false' in text
    assert rows[0]['after']==str(preparation.CAMPAIGN/'data')
    data=next(v for k,v in node.value if k.value=='data')
    alias=next(v for k,v in node.value if k.value=='alias')
    assert alias is data


def test_false_release_rejected_before_source_or_checkpoint_reads(tmp_path):
    plan=tmp_path/'manifest.json';release=tmp_path/'root_release.json'
    plan.write_text('{}');release.write_text(json.dumps(dict(reviewer='root',capture_authorized=False)))
    with pytest.raises(ValueError,match='No exact root capture release'):
        worker.authorize(plan,release,tmp_path)


@pytest.mark.parametrize('corrupt',[False,True])
def test_actual_imported_helper_bytes_are_bound(tmp_path,monkeypatch,corrupt):
    helper=tmp_path/'helpers/tool.py';helper.parent.mkdir();helper.write_text('immutable helper bytes\n')
    policy=dict(bound_inputs={str(helper):worker.sha(helper)},code_sha256={},
        branch=str(tmp_path/'branch'),scientific_files_sha256={},helpers=str(helper.parent))
    if corrupt:helper.write_text('changed helper bytes\n')
    monkeypatch.setattr(worker,'sys',SimpleNamespace(modules={'_lc_tool':SimpleNamespace(__file__=str(helper))}))
    if corrupt:
        with pytest.raises(ValueError,match='Unbound actual project/helper import'):
            worker.project_import_bindings(policy,tmp_path)
    else:
        assert worker.project_import_bindings(policy,tmp_path)['_lc_tool']['sha256']==worker.sha(helper)


def test_continuous_guard_stops_on_foreign_GPU_process(tmp_path,monkeypatch):
    uuid='GPU-test';killed=[]
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=f'{uuid},12345,1\n'))
    monkeypatch.setattr(worker.os,'kill',lambda pid,sig:killed.append((pid,sig)))
    policy=dict(max_worker_RSS_gib=1000,max_process_mib=10240)
    worker.monitor_resources(policy,uuid,tmp_path,threading.Event())
    assert killed==[(os.getpid(),signal.SIGTERM)]
    assert (tmp_path/'resource_failure.json').exists()
    assert (tmp_path/'resource_checks.jsonl').exists()


def test_three_source_MODEL_parity_views_never_access_target_dataset(tmp_path):
    renders=tmp_path/'references';renders.mkdir()
    helper=Path(__file__).resolve().parents[1]/'scripts/probe_marshawn_source_W32.py'
    reference=dict(helper=str(helper),helper_sha256=worker.sha(helper),renders=str(renders),
        files_sha256={},checkpoint_sha256='source-sha')
    for cam in ['DC','HC','KA']:
        for suffix,array in [('.png',np.full((2,2,3),64,dtype=np.uint8)),('_alpha.png',np.full((2,2),65535,dtype=np.uint16))]:
            file=cam+suffix;Image.fromarray(array).save(renders/file);reference['files_sha256'][file]=worker.sha(renders/file)
    class Cameras:
        def __getitem__(self,key):return self
        def to(self,device):return self
    class Dataset:
        image_filenames=[Path('eval_'+c+'.png')for c in ['DC','HC','KA']]
        cameras=Cameras()
        def __getitem__(self,index):raise AssertionError('Heldout target/mask access forbidden')
    class Model:
        device='cpu'
        def get_outputs_for_camera(self,camera):
            return dict(rgb=torch.full((2,2,3),64/255),accumulation=torch.ones(2,2,1))
    pipe=SimpleNamespace(datamanager=SimpleNamespace(eval_dataset=Dataset()))
    result=worker.prove_source_FHD(pipe,reference,tmp_path,Model(),torch)
    assert len(result)==6 and all(r['restored_bitwise_equal']for r in result.values())
    receipt=json.loads((tmp_path/'source_FHD_parity/parity.json').read_text())
    assert receipt['EVAL_target_RGB_or_masks_read'] is False
    assert receipt['TRAIN_capture_started'] is False
