"""Receipts must bind actual map bytes/metadata, masks, RGB and the selected box."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_marshawn_tuning import refresh_inventory,verify_inventory,sha,write


def dataset(tmp_path):
    for folder in ('images','masks','semantics','lookcloser_frequencies'):(tmp_path/folder).mkdir()
    image='images/train_AA.png';mask='masks/train_AA.png';semantic='semantics/AA.npz'
    for name in (image,mask,semantic):(tmp_path/name).write_bytes(name.encode())
    tf=dict(train_filenames=[image],frames=[dict(file_path=image,mask_path=mask)])
    write(tmp_path/'transforms.json',tf)
    box=dict(aabb=[[0,0,0],[1,1,1]],world_to_box=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    write(tmp_path/'box.json',box)
    write(tmp_path/'ready.json',dict(**box,transforms_sha256=sha(tmp_path/'transforms.json'),
        records=[dict(camera='AA',image=image,rgb_sha256=sha(tmp_path/image),semantic_sha256=sha(tmp_path/semantic))]))
    frequency=tmp_path/'lookcloser_frequencies/train_AA.pt';frequency.write_bytes(b'fixed map bytes')
    write(frequency.with_suffix('.json'),dict(value_type='scalar_resolution',source_image_sha256=sha(tmp_path/image),
        train_steps_per_level=1000,patch_size=8,stride=8,n_levels=16,min_res=16,max_res=8192,
        eval_patch_batch_size=1024,ssim_threshold=.95,fit_script_sha256='legacy-fitter-sha'))
    return tmp_path


def test_seal_then_verify_idempotent_does_not_edit_payload(tmp_path):
    data=dataset(tmp_path)
    before={str(p.relative_to(data)):sha(p) for p in data.rglob('*') if p.is_file() and p.name!='ready.json'}
    ready=refresh_inventory(data)
    assert ready['box_sha256']==sha(data/'box.json')
    assert ready['inventory']['frequency_recipe']['ssim_threshold']==.95
    assert ready['inventory']['frequency_sources']['train_AA']['preprocess_source_sha256'] is None
    ready_hash=sha(data/'ready.json')
    assert refresh_inventory(data)==verify_inventory(data)
    assert sha(data/'ready.json')==ready_hash
    assert before=={str(p.relative_to(data)):sha(p) for p in data.rglob('*') if p.is_file() and p.name!='ready.json'}


@pytest.mark.parametrize('relative',['lookcloser_frequencies/train_AA.pt','lookcloser_frequencies/train_AA.json',
                                    'box.json','masks/train_AA.png','images/train_AA.png','semantics/AA.npz'])
def test_perturbed_artifacts_cannot_be_silently_resealed(tmp_path,relative):
    data=dataset(tmp_path);refresh_inventory(data)
    path=data/relative
    if path.suffix=='.json':
        payload=json.loads(path.read_text());payload['tampered']=True;write(path,payload)
    else:path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):verify_inventory(data)
    with pytest.raises(ValueError):refresh_inventory(data)


def test_legacy_maps_wrong_threshold_schedule_rejected(tmp_path):
    data=dataset(tmp_path);p=data/'lookcloser_frequencies/train_AA.json'
    meta=json.loads(p.read_text());meta['train_steps_per_level']=200;write(p,meta)
    with pytest.raises(ValueError,match='recipe mismatch'):refresh_inventory(data)
