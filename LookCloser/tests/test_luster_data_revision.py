"""A reviewed local mask correction must not overwrite source or unrelated data."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from archive_luster_checkpoint import sha
from prepare_luster_video import export_directory
from repair_luster_background_polygon import main


def setup_case(tmp_path,monkeypatch,train=True):
    frame=tmp_path/'frames/000475';source=frame/'source';data=frame/'data'
    for folder in ['frame/images','masks/cam_11']:(source/folder).mkdir(parents=True)
    for folder in ['images','masks','lookcloser_frequencies']:(data/folder).mkdir(parents=True)
    name='cam_011_000475';rgb=source/'frame/images'/f'{name}.jpg';mask=source/'masks/cam_11/000475.png'
    Image.new('RGB',(12,12),'white').save(rgb);Image.new('L',(12,12),255).save(mask)
    Image.new('RGB',(6,6),'white').save(data/'images'/f'{name}.png');Image.new('L',(6,6),255).save(data/'masks'/f'{name}.png')
    (data/'transforms.json').write_text(json.dumps({'train_filenames':[f'images/{name}.png'] if train else []}))
    for suffix in ['pt','json','receipt.json']:(data/'lookcloser_frequencies'/f'{name}.{suffix}').write_text('old cache')
    (data/'lookcloser_frequencies/unrelated.pt').write_text('keep')
    (data/'audit_ready.json').write_text('{}');(data/'frequency_complete.json').write_text('{}')
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(frame='000475',camera=11,id='gap_v1',reviewed=True,reason='test background',
         rgb_sha256=sha(rgb),mask_sha256=sha(mask),polygon_full_resolution=[[2,2],[8,2],[8,8],[2,8]])))
    monkeypatch.setattr(sys,'argv',['repair',str(tmp_path),str(spec)])
    return data,rgb,mask


def test_repair_preserves_sources_and_other_maps(tmp_path,monkeypatch):
    data,rgb,mask=setup_case(tmp_path,monkeypatch);original=(sha(rgb),sha(mask))
    main()
    assert (sha(rgb),sha(mask))==original
    assert (data/'lookcloser_frequencies/unrelated.pt').read_text()=='keep'
    assert not (data/'lookcloser_frequencies/cam_011_000475.pt').exists()
    assert not (data/'audit_ready.json').exists()
    assert not (data/'frequency_complete.json').exists()
    assert (data/'revisions/gap_v1/before/lookcloser_frequencies/cam_011_000475.pt').exists()
    assert (np.array(Image.open(data/'masks/cam_011_000475.png'))<255).any()
    with pytest.raises(ValueError,match='already exists'):main()


def test_eval_repair_requires_another_protocol(tmp_path,monkeypatch):
    data,_,_=setup_case(tmp_path,monkeypatch,train=False)
    with pytest.raises(ValueError,match='train views only'):main()
    assert (data/'audit_ready.json').exists()


def test_export_cache_changes_with_data_revision(tmp_path):
    assert export_directory(tmp_path,16000)==tmp_path/'export_s016000'
    (tmp_path/'data').mkdir();revision=tmp_path/'data/revision.json'
    revision.write_text(json.dumps({'id':'gap_v1'}))
    assert export_directory(tmp_path,16000)==tmp_path/'export_s016000_gap_v1'
    revision.write_text(json.dumps({'id':'../escape'}))
    with pytest.raises(ValueError):export_directory(tmp_path,16000)
