"""Archive verification must precede local cache release or checkpoint reuse."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
import archive_luster_checkpoint as checkpoints
import archive_luster_originals as originals


def test_failed_remote_checkpoint_verification_keeps_local_file(tmp_path,monkeypatch):
    checkpoint=tmp_path/'frames/000471/runs/test/step-000006000.ckpt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'valuable-model')
    monkeypatch.setattr(checkpoints.subprocess,'run',lambda *a,**kw:None)
    monkeypatch.setattr(checkpoints.subprocess,'check_output',lambda *a,**kw:json.dumps(dict(sha256='wrong',bytes=14)))
    with pytest.raises(ValueError,match='differs'):
        checkpoints.archive(tmp_path,checkpoint,'/archive',release=True)
    assert checkpoint.read_bytes()==b'valuable-model'


def test_only_campaign_checkpoints_can_be_released(tmp_path):
    checkpoint=tmp_path/'source.png';checkpoint.write_bytes(b'image')
    with pytest.raises(ValueError):checkpoints.archive(tmp_path,checkpoint,'/archive',release=True)
    assert checkpoint.exists()


def test_bad_restore_does_not_create_a_reusable_checkpoint(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.ckpt'
    checkpoint.with_suffix('.archive.json').write_text(json.dumps(dict(host='dev3',remote_path='/model.ckpt',sha256=hashlib.sha256(b'good').hexdigest())))
    def corrupt_download(command,**kwargs):Path(command[-1]).write_bytes(b'bad')
    monkeypatch.setattr(checkpoints.subprocess,'run',corrupt_download)
    with pytest.raises(ValueError,match='differs'):checkpoints.restore(checkpoint)
    assert not checkpoint.exists() and not checkpoint.with_suffix('.restore-partial').exists()


def test_hd_originals_require_matching_remote_hashes(tmp_path,monkeypatch):
    (tmp_path/'original_hd_archive.json').write_text(json.dumps(dict(host='dev3',remote_dir='/originals',files={'cam.png':dict(sha256='expected',size=[10,20])})))
    monkeypatch.setattr(originals.subprocess,'check_output',lambda *a,**kw:json.dumps({'cam.png':'different'}))
    with pytest.raises(ValueError,match='differ'):originals.verify_archived_originals(tmp_path)
