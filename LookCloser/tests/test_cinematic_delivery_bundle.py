import json
from pathlib import Path
import sys
import zipfile
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_temporal_texture import sha
from bundle_cinematic_choices import bundle,NAMES
from audit_cinematic_delivery import audit


def fixture(tmp_path):
    records=[]
    for name in NAMES:
        folder=tmp_path/name;folder.mkdir()
        video=folder/'video.mp4';video.write_bytes(('dummy encoded payload '+name).encode())
        publication=folder/'publication.json';publication.write_text('{}')
        records.append(dict(variant=name,video=str(video),video_sha256=sha(video),publication_sha256=sha(publication)))
    path=tmp_path/'delivery_audit.json'
    path.write_text(json.dumps(dict(all_four=True,status='delivery_integrity_pass_with_disclosed_visual_residuals',records=records)))
    return records


def test_bundle_preserves_bytes_and_declares_live_action(tmp_path):
    records=fixture(tmp_path);bundle(tmp_path)
    with zipfile.ZipFile(tmp_path/'cinematic_choices.zip') as archive:
        assert len(archive.namelist())==6
        for row in records:assert archive.read(NAMES[row['variant']])==Path(row['video']).read_bytes()
        manifest=json.loads(archive.read('manifest.json'))
        assert not manifest['all_frames_are_3d_renders'] and not manifest['artifact_free_approval']
        assert 'not a 3D reconstruction' in archive.read('README.txt').decode()


def test_bundle_rejects_changed_movie(tmp_path):
    records=fixture(tmp_path);Path(records[0]['video']).write_bytes(b'changed')
    with pytest.raises(AssertionError):bundle(tmp_path)
    assert not (tmp_path/'cinematic_choices.zip').exists()


def test_delivery_rejects_unfinished_publication(tmp_path):
    folder=tmp_path/'locked_arc';folder.mkdir()
    (folder/'publication.json').write_text(json.dumps(dict(status='review_in_progress')))
    with pytest.raises(AssertionError):audit(tmp_path,['locked_arc'])
