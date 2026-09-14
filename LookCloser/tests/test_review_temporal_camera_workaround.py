import importlib.util
from pathlib import Path
import sys
import json
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('camera_workaround_review', SCRIPTS / 'review_temporal_camera_workaround.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from finalize_temporal_camera_review import finalize
from joint_temporal_texture import sha


def test_native_crop_follows_landmark_without_rescaling():
    box = module.crop_box({'scene_landmark_portrait_xy': [348.0, 959.0]})
    assert box == (98, 984, 598, 1284)
    other = module.crop_box({'scene_landmark_portrait_xy': [448.0, 909.0]})
    assert tuple(b - a for a, b in zip(box, other)) == (100, -50, 100, -50)


def review_fixture(root):
    source = root / 'source.json'
    source.write_text('{}')
    rgb = root / 'frame.png'
    rgb.write_bytes(b'hashed-image-fixture')
    sheet = root / 'jaw_000.png'
    sheet.write_bytes(b'hashed-sheet-fixture')
    request = dict(source_request=str(source), source_request_sha256=sha(source),
                   inputs=[dict(frame_id='000899', path=str(rgb), sha256=sha(rgb))])
    (root / 'request.json').write_text(json.dumps(request))
    sheets = dict(request_sha256=sha(root / 'request.json'), sheets=[dict(
        path=str(sheet), sha256=sha(sheet), frame_ids=['000899'])])
    (root / 'sheets.json').write_text(json.dumps(sheets))
    notes = root / 'notes.json'
    notes.write_text(json.dumps(dict(summary='Residual artifact', sheets={'jaw_000.png': 'Directly reviewed; seam remains'})))
    return notes


def test_review_binds_all_inputs_without_artifact_free_claim(tmp_path):
    notes = review_fixture(tmp_path)
    finalize(tmp_path, notes)
    result = json.loads((tmp_path / 'visual_review.json').read_text())
    assert result['reviewed_frame_count'] == 1
    assert result['artifact_free'] is False
    assert result['whole_actor_accepted'] is False


@pytest.mark.parametrize('tamper', ['image', 'sheet', 'missing_notes', 'duplicate_frame'])
def test_review_rejects_changed_or_unreviewed_input(tmp_path, tamper):
    notes = review_fixture(tmp_path)
    if tamper == 'image':
        (tmp_path / 'frame.png').write_bytes(b'changed')
    elif tamper == 'sheet':
        (tmp_path / 'jaw_000.png').write_bytes(b'changed')
    elif tamper == 'missing_notes':
        notes.write_text(json.dumps(dict(summary='No review', sheets={})))
    else:
        path = tmp_path / 'sheets.json'
        data = json.loads(path.read_text())
        data['sheets'][0]['frame_ids'].append('000899')
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        finalize(tmp_path, notes)
    assert not (tmp_path / 'visual_review.json').exists()
