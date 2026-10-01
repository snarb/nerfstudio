"""Encode a short real-frame fixture and reject stale model/review identities."""
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import assemble_luster_video as assembly
from prepare_luster_video import write


def fixture(root):
    frames = ['000470', '000471']
    write(root/'manifest.json', {'frames': frames, 'fps': 30})
    write(root/'video_cameras.json', {'fixed': True})
    for index, frame in enumerate(frames):
        images = {}
        for kind in ['body', 'detail']:
            path = root/'video_frames'/kind/f'{frame}.png'; path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new('RGB', (1080,1920), 'black')
            ImageDraw.Draw(image).rectangle((200+index*20,300,500+index*20,1500), fill='green')
            image.save(path); images[kind] = dict(path=str(path), sha256=assembly.sha(path))
        write(root/'selection'/f'{frame}.json', dict(step=8000, per_view=[dict(split='eval',
              physical_camera='cam_012', psnr=30., ssim=.95, lpips=.05, foreground_psnr=25., rois={})]))
        write(root/'video_frames/receipts'/f'{frame}.json', dict(frame=frame, checkpoint_sha256=f'checkpoint{index}',
              cameras_sha256=assembly.sha(root/'video_cameras.json'), images=images, field_parameters_sha256=f'field{index}'))
        write(root/'snapshots'/f'{frame}.json', dict(archived_checkpoint={'sha256': f'checkpoint{index}'},
              selection=str(root/'selection'/f'{frame}.json')))
        write(root/'visual_reviews'/f'{frame}.json', dict(accepted=True, checkpoint_sha256=f'checkpoint{index}'))


def test_two_frames_encode_at_source_rate_without_repeating_to_two_seconds(tmp_path, monkeypatch):
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe required for encoding integration')
    fixture(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['assemble', str(tmp_path)])
    assembly.main()
    manifest = json.loads((tmp_path/'final_video/manifest.json').read_text())
    assert manifest['duration_seconds'] == 2/30
    assert manifest['visual_review'].startswith('pending')
    for result in manifest['videos'].values():
        assert result['probe']['nb_read_frames'] == '2'
        assert result['probe']['avg_frame_rate'] == '30/1'
        assert result['probe']['color_space'] == 'bt709'
        assert result['decoded']


def test_assembly_rejects_stale_visual_acceptance(tmp_path, monkeypatch):
    fixture(tmp_path)
    write(tmp_path/'visual_reviews/000471.json', dict(accepted=True, checkpoint_sha256='different'))
    monkeypatch.setattr(sys, 'argv', ['assemble', str(tmp_path)])
    with pytest.raises(ValueError, match='checkpoint-bound'):
        assembly.main()


def test_assembly_rejects_duplicate_learned_models(tmp_path, monkeypatch):
    fixture(tmp_path)
    path=tmp_path/'video_frames/receipts/000471.json'; receipt=json.loads(path.read_text())
    receipt['field_parameters_sha256']='field0'; write(path,receipt)
    monkeypatch.setattr(sys, 'argv', ['assemble', str(tmp_path)])
    with pytest.raises(ValueError, match='Repeated learned fields'):
        assembly.main()
