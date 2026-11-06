import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_marshawn_4k_frequency_queue.py'
    spec = importlib.util.spec_from_file_location('queue4k', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_camera_remapping_only_changes_metadata_paths_and_progress():
    source = (Path(__file__).resolve().parents[1] / 'scripts/fit_marshawn_4k_AA_frequency.py').read_text()
    output = module().camera_worker(source, 'NE')
    assert output.replace("'camera':'NE'", "'camera':'AA'").replace('train_NE.pt', 'train_AA.pt').replace('fitting_frequency_NE', 'fitting_frequency_AA') == source
    compile(output, 'camera_worker', 'exec')


@pytest.mark.parametrize('camera', ['DC', 'HC', 'KA', '../AA'])
def test_eval_and_invalid_cameras_rejected(camera):
    with pytest.raises(ValueError, match='TRAIN'):
        module().camera_worker('', camera)
