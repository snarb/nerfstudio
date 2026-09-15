"""The wider domain wrapper must not modify any admission calculation."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_wrapper_only_rebinds_paths(monkeypatch):
    wrapper = importlib.import_module('run_mhr_confidence_domain_admission')
    module = wrapper.admission
    before = {name: getattr(module, name) for name in [
        'certificates', 'initial_admission', 'interpolation_admission',
        'native_guard', 'train_reference_votes', 'footprint_veto', 'measured_pixel_veto',
        'ARMS', 'SOURCE', 'PRIOR', 'FRAME', 'HELPERS', 'Scene2',
    ]}
    monkeypatch.setattr(module, 'CANDIDATES', Path('/example/old-candidates'))
    monkeypatch.setattr(module, 'OUT', Path('/example/old-output'))
    wrapper.configure()
    assert module.CANDIDATES == wrapper.CANDIDATES
    assert module.OUT == wrapper.OUT
    assert all(getattr(module, name) is value for name, value in before.items())


def test_request_extension_preserves_algorithm_and_restores_save(monkeypatch):
    wrapper = importlib.import_module('run_mhr_confidence_domain_admission')
    module = wrapper.admission
    monkeypatch.setattr(module, 'CANDIDATES', module.CANDIDATES)
    monkeypatch.setattr(module, 'OUT', module.OUT)
    monkeypatch.setattr(sys, 'argv', ['run_mhr_confidence_domain_admission.py'])
    proof = {'algorithm_and_thresholds_unchanged': True}
    monkeypatch.setattr(wrapper, 'binding', lambda: proof)
    writes = []
    saver = lambda path, value: writes.append((path, value))
    monkeypatch.setattr(module, 'save', saver)
    original_request = {'interpolation': {'seed_radius': 0.003}, 'final_cameras': 62}

    def run():
        module.save(wrapper.OUT / 'request.json', original_request)
        module.save(wrapper.OUT / 'result.json', {'added': 0})

    monkeypatch.setattr(module, 'run', run)
    wrapper.main()
    assert writes[0][1] == dict(original_request, control_wrapper=proof)
    assert writes[1][1] == {'added': 0}
    assert 'control_wrapper' not in original_request
    assert module.save is saver
