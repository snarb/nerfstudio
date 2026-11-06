import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
def module(name):
    sp=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def test_extension_rejects_unreviewed_or_changed_pilot(tmp_path):
    m=module('build_marshawn_4k_train');review=tmp_path/'root_review.json';review.write_text('{}')
    with pytest.raises(ValueError,match='root review'):m.validate_review(review,tmp_path/'data')
    (tmp_path/'complete.json').write_text('{}')
    review.write_text(json.dumps({'reviewer':'root','authorized_next':'CPU export all63 TRAIN using exact same builder','complete_sha256':'wrong'}))
    with pytest.raises(ValueError,match='completion changed'):m.validate_review(review,tmp_path/'data')


def test_no_eval_leak_into_63_scope(tmp_path,monkeypatch):
    m=module('build_marshawn_4k_train');data=tmp_path/'data';data.mkdir();monkeypatch.setattr(m,'validate_review',lambda *args:{})
    frames=[{'physical_camera':str(i),'file_path':str(i)+'.png'} for i in range(62)]+[{'physical_camera':'DC','file_path':'DC.png'}]
    (data/'transforms.json').write_text(json.dumps({'frames':frames,'train_filenames':[r['file_path'] for r in frames]}));(data/'ready.json').write_text('{}')
    with pytest.raises(ValueError,match='TRAIN'):m.export(data,tmp_path/'out',tmp_path/'review.json')
    assert not (tmp_path/'out').exists()


def test_frequency_release_fails_before_cuda_import(tmp_path):
    m=module('fit_marshawn_4k_AA_frequency');release=tmp_path/'release.json';release.write_text('{}');spec=tmp_path/'spec.json';spec.write_text(json.dumps({'release':str(release)}))
    with pytest.raises(ValueError,match='root GPU release'):m.validate(spec)


def test_exact_progressive_censoring_accounting():
    m=module('fit_marshawn_4k_AA_frequency');counts={};remaining=100
    for level in range(16):
        passed=10 if level<3 else 0;m.add_counts(counts,level,remaining,passed);remaining-=passed
    assert m.check_counts(counts,100)==70
    counts['5']['evaluated']-=1
    with pytest.raises(ValueError,match='coverage'):m.check_counts(counts,100)
    assert m.check_counts({'0':{'evaluated':100,'passed':100}},100)==0
    with pytest.raises(ValueError,match='final level'):m.check_counts({'0':{'evaluated':100,'passed':10}},100)
