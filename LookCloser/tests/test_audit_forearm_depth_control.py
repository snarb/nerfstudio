import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_temporal_texture import atomic_json,sha
from audit_forearm_depth_control import audit


def control(tmp_path):
    r=tmp_path/'control';atomic_json(r/'request.json',{})
    digest=sha(r/'request.json');cfg=r/'pipeline/dense/stereo/patch-match.cfg'
    cfg.parent.mkdir(parents=True);cfg.write_text('explicit12sources\n')
    for stage,value in [('undistort','old_default_config'),('patch-config',sha(cfg))]:
        atomic_json(r/'stages'/f'{stage}.json',dict(request_sha256=digest,retained_hashes={str(cfg):value}))
    atomic_json(r/'complete.json',dict(request_sha256=digest,hashes={}))
    for v in ['fuse-original','fuse-full-block']:
        out=r/(v+'_render');atomic_json(out/'request.json',{})
        atomic_json(out/'frames/001033/complete.json',dict(request_sha256=sha(out/'request.json'),hashes={}))
    atomic_json(r/'depth_qc.json',dict(maps=62,shape=[1080,1920]))
    evidence=[]
    for name in ['rgb.png','clay.png']:
        p=r/name;p.write_bytes(b'fixture');evidence.append(dict(path=str(p),sha256=sha(p)))
    atomic_json(r/'visual_review.json',dict(status='rejected_as_forearm_repair',evidence=evidence))
    return r,cfg


def test_explicit_same_request_config_successor_is_allowed(tmp_path):
    r,_=control(tmp_path);audit(r)
    assert (r/'final_audit.json').exists()


def test_later_config_tampering_is_not_excused(tmp_path):
    r,cfg=control(tmp_path);cfg.write_text('unexpected mutation\n')
    with pytest.raises(ValueError,match='Unexplained changed artifact'):audit(r)


def test_missing_visual_evidence_fails(tmp_path):
    r,_=control(tmp_path);atomic_json(r/'visual_review.json',dict(status='pending',evidence=[]))
    with pytest.raises(ValueError,match='Explicit matched RGB/clay'):audit(r)
