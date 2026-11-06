"""A pruning race must preserve surviving evidence without inventing retained weights."""
import importlib.util,json,os
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'scripts/watch_marshawn_semantic_evaluations.py'
spec=importlib.util.spec_from_file_location('archive',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path,keep=True):
    c=tmp_path/'campaign';(c/'independent_occupancy').mkdir(parents=True)
    b=c/'branch';run=b/'run';renders=run/'renders/step-000121316';renders.mkdir(parents=True)
    (renders/'metrics.json').write_text('{}');(renders/'DC.png').write_bytes(b'render')
    occ=run/'occupancy';occ.mkdir();grid=occ/'step-000121316_grid.json';npz=grid.with_suffix('.npz');npz.write_bytes(b'actual grid')
    cp=run/'checkpoints/step-000121316.ckpt';cp.parent.mkdir();cp.write_bytes(b'checkpoint');digest=m.sha(cp)
    grid.write_text(json.dumps(dict(checkpoint_sha256=digest,grid_archive_sha256=m.sha(npz))))
    if not keep:cp.unlink()
    config=run/'worker/lookcloser/frozen/config.yml';config.parent.mkdir(parents=True);config.write_text('actual config')
    (b/'spec.json').write_text('{}');(b/'source_manifest.json').write_text('{}')
    (run/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),time=0,step=124000)))
    return b,c,cp,grid

@pytest.mark.parametrize('keep',[True,False])
def test_retained_weight_claim_is_exact(tmp_path,keep):
    b,c,cp,grid=fixture(tmp_path,keep);r=m.archive(b,c)['archives'][0]
    assert r['checkpoint_retained'] is keep and r['immutable_hardlink'] is keep
    dest=Path(r['archive']);assert (dest/'renders/DC.png').exists()
    assert (dest/'checkpoint.ckpt').exists() is keep
    assert bool(r['checkpoint_unavailable_reason']) is (not keep)
    assert m.archive(b,c)['archives']==[r]

def test_hash_mismatch_never_certified(tmp_path):
    b,c,cp,grid=fixture(tmp_path);cp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checkpoint differs'):m.archive(b,c)
    assert not(c/'independent_occupancy/semantic_actor_progress/step-000121316/archive_receipt.json').exists()
