"""Frame isolation and conservative self-only GPU admission."""
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from guarded_transfer_inference import safe


def test_guard_requires_large_reserve_and_bounded_own_memory():
    good=dict(total_mib=97887,used_mib=13000,free_mib=84000,own_mib=4500)
    assert safe(good)
    for field,value in [('free_mib',65535),('used_mib',24576),('own_mib',6144)]:
        assert not safe({**good,field:value})


def test_frame_context_is_process_local_and_rejects_foreign_geometry(tmp_path):
    for frame in [545,599]:
        context=tmp_path/f'{frame}.json'
        context.write_text(json.dumps(dict(frame=frame,origin='/example/origin',
            source=f'/example/fullres/{frame:06d}',checkpoint='/example/frozen.ckpt',
            checkpoint_sha256='a'*64)))
        root=tmp_path/f'root{frame}'
        geometry=root/'geometry/cam007';geometry.mkdir(parents=True)
        (geometry/'receipt.json').write_text(json.dumps(dict(state='complete',checkpoint_sha256='b'*64)))
        env={**os.environ,'PYTHONPATH':str(SCRIPTS),'LUSTER_HYBRID_FRAME_MANIFEST':str(context),
             'CUDA_VISIBLE_DEVICES':''}
        code=f"""
import luster_hybrid_common as c
from pathlib import Path
assert c.FRAME=={frame}
assert str(c.source_path(7))=='/example/fullres/{frame:06d}/images/cam_007_{frame:06d}.jpg'
try:
    c.geometry(Path({str(root)!r}),7)
except ValueError:
    pass
else:
    raise AssertionError('Foreign checkpoint accepted')
"""
        subprocess.run([sys.executable,'-c',code],env=env,check=True)


def test_default_540_is_unchanged():
    env={**os.environ,'PYTHONPATH':str(SCRIPTS),'CUDA_VISIBLE_DEVICES':''}
    env.pop('LUSTER_HYBRID_FRAME_MANIFEST',None)
    subprocess.run([sys.executable,'-c',
        "import luster_hybrid_common as c; assert c.FRAME==540; assert c.source_path(7).name=='cam_007_000540.jpg'; assert c.DIGEST.startswith('f07369d4')"],
        env=env,check=True)


def test_aabb_projection_contains_dense_surface_and_handles_camera_crossing():
    import numpy as np
    from bounded_native_luster_hybrid import projected_box
    from luster_hybrid_common import project
    row=dict(w=1000,h=800,fl_x=500.,fl_y=500.,cx=500.,cy=400.,
             transform_matrix=np.eye(4).tolist())
    bounds=np.array([[-.4,-.5,-3],[.6,.7,-1]],np.float32)
    rng=np.random.default_rng(42)
    points=rng.uniform(bounds[0],bounds[1],(10000,3)).astype(np.float32)
    uv,_=project(points,row);x0,y0,x1,y1=projected_box(bounds,row)
    inside=(uv[:,0]>=0)&(uv[:,0]<1000)&(uv[:,1]>=0)&(uv[:,1]<800)
    assert np.all((uv[inside,0]>=x0)&(uv[inside,0]<x1)&(uv[inside,1]>=y0)&(uv[inside,1]<y1))
    assert projected_box(np.array([[-1,-1,-1],[1,1,1]],np.float32),row)==[0,0,1000,800]


def test_saved_cell_union_covers_points_without_filling_gap():
    import numpy as np
    from occupancy_native_luster_hybrid import project_cell_union
    from luster_hybrid_common import project
    row=dict(w=1000,h=800,fl_x=500.,fl_y=500.,cx=500.,cy=400.,
             transform_matrix=np.eye(4).tolist())
    lo=np.array([[-1,-.1,-2],[.8,-.1,-2]],np.float32)
    hi=lo+np.array([.2,.2,.2],np.float32)
    mask=project_cell_union(lo,hi,row)
    rng=np.random.default_rng(43)
    for a,b in zip(lo,hi):
        uv,_=project(rng.uniform(a,b,(1000,3)).astype(np.float32),row)
        xy=np.rint(uv).astype(int)
        assert mask[xy[:,1],xy[:,0]].all()
    assert not mask[400,500]
    assert not project_cell_union(lo[:0],hi[:0],row).any()


def test_witness_pool_is_frozen_and_all_demo_heldout_witnesses_excluded(tmp_path):
    from source_surface_confidence_luster_hybrid import frozen_witness_pool
    from luster_hybrid_common import assert_source,read,write
    (tmp_path/'context.json').write_text('{}')
    write(tmp_path/'preflight/sources.json',dict(all_sources=[7,12,95,150,11,13]))
    path=frozen_witness_pool(tmp_path)
    assert read(path)['cameras']==[11,13]
    (tmp_path/'geometry/cam014').mkdir(parents=True)
    assert read(frozen_witness_pool(tmp_path))['cameras']==[11,13]
    rows={7:dict(train=True),12:dict(train=False),11:dict(train=True)}
    assert_source(rows,12,7)  # Camera007 is a legitimate TRAIN texture for another target.
    import pytest
    with pytest.raises(ValueError):assert_source(rows,7,7)
    with pytest.raises(ValueError):assert_source(rows,7,12)
