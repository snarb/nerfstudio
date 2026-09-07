import json
from argparse import Namespace
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_patchmatch_camera_path import calibration_path_intervals
from run_patchmatch_tsdf_temporal_flythrough import (
    CALIBRATION,
    EXISTING_ROOT,
    FRAME_COUNT,
    PATH_ANCHORS,
    PATH_INTERVALS,
    adopted_geometry,
    claim,
    grid_position,
    quarantine_path,
    run_logged,
    validate_outer_path,
)
from run_colmap_patchmatch_tsdf_geometry_worker import commands as geometry_commands
from recover_patchmatch_tsdf_temporal_small_components import small_component_policy
from run_colmap_patchmatch_tsdf_geometry_component_gate import component_policy
from run_patchmatch_tsdf_temporal_component_gate_controller import component_geometry_is_eligible


def calibrated_camera(name: str, index: int) -> dict:
    matrix=np.eye(4)
    matrix[0,3]=index
    return {
        'physical_camera':name,'transform_matrix':matrix.tolist(),
        'fl_x':100.,'fl_y':100.,'cx':40.,'cy':30.,'w':80,'h':60,
    }


def test_dec5_outer_path_has_150_closed_ordered_poses():
    calibration={'frames':[calibrated_camera(name,index) for index,name in enumerate(dict.fromkeys(PATH_ANCHORS))]}
    path=calibration_path_intervals(calibration,list(PATH_ANCHORS),list(PATH_INTERVALS))
    validate_outer_path(calibration,path)
    assert len(path)==FRAME_COUNT
    assert sum(PATH_INTERVALS)==FRAME_COUNT-1
    np.testing.assert_allclose(path[0]['transform_matrix'],path[-1]['transform_matrix'],rtol=0,atol=1e-12)


def test_every_outer_anchor_is_at_least_two_grid_steps_from_center():
    center=grid_position('H004_C005_center')
    for anchor in PATH_ANCHORS:
        position=grid_position(anchor)
        assert max(abs(position[0]-center[0]),abs(position[1]-center[1]))>=2


def test_monitoring_failure_is_nonfatal_and_does_not_orphan_child(tmp_path):
    checks=[]
    def broken_check(process):
        checks.append(process.pid)
        raise RuntimeError('transient monitor failure')
    run_logged(
        [sys.executable,'-c','print("done")'],tmp_path/'run.log',env=dict(os.environ),check=broken_check
    )
    assert checks
    assert 'monitor_warning=' in (tmp_path/'run.log').read_text()


def test_claim_reclaims_dead_stale_owner_and_quarantines_receipt(tmp_path):
    for name in ('frames','claims','quarantine'):
        (tmp_path/name).mkdir()
    stale=tmp_path/'claims/000899'
    stale.mkdir()
    (stale/'claim.json').write_text(json.dumps({
        'pid':99999999,'owner_hostname':os.uname().nodename,
        'claimed_at':'2000-01-01T00:00:00+00:00',
    }))
    args=Namespace(output_root=tmp_path,retry_failed=False,reclaim_stale_hours=1.0)
    new=claim(args,{'request_sha256':'abc'},'000899','local')
    assert new == stale
    assert json.loads((new/'claim.json').read_text())['request_sha256']=='abc'
    assert any((tmp_path/'quarantine').iterdir())


def test_claim_does_not_steal_live_owner(tmp_path):
    for name in ('frames','claims','quarantine'):
        (tmp_path/name).mkdir()
    live=tmp_path/'claims/000899'
    live.mkdir()
    (live/'claim.json').write_text(json.dumps({
        'pid':os.getpid(),'owner_hostname':os.uname().nodename,
        'claimed_at':datetime.now(timezone.utc).isoformat(),
    }))
    args=Namespace(output_root=tmp_path,retry_failed=True,reclaim_stale_hours=0.0)
    assert claim(args,{'request_sha256':'abc'},'000899','local') is None


def test_cross_device_quarantine_keeps_scratch_native_and_writes_receipt(tmp_path,monkeypatch):
    source_root=tmp_path/'scratch'
    source_root.mkdir()
    source=source_root/'attempt_0'
    source.mkdir()
    (source/'payload').write_text('kept')
    output=tmp_path/'output'
    (output/'quarantine').mkdir(parents=True)
    real_stat=Path.stat
    def different_devices(path):
        result=real_stat(path)
        if Path(path)==output/'quarantine':
            values=list(result)
            values[2]=result.st_dev+1
            return os.stat_result(values)
        return result
    monkeypatch.setattr(Path,'stat',different_devices)
    destination=quarantine_path(Namespace(output_root=output),source,'frame.attempt0.scratch')
    assert destination.parent==source_root/'.quarantine'
    assert (destination/'payload').read_text()=='kept'
    assert list((output/'quarantine').glob('*.json'))


def test_first50_adoption_validates_full_retained_manifest(tmp_path):
    args=Namespace(existing_root=EXISTING_ROOT)
    mesh,metadata,result=adopted_geometry(args,'000899',tmp_path)
    assert mesh.is_file() and metadata.is_file()
    assert result['mesh_components']==1
    assert result['source_campaign_request_sha256']


def test_geometry_only_worker_keeps_frozen_patchmatch_and_tsdf_arguments(tmp_path):
    rows={name:command for name,command,_,_ in geometry_commands(tmp_path,tmp_path/'out',Path('/bin/true'),'0')}
    assert set(rows)=={
        'export-fixed-model','undistort','patch-config','patchmatch-photometric',
        'patchmatch-geometric','import-depth','fuse-tsdf',
    }
    geometric=rows['patchmatch-geometric']
    assert geometric[geometric.index('--PatchMatchStereo.filter_min_num_consistent')+1]=='2'
    assert geometric[geometric.index('--PatchMatchStereo.geom_consistency_max_cost')+1]=='6.0'
    assert geometric[geometric.index('--PatchMatchStereo.filter_geom_consistency_max_cost')+1]=='2.0'
    fuse=rows['fuse-tsdf']
    assert fuse[fuse.index('--voxel-length')+1]=='0.0005'
    assert fuse[fuse.index('--tensor-weight-threshold')+1]=='2.0'


def test_small_component_amendment_is_global_and_conservative():
    policy=small_component_policy([153_941,437])
    assert policy['geometry_changed'] is False
    assert policy['secondary_triangle_total']==437
    assert policy['secondary_to_largest_fraction'] < .005


def test_small_component_amendment_rejects_material_geometry():
    import pytest
    with pytest.raises(ValueError,match='exceed'):
        small_component_policy([100_000,1001])
    with pytest.raises(ValueError,match='exceed'):
        small_component_policy([100_000,600],max_secondary_triangles=1000,max_secondary_fraction=.005)


def test_pending_visual_component_gate_accepts_bounded_articulated_surface():
    policy=component_policy([130_371,15_722,1_323])
    policy.update({'gate_script_sha256':'abc','visual_status':'pending'})
    geometry={
        'validation_status':'pass_pending_multicomponent_visual_gate',
        'mesh_components':3,'mesh_component_triangles':[130_371,15_722,1_323],
        'mesh_triangles':147_416,'quality_gate_amendment':policy,
    }
    assert component_geometry_is_eligible(geometry,'abc')
    assert policy['automatic_geometry_retry'] is False


def test_pending_visual_component_gate_rejects_large_secondary_surface():
    import pytest
    with pytest.raises(ValueError,match='exceed'):
        component_policy([100_000,40_001])
    with pytest.raises(ValueError,match='exceed'):
        component_policy([100_000,25_001],max_secondary_triangles=40_000,max_secondary_fraction=.25)
