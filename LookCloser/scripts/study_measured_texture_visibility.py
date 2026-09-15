"""Matched train/moving-view control for negative measured-depth texture evidence."""
import argparse
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import load_real,project_integer
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_measured_free_surface import VIEWS,baseline
from measured_texture_visibility import reject_farther_source

ROOT=Path('/mnt/data/dec5_measured_texture_visibility_native_depth_v2')
FRAME='000995'


def current_source(renderer):
    from study_native_texture_footprint import transform
    source=transform(inspect.getsource(renderer.render_one))
    pairs={
        'np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2':"_view_quality((directions*normal).sum(-1),length,_quality_mode)",
        'np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2':"_view_quality((direction*normal[f]).sum(-1),length,_quality_mode)*angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0][:,None]",
    }
    for old,new in pairs.items():
        assert source.count(old)==1
        source=source.replace(old,new)
    return source


def run(view):
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    from wide_dynamic_camera_flight import install_source_masks
    parent=renderer.verify_request(baseline(FRAME,view))
    recipe=parent['recipe']
    assert recipe['source_incidence_power']==2 and recipe['static_registration'] is False
    assert recipe['pixel_fallback_angle_prior'] is True
    source=current_source(renderer)
    original_hash=hashlib.sha256(source.encode()).hexdigest()
    assert original_hash==install()==parent['source_quality_implementation_sha256']
    old='return q,zq,valid'
    assert source.count(old)==1
    source=source.replace(old,'return q,zq,_measured_guard(points,valid)')
    implementation=hashlib.sha256(source.encode()).hexdigest()
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME)
    actual_rows,_,_=renderer.cameras(FRAME)
    assert [r['physical_camera'] for r in rows]==[r['physical_camera'] for r in actual_rows]
    counts=[]
    def guard(points,valid):
        # Raw PM depth uses the existing integer-lattice camera convention.
        # Display RGB query UV subtracts .5 and must NOT index these maps directly.
        allowed=valid.cpu().numpy();rejected=[]
        for row,d,ok in zip(rows,depths,allowed):
            uv,z=project_integer(row,points)
            rejected.append(reject_farther_source(d,uv,z,ok))
        rejected=np.stack(rejected)
        counts.append(dict(samples=len(points),eligible_by_camera=allowed.sum(1).tolist(),
                           rejected_by_camera=rejected.sum(1).tolist()))
        return valid & ~renderer.torch.tensor(rejected,device=valid.device)
    renderer.__dict__['_measured_guard']=guard
    exec(compile(source,__file__+':measured_visibility','exec'),renderer.__dict__)
    install_source_masks(renderer)
    renderer.torch.set_num_threads(2)
    request=deepcopy(parent)
    request['inventory']=[r for r in request['inventory'] if r['frame_id']==FRAME]
    request['source_rows']=[r for r in request['source_rows'] if Path(r['source_dataset']).name==FRAME]
    request['ordered_frame_ids']=[FRAME]
    request.update(measured_texture_visibility=dict(
        mode='reject_stable_farther_measured_layer',minimum_gap=.005,relative_gap=.01,
        native_radius=2,far_tap_fraction=.8,middle_spread_fraction=.005,
        missing_depth='unknown_retained',corroboration_extra_views=0,
        graph_and_pixel_query=True,occluding_nearer_layer_not_tested=True,
        raw_depth_projection='project_integer on actual world samples; no display -.5 UV',
        depth_receipt=receipt,implementation_sha256=implementation),
        baseline_request_sha256=sha(baseline(FRAME,view)/'request.json'),
        geometry_changed=False,partial_diagnostic_only=True,full_video_candidate=False,
        artifact_free_approval=False,production_changed=False)
    for name in [Path(__file__).name,'measured_texture_visibility.py','carve_patchmatch_mesh_free_space.py','study_confidence_depth_prior.py']:
        request['script_hashes'][name]=sha(Path(__file__).with_name(name))
    out=ROOT/FRAME/view;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    atomic_json(out/'request.json',request)
    renderer.render(out,[FRAME])
    atomic_json(out/'guard_audit.json',dict(request_sha256=sha(out/'request.json'),
        frame_complete_sha256=sha(out/'frames'/FRAME/'complete.json'),query_batches=counts,
        source_camera_order=[r['physical_camera'] for r in rows],
        original_implementation_sha256=original_hash,guarded_implementation_sha256=implementation,
        geometry_changed=False,visual_status='pending',script_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',choices=VIEWS,required=True)
    run(p.parse_args().view)
