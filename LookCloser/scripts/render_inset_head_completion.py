"""Matched production/inferred-shell RGB controls, with unchanged texture masks."""
import argparse
from copy import deepcopy
from pathlib import Path
from joint_temporal_texture import read, sha, atomic_json, cameras
from probe_inset_head_completion import ROOT, MOVIE, FRAMES
from study_confidence_depth_prior import REGIONS


def render(frame):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    geometry=ROOT/frame/'guarded';g=read(geometry/'result.json')
    assert g['native_free_space_guard_passed'] and sha(geometry/'mesh.ply')==g['hashes']['mesh.ply']
    rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
    for view in ['moving','native_unmasked']:
        for variant in ['baseline','completion']:
            if view=='moving' and variant=='baseline':continue
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
            if view=='native_unmasked':
                target=deepcopy(next(r for r in rows if r['physical_camera']==name))
                target['physical_camera']='diagnostic_unmasked_target_'+name
                target['reference_physical_camera']=name;entry['camera']=target
            if variant=='completion':entry.update(mesh=str(geometry/'mesh.ply'),mesh_sha256=sha(geometry/'mesh.ply'))
            q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=variant=='completion',
                inset_shell_variant=variant,source_quality_implementation_sha256=implementation,
                inset_guard_request_sha256=sha(geometry/'request.json'),inset_guard_result_sha256=sha(geometry/'result.json'),
                native_target_mask_disabled=view=='native_unmasked',texture_source_masks_unchanged=True,
                actual_native_physical_camera=name if view=='native_unmasked' else None,
                inferred_not_measured_geometry=True,observed_neighborhood_certificates=False)
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            dest=ROOT/'rgb'/frame/view/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            if (dest/'request.json').exists() and read(dest/'request.json')!=q:raise ValueError('Changed inset RGB request')
            atomic_json(dest/'request.json',q);engine.render(dest,[frame])


def review():
    # Same matched depth/RGB counts, native GT transform, and crop definitions.
    import render_measured_head_mask_completion as helper
    original_panel=helper.panel
    def labelled_panel(path,images,labels,box):
        return original_panel(path,images,[x.replace('measured-mask geometry','inferred inset shell') for x in labels],box)
    helper.panel=labelled_panel
    helper.ROOT=ROOT;helper.review()
    record=read(ROOT/'review/result.json')
    record['review_adapter_sha256']=sha(__file__)
    record['candidate']='inferred radial inset shell, not measured-mask-only geometry'
    atomic_json(ROOT/'review/result.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review'])
    p.add_argument('--frame',choices=FRAMES);a=p.parse_args()
    if a.action=='render':
        if a.frame is None:p.error('render requires --frame')
        render(a.frame)
    else:review()
