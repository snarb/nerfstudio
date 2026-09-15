"""C2 short-acceleration cinematic timing and earlier disclosed lens tightening.

Preserves failed v1 clay requests. Fourth shot trades radial extent for a
broader calibrated arc; actual movement is never inferred from lens change.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import cinematic_pushin_choices as core
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

BASE=Path('/mnt/data/dec5_cinematic_pushin_v2');PARENT=core.PARENT
VARIANTS=core.VARIANTS;CANARIES=core.CANARIES;CANARY_INDICES=core.CANARY_INDICES
quintic=core.ease

def timing(x):
    t=np.clip(np.asarray(x)*126,0,126)
    def integral(s):return 2.5*s**4-3*s**5+s**6
    s=np.clip((t-94)/32,0,1)
    return np.where(t<10,10*integral(t/10),np.where(t<94,t-5,89+32*(s-integral(s))))/105

def path_for(variant,parent):
    original_ease=core.ease;original_controls=deepcopy(core.CONTROLS)
    core.ease=timing
    core.CONTROLS['soft_diagonal']=[
        [('J004_B005',.30),('L004_C005',.70)],
        [('J004_B005',.24),('L004_C005',.35),('H004_C005',.41)],
        [('J004_B005',.025),('H004_C005',.975)], [('H004_C005',1.)]]
    try:rows,report=core.path_for(variant,parent)
    finally:core.ease=original_ease;core.CONTROLS=original_controls
    for i,row in enumerate(rows):
        previous=.85+.60*timing(i/126)
        focal=.85+.60*quintic(i/95)
        row['fl_x']*=focal/previous;row['fl_y']*=focal/previous
        normalized=normalize_frame(row,read(CALIBRATION),read(report['reference_metadata']))
        row['scene_landmark_portrait_xy']=portrait_projection(np.array(report['fixed_target']),normalized).tolist()
    report.update(easing='Integrated velocity: quintic 10-frame acceleration, 84-frame cruise, 32-frame quintic deceleration; exact rest126..149',
        focal_easing='Independent disclosed 0.85..1.45 virtual focal ramp, quintic over indices0..95, constant thereafter',
        previous_failed_clay_root=str(core.BASE),boundary_tradeoff='Contributing columns D..L (no A/B/M/N). Actual y strictly inside B/D; no A/E. Fourth arc has smaller true radial approach.',
        gesture_tradeoff='Fourth arc maximizes restrained angular sweep at cost of radial dolly; not equal physical push to other three')
    report['projected_landmark_extent_pixels']=np.ptp([r['scene_landmark_portrait_xy'] for r in rows],axis=0).tolist()
    return rows,report

def initialize(dry=False):
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        rows,report=path_for(variant,parent);print(variant,report,flush=True)
        if dry:continue
        request=deepcopy(parent)
        for record,row in zip(request['inventory'],rows):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        request['camera_path_report']=report;request['recipe'].update(camera_path_variant='cinematic_v2_'+variant,camera_periodic=False)
        request.update(partial_diagnostic_only=False,full_video_candidate=True,artifact_free_approval=False,inherited_actor_inventory_unchanged=True,
            inherited_camera_and_actor_inventory_unchanged=False,required_initial_rgb_gate=CANARIES,
            initial_gate=dict(status='requires_clay_rgb_and_final_hold_review'),camera_workaround_parent=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json')))
        for name in ['cinematic_pushin_choices.py',Path(__file__).name]:request['script_hashes'][name]=sha(Path(__file__).with_name(name))
        out=BASE/variant;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==request
        atomic_json(out/'request.json',request)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['describe','init','screen']);a=p.parse_args()
    if a.action in ['describe','init']:initialize(a.action=='describe')
    else:
        import large_motion_camera_choices as renderer
        renderer.BASE=BASE;renderer.VARIANTS=VARIANTS;renderer.CANARY_INDICES=CANARY_INDICES;renderer.screen()
