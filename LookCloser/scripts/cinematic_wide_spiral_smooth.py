"""Same broad spiral with physical arc-length timing, avoiding a late speed spike."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import cinematic_wide_spiral as core
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request
from render_patchmatch_camera_path import normalize_frame

BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_v2')
PARENT=core.PARENT;VARIANTS=core.VARIANTS;CANARIES=core.CANARIES;CANARY_INDICES=core.CANARY_INDICES;RAW_IDS=core.RAW_IDS


def arc_length_map():
    rows,_,_=cameras('000973');lookup={r['physical_camera'][:9]:r for r in rows}
    corners=np.array([lookup[n]['transform_matrix'] for n in ['C004_B005','K004_B005','K004_D005','C004_D005']])[:,:3,3]
    end=np.array(lookup['H004_C005']['transform_matrix'])[:3,3]
    times=np.linspace(0,118,20001);_,a,r,_,settle=core.spiral_parameters(times)
    x=(-1+3.6*r*np.cos(a)+5)/8;y=(1-.9*r*np.sin(a))/2
    w=np.column_stack(((1-x)*(1-y),x*(1-y),x*y,(1-x)*y))*(1-settle[:,None])
    positions=w@corners+settle[:,None]*end
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    take=np.r_[True,np.diff(distance)>1e-12]
    return times[take],distance[take]/distance[-1]


def path_for(variant,parent):
    times,fraction=arc_length_map();original=core.spiral_parameters
    def parameters(indices):
        desired=core.timing(np.minimum(np.asarray(indices,dtype=float)/118,1))
        return original(np.interp(desired,fraction,times))
    core.spiral_parameters=parameters
    try:rows,report=core.path_for(variant,parent)
    finally:core.spiral_parameters=original
    report.update(easing='Physical arc-length resampling of the exact wide spiral; short C2 speed ramp, cruise, long deceleration to118.',
        arc_length_resampled=True,arc_length_table_samples=20001,
        previous_speed_spike_control=str(core.BASE),same_geometric_curve_as_v1=True)
    return rows,report


def initialize(dry=False):
    parent=verify_request(PARENT);previous=verify_request(core.PREVIOUS);cal=read(CALIBRATION)
    for variant in VARIANTS:
        path,report=path_for(variant,parent)
        print(variant,'angle',report['center_ray_angle_extent_degrees'],'extent ratio',report['first95_extent_ratio_vs_previous_free'],'first turn',report['first_turn_end_index'],flush=True)
        if dry:continue
        q=deepcopy(previous)
        for record,row in zip(q['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        q.update(camera_path_report=report,required_initial_rgb_gate=CANARIES,raw_render_frame_ids=RAW_IDS,
            previous_cinematic_request_sha256=sha(core.PREVIOUS/'request.json'),previous_cinematic_source=str(core.PREVIOUS),
            inherited_actor_inventory_unchanged=True,geometry_changed=False,
            initial_gate=dict(status='requires_wide_spiral_arc_length_RGB_review'),artifact_free_approval=False)
        q['recipe']['camera_path_variant']='arc_length_'+variant
        for name in ['cinematic_wide_spiral.py',Path(__file__).name]:q['script_hashes'][name]=sha(Path(__file__).with_name(name))
        root=BASE/variant;root.mkdir(parents=True,exist_ok=True);(root/'frames').mkdir(exist_ok=True)
        if (root/'request.json').exists():assert read(root/'request.json')==q
        atomic_json(root/'request.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['describe','init','screen','canary','render','motion','panels','sheets','raw_audit','audit','encode','publish','record'])
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--groups',nargs='*',type=int,default=[]);p.add_argument('--images',nargs='*',default=[]);p.add_argument('--note');a=p.parse_args()
    if a.action in ['describe','init']:initialize(a.action=='describe')
    elif a.action=='screen':
        import large_motion_camera_choices as screen
        screen.BASE=BASE;screen.VARIANTS=VARIANTS;screen.CANARY_INDICES=CANARY_INDICES;screen.screen()
    elif a.action in ['canary','render']:
        import supervise_large_motion_choices as lifecycle
        lifecycle.BASE=BASE;lifecycle.VARIANTS=VARIANTS;lifecycle.CANARIES=CANARIES
        original=lifecycle.verify_request;lifecycle.verify_request=lambda root:dict(original(root),ordered_frame_ids=RAW_IDS)
        lifecycle.supervise(canary=a.action=='canary')
    else:
        import finalize_cinematic_wide_spiral as final
        final.BASE=BASE;final.path_for=path_for;final.configure()
        for variant in ([a.variant] if a.variant else VARIANTS):
            if a.action in ['motion','publish']:getattr(final,a.action)(variant)
            elif a.action=='record':final.shared.record_review(variant,a.groups,a.images.copy(),a.note)
            else:getattr(final.shared,a.action)(variant)
