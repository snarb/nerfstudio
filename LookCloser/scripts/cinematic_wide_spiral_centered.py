"""A visibly broad first spiral turn in the interior frontal camera rig.

The path is a contracting ellipse across the calibrated rig, not a 360-degree
orbit behind the actor. All camera centers are convex train-camera mixtures.
Keep the 150 moving actor times and the previously requested actual RGB ending.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from cinematic_pushin_framed import timing
from cinematic_pushin_choices import ease
from screen_travel_camera_flight import portrait_projection

BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_v3')
PARENT=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
PREVIOUS=Path('/mnt/data/dec5_cinematic_pushin_v4/free_arc')
VARIANTS=('wide_spiral_lookat','wide_spiral_free')
CANARY_INDICES=[0,12,24,36,48,60,69,80,94,108,118,125]
CANARIES=[f'{899+2*i:06d}' for i in CANARY_INDICES]
RAW_IDS=[f'{899+2*i:06d}' for i in range(126)]


def geometric_parameters(indices):
    u=timing(np.minimum(np.asarray(indices,dtype=float)/118,1))
    angle=np.pi/4-2*np.pi*1.2*u
    radius=(1-.15*ease(u/(5/6)))*(1-ease((u-.80)/.20))
    xy=np.column_stack((0.+2.7*radius*np.cos(angle),.9*radius*np.sin(angle)))
    # Convex C..K / B..D interpolation. Blend to the exact physical H/C
    # center late; the interpolated surface center need not equal that camera.
    settle=ease((u-.80)/.20)
    xy=(1-settle[:,None])*xy
    return u,angle,radius,xy,settle


def arc_length_map():
    """Physical length of the full convex path, independent of lens/rotation."""
    from functools import lru_cache
    return _arc_length_map()

from functools import lru_cache

@lru_cache(maxsize=1)
def _arc_length_map():
    rows,_,_=cameras('000973'); lookup={r['physical_camera'][:9]:r for r in rows}
    corners=np.array([lookup[n]['transform_matrix'] for n in ['C004_B005','K004_B005','K004_D005','C004_D005']])[:,:3,3]
    end=np.array(lookup['H004_C005']['transform_matrix'])[:3,3]
    times=np.linspace(0,118,20001);_,a,r,_,settle=geometric_parameters(times)
    x=(2.7*r*np.cos(a)+5)/8;y=(1-.9*r*np.sin(a))/2
    w=np.column_stack(((1-x)*(1-y),x*(1-y),x*y,(1-x)*y))*(1-settle[:,None])
    positions=w@corners+settle[:,None]*end
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    take=np.r_[True,np.diff(distance)>1e-12]
    return times[take],distance[take]/distance[-1]

def spiral_parameters(indices):
    times,fraction=arc_length_map()
    desired=timing(np.minimum(np.asarray(indices,dtype=float)/118,1))
    return geometric_parameters(np.interp(desired,fraction,times))


def path_for(variant,parent):
    if variant not in VARIANTS:raise ValueError('Unknown spiral variant')
    rows,_,metadata=cameras('000973');meta=read(metadata);cal=read(CALIBRATION)
    lookup={r['physical_camera'][:9]:r for r in rows}
    end=lookup['H004_C005'];endpose=np.array(end['transform_matrix'])
    anchors=[lookup[n] for n in ['C004_B005','K004_B005','K004_D005','C004_D005']]
    corners=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    old_report=read(PREVIOUS/'request.json')['camera_path_report']
    target=np.array(old_report['fixed_target']);up=endpose[:3,0]
    index=np.arange(150);u,angle,radius,xy,settle=spiral_parameters(index)
    # Unsettled ellipse coordinates are used for the rectangle; the exact
    # endpoint has its own positive convex weight.
    ellipse=np.column_stack((0.+2.7*radius*np.cos(angle),.9*radius*np.sin(angle)))
    x=(ellipse[:,0]+5)/8;y=(1-ellipse[:,1])/2
    weights=np.column_stack(((1-x)*(1-y),x*(1-y),x*y,(1-x)*y))*(1-settle[:,None])
    allweights=np.column_stack((weights,settle));positions=weights@corners+settle[:,None]*endpose[:3,3]
    assert allweights.min()>=-1e-12 and np.allclose(allweights.sum(1),1)
    result=[]
    for i,position in enumerate(positions):
        row=deepcopy(end);q=min(i*126/118,126);late=float(ease((q-80)/46))
        focal=.85+.6*ease(q/95)+.45*late
        row['fl_x']*=float(focal);row['fl_y']*=float(focal);row['cx']+=400*late
        z=position-target;z/=np.linalg.norm(z);yaxis=np.cross(z,up);yaxis/=np.linalg.norm(yaxis)
        look=np.column_stack((np.cross(yaxis,z),yaxis,z));rotation=look
        if variant=='wide_spiral_free':
            # Deliberate changing composition, not a post-render crop. This
            # makes translation legible even against the absent room background.
            offset=np.array([180*np.cos(angle[i]),130*np.sin(angle[i])])*radius[i]*(1-settle[i])
            desired=np.array([row['cy'],row['w']-1-row['cx']])+offset
            native=np.array([row['w']-1-desired[1],desired[0]])
            ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1.]);ray/=np.linalg.norm(ray)
            rotation=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=position
        if i>=118:pose=endpose.copy()
        row.update(transform_matrix=pose.tolist(),physical_camera=f'{variant}_{i:05d}',rig_offset_xy=xy[i].tolist(),
            convex_weights=allweights[i].tolist(),spiral_phase_radians=float(angle[i]),spiral_radius_factor=float(radius[i]))
        row['scene_landmark_portrait_xy']=portrait_projection(target,row).tolist()
        result.append(calibration_pose(row,cal,meta))
    rawposes=np.array([r['transform_matrix'] for r in result]);rawcenters=rawposes[:,:3,3]
    hull=ConvexHull(np.array([r['transform_matrix'] for r in cal['frames'] if r['physical_camera'] not in HELD_CAMERAS])[:,:3,3])
    residual=float((rawcenters@hull.equations[:,:3].T+hull.equations[:,3]).max())
    rays=positions-target;radii=np.linalg.norm(rays,axis=1);rays/=radii[:,None]
    span=float(np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max())
    previous=read(PREVIOUS/'request.json')
    old=[normalize_frame(calibration_pose(r['camera'],cal,read(r['metadata'])),cal,meta) for r in previous['inventory']]
    oldpos=np.array([r['transform_matrix'] for r in old])[:,:3,3]
    # End-camera transverse axes: horizontal/vertical displacement in space,
    # measured independently of look-at rotations, focal length and sensor shift.
    axes=np.column_stack((endpose[:3,1],endpose[:3,0]))
    transverse=(positions-endpose[:3,3])@axes;oldtransverse=(oldpos-endpose[:3,3])@axes
    ratios=np.ptp(transverse[:95],axis=0)/np.maximum(np.ptp(oldtransverse[:95],axis=0),1e-9)
    assert residual<1e-7 and span>30 and ratios[0]>1.4 and ratios[1]>1.1,(residual,span,ratios)
    assert xy[:,0].min()>-5 and xy[:,0].max()<3 and xy[:,1].min()>-.95 and xy[:,1].max()<.95
    report=dict(variant=variant,trajectory='1.2-turn contracting ellipse across interior frontal train rig',
        fixed_target=target.tolist(),reference_metadata=str(metadata),reference_centers=positions.tolist(),
        endpoint_train_camera=end['physical_camera'],actual_endpoint_calibration_pose=calibration_pose(end,cal,meta)['transform_matrix'],
        anchors=[r['physical_camera'] for r in anchors]+[end['physical_camera']],reference_anchor_centers=np.vstack([corners,endpose[:3,3]]).tolist(),
        hold_start_index=118,hold_count=32,hold_actual_times=[f'{899+2*i:06d}' for i in range(118,150)],
        rig_parameter_min=xy.min(0).tolist(),rig_parameter_max=xy.max(0).tolist(),
        first_turn_nominal_horizontal_radius_columns=2.7,first_turn_nominal_vertical_radius_rows=.9,
        first_turn_end_index=int(np.flatnonzero(u>=5/6)[0]),first_turn_radius_at_completion=float(radius[np.flatnonzero(u>=5/6)[0]]),
        first95_transverse_extent=np.ptp(transverse[:95],axis=0).tolist(),
        previous_first95_transverse_extent=np.ptp(oldtransverse[:95],axis=0).tolist(),first95_extent_ratio_vs_previous_free=ratios.tolist(),
        center_ray_angle_extent_degrees=span,train_hull_max_residual=residual,minimum_convex_weight=float(allweights.min()),
        physical_path_length_calibration_units=float(np.linalg.norm(np.diff(rawcenters,axis=0),axis=1).sum()),
        reference_radius_start_end=[float(radii[0]),float(radii[-1])],radial_approach_fraction=float(1-radii[-1]/radii[0]),
        radial_largest_outward_step=float(np.diff(radii).max()),monotonic_radial_dolly=False,
        virtual_focal_multiplier_start_end=[.85,1.9],virtual_sensor_principal_x_shift_pixels=400,
        projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in result],axis=0).tolist(),
        image_crop=False,refiner=False,inpainting=False,actor_time_unchanged=True,
        easing='Physical arc-length timing; short C2 acceleration, cruise, long deceleration to exact H/C118',
        arc_length_resampled=True,ellipse_center_columns=0.,initial_phase_radians=float(np.pi/4),
        rejected_wider_control='/mnt/data/dec5_cinematic_wide_spiral_v2',
        boundary_tradeoff='C..K contributing columns, no A/B/M/N; actual rows strictly between B/D, no A/E. Not a 360-degree orbit behind actor.',
        presentation=dict(raw_mesh_indices=[0,125],dissolve_indices=[118,125],pure_train_indices=[126,149],
            endpoint_source='H004_C005_1210SZ',native_background_preserved=True,final_second_is_3d=False))
    return result,report


def initialize(dry=False):
    parent=verify_request(PARENT);previous=verify_request(PREVIOUS);cal=read(CALIBRATION)
    for variant in VARIANTS:
        path,report=path_for(variant,parent)
        print(variant,'angle',report['center_ray_angle_extent_degrees'],'first95 extent ratio',report['first95_extent_ratio_vs_previous_free'],flush=True)
        if dry:continue
        q=deepcopy(previous)
        for record,row in zip(q['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        q.update(camera_path_report=report,required_initial_rgb_gate=CANARIES,raw_render_frame_ids=RAW_IDS,
            previous_cinematic_request_sha256=sha(PREVIOUS/'request.json'),previous_cinematic_source=str(PREVIOUS),
            inherited_actor_inventory_unchanged=True,geometry_changed=False,
            initial_gate=dict(status='requires_broad_first_turn_clay_RGB_review'),artifact_free_approval=False)
        q['recipe']['camera_path_variant']=variant
        q['script_hashes'][Path(__file__).name]=sha(__file__)
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

