"""Opt-in calibrated-hull dolly/arc shots with a one-second real-pose hold.

No geometry, source selection, radiometry or actor-time changes. Physical
camera motion and the disclosed virtual focal ramp are audited separately.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

PARENT=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
BASE=Path('/mnt/data/dec5_cinematic_pushin_v1')
VARIANTS=('locked_arc','free_arc','rising_arc','soft_diagonal')
CANARY_INDICES=[0,20,48,54,65,69,73,100,112,126,147,149]
CANARIES=[f'{899+2*i:06d}' for i in CANARY_INDICES]
# Every control is a convex mixture of real train centers. B/D anchors may
# contribute, but their row coordinates are never reached by actual samples.
CONTROLS={
 'locked_arc': [[('J004_B005',.8),('D004_D005',.2)], [('J004_B005',.6),('F004_D005',.4)], [('J004_B005',.3),('H004_C005',.7)], [('H004_C005',1.)]],
 'free_arc': [[('J004_B005',.8),('D004_D005',.2)], [('J004_B005',.6),('F004_D005',.4)], [('J004_B005',.3),('H004_C005',.7)], [('H004_C005',1.)]],
 'rising_arc': [[('J004_B005',.65),('K004_D005',.35)], [('J004_B005',.6),('I004_D005',.4)], [('J004_B005',.25),('H004_C005',.75)], [('H004_C005',1.)]],
 'soft_diagonal': [[('J004_B005',.65),('E004_D005',.35)], [('J004_B005',.55),('G004_D005',.45)], [('J004_B005',.2),('H004_C005',.8)], [('H004_C005',1.)]],
}

def ease(t):
    t=np.clip(t,0,1);return t**3*(10-15*t+6*t*t)

def path_for(variant,parent):
    cal=read(CALIBRATION);rows,_,metapath=cameras('000973');meta=read(metapath)
    lookup={r['physical_camera'][:9]:r for r in rows};end=lookup['H004_C005'];endpose=np.array(end['transform_matrix'])
    previous_target=np.array(parent['camera_path_report']['fixed_target'])
    target=endpose[:3,3]-np.linalg.norm(endpose[:3,3]-previous_target)*endpose[:3,2];up=endpose[:3,0]
    control=[];rig=[]
    for spec in CONTROLS[variant]:
        assert abs(sum(w for _,w in spec)-1)<1e-10
        control.append(sum(w*np.array(lookup[n]['transform_matrix'])[:3,3] for n,w in spec))
        rig.append(sum(w*np.array([ord(n[0])-ord('H'),ord('C')-ord(n[5])]) for n,w in spec))
    control=np.array(control);rig=np.array(rig);index=np.arange(150);u=ease(index/126.)
    b=np.column_stack(((1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3));positions=b@control;xy=b@rig
    # Optical zoom is explicit, not mislabeled dolly: close-up on the exact
    # endpoint train extrinsic needs a tighter virtual sensor field of view.
    focal=.85+.60*ease(index/126.)
    result=[];projection=[];look_errors=[]
    for i,position in enumerate(positions):
        row=deepcopy(end);row['fl_x']*=focal[i];row['fl_y']*=focal[i]
        z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y);look=np.column_stack((np.cross(y,z),y,z))
        if variant=='locked_arc':rotation=look
        else:
            # One restrained compositional gesture, tapering C2 to the exact
            # physical camera orientation, without a per-frame tracked crop.
            amp={'free_arc':(100.,90.),'rising_arc':(-70.,100.),'soft_diagonal':(75.,-60.)}[variant]
            gesture=np.sin(np.pi*u[i])**2
            ray=np.array([-amp[1]*gesture/row['fl_x'],-amp[0]*gesture/row['fl_y'],-1.]);ray/=np.linalg.norm(ray)
            rotation=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=position
        if i>=126:pose=endpose.copy()
        row.update(transform_matrix=pose.tolist(),physical_camera=f'cinematic_{variant}_{i:05d}',rig_offset_xy=xy[i].tolist())
        projected=portrait_projection(target,row);row['scene_landmark_portrait_xy']=projected.tolist();projection.append(projected)
        look_errors.append(float(np.degrees(np.arccos(np.clip(np.dot(-pose[:3,2],(target-position)/np.linalg.norm(target-position)),-1,1)))))
        result.append(calibration_pose(row,cal,meta))
    radius=np.linalg.norm(positions-target,axis=1);rays=(positions-target)/radius[:,None]
    angle=float(np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max());rawpos=np.array([r['transform_matrix'] for r in result])[:,:3,3]
    hull=ConvexHull(np.array([r['transform_matrix'] for r in read(CALIBRATION)['frames'] if r['physical_camera'] not in HELD_CAMERAS])[:,:3,3])
    residual=float((rawpos@hull.equations[:,:3].T+hull.equations[:,3]).max())
    endpoint=calibration_pose(end,cal,meta)
    np.testing.assert_allclose(np.array([r['transform_matrix'] for r in result[126:]]),np.repeat(np.array(endpoint['transform_matrix'])[None],24,axis=0),atol=1e-10)
    assert residual<1e-7 and xy[:,1].min()>-.95 and xy[:,1].max()<.95
    report=dict(variant=variant,endpoint_train_camera=end['physical_camera'],fixed_target=target.tolist(),
        control_mixtures=CONTROLS[variant],reference_control_centers=control.tolist(),reference_metadata=str(metapath),
        reference_radius_start_end=[float(radius[0]),float(radius[-1])],radial_approach_fraction=float(1-radius[-1]/radius[0]),
        radial_largest_outward_step=float(np.diff(radius).max()),center_ray_angle_extent_degrees=angle,
        physical_path_length_calibration_units=float(np.linalg.norm(np.diff(rawpos,axis=0),axis=1).sum()),
        physical_endpoint_displacement_calibration_units=float(np.linalg.norm(rawpos[-1]-rawpos[0])),
        rig_parameter_min=xy.min(0).tolist(),rig_parameter_max=xy.max(0).tolist(),train_hull_max_residual=residual,
        look_at_angular_error_max_degrees=max(look_errors),projected_landmark_extent_pixels=np.ptp(projection,axis=0).tolist(),
        actual_endpoint_calibration_pose=endpoint['transform_matrix'],hold_start_index=126,hold_count=24,hold_actual_times=[f'{899+2*i:06d}' for i in range(126,150)],
        virtual_focal_multiplier_start_end=[float(focal[0]),float(focal[-1])],focal_reference='real endpoint train-camera intrinsics',
        image_crop=False,refiner=False,inpainting=False,actor_time_unchanged=True,
        easing='quintic smoothstep(i/126): continuous first and second derivatives zero at start/hold; cubic Bezier centers',
        boundary_tradeoff='Only D..K contributing columns. Actual row-coordinate strictly inside B/D; five-row rig A..E prevents avoiding both penultimate neighborhoods while moving vertically.')
    assert report['radial_approach_fraction']>.05 and angle>5 and report['radial_largest_outward_step']<1e-8,report
    return result,report

def initialize(dry=False):
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        path,report=path_for(variant,parent);print(variant,report,flush=True)
        if dry:continue
        request=deepcopy(parent)
        for record,row in zip(request['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        request['camera_path_report']=report;request['recipe'].update(camera_path_variant='cinematic_'+variant,camera_periodic=False)
        request.update(partial_diagnostic_only=False,full_video_candidate=True,artifact_free_approval=False,
            inherited_actor_inventory_unchanged=True,inherited_camera_and_actor_inventory_unchanged=False,
            required_initial_rgb_gate=CANARIES,initial_gate=dict(status='requires_clay_rgb_and_final_hold_review'),
            camera_workaround_parent=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json')))
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        out=BASE/variant;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==request
        atomic_json(out/'request.json',request)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['describe','init','screen']);a=p.parse_args()
    if a.action in ['describe','init']:initialize(a.action=='describe')
    else:
        import large_motion_camera_choices as renderer
        renderer.BASE=BASE;renderer.VARIANTS=VARIANTS;renderer.CANARY_INDICES=CANARY_INDICES;renderer.screen()
