"""Explicit late optical/sensor-window framing, unchanged physical v2 paths.

Two whole-head endings and two beauty endings with intentionally cropped upper
hair. No post-render crop, synthetic refinement or claim of extra dolly motion.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import cinematic_pushin_framed as previous
from cinematic_pushin_choices import ease
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION
from render_patchmatch_camera_path import normalize_frame
from render_smooth_temporal_mesh_video import verify_request
from screen_travel_camera_flight import portrait_projection

BASE=Path('/mnt/data/dec5_cinematic_pushin_v3');PARENT=previous.PARENT
VARIANTS=previous.VARIANTS;CANARIES=previous.CANARIES;CANARY_INDICES=previous.CANARY_INDICES;timing=previous.timing

def path_for(variant,parent):
    rows,report=previous.path_for(variant,parent);beauty=variant in ['free_arc','soft_diagonal'];boost=.45 if beauty else .20
    for i,row in enumerate(rows):
        late=float(ease((i-80)/46));base=.85+.60*ease(i/95)
        row['fl_x']*=float((base+boost*late)/base);row['fl_y']*=float((base+boost*late)/base)
        row['cx']+=400*late if beauty else 0
        normalized=normalize_frame(row,read(CALIBRATION),read(report['reference_metadata']))
        row['scene_landmark_portrait_xy']=portrait_projection(np.array(report['fixed_target']),normalized).tolist()
    report.update(virtual_focal_multiplier_start_end=[.85,1.9 if beauty else 1.65],
        focal_easing='v2 lens0.85..1.45 over0..95 plus late quintic boost80..126; fixed last24',
        virtual_sensor_principal_x_shift_pixels=400 if beauty else 0,virtual_sensor_shift_easing='quintic80..126, fixed126..149',
        endpoint_framing='beauty face/eyes; upper hair intentionally outside virtual sensor window' if beauty else 'whole-head close portrait',
        post_render_crop=False,physical_path_identical_to_v2=True,previous_rgb_gate_root=str(previous.BASE),
        projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in rows],axis=0).tolist())
    return rows,report

def initialize():
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        rows,report=path_for(variant,parent);req=deepcopy(verify_request(previous.BASE/variant))
        for record,row in zip(req['inventory'],rows):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        req['camera_path_report']=report;req['recipe']['camera_path_variant']='cinematic_v3_'+variant
        req['script_hashes'][Path(__file__).name]=sha(__file__)
        root=BASE/variant;root.mkdir(parents=True,exist_ok=True);(root/'frames').mkdir(exist_ok=True)
        if (root/'request.json').exists():assert read(root/'request.json')==req
        atomic_json(root/'request.json',req);print(variant,report['endpoint_framing'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','screen']);a=p.parse_args()
    if a.action=='init':initialize()
    else:
        import large_motion_camera_choices as renderer
        renderer.BASE=BASE;renderer.VARIANTS=VARIANTS;renderer.CANARY_INDICES=CANARY_INDICES;renderer.screen()
