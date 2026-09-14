"""Five-time RGB canary for a uniformly raised, still-moving camera path."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from joint_temporal_texture import atomic_json, sha, read
from render_smooth_temporal_mesh_video import verify_request
from probe_elevated_jaw_visibility import PARENT, FRAMES, camera_at_height

OUTPUT=Path('/mnt/data/dec5_elevated_jaw_rgb_plus04')


def initialize(output):
    parent=verify_request(PARENT); request=deepcopy(parent)
    for record in request['inventory']:
        record['camera']=camera_at_height(record,parent['camera_path_report'],.4)
    request['recipe']['camera_path_variant']='elevated_plus_04_rows_uniform'
    request['camera_path_report']['camera_envelope'].update(bottom=.6,top=1.35)
    request['camera_path_report']['uniform_rig_height_offset']=.4
    request.update(partial_diagnostic_only=True,canary_frames=FRAMES,
                   additional_gradient_color_correction=False,
                   source_parent_request_sha256=sha(PARENT/'request.json'))
    request['script_hashes'].update({n:sha(Path(__file__).with_name(n)) for n in ['study_elevated_jaw_rgb.py','probe_elevated_jaw_visibility.py']})
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    atomic_json(output/'request.json',request)
    # Positions are in per-frame normalized spaces: motion proof uses the path's
    # calibration-rig coordinates, not mixed normalized camera translations.
    xy=np.array([r['camera']['rig_offset_xy'] for r in request['inventory']])
    delta=np.diff(np.vstack([xy,xy[:1]]),axis=0)
    atomic_json(output/'camera_checks.json',dict(actual_source_times=len(request['inventory']),
        distinct_camera_positions=len(np.unique(xy,axis=0)),rig_min=xy.min(0).tolist(),rig_max=xy.max(0).tolist(),
        maximum_rig_step=float(np.linalg.norm(delta,axis=1).max()),
        maximum_rig_second_difference=float(np.linalg.norm(np.diff(np.vstack([delta,delta[:1]]),axis=0),axis=1).max()),
        same_horizontal_path=True,uniform_vertical_offset=.4,geometry_unchanged=True,
        fixed_intrinsics_unchanged=True,temporal_order_unchanged=True,rendered_times_requested=FRAMES,
        published_movie_replaced=False))


def render(output):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2);install(renderer);install_source_masks(renderer)
    renderer.render(output,FRAMES)


def panels(output):
    from PIL import Image,ImageDraw
    root=output/'review';root.mkdir(exist_ok=True)
    rows=[]
    for f in FRAMES:
        paths=[PARENT/'frames'/f/'frame.png',output/'frames'/f/'frame.png']
        for name,box in [('jaw',(350,930,710,1280)),('head',(0,500,1080,1300))]:
            w,h=box[2]-box[0],box[3]-box[1]
            panel=Image.new('RGB',(w*2,h+24));draw=ImageDraw.Draw(panel)
            for i,p in enumerate(paths):
                panel.paste(Image.open(p).crop(box),(i*w,24));draw.text((i*w+3,4),['published camera','same frame, camera +0.4 rows'][i],fill='white')
            panel.save(root/f'{f}_{name}.png')
        rows.append(dict(frame=f,baseline_sha256=sha(paths[0]),candidate_sha256=sha(paths[1]),
            jaw_comparison_sha256=sha(root/f'{f}_jaw.png'),head_comparison_sha256=sha(root/f'{f}_head.png')))
    atomic_json(root/'receipt.json',dict(rows=rows,request_sha256=sha(output/'request.json'),visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render','panels'])
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'init':initialize,'render':render,'panels':panels}[a.action](a.output)
