"""Phase-shifted dynamic video candidate with a required nine-time RGB gate."""
from copy import deepcopy
from pathlib import Path
import argparse
from joint_temporal_texture import CALIBRATION, read, sha, atomic_json
from render_smooth_temporal_mesh_video import verify_request
from probe_temporal_camera_phase import PARENT, FRAMES, shifted_camera

OUTPUT=Path('/mnt/data/dec5_phase30_dynamic_150')


def initialize(output):
    parent=verify_request(PARENT);request=deepcopy(parent);cal=read(CALIBRATION)
    for i,r in enumerate(request['inventory']):
        r['camera']=shifted_camera(parent['inventory'],i,30,cal)
    request['recipe'].update(camera_path_variant='elevated_periodic_phase_plus30',camera_phase_frames=30)
    request['camera_path_report']['phase_offset_samples']=30
    for k,v in request['camera_path_report'].get('extrema_indices',{}).items():
        request['camera_path_report']['extrema_indices'][k]=(v-30)%150
    request.update(partial_diagnostic_only=False,full_video_candidate=True,required_initial_rgb_gate=FRAMES,
        original_elevated_request_sha256=sha(PARENT/'request.json'),camera_path_set_unchanged=True,
        actor_time_order_unchanged=True,additional_color_correction=False)
    request['script_hashes'].update({n:sha(Path(__file__).with_name(n)) for n in ['study_camera_phase_rgb.py','probe_temporal_camera_phase.py']})
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    atomic_json(output/'request.json',request)


def render_canary(output):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2);install(renderer);install_source_masks(renderer)
    renderer.render(output,FRAMES)


def review_panels(output):
    from PIL import Image,ImageDraw
    root=output/'initial_review';root.mkdir(exist_ok=True);rows=[]
    for f in FRAMES:
        paths=[PARENT/'frames'/f/'frame.png',output/'frames'/f/'frame.png']
        panel=Image.new('RGB',(2160,824));draw=ImageDraw.Draw(panel)
        for i,p in enumerate(paths):
            panel.paste(Image.open(p).crop((0,500,1080,1300)),(1080*i,24))
            draw.text((1080*i+3,4),['published phase','same actor time; camera phase +30'][i],fill='white')
        panel.save(root/f'{f}_head.png')
        rows.append(dict(frame=f,baseline_sha256=sha(paths[0]),candidate_sha256=sha(paths[1]),
                         comparison_sha256=sha(root/f'{f}_head.png')))
    atomic_json(root/'receipt.json',dict(rows=rows,request_sha256=sha(output/'request.json'),visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render-canary','panels'])
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'init':initialize,'render-canary':render_canary,'panels':review_panels}[a.action](a.output)
