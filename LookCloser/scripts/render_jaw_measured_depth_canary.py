"""Matched production-wrapper RGB of the two observed-depth notch canaries."""
from pathlib import Path
from copy import deepcopy
import argparse
from joint_temporal_texture import read, sha, atomic_json
from study_jaw_boundary_notches import PARENT, PHASE


def run(root, frame):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2);install(renderer);install_source_masks(renderer)
    for label,parent in [('old_moving',PARENT),('phase_moving',PHASE)]:
        baseline=renderer.verify_request(parent)
        for variant in ['baseline','guarded']:
            output=root/'rgb'/frame/label/variant
            request=deepcopy(baseline)
            request['inventory']=[r for r in request['inventory'] if r['frame_id'] == frame]
            request.update(partial_diagnostic_only=True,full_video_candidate=False,
                diagnostic_parent_request_sha256=sha(parent/'request.json'),jaw_control_variant=variant)
            # Isolated one-time canary; the same code/rule is used for both times.
            if variant=='guarded':
                for row in request['inventory']:
                    folder=root/'guarded'/row['frame_id']; result=read(folder/'result.json')
                    if not result['observed_free_space_guard_passed']:raise ValueError('Failed observed-depth guard')
                    row['mesh']=str(folder/'mesh.ply');row['mesh_sha256']=sha(folder/'mesh.ply')
                    if row['mesh_sha256']!=result['mesh_sha256']:raise ValueError('Changed candidate')
                    row['jaw_guard_result_sha256']=sha(folder/'result.json')
            request['script_hashes'][Path(__file__).name]=sha(__file__)
            output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
            if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Changed canary request')
            atomic_json(output/'request.json',request);renderer.render(output,[frame])
    from PIL import Image,ImageDraw
    folder=root/'rgb_review'/frame;folder.mkdir(parents=True,exist_ok=True);panels=[]
    for label in ['old_moving','phase_moving']:
        panel=Image.new('RGB',(1600,824));draw=ImageDraw.Draw(panel)
        for i,variant in enumerate(['baseline','guarded']):
            path=root/'rgb'/frame/label/variant/'frames'/frame/'frame.png'
            panel.paste(Image.open(path).crop((170,450,970,1250)),(800*i,24));draw.text((800*i+5,5),variant,fill='white')
        dest=folder/f'{label}_native.png';panel.save(dest);panels.append(dict(path=str(dest),sha256=sha(dest)))
    atomic_json(folder/'receipt.json',dict(panels=panels,visual_status='pending',same_camera_and_actor=True,
        production_wrappers=['temporal_texture_view_prior','wide_dynamic_camera_flight.install_source_masks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_measured_depth'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);args=p.parse_args();run(args.root,args.frame)
