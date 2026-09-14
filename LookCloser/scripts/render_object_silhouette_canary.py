"""Matched target-view renders for the independent object-silhouette experiment."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from test_temporal_object_silhouette import PARENT
import render_smooth_temporal_mesh_video as renderer


def run(output,experiment,frames):
    request=deepcopy(renderer.verify_request(PARENT));request['recipe']['train_foreground_guard']=False
    request['purpose']='matched local object-silhouette canary, not a complete dynamic delivery'
    request['inherited_foreground_masks']={'parent':str(PARENT),'request_sha256':sha(PARENT/'request.json')}
    for record in request['inventory']:
        if record['frame_id'] in frames:
            root=experiment/record['frame_id'];result=read(root/'result.json')
            if result['status']!='candidate_requires_render_and_review':raise ValueError('No object candidate')
            record['mesh']=str(root/'mesh.ply');record['mesh_sha256']=sha(root/'mesh.ply')
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['script_hashes']['test_temporal_object_silhouette.py']=sha(Path(__file__).with_name('test_temporal_object_silhouette.py'))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Canary request mismatch')
    atomic_json(output/'request.json',request)
    from temporal_texture_view_prior import install
    install(renderer);original_render=renderer.render_one;original_depth=renderer.camera_depth
    def render(root,record,source):
        guard=PARENT/'foreground_guard'/record['frame_id'];masks=np.load(guard/'masks.npz')['masks']
        lookup=dict(zip(read(guard/'cameras.json'),masks))
        def depth(scene,row):
            d,ids,b=original_depth(scene,row)
            if row['physical_camera'] in lookup:d=d.copy();d[lookup[row['physical_camera']]==0]=np.inf
            return d,ids,b
        renderer.camera_depth=depth
        try:return original_render(root,record,source)
        finally:renderer.camera_depth=original_depth
    renderer.render_one=render;renderer.render(output,frames)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_dynamic_object_canary'))
    p.add_argument('--experiment',type=Path,default=Path('/mnt/data/dec5_dynamic_object_silhouette'))
    p.add_argument('--frames',nargs='+',default=['000973','000975']);a=p.parse_args();renderer.torch.set_num_threads(2);run(a.output,a.experiment,a.frames)
