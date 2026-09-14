"""Opt-in matched RGB control: apply target-angle prior before incidence culling.

Only the centroid source-label admission order changes. Pixel fallback,
visibility, geometry, camera response, sampling and graph settings stay frozen.
The original renderer file and defaults are not edited.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import inspect
import ast
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
import render_smooth_temporal_mesh_video as renderer
from temporal_texture_view_prior import angle_weights
from wide_dynamic_camera_flight import install_source_masks

BASE=Path('/mnt/data/dec5_forearm_color_qualified_curved')


def early_quality(quality,weights):
    q=np.asarray(quality)*np.asarray(weights)[:,None]
    return np.where(q>=q.max(0)*.12,q,0)


def transform_source(source):
    old='quality=np.where(quality>=quality.max(0)*.12,quality,0)'
    new="quality=_early_quality(quality,angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0])"
    if source.count(old)!=1:raise ValueError('Renderer source no longer matches tested admission transform')
    return source.replace(old,new)


def install():
    # The legacy renderer has no admission hook. Compile a single verified local
    # statement substitution in this isolated process; bind its exact source hash.
    source=transform_source(inspect.getsource(renderer.render_one))
    renderer.__dict__.update(_early_quality=early_quality,angle_weights=angle_weights)
    exec(compile(source,__file__+':admission_control','exec'),renderer.__dict__)
    install_source_masks(renderer)
    return hashlib.sha256(source.encode()).hexdigest()


def run(root,frame):
    renderer.torch.set_num_threads(2);implementation_sha=install()
    for view in ['moving','H004_A005_1210M6']:
        baseline=BASE/'rgb'/frame/view/'guarded';parent=read(baseline/'request.json');ancestry={}
        for name,digest in parent['script_hashes'].items():
            current=Path(__file__).with_name(name)
            if sha(current)==digest:continue
            archived=BASE/'config'/name
            if name!='study_forearm_production_delta.py' or not archived.exists() or sha(archived)!=digest:
                raise ValueError('Changed parent execution dependency: '+name)
            # That controller added a preparation-only plane flag. Its render
            # function is identical; archive/hash the old producer explicitly.
            def body(path):
                tree=ast.parse(path.read_text())
                return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render'))
            if body(archived)!=body(current):raise ValueError('Parent controller render function changed')
            ancestry[name]=dict(original_sha256=digest,archive=str(archived),current_sha256=sha(current),render_ast_equal=True)
            parent['script_hashes'][name]=sha(current)
        target=root/'rgb'/frame/view/'guarded';target.mkdir(parents=True,exist_ok=True);(target/'frames').mkdir(exist_ok=True)
        request=deepcopy(parent);request['recipe']['texture_source_prior']='target_angle_before_incidence_clip'
        request.update(admission_transform_sha256=implementation_sha,matched_late_prior_request_sha256=sha(baseline/'request.json'),
            geometry_changed=False,pixel_fallback_changed=False,full_video_candidate=False,controller_code_ancestry=ancestry)
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        if (target/'request.json').exists() and read(target/'request.json')!=request:raise ValueError('Frozen early-prior mismatch')
        atomic_json(target/'request.json',request);renderer.render(target,[frame])
        # The scorer's production baseline remains the unchanged production mesh,
        # while the visual pair below isolates texture on the same repaired mesh.
        link=root/'rgb'/frame/view/'baseline';source=BASE/'rgb'/frame/view/'baseline'
        if link.exists():
            if not link.is_symlink() or link.resolve()!=source.resolve():raise ValueError('Unexpected baseline link')
        else:link.symlink_to(source,target_is_directory=True)
        panel=Image.new('RGB',(1080,870));draw=ImageDraw.Draw(panel)
        for i,(label,folder) in enumerate([('same mesh: late angle prior',baseline),('same mesh: early angle prior',target)]):
            panel.paste(Image.open(folder/'frames'/frame/'frame.png').crop((0,1080,540,1920)),(i*540,30));draw.text((i*540+4,5),label,fill='white')
        out=root/frame;out.mkdir(exist_ok=True);panel.save(out/(view+'_matched_native.png'))
    print(frame,'matched early texture control complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_early_texture_prior'))
    p.add_argument('--frame',required=True,choices=['001029','001033','001037']);a=p.parse_args();run(a.root,a.frame)
