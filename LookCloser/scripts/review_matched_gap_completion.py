"""Render/review the identical-Poisson counterfactual to centroid-gap removal."""
import argparse
from copy import deepcopy
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from render_close_boundary_completion import ROOT,MOVIE
from review_jaw_repair_transfer import verified_image,panel


def render(frame):
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    from pathlib import Path
    implementation=install();renderer.torch.set_num_threads(2)
    parent=renderer.verify_request(MOVIE);rows,_,_=cameras(frame)
    mesh=ROOT/frame/'matched_gap/mesh.ply';result=read(mesh.parent/'result.json')
    if sha(mesh)!=result['hashes']['mesh.ply'] or not result['observed_guard_passed']:raise ValueError('Unverified control')
    for view in ['moving','F004_E005_1210FP']:
        q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
        if view!='moving':entry['camera']=next(r for r in rows if r['physical_camera']==view)
        entry.update(mesh=str(mesh),mesh_sha256=sha(mesh))
        q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,
            matched_centroid_gap_control_sha256=sha(mesh.parent/'request.json'),source_quality_implementation_sha256=implementation)
        q['script_hashes'][Path(__file__).name]=sha(__file__)
        out=ROOT/frame/'rgb'/view/'matched_gap';out.mkdir(exist_ok=False);(out/'frames').mkdir();atomic_json(out/'request.json',q)
        renderer.render(out,[frame])


def review(frame):
    out=ROOT/frame/'matched_review';out.mkdir(exist_ok=False);records=[]
    for view in ['moving','F004_E005_1210FP']:
        roots=[ROOT/frame/'rgb'/view/variant for variant in ['matched_gap','close_boundary']]
        images=[];depths=[]
        for root in roots:
            im,_=verified_image(root,frame);images.append(im)
            depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        labels=['same raw surface: minimum gap','same raw surface: no minimum gap']
        panel(out/(view+'_head.png'),images,labels,(170,450,970,1350))
        box=(570,1070,780,1240) if view!='moving' else (370,1080,670,1320)
        panel(out/(view+'_detail.png'),images,labels,box)
        # Full-image counts below are ray-support diagnostics, not image quality.
        records.append(dict(view=view,changed_rgb_pixels=int(np.any(images[0]!=images[1],axis=2).sum()),
            depth_gained=int(((depths[0]==0)&(depths[1]>0)).sum()),depth_lost=int(((depths[0]>0)&(depths[1]==0)).sum()),
            inputs={str(root/'frames'/frame/'frame.png'):sha(root/'frames'/frame/'frame.png') for root in roots}))
    atomic_json(out/'result.json',dict(records=records,visual_status='pending',quality_metrics_computed=False,script_sha256=sha(__file__)))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);p.add_argument('--frame',choices=['001193','001195'],required=True)
    a=p.parse_args();{'render':render,'review':review}[a.action](a.frame)
