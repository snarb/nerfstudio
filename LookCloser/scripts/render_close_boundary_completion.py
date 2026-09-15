"""Matched current-texture evaluation of minimum-gap-free local completion."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from review_jaw_repair_transfer import verified_image,panel
from study_close_boundary_completion import ROOT

MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
HELD=Path('/mnt/data/dec5_unwarped_head_texture/unwarped')


def previous(frame):
    root=Path('/mnt/data/dec5_poisson_jaw_completion') if frame=='001193' else Path('/mnt/data/dec5_neighborhood_completion_transfer')/frame
    return root/'interpolated'/frame


def mesh_for(frame,variant):
    root=previous(frame) if variant=='previous' else ROOT/frame/'interpolated'/frame
    result=read(root/'result.json')
    if not result['observed_guard_passed'] or sha(root/'mesh.ply')!=result['hashes']['mesh.ply']:
        raise ValueError('Unverified geometry')
    return root/'mesh.ply'


def render(frame,heldout=False):
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    implementation=install();renderer.torch.set_num_threads(2)
    parent=renderer.verify_request(HELD if heldout else MOVIE)
    rows,_,_=cameras(frame)
    for view in (['heldout'] if heldout else ['moving','F004_E005_1210FP']):
        for variant in (['close_boundary'] if heldout else ['baseline','previous','close_boundary']):
            if view=='moving' and variant=='baseline':continue
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
            if view not in ['moving','heldout']:entry['camera']=next(r for r in rows if r['physical_camera']==view)
            if variant!='baseline':
                mesh=mesh_for(frame,variant);entry.update(mesh=str(mesh),mesh_sha256=sha(mesh))
            q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=variant!='baseline',
                geometry_comparison=variant,source_quality_implementation_sha256=implementation,
                minimum_gap_study_controller_sha256=sha(ROOT/frame/'controller_request.json'))
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            dest=ROOT/'heldout'/variant if heldout else ROOT/frame/'rgb'/view/variant
            dest.mkdir(parents=True,exist_ok=False);(dest/'frames').mkdir();atomic_json(dest/'request.json',q)
            renderer.render(dest,[frame])


def review(frame):
    import cv2
    from diagnose_jaw_transfer_support import POLYGONS
    dest=ROOT/frame/'review';dest.mkdir(exist_ok=False);records=[]
    for view in ['moving','F004_E005_1210FP']:
        roots=[MOVIE if view=='moving' else ROOT/frame/'rgb'/view/'baseline']
        roots += [ROOT/frame/'rgb'/view/v for v in ['previous','close_boundary']]
        images=[];depths=[];results=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);results.append(r)
            depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:
            if not all(r[key]==results[0][key] for r in results):raise ValueError('Unmatched comparison')
        record=dict(view=view,inputs={str(root/'frames'/frame/'frame.png'):sha(root/'frames'/frame/'frame.png') for root in roots},
            changed_rgb_pixels=[int(np.any(im!=images[0],axis=2).sum()) for im in images])
        labels=['production','previous completion','no minimum gap']
        if view!='moving':
            gtpath=Path('/mnt/data/dec5_jaw_repair_transfer/review')/frame/(view+'_gt.png')
            gt=np.array(Image.open(gtpath));record['gt_sha256']=sha(gtpath)
            for name,polygon in [('interior',POLYGONS[frame]),('edge',[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]])]:
                mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
                record[name]=dict(polygon=polygon,depth_misses=[int((mask&(d==0)).sum()) for d in depths],
                    black_rgb=[int((mask&(im.max(2)==0)).sum()) for im in images])
            images=[gt,*images];labels=['real train GT',*labels]
        else:
            # Fixed rectangle from the published moving-frame diagnosis. Not a
            # candidate-defined face metric and not an anatomical hole count.
            box=(590,1170,610,1185) if frame=='001193' else (570,1140,670,1240)
            x0,y0,x1,y1=box
            record['diagnostic_rectangle']=dict(box=box,includes_background=frame!='001193',
                depth_misses=[int((d[y0:y1,x0:x1]==0).sum()) for d in depths],
                black_rgb=[int((im[y0:y1,x0:x1].max(2)==0).sum()) for im in images])
        panel(dest/(view+'_head.png'),images,labels,(170,450,970,1350))
        panel(dest/(view+'_detail.png'),images,labels,(570,1070,780,1240) if view!='moving' else (370,1080,670,1320))
        records.append(record)
    atomic_json(dest/'result.json',dict(records=records,visual_status='pending',quality_metrics_computed=False,
        script_sha256=sha(__file__),same_texture_for_all_variants=True))
    print(records,flush=True)


def score():
    import evaluate_head_source_quality as evaluator
    root=ROOT/'heldout';(root/'baseline').symlink_to(HELD,target_is_directory=True)
    evaluator.ROOT=root;evaluator.MODES=('baseline','close_boundary');evaluator.score()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['render','review','heldout','score']);p.add_argument('--frame',choices=['001193','001195'],default='001193')
    a=p.parse_args()
    if a.action=='score':score()
    elif a.action=='review':review(a.frame)
    else:render(a.frame,a.action=='heldout')
