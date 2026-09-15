"""Matched current RGB review of a uniform measured-free-space mesh subset."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from review_full_block_transfer import ROOT as DEPTH_ROOT,VIDEO
from prune_measured_free_surface import ROOT
from review_jaw_repair_transfer import verified_image,panel

VIEWS=['moving','H004_C005_1210SZ','K004_B005_1210DS']


def baseline(frame,view):
    return VIDEO if view=='moving' else DEPTH_ROOT/frame/'rgb'/view/'production'


def prepare(frame):
    root=ROOT/frame; result=read(root/'result.json')
    assert result['request_sha256']==sha(root/'request.json')
    for name,digest in result['hashes'].items():assert sha(root/name)==digest
    for view in VIEWS:
        before=baseline(frame,view);q=deepcopy(read(before/'request.json'))
        q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame]
        assert len(q['inventory'])==1
        q['inventory'][0].update(mesh=str(root/'mesh.ply'),mesh_sha256=result['hashes']['mesh.ply'])
        q['source_rows']=[r for r in q['source_rows'] if Path(r['source_dataset']).name==frame]
        q['ordered_frame_ids']=[frame]
        q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,
            artifact_free_approval=False,measured_free_pruning_sha256=sha(root/'result.json'),
            pruning_request_sha256=sha(root/'request.json'),texture_source_masks_unchanged=True,
            current_production_mesh_is_parent=True)
        q['script_hashes'][Path(__file__).name]=sha(__file__)
        out=root/'rgb'/view;(out/'frames').mkdir(parents=True,exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==q
        else:atomic_json(out/'request.json',q)


def render(frame,view):
    from run_view_consistent_dynamic_video import install
    import render_smooth_temporal_mesh_video as engine
    implementation=install();engine.torch.set_num_threads(2)
    out=ROOT/frame/'rgb'/view
    assert implementation==read(out/'request.json')['source_quality_implementation_sha256']
    engine.render(out,[frame])


def review(frame):
    root=ROOT/frame;records=[]
    boxes={'moving':(380,1400,680,1870),'H004_C005_1210SZ':(230,1080,530,1550),
        'K004_B005_1210DS':(40,1120,340,1590)}
    heads={'moving':(150,900,850,1490),'H004_C005_1210SZ':(100,500,1000,1200),
        'K004_B005_1210DS':(100,450,950,1210)}
    for view in VIEWS:
        old=baseline(frame,view);new=root/'rgb'/view
        a,ar=verified_image(old,frame);b,br=verified_image(new,frame)
        for k in ['camera','source_cameras','fixed_exposure']:assert ar[k]==br[k]
        aq,bq=read(old/'request.json'),read(new/'request.json')
        for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert aq[k]==bq[k]
        ad=np.load(old/'frames'/frame/'target_depth.npz')['depth']
        bd=np.load(new/'frames'/frame/'target_depth.npz')['depth']
        assert ad.shape==bd.shape and np.isfinite(ad).all() and np.isfinite(bd).all()
        assert not ((ad==0)&(bd>0)).any(), 'Deletion cannot add hits'
        assert not ((ad>0)&(bd>0)&(bd<ad-1e-6)).any(), 'Deletion cannot move hits nearer'
        images=[a,b];labels=['production','measured-free subset']
        if view!='moving':
            images.insert(0,np.asarray(Image.open(DEPTH_ROOT/frame/'review'/view/'train_gt.png')))
            labels.insert(0,'real train GT')
        out=root/'review'/view
        panel(out/'lipstick_native.png',images,labels,boxes[view])
        panel(out/'head_native.png',images,labels,heads[view])
        for name,image in [('production',a),('candidate',b)]:
            Image.fromarray(image).crop(boxes[view]).save(out/(name+'_lipstick_native.png'))
        records.append(dict(view=view,baseline_request_sha256=sha(old/'request.json'),
            candidate_request_sha256=sha(new/'request.json'),baseline_png_sha256=sha(old/'frames'/frame/'frame.png'),
            candidate_png_sha256=sha(new/'frames'/frame/'frame.png'),
            lost_depth=int(((ad>0)&(bd==0)).sum()),
            farther_depth=int(((ad>0)&(bd>ad+1e-6)).sum()),
            changed_rgb=int(np.any(a!=b,axis=2).sum()),
            hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},visual_status='pending'))
    atomic_json(root/'review/result.json',dict(records=records,counts_not_quality_metrics=True,
        production_promoted=False,geometry_is_original_subset=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','review'])
    p.add_argument('--frame',default='000995',choices=['000995']);p.add_argument('--view',choices=VIEWS)
    a=p.parse_args()
    if a.action=='render':
        if not a.view:p.error('render requires --view')
        render(a.frame,a.view)
    else:globals()[a.action](a.frame)
