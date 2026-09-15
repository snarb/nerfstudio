"""Matched RGB comparison of strict and layer-qualified foreground protection."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image,panel
from study_forearm_layer_qualified_guard import ROOT,LOWER,FRAME

MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
VIEWS=['H004_A005_1210M6','E004_D005_1210L4','moving']


def render(worker,workers):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    if workers<1 or not 0<=worker<workers:raise ValueError('Invalid worker partition')
    implementation=install();engine.torch.set_num_threads(2)
    qualification=read(ROOT/'qualification_request.json')
    for path,digest in qualification['script_hashes'].items():assert sha(path)==digest
    result=read(ROOT/'geometry/result.json');assert sha(ROOT/'geometry/mesh.ply')==result['mesh_sha256']
    assert read(ROOT/'qualification_result.json')['geometry_result_sha256']==sha(ROOT/'geometry/result.json')
    for i,view in enumerate(VIEWS):
        if i%workers!=worker:continue
        parent=LOWER/'rgb'/view/'candidate';q=deepcopy(engine.verify_request(parent))
        assert len(q['inventory'])==1 and q['inventory'][0]['frame_id']==FRAME
        q['inventory'][0].update(mesh=str(ROOT/'geometry/mesh.ply'),mesh_sha256=result['mesh_sha256'])
        q.update(qualification_request_sha256=sha(ROOT/'qualification_request.json'),
            qualification_result_sha256=sha(ROOT/'qualification_result.json'),
            effective_guard_changed=True,qualified_guard_geometry_result_sha256=sha(ROOT/'geometry/result.json'),
            strict_guard_parent_request_sha256=sha(parent/'request.json'),source_quality_implementation_sha256=implementation,
            full_video_candidate=False,artifact_free_approval=False,partial_diagnostic_only=True)
        q['script_hashes'][Path(__file__).name]=sha(__file__)
        dest=ROOT/'rgb'/view;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
        if (dest/'request.json').exists() and read(dest/'request.json')!=q:raise ValueError('Changed RGB request')
        atomic_json(dest/'request.json',q);engine.render(dest,[FRAME])


def review():
    records=[]
    for view in VIEWS:
        original=MOVIE if view=='moving' else LOWER/'rgb'/view/'baseline'
        roots=[original,LOWER/'rgb'/view/'candidate',ROOT/'rgb'/view];ims=[];rs=[];ds=[]
        for root in roots:
            im,r=verified_image(root,FRAME);ims.append(im);rs.append(r)
            ds.append(np.rot90(np.load(root/'frames'/FRAME/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:assert all(r[key]==rs[0][key] for r in rs)
        unchanged=int(np.any(ims[-1][:1200]!=ims[1][:1200],axis=2).sum())
        labels=['production','strict prior guard','layer-qualified prior guard']
        if view!='moving':
            gt=LOWER/FRAME/(view+'.png') if view=='E004_D005_1210L4' else Path('/mnt/data/dec5_wrist_observations')/FRAME/(view+'.png')
            ims.insert(0,np.array(Image.open(gt)));labels.insert(0,'real train GT')
        paths=[]
        boxes={'overview':(0,1320,700,1920),
               'detail':(90,1650,430,1920) if view=='moving' else ((40,1650,300,1920) if view.startswith('H') else (80,1650,380,1920))}
        for name,box in boxes.items():
            path=ROOT/'review'/(view+'_'+name+'.png');panel(path,ims,labels,box);paths.append(dict(path=str(path),sha256=sha(path)))
        records.append(dict(view=view,panels=paths,upper_1200_rows_changed_pixels=unchanged,
            strict_to_qualified_rgb_changes=int(np.any(ims[-1]!=ims[-2],axis=2).sum()),
            strict_to_qualified_new_depth_pixels=int(((ds[-2]==0)&(ds[-1]>0)).sum()),
            strict_to_qualified_new_black_pixels=int(((ims[-1].max(2)==0)&(ims[-2].max(2)>0)).sum()),
            counts_not_anatomical_metrics=True,visual_status='pending'))
    atomic_json(ROOT/'review/result.json',dict(records=records,full_frame_quality_metrics=False,
        script_sha256=sha(__file__),visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.action=='render':render(a.worker,a.workers)
    else:review()
