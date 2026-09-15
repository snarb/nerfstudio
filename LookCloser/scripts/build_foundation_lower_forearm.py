"""Apply the unchanged dual-pair foreground recipe to lower-row observations."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_foundation_lower_forearm import ROOT,FRAME
from review_jaw_repair_transfer import panel,verified_image

MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')


def geometry():
    import build_foundation_consensus_patch as empty
    import build_foundation_foreground_patch as front
    empty.ROOT=ROOT/'empty_ray';empty.BIAS=ROOT/'bias';empty.SOURCES=[ROOT/FRAME]
    empty.run()
    front.ROOT=ROOT/'foreground';front.CONTROL=empty.ROOT;front.BIAS=empty.BIAS
    front.run()
    atomic_json(ROOT/'geometry_adapter.json',dict(script_sha256=sha(__file__),
        stage_request_sha256=sha(ROOT/FRAME/'request.json'),bias_adapter_sha256=sha(ROOT/'bias_adapter.json'),
        producers={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [empty,front]},
        foreground_request_sha256=sha(ROOT/'foreground/request.json'),
        thresholds_unchanged_from_previous_hand_canary=True,production_updated=False))


def review():
    from review_hand_silhouette_volume import shaded
    root=ROOT/'foreground';request=read(root/'request.json');result=read(root/'result.json')
    assert result['mesh_sha256']==sha(root/'mesh.ply')
    old=o3d.io.read_triangle_mesh(request['source_mesh']);data=np.load(root/'proposal.npz')
    v=np.concatenate((np.asarray(old.vertices),data['added_vertices']))
    t=np.concatenate((np.asarray(old.triangles),data['added_triangles']+len(old.vertices)))
    proposed=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    final=o3d.io.read_triangle_mesh(str(root/'mesh.ply'));rows,_,_=cameras(FRAME)
    views={r['physical_camera']:r for r in rows if r['physical_camera'] in ['H004_A005_1210M6','E004_D005_1210L4']}
    views['moving']=next(r['camera'] for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==FRAME)
    records=[]
    for name,row in views.items():
        ims=[];ds=[]
        for mesh in [old,proposed,final]:
            im,d=shaded(mesh,row);ims.append(im);ds.append(d)
        labels=['original',str(result['proposed_triangles'])+' proposed',str(result['retained_triangles'])+' guarded']
        gt=ROOT/FRAME/(name+'.png') if name=='E004_D005_1210L4' else Path('/mnt/data/dec5_wrist_observations')/FRAME/(name+'.png')
        if name!='moving':ims.insert(0,np.array(Image.open(gt)));labels.insert(0,'real train GT')
        box=(0,1350,600,1920) if name!='moving' else (100,1360,720,1920)
        path=ROOT/'geometry_review'/(name+'.png');panel(path,ims,labels,box)
        records.append(dict(view=name,panel=str(path),sha256=sha(path),
            changed_depth=[int((abs(d-ds[0])>1e-6).sum()) for d in ds],
            newly_visible=[int(((d>0)&(ds[0]==0)).sum()) for d in ds]))
    atomic_json(ROOT/'geometry_review/result.json',dict(records=records,visual_status='pending',
        images_are_geometry_shading_not_rgb_predictions=True))


def render(worker,workers):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    if workers<1 or not 0<=worker<workers:raise ValueError('Bad partition')
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    result=read(ROOT/'foreground/result.json')
    if not result['guard_passed'] or result['retained_triangles']==0:raise ValueError('No guarded candidate')
    assert sha(ROOT/'foreground/mesh.ply')==result['mesh_sha256']
    rows,_,_=cameras(FRAME);names=['H004_A005_1210M6','E004_D005_1210L4','moving']
    for i,name in enumerate(names):
        if i%workers!=worker:continue
        for variant in ['baseline','candidate']:
            if variant=='baseline' and name=='moving':continue
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==FRAME];entry=q['inventory'][0]
            if name!='moving':entry['camera']=next(r for r in rows if r['physical_camera']==name)
            if variant=='candidate':entry.update(mesh=str(ROOT/'foreground/mesh.ply'),mesh_sha256=result['mesh_sha256'])
            q.update(partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False,
                source_quality_implementation_sha256=implementation,geometry_changed=variant=='candidate',
                lower_forearm_geometry_result_sha256=sha(ROOT/'foreground/result.json'))
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            dest=ROOT/'rgb'/name/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            if (dest/'request.json').exists() and read(dest/'request.json')!=q:raise ValueError('Changed RGB request')
            atomic_json(dest/'request.json',q);engine.render(dest,[FRAME])


def rgb_panels():
    records=[]
    for name in ['H004_A005_1210M6','E004_D005_1210L4','moving']:
        roots=[MOVIE if name=='moving' else ROOT/'rgb'/name/'baseline',ROOT/'rgb'/name/'candidate']
        images=[];results=[]
        for root in roots:
            im,r=verified_image(root,FRAME);images.append(im);results.append(r)
        for key in ['camera','source_cameras','fixed_exposure']:assert results[0][key]==results[1][key]
        labels=['production mesh','lower-row prior addition']
        if name!='moving':
            gt=ROOT/FRAME/(name+'.png') if name=='E004_D005_1210L4' else Path('/mnt/data/dec5_wrist_observations')/FRAME/(name+'.png')
            images.insert(0,np.array(Image.open(gt)));labels.insert(0,'real train GT')
        path=ROOT/'rgb_review'/(name+'.png');panel(path,images,labels,(0,1320,700,1920))
        records.append(dict(view=name,path=str(path),sha256=sha(path),visual_status='pending'))
    atomic_json(ROOT/'rgb_review/result.json',dict(records=records,full_frame_metrics=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['geometry','review','render','rgb-panels'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.action=='render':render(a.worker,a.workers)
    else:{'geometry':geometry,'review':review,'rgb-panels':rgb_panels}[a.action]()
