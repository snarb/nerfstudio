"""Exact central train poses at a difficult actor time; no trajectory/video change."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras
from calibrated_depth_witness import load_images

ROOT=Path('/mnt/data/dec5_central_train_pose_probe')
PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
FRAME='001037'
COLUMNS='EFGHIJK'


def initialize():
    import render_smooth_temporal_mesh_video as renderer
    parent=renderer.verify_request(PARENT);entry=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    rows,_,metadata=cameras(FRAME);a=read(metadata);b=read(entry['metadata'])
    for key in ['dataparser_scale','dataparser_transform']:np.testing.assert_array_equal(a[key],b[key])
    assert sha(entry['mesh'])==entry['mesh_sha256']
    selected=[next(r for r in rows if r['physical_camera'].startswith(c+'004_C005_')) for c in COLUMNS]
    ROOT.mkdir(parents=True,exist_ok=False);images,_,receipt=load_images(FRAME);(ROOT/'gt').mkdir()
    for row in selected:
        Image.fromarray(np.rot90(images[row['physical_camera']])).save(ROOT/'gt'/(row['physical_camera']+'.png'))
    variants=[('moving',entry['camera'],None)]
    intrinsic_keys=['fl_x','fl_y','cx','cy','w','h','k1','k2','p1','p2','camera_model']
    for row in selected:
        for kind in ['native','flight_intrinsics']:
            camera=deepcopy(row);camera.pop('file_path',None)
            camera['physical_camera']='diagnostic_central_'+kind+'_'+row['physical_camera']
            camera['train_pose_physical_camera']=row['physical_camera']
            # A virtual target ID is mandatory: a native source mask cannot be
            # indexed directly on a target with a different intrinsic matrix.
            if kind=='flight_intrinsics':
                for key in intrinsic_keys:
                    if key in entry['camera']:camera[key]=entry['camera'][key]
                    else:camera.pop(key,None)
            variants.append((kind+'_'+row['physical_camera'],camera,row['physical_camera']))
    records=[]
    for view,camera,physical in variants:
        request=deepcopy(parent);request['inventory']=[deepcopy(entry)];request['inventory'][0]['camera']=camera
        request.update(partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False,
            central_pose_probe=True,actor_frame_unchanged=True,mesh_unchanged=True,
            target_mask_not_reused_in_foreign_intrinsics=True,
            probe_parent_sha256=sha(PARENT/'request.json'))
        for name in ['probe_central_train_camera_workaround.py','study_native_texture_footprint.py','native_texture_footprint.py']:
            request['script_hashes'][name]=sha(Path(__file__).with_name(name))
        dest=ROOT/view;dest.mkdir();(dest/'frames').mkdir();atomic_json(dest/'request.json',request)
        records.append(dict(view=view,physical_camera=physical,camera=camera,request_sha256=sha(dest/'request.json')))
    atomic_json(ROOT/'probe_request.json',dict(frame=FRAME,parent_request_sha256=sha(PARENT/'request.json'),
        mesh=entry['mesh'],mesh_sha256=entry['mesh_sha256'],views=records,selected_columns=COLUMNS,
        physical_vertical_row='C',physical_rows_from_vertical_edge=2,
        intrinsics_control='native vs unchanged flight intrinsics; identical physical pose in each pair',
        source_masks_unchanged=True,no_target_mask_warp_or_crop=True,rgb_receipt=receipt,
        script_sha256=sha(__file__),not_a_dynamic_video=False,probe_only=True))
    # GT-only context is generated before any candidate render is consumed.
    overview=Image.new('RGB',(7*270,500));draw=ImageDraw.Draw(overview)
    for i,row in enumerate(selected):
        im=Image.open(ROOT/'gt'/(row['physical_camera']+'.png')).resize((270,480))
        overview.paste(im,(i*270,20));draw.text((i*270+3,3),row['physical_camera'],fill='white')
    overview.save(ROOT/'gt_overview.png');print('Prepared',len(records),'views',flush=True)


def render(worker,workers):
    import render_smooth_temporal_mesh_video as renderer
    from study_native_texture_footprint import install
    install();renderer.torch.set_num_threads(2);request=read(ROOT/'probe_request.json')
    if workers<1 or not 0<=worker<workers:raise ValueError('Invalid worker partition')
    assert sha(__file__)==request['script_sha256']
    for i,record in enumerate(request['views']):
        if i%workers!=worker:continue
        dest=ROOT/record['view'];assert sha(dest/'request.json')==record['request_sha256']
        renderer.render(dest,[FRAME]);print('completed view',record['view'],flush=True)


def review():
    from review_jaw_repair_transfer import verified_image,panel
    request=read(ROOT/'probe_request.json');dest=ROOT/'review';dest.mkdir(exist_ok=False);records=[]
    for record in request['views']:
        view=record['view'];root=ROOT/view;im,result=verified_image(root,FRAME)
        assert not result['rgb_averaging'] and not result['target_rgb_read']
        images=[im];labels=[view]
        if view.startswith('native_'):
            gt=np.array(Image.open(ROOT/'gt'/(record['physical_camera']+'.png')))
            images=[gt,im];labels=['real train GT','same-pose reconstruction']
        panel(dest/(view+'_hand.png'),images,labels,(0,1260,720,1920))
        # Keep uncropped full-resolution images in the render outputs.
        thumb=Image.fromarray(im).resize((540,960));thumb.save(dest/(view+'_overview.png'))
        depth=np.rot90(np.load(root/'frames'/FRAME/'target_depth.npz')['depth'])
        y,x=np.nonzero(depth>0)
        records.append(dict(view=view,prediction_sha256=sha(root/'frames'/FRAME/'frame.png'),
            render_result_sha256=sha(root/'frames'/FRAME/'result.json'),
            foreground_bbox=None if not len(x) else [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
            bbox_is_context_not_anatomical_completeness=True,visual_status='pending'))
    atomic_json(dest/'result.json',dict(frame=FRAME,records=records,probe_request_sha256=sha(ROOT/'probe_request.json'),
        no_image_quality_metrics=True,video_updated=False,visual_status='pending'))
    print('Prepared',len(records),'review pairs',flush=True)


def check():
    ps=subprocess.check_output(['ps','-eo','pid,etime,pcpu,rss,args'],text=True)
    processes=[line.strip() for line in ps.splitlines() if 'python scripts/probe_central_train_camera_workaround.py render' in line and '/bin/bash' not in line]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    request=read(ROOT/'probe_request.json');complete=[r['view'] for r in request['views'] if (ROOT/r['view']/'frames'/FRAME/'complete.json').exists()]
    stat=dict(unix_time=time.time(),processes=processes,gpu=gpu,completed_views=complete,total=len(request['views']))
    import json
    with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(stat)+'\n')
    print('completed',len(complete),'/',len(request['views']),'workers',len(processes),'GPU',gpu,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render','review','check'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.action=='render':render(a.worker,a.workers)
    else:{'init':initialize,'review':review,'check':check}[a.action]()
