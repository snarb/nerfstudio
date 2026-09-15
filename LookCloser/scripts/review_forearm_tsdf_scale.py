"""Matched native geometry/RGB controls for measured-depth TSDF scale."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from review_hand_silhouette_volume import shaded
from review_jaw_repair_transfer import panel,verified_image
from study_forearm_tsdf_scale import ROOT,SCALES

MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
NAMES=['H004_A005_1210M6','E004_C005_1210YM']


def views(frame):
    rows,_,_=cameras(frame);request=read(MOVIE/'request.json')
    result={r['physical_camera']:r for r in rows if r['physical_camera'] in NAMES}
    result['moving']=next(r['camera'] for r in request['inventory'] if r['frame_id']==frame)
    return result


def geometry(frame):
    root=ROOT/frame;complete=read(root/'complete.json');assert complete['request_sha256']==sha(root/'request.json')
    meshes={}
    for r in complete['variants']:
        path=root/r['variant']/'mesh.ply';assert sha(path)==r['mesh_sha256']
        meshes[r['variant']]=o3d.io.read_triangle_mesh(str(path))
    dest=root/'geometry_review';dest.mkdir(exist_ok=False);receipt=[]
    for name,camera in views(frame).items():
        images={};maps={}
        for variant,mesh in meshes.items():images[variant],maps[variant]=shaded(mesh,camera)
        box=(0,1400,500,1920) if name!='moving' else (100,1360,720,1920)
        for variant in list(meshes)[1:]:
            path=dest/(name+'_'+variant+'.png')
            panel(path,[images['fine'],images[variant]],['fine geometry',variant+' geometry'],box)
            receipt.append(dict(path=str(path),sha256=sha(path)))
        np.savez_compressed(dest/(name+'_depths.npz'),**maps)
        print('geometry review',frame,name,flush=True)
    atomic_json(dest/'result.json',dict(panels=receipt,script_sha256=sha(__file__),visual_status='pending'))


def render(frame,variants):
    import run_view_consistent_dynamic_video as source
    source.install();renderer=source.lifecycle.renderer;renderer.torch.set_num_threads(2)
    parent=renderer.verify_request(MOVIE)
    entry=next(r for r in parent['inventory'] if r['frame_id']==frame)
    for name,camera in views(frame).items():
        if name=='E004_C005_1210YM':continue
        for variant in variants:
            dest=ROOT/frame/'rgb'/name/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            path=ROOT/frame/variant/'mesh.ply';expected=read(path.parent/'complete.json')
            assert sha(path)==expected['mesh_sha256']
            request=deepcopy(parent);record=deepcopy(entry);record.update(camera=deepcopy(camera),mesh=str(path),mesh_sha256=sha(path))
            # Keep real-camera intrinsics; deliberately virtual ID avoids target/source-mask identity assumptions.
            record['camera']['physical_camera']='tsdf_scale_'+name
            request.update(inventory=[record],partial_diagnostic_only=True,full_video_candidate=False,
                artifact_free_approval=False,geometry_changed_from_texture_parent=True,
                tsdf_scale_variant=variant,matched_frame=frame)
            request['script_hashes'][Path(__file__).name]=sha(__file__)
            if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Changed RGB request')
            atomic_json(dest/'request.json',request);renderer.render(dest,[frame])
            print('rendered',frame,name,variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['geometry','render'])
    p.add_argument('--frame',default='001037');p.add_argument('--variants',nargs='+',choices=list(SCALES),default=['fine','coarse'])
    a=p.parse_args();geometry(a.frame) if a.action=='geometry' else render(a.frame,a.variants)
