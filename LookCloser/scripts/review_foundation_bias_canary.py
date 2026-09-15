"""Measured-depth bias and rejected foreground additions: actual visual evidence."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_foundation_anchor_bias import ROOT as BIAS
from build_foundation_foreground_patch import ROOT as PATCH
from build_foundation_consensus_patch import MOVIE
from review_hand_silhouette_volume import shaded
from review_jaw_repair_transfer import panel


def panels():
    request=read(PATCH/'request.json');result=read(PATCH/'result.json')
    assert sha(PATCH/'mesh.ply')==result['mesh_sha256']
    assert sha(request['source_mesh'])==request['source_mesh_sha256']
    old=o3d.io.read_triangle_mesh(request['source_mesh']);data=np.load(PATCH/'proposal.npz')
    v=np.concatenate((np.asarray(old.vertices),data['added_vertices']))
    t=np.concatenate((np.asarray(old.triangles),data['added_triangles']+len(old.vertices)))
    raw=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    guarded=o3d.io.read_triangle_mesh(str(PATCH/'mesh.ply'));rows,_,_=cameras('001037')
    views={r['physical_camera']:r for r in rows if r['physical_camera']=='H004_A005_1210M6'}
    views['moving']=next(r['camera'] for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']=='001037')
    records=[]
    for name,row in views.items():
        images=[];depths=[]
        for mesh in [old,raw,guarded]:
            im,d=shaded(mesh,row);images.append(im);depths.append(d)
        labels=['original geometry','290 proposed triangles','6 after observed-depth guard']
        if name!='moving':
            gt=Path('/mnt/data/dec5_wrist_observations/001037')/(name+'.png')
            images.insert(0,np.array(Image.open(gt)));labels.insert(0,'real train GT (context)')
        box=(0,1400,500,1920) if name!='moving' else (100,1360,720,1920)
        path=PATCH/'review'/(name+'.png');panel(path,images,labels,box)
        records.append(dict(view=name,path=str(path),sha256=sha(path),
            changed_depth_pixels=[int((abs(d-depths[0])>1e-6).sum()) for d in depths],
            newly_visible=[int(((d>0)&(depths[0]==0)).sum()) for d in depths]))
    atomic_json(PATCH/'review/result.json',dict(records=records,geometry_not_rgb_prediction=True,
        script_sha256=sha(__file__),visual_status='pending'))


if __name__=='__main__':panels()
