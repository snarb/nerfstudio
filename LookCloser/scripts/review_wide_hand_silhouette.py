"""Matched native geometric controls for six/twelve-camera hand silhouettes."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from review_hand_silhouette_volume import shaded
from review_jaw_repair_transfer import panel
import study_wide_hand_silhouette as study


def run():
    root=study.ROOT;request=read(root/'request.json');a=read(Path(request['original'])/'request.json');b=read(Path(request['extra'])/'request.json')
    assert sha(Path(request['original'])/'request.json')==request['original_request_sha256']
    assert sha(Path(request['extra'])/'request.json')==request['extra_request_sha256']
    dest=root/'review';dest.mkdir(exist_ok=False)
    meshes={'production':o3d.io.read_triangle_mesh(a['source_mesh'])}
    for n in ['six/raw','six/domain_safe','twelve/domain_safe']:
        path=root/n/'hull.ply';assert sha(path)==read(path.parent/'result.json')['mesh_sha256']
        meshes[n]=o3d.io.read_triangle_mesh(str(path))
    names=['G004_A005_121071','H004_A005_1210M6','E004_C005_1210YM','F004_D005_1210KW']
    rows=[r['camera'] for r in a['cameras']+b['cameras'] if r['camera']['physical_camera'] in names]
    movie=read(study.base.MOVIE/'request.json');rows.append(next(r['camera'] for r in movie['inventory'] if r['frame_id']==study.base.FRAME))
    sources={r['camera']['physical_camera']:study.base.OBS/study.base.FRAME/(r['camera']['physical_camera']+'.png') for r in a['cameras']}
    sources.update({r['camera']['physical_camera']:Path('/mnt/data/dec5_wrist_wide_observations')/study.base.FRAME/(r['camera']['physical_camera']+'.png') for r in b['cameras']})
    records=[]
    for row in rows:
        n=row['physical_camera'];images=[];labels=[];maps={}
        if n in sources:images.append(np.array(Image.open(sources[n])));labels.append('real train GT')
        for variant,mesh in meshes.items():
            im,depth=shaded(mesh,row);images.append(im);labels.append(variant);maps[variant.replace('/','_')]=depth
        box=(0,1250,500,1920) if n in sources else (100,1360,720,1920)
        path=dest/(n+'.png');panel(path,images,labels,box)
        np.savez_compressed(dest/(n+'_depths.npz'),**maps)
        records.append(dict(camera=n,path=str(path),sha256=sha(path)))
        print('review',n,flush=True)
    atomic_json(dest/'result.json',dict(panels=records,script_sha256=sha(__file__),visual_status='pending',rgb_prediction=False))


if __name__=='__main__':run()
