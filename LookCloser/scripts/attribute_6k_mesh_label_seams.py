"""Distinguish frozen face labels from native pixel visibility fallback."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from diagnose_6k_source_seams import ROOT, NEW


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    out=ROOT/'mesh_label_attribution';out.mkdir(exist_ok=False)
    evidence=read(ROOT/'result.json');frame=evidence['frame']
    folder=NEW/'frames'/frame;render=read(folder/'result.json');request=read(NEW/'request.json')
    source=next(r for r in read(Path(request['parent'])/'request.json')['inventory'] if r['frame_id']==frame)
    assert sha(source['mesh'])==source['mesh_sha256']
    labels=np.load(folder/'face_source_labels.npy')
    mesh=o3d.io.read_triangle_mesh(source['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    native=np.rot90(np.asarray(Image.open(folder/'source_ids.png')))
    rgb=Image.open(folder/'frame.png');records=[]
    for row in evidence['records']:
        x0,y0,x1,y1=row['box'];camera=deepcopy(render['camera'])
        # rot90 maps portrait(y,x) to landscape(x,W-1-y).
        camera['cx']-=camera['w']-y1;camera['cy']-=x0
        camera['w']=y1-y0;camera['h']=x1-x0
        d,face,_=camera_depth(scene,camera);face=np.rot90(face);valid=np.rot90(np.isfinite(d))
        preferred=np.full(face.shape,255,np.int64);preferred[valid]=labels[face[valid]]
        selected=native[y0:y1,x0:x1];different=(preferred!=selected)&valid
        crop=np.asarray(rgb.crop(row['box']));overlay=crop.copy();overlay[different]=[255,255,0]
        h,w=face.shape;panel=Image.new('RGB',(w*2,h+24));draw=ImageDraw.Draw(panel)
        for i,(title,im) in enumerate([('Native RGB',crop),('Yellow: differs from frozen mesh-face source',overlay)]):
            panel.paste(Image.fromarray(im),(i*w,24));draw.text((i*w+4,4),title,fill='white')
        path=out/f'{frame}_{row["region"]}.png';panel.save(path)
        artifact=out/f'{frame}_{row["region"]}.npz'
        np.savez_compressed(artifact,face_ids=face,valid=valid,preferred=preferred,selected=selected)
        records.append(dict(region=row['region'],pixels=face.size,mesh_misses=int((~valid).sum()),
            preferred_source_values=np.unique(preferred).tolist(),fallback_pixels=int(different.sum()),
            image=str(path),image_sha256=sha(path),evidence=str(artifact),evidence_sha256=sha(artifact)))
    atomic_json(out/'result.json',dict(records=records,script_sha256=sha(__file__),
        input_hashes={str(ROOT/'result.json'):sha(ROOT/'result.json'),source['mesh']:sha(source['mesh']),
            str(folder/'face_source_labels.npy'):sha(folder/'face_source_labels.npy'),
            str(folder/'result.json'):sha(folder/'result.json')},geometry_changed=False))
    print(records,flush=True)


if __name__=='__main__':main()
