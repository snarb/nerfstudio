"""Geometry-only checks for silhouette envelopes, before any RGB rollout."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel
import study_hand_silhouette_volume as study


def shaded(mesh, row):
    mesh.compute_triangle_normals()
    depth,ids,_ = camera_depth(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),row)
    good = np.isfinite(depth)
    out=np.zeros((*good.shape,3),np.uint8)
    normal=np.asarray(mesh.triangle_normals)[ids[good]]
    forward=np.asarray(row['transform_matrix'])[:3,2]
    intensity=.25+.75*np.abs(normal@forward)
    out[good]=(intensity[:,None]*np.array([200,180,155])).clip(0,255).astype(np.uint8)
    return np.rot90(out),np.rot90(np.where(good,depth,0))


def run():
    q=study.verify();root=study.ROOT/study.FRAME;dest=root/'geometry_review';dest.mkdir(exist_ok=False)
    data=np.load(root/'silhouettes.npz');meshes={};paths={'production':Path(q['source_mesh'])}
    paths.update({f'margin{i}':root/f'margin{i}/hull.ply' for i in [0,3]})
    for variant,path in paths.items():
        if variant=='production':assert sha(path)==q['source_mesh_sha256']
        else:assert sha(path)==read(path.parent/'result.json')['mesh_sha256']
        meshes[variant]=o3d.io.read_triangle_mesh(str(path))
    rows=[r['camera'] for r in q['cameras']]
    movie=read(study.MOVIE/'request.json')
    rows += [next(r['camera'] for r in movie['inventory'] if r['frame_id']==study.FRAME)]
    summaries=[];outputs={}
    for row in rows:
        name=row['physical_camera'];images=[];labels=[];metrics=[];maps={}
        if name+'_mask' in data:
            images.append(np.array(Image.open(study.OBS/study.FRAME/(name+'.png'))));labels.append('real train GT')
        for variant,mesh in meshes.items():
            im,depth=shaded(mesh,row);images.append(im);labels.append(variant+' geometry')
            maps[variant]=depth
            metric=dict(variant=variant)
            if name+'_mask' in data:
                mask=np.rot90(data[name+'_mask']);domain=np.rot90(data[name+'_domain'])
                metric.update(mask_pixels=int(mask.sum()),missing_mask_depth=int((mask&(depth==0)).sum()),
                              outside_mask_depth=int((domain&~mask&(depth>0)).sum()))
            metrics.append(metric)
        box=(0,1400,500,1920) if name+'_mask' in data else (100,1360,720,1920)
        path=dest/(name+'.png');panel(path,images,labels,box);outputs[str(path)]=sha(path)
        np.savez_compressed(dest/(name+'_depths.npz'),**maps)
        summaries.append(dict(camera=name,rows=metrics))
        print(name,metrics,flush=True)
    atomic_json(dest/'result.json',dict(request_sha256=sha(root/'request.json'),script_sha256=sha(__file__),
        rows=summaries,panels=outputs,visual_status='pending',new_rgb_render=False,
        geometry_envelope_only=True,these_are_mask_fit_diagnostics_not_quality_metrics=True))


if __name__=='__main__':run()
