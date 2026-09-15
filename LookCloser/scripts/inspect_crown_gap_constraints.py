"""Attribute local proposal rejection and show real-camera silhouette vetoes."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from diagnose_crown_completion_gap import ROOT,RAW,INSET,FRAMES
from probe_inset_head_completion import inset_vertices
from refine_measured_head_masks import ROOT as MASKS
from diffusion_mesh_repair import scene_for
from calibrated_depth_witness import load_images


def run(frame):
    out=ROOT/frame/'constraints';out.mkdir(exist_ok=False)
    config=read(INSET/frame/'request.json');a=np.load(ROOT/frame/'evidence.npz')
    raw=o3d.io.read_triangle_mesh(str(RAW/frame/'poisson_raw.ply'));raw.compute_vertex_normals()
    rv=inset_vertices(np.asarray(raw.vertices),np.asarray(config['center']),.001);rt=np.asarray(raw.triangles)
    original=o3d.io.read_triangle_mesh(config['source_mesh']);original.compute_triangle_normals()
    v=np.asarray(original.vertices);t=np.asarray(original.triangles);ids=a['raw_triangle_ids'];selected=rt[ids]
    points=rv[selected];closest=scene_for(v,t).compute_closest_points(o3d.core.Tensor(points.reshape(-1,3).astype(np.float32)))
    dist=np.linalg.norm(points.reshape(-1,3)-closest['points'].numpy(),axis=1).reshape(-1,3)
    edges,n=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    bd=cKDTree(v[np.unique(edges[n==1])]).query(points.reshape(-1,3))[0].reshape(-1,3)
    normals=np.asarray(raw.vertex_normals)[selected].reshape(-1,3)
    dot=np.sum(normals*np.asarray(original.triangle_normals)[closest['primitive_ids'].numpy()],axis=1).reshape(-1,3)
    length=np.linalg.norm(points-points[:,[1,2,0]],axis=2).max(1)
    fails=dict(surface_distance=(dist>config['maximum_surface_distance']).any(1),
        boundary_distance=(bd>config['maximum_boundary_distance']).any(1),
        normal_dot=(dot<config['minimum_normal_dot']).any(1),
        head_coordinate=(points[:,:,0]<=config['min_head_x']).any(1),
        edge_length=length>config['maximum_triangle_edge'])
    islocal=~np.logical_or.reduce(list(fails.values()))
    np.testing.assert_array_equal(islocal,a['triangle_stage']>0)
    pixels=a['native_full_prior_triangle_ids'][a['crown_region']&a['remaining_query']]
    weights=np.bincount(pixels.astype(int),minlength=len(rt))[ids]
    summary={k:dict(triangles=int(x.sum()),remaining_crown_pixels=int(weights[x].sum())) for k,x in fails.items()}
    rows,_,_=cameras(frame);masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
    image_map,_,rgb_receipt=load_images(frame)
    candidates=np.flatnonzero(a['triangle_stage']==1);order=candidates[np.argsort(-weights[candidates])]
    chosen=[]
    for k in order:
        if all(np.linalg.norm(points[k].mean(0)-points[j].mean(0))>.001 for j in chosen):chosen.append(int(k))
        if len(chosen)==3:break
    records=[]
    for number,k in enumerate(chosen):
        sample=np.concatenate([points[k],points[k].mean(0)[None]])
        uv,z=project(sample,rows);xy=np.rint(uv).astype(int);outside=[];inside=[]
        for ci,row in enumerate(rows):
            valid=(z[ci]>0)&(xy[ci,:,0]>2)&(xy[ci,:,0]<1917)&(xy[ci,:,1]>2)&(xy[ci,:,1]<1077)
            hit=np.zeros(4,bool);hit[valid]=masks[names.index(row['physical_camera']),xy[ci,valid,1],xy[ci,valid,0]]
            outside.append(int((valid&~hit).sum()));inside.append(hit)
        outside=np.array(outside);order=np.argsort(-outside,kind='stable');veto=order[outside[order]>0]
        cams=list(veto[:2])+list(veto[-2:]) if len(veto)>3 else list(veto)
        canvas=Image.new('RGB',(len(cams)*320,350));draw=ImageDraw.Draw(canvas)
        for j,ci in enumerate(cams):
            row=rows[ci];rgb=image_map[row['physical_camera']];assert rgb.dtype==np.uint8
            x,y=np.rint(uv[ci].mean(0)).astype(int);box=(x-150,y-150,x+150,y+150)
            crop=Image.fromarray(rgb).crop(box);cd=ImageDraw.Draw(crop)
            for point,hit in zip(uv[ci],inside[ci]):
                u,w=point-[x-150,y-150];cd.ellipse((u-3,w-3,u+3,w+3),outline='lime' if hit else 'red',width=2)
            canvas.paste(crop,(j*320,45));draw.text((j*320,4),row['physical_camera'],fill='white')
            draw.text((j*320,22),f'outside samples {outside[ci]}/4',fill='white')
        path=out/f'mask_veto_{number:02d}.png';canvas.save(path)
        records.append(dict(raw_triangle=int(ids[k]),remaining_crown_pixels=int(weights[k]),
            outside_camera_count=int((outside>0).sum()),all_four_outside_camera_count=int((outside==4).sum()),
            inspected_camera_candidates=[rows[ci]['physical_camera'] for ci in cams],
            outside_counts=outside.tolist(),sample_points=sample.tolist(),panel=path.name,panel_sha256=sha(path)))
    np.savez_compressed(out/'evidence.npz',triangle_ids=ids,pixel_weights=weights,surface_distance=dist,boundary_distance=bd,
        normal_dot=dot,maximum_edge=length,**{k+'_fail':x for k,x in fails.items()})
    atomic_json(out/'result.json',dict(source_result_sha256=sha(ROOT/frame/'result.json'),
        original_inset_config_sha256=sha(INSET/frame/'request.json'),script_sha256=sha(__file__),
        local_failure_counts=summary,failures_overlap=True,mask_examples=records,rgb_receipt=rgb_receipt,
        evidence_sha256=sha(out/'evidence.npz'),geometry_changed=False,visual_status='pending'))
    print(frame,summary,[(r['raw_triangle'],r['outside_camera_count']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=FRAMES);run(p.parse_args().frame)
