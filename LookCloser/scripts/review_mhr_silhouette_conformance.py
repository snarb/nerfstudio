"""Anatomical/topological and native evidence for the one frozen silhouette fit."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from fit_mhr_silhouette_conformance import ROOT, SOURCE, project_jacobian, sample_sdf, prepare
from study_multiview_face_prior import read, save, sha, portrait_to_native, CROP
from study_mhr_local_head_prior import RGB, FRAME
from study_confidence_depth_prior import unproject
from admit_mhr_local_patch_depth import Scene2
from check_mhr_conformance_crossings import transverse_crossings
from triangulate_face_prior import quantiles


def main():
    import open3d as o3d
    from scipy.ndimage import distance_transform_edt
    from bake_joint_temporal_mesh import camera_depth
    dest = ROOT/'review_v2'; dest.mkdir(exist_ok=False)
    protocol = read(ROOT/'protocol.json'); fit = np.load(ROOT/'fit.npz')
    tri, neutral, active = fit['triangles'], fit['neutral'], fit['active']
    parent, rows, masks, names, evidence, validation = prepare()
    variants = [('baseline', fit['baseline']), ('silhouette', fit['vertices'])]
    geometry, scenes, bindings, files, diagnostics = {}, {}, {}, [], []
    for p in [ROOT/'protocol.json', ROOT/'fit.npz', ROOT/'result.json', RGB/FRAME/'input.json', RGB/'inference.json']:
        bindings[str(p)] = sha(p)
    for label, vertices in variants:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(tri))
        mesh.compute_triangle_normals()
        pairs = np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1,2)
        proper = pairs[transverse_crossings(vertices[tri[pairs[:,0]]], vertices[tri[pairs[:,1]]])]
        cross = np.cross(vertices[tri[:,1]]-vertices[tri[:,0]], vertices[tri[:,2]]-vertices[tri[:,0]])
        if label == 'baseline': base_cross = cross; base_pairs = set(map(tuple, proper))
        reversed_ids = np.flatnonzero(np.sum(base_cross*cross, axis=1) <= 0)
        touches_active = active[tri].any(1)
        geometry[label] = dict(nonadjacent_intersection_pairs=len(pairs), strict_transverse_pairs=len(proper),
             strict_pairs_touching_active=int(touches_active[proper].any(1).sum()),
             new_strict_pair_ids=len(set(map(tuple,proper))-base_pairs),
             normal_changes_over90=len(reversed_ids), normal_changes_touching_active=int(touches_active[reversed_ids].sum()),
             minimum_triangle_area=float(np.linalg.norm(cross,axis=1).min()/2),
             min_area_ratio=float((np.linalg.norm(cross,axis=1)/np.maximum(np.linalg.norm(base_cross,axis=1),1e-30)).min()))
        np.savez_compressed(dest/(label+'_topology.npz'), intersections=pairs, strict_pairs=proper, reversed_triangles=reversed_ids)
        scenes[label] = (Scene2(vertices, tri), np.asarray(mesh.triangle_normals))
    original = o3d.io.read_triangle_mesh(protocol['original_mesh']); original.compute_triangle_normals()
    bindings[protocol['original_mesh']] = sha(protocol['original_mesh'])
    oldscene = Scene2(np.asarray(original.vertices), np.asarray(original.triangles))
    scenes['original'] = (oldscene, np.asarray(original.triangle_normals))
    predictions = {r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1}

    def panel(images, path):
        w,h = images[0][1].size; out = Image.new('RGB',(len(images)*w,h+24)); draw=ImageDraw.Draw(out)
        for index,(name,im) in enumerate(images): out.paste(im,(index*w,24));draw.text((index*w+2,4),name,fill='white')
        out.save(path); files.append(dict(path=str(path),sha256=sha(path)))

    for prefix in ['C004_E','E004_D','G004_B','M004_B']:
        row = next(r for r in rows if r['physical_camera'].startswith(prefix)); name=row['physical_camera']
        x0,y0,x1,y1 = predictions[name]['native_review_box']; y1=min(1500,y1+180)
        rp=RGB/FRAME/(name+'.png');bindings[str(rp)]=sha(rp)
        rgb=Image.open(rp).convert('RGB').crop((x0,y0-CROP[1],x1,y1-CROP[1]))
        yy,xx=np.mgrid[y0:y1,x0:x1];xy=portrait_to_native(np.c_[xx.ravel(),yy.ravel()])
        center=np.asarray(row['transform_matrix'])[:3,3];direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        unit=direction/np.linalg.norm(direction,axis=1,keepdims=True)
        rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
        images=[('train RGB',rgb)]
        for label in ['original','baseline','silhouette']:
            scene,normals=scenes[label];hit=scene.cast_rays(rays);d=hit['t_hit'].numpy();ids=hit['primitive_ids'].numpy();valid=np.isfinite(d)
            color=np.full((len(d),3),20,np.uint8);color[valid]=(70+170*abs(np.sum(normals[ids[valid]]*-unit[valid],axis=1)))[:,None]
            images.append((label,Image.fromarray(color.reshape(*yy.shape,3))))
        panel(images,dest/(name+'_clay.png'))
        mask=masks[names.index(name)].astype(bool);sdf=(distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32)
        overlays=[]
        for label,v in variants:
            uv,z,_=project_jacobian(v[active],row);available=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
            values,_=sample_sdf(sdf,uv[available]);coords=np.c_[uv[available,1],1919-uv[available,0]]-np.array([x0,y0])
            im=rgb.copy();draw=ImageDraw.Draw(im)
            for (x,y),value in zip(coords,values):draw.ellipse((x-1,y-1,x+1,y+1),fill='red' if value>2 else (0,220,100))
            overlays.append((label+' red=outside>2px',im))
            nd=neutral[active][available];inside=values>2
            diagnostics.append(dict(camera=name,arm=label,available=len(values),outside=int(inside.sum()),
                 outside_front=int((inside&(nd[:,2]>=0)).sum()),outside_back=int((inside&(nd[:,2]<0)).sum()),
                 excess_pixels=quantiles(np.maximum(values-2,0))))
        panel(overlays,dest/(name+'_projection.png'))
    parentpath=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json'); spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    for p in [parentpath,spots]:bindings[str(p)]=sha(p)
    camera=next(r['camera'] for r in read(parentpath)['inventory'] if r['frame_id']==FRAME)
    x0,y0,x1,y1=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    images=[]; target=[]
    for label in ['original','baseline','silhouette']:
        scene,normals=scenes[label];depth,ids,_=camera_depth(scene,camera);ok=np.isfinite(depth);color=np.full((*depth.shape,3),20,np.uint8)
        color[ok]=(60+170*abs(normals[ids[ok]]@np.array([.3,.4,.866])))[:,None]
        images.append((label,Image.fromarray(np.rot90(color)).crop((x0-65,y0-65,x1+66,y1+66))))
        d=np.rot90(depth)[y0:y1+1,x0:x1+1]
        if label=='original': missing=~np.isfinite(d)
        target.append(dict(arm=label,original_missing=int(missing.sum()),prior_hits_at_original_misses=int(np.isfinite(d[missing]).sum()),depths=[float(x) if np.isfinite(x) else None for x in d[missing]]))
    panel(images,dest/'requested_hole_prior_only.png')
    save(dest/'result.json',dict(geometry=geometry,diagnostics=diagnostics,target_prior_only=target,
         displacement=quantiles(np.linalg.norm(fit['displacement'][active],axis=1)),
         files=files,input_hashes=bindings,script_sha256=sha(__file__),
         crossing_helper_sha256=sha(Path(__file__).with_name('check_mhr_conformance_crossings.py')),
         original_geometry_changed=False,target_used_posthoc_only=True,production_accepted=False))


if __name__=='__main__':main()
