"""Matched native geometry review of cached-depth extraction controls."""
import numpy as np
from PIL import Image,ImageDraw
from study_jaw_tsdf_extraction_weight import ROOT, SOURCE, FRAMES, ARMS
from joint_temporal_texture import read,sha,atomic_json,cameras,geometry_paths
from review_jaw_depth_controls import normalization_matrix


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    parent='/mnt/data/dec5_elevated_camera_dynamic_150/request.json'
    spots='/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json'
    manifest=read(ROOT/'result.json'); assert manifest['request_sha256']==sha(ROOT/'request.json')
    for frame in FRAMES:
        root=ROOT/frame; out=root/'review'; out.mkdir(exist_ok=False)
        original,meta=geometry_paths(frame); base=read(meta); rows,_,_=cameras(frame)
        moving=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==frame)
        box=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']==frame)
        x0,y0,x1,y1=box; crop=(x0-65,y0-65,x1+66,y1+66)
        views=[('old_moving',moving)]+[(n,next(r for r in rows if r['physical_camera']==n)) for n in ['F004_E005_1210FP','M004_B005_12109O']]
        paths=[('published',original)]+[(a,root/a/'mesh.ply') for a in ARMS]
        models=[]; bindings={str(original):sha(original),str(meta):sha(meta),parent:sha(parent),spots:sha(spots)}; mappings={}
        for name,path in paths:
            mesh=o3d.io.read_triangle_mesh(str(path)); bindings[str(path)]=sha(path)
            if name!='published':
                metadata=read(root/name/'mesh.json'); bindings[str(root/name/'mesh.json')]=sha(root/name/'mesh.json')
                mapping=normalization_matrix(base)@np.linalg.inv(normalization_matrix(metadata)); mappings[name]=mapping.tolist(); mesh.transform(mapping)
            mesh.compute_triangle_normals(); models.append((name,scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),np.asarray(mesh.triangle_normals),len(mesh.vertices),len(mesh.triangles)))
        records=[]; files=[]
        def panel(images,name):
            w,h=images[0][1].size; p=Image.new('RGB',(w*len(images),h+24)); d=ImageDraw.Draw(p)
            for j,(label,im) in enumerate(images):p.paste(im,(j*w,24));d.text((j*w+2,4),label,fill='white')
            path=out/name;p.save(path);files.append(dict(path=str(path),sha256=sha(path)))
        for name,camera in views:
            face=[];detail=[]
            for label,scene,normals,nv,nt in models:
                depth,ids,_=camera_depth(scene,camera); ok=np.isfinite(depth); rgb=np.zeros((*depth.shape,3),np.uint8)
                rgb[ok]=(60+170*abs(normals[ids[ok]]@np.array([.3,.4,.866])))[:,None]
                im=Image.fromarray(np.rot90(rgb));face.append((label,im.crop((290,640,790,1420))))
                record=dict(frame=frame,camera=name,arm=label,vertices=nv,triangles=nt)
                if name=='old_moving':
                    detail.append((label,im.crop(crop)));record['selected_spot_misses']=int((~np.rot90(ok)[y0:y1+1,x0:x1+1]).sum())
                records.append(record)
            panel(face,name+'_face.png')
            if detail:panel(detail,name+'_spot.png')
        atomic_json(out/'result.json',dict(records=records,files=files,input_hashes=bindings,render_only_normalization=mappings,
            script_sha256=sha(__file__),production_accepted=False,visual_status='pending',target_used_posthoc_only=True))
        print(frame,[r for r in records if r['camera']=='old_moving'],flush=True)


if __name__=='__main__':main()
