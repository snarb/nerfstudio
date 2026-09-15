"""Post-hoc native clay diagnostics of raw candidates; no fitting or approval."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from build_mhr_local_patch_candidates import ROOT, ARMS
from build_train_hair_semantics import read, sha, write


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from study_mhr_local_head_prior import RGB, FRAME
    from study_multiview_face_prior import portrait_to_native, CROP
    from study_confidence_depth_prior import unproject
    from bake_joint_temporal_mesh import camera_depth
    request=read(ROOT/'request.json'); dest=ROOT/'raw_review'; dest.mkdir(exist_ok=False)
    assert sha(request['source_mesh'])==request['source_mesh_sha256']
    paths=[('original',Path(request['source_mesh']))]+[(a,ROOT/a/'local_raw.ply') for a in ARMS]
    models=[]; bindings={str(p):sha(p) for _,p in paths}
    for label,path in paths:
        mesh=o3d.io.read_triangle_mesh(str(path)); mesh.compute_triangle_normals()
        v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
        models.append((label,scene_for(v,t),np.asarray(mesh.triangle_normals)))
    nt=read(ROOT/ARMS[0]/'result.json')['original_triangles']
    source=read(RGB/FRAME/'input.json'); rows=[i['camera'] for i in source['inputs']]
    inference=read(RGB/'inference.json')
    boxes={r['camera']:r for r in inference['records'] if r['frame']==FRAME and r['detected']==1}
    panels=[]; stats=[]
    def panel(images,path):
        w,h=images[0][1].size; out=Image.new('RGB',(w*len(images),h+24)); draw=ImageDraw.Draw(out)
        for j,(label,im) in enumerate(images):
            out.paste(im,(j*w,24)); draw.text((j*w+2,4),label,fill='white')
        out.save(path); panels.append(dict(path=str(path),sha256=sha(path)))
    for prefix in ['G004_A','G004_B','M004_A','M004_B','E004_B','H004_C']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix)); name=row['physical_camera']
        x0,y0,x1,y1=boxes[name]['native_review_box']; y1=min(1550,y1+100); w=x1-x0; h=y1-y0
        yy,xx=np.mgrid[y0:y1,x0:x1]; xy=portrait_to_native(np.c_[xx.ravel(),yy.ravel()])
        center=np.asarray(row['transform_matrix'])[:3,3]; directions=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        unit=directions/np.linalg.norm(directions,axis=1,keepdims=True)
        rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,directions.shape),directions].astype(np.float32))
        imagepath=RGB/FRAME/(name+'.png'); bindings[str(imagepath)]=sha(imagepath)
        rgb=Image.open(imagepath).convert('RGB').crop((x0,y0-CROP[1],x1,y1-CROP[1]))
        images=[('train RGB',rgb)]; markers=[('train RGB',rgb)]
        for label,scene,normals in models:
            hit=scene.cast_rays(rays); depth=hit['t_hit'].numpy(); ids=hit['primitive_ids'].numpy(); valid=np.isfinite(depth)
            color=np.full((len(xy),3),20,np.uint8)
            color[valid]=(70+170*abs(np.sum(normals[ids[valid]]*-unit[valid],axis=1)))[:,None]
            images.append((label,Image.fromarray(color.reshape(h,w,3))))
            if label=='original': old=valid.copy()
            added=valid&(ids>=nt) if label!='original' else np.zeros_like(valid)
            color[added]=[240,60,40]; markers.append((label+' added',Image.fromarray(color.reshape(h,w,3))))
            stats.append(dict(camera=name,arm=label,newly_visible=int((valid&~old).sum()),added_visible=int(added.sum())))
        panel(images,dest/(name+'.png')); panel(markers,dest/(name+'_added.png'))
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    cam=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME)
    spot=next(r for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    x0,y0,x1,y1=spot['bbox_inclusive']; crop=(x0-65,y0-65,x1+66,y1+66); images=[]; missing=None
    for label,scene,normals in models:
        d,ids,_=camera_depth(scene,cam); ok=np.isfinite(d); rgb=np.full((*d.shape,3),20,np.uint8)
        rgb[ok]=(60+170*abs(normals[ids[ok]]@np.array([.3,.4,.866])))[:,None]
        images.append((label,Image.fromarray(np.rot90(rgb)).crop(crop)))
        valid=np.rot90(ok)[y0:y1+1,x0:x1+1]
        if missing is None: missing=~valid
        stats.append(dict(camera='requested_hole_posthoc',arm=label,original_missing=int(missing.sum()),
                          remaining_missing=int((missing&~valid).sum())))
    panel(images,dest/'requested_hole.png')
    for p in [parent,spots,RGB/'inference.json',RGB/FRAME/'input.json',ROOT/'request.json']:
        bindings[str(p)]=sha(p)
    write(dest/'result.json',dict(files=panels,statistics=stats,input_hashes=bindings,
        script_sha256=sha(__file__),raw_not_depth_approved=True,production_accepted=False,
        targets_used_posthoc_only=True,visual_status='pending'))
    print('Raw review ready',stats[-4:],flush=True)


if __name__=='__main__': main()
