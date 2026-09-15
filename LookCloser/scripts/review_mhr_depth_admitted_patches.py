"""Native clay comparison of depth-admitted MHR controls, not RGB approval."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from build_train_hair_semantics import read, sha, write


def review(root):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from study_mhr_local_head_prior import RGB, FRAME
    from study_multiview_face_prior import portrait_to_native, CROP
    from study_confidence_depth_prior import unproject
    request=read(root/'request.json'); original=Path(request['inputs']['actual_source_mesh'])
    assert sha(original)==request['inputs']['actual_source_mesh_sha256']
    arms=request['arms']; paths=[('original',original)]+[(a,root/a/'strict/mesh.ply') for a in arms]
    dest=root/'native_clay_review'; dest.mkdir(exist_ok=False)
    bindings={str(original):sha(original),str(root/'request.json'):sha(root/'request.json')}
    models=[]
    for label,path in paths:
        if label!='original':
            result=read(path.parent/'result.json'); assert result['observed_guard_passed']
            assert sha(path)==result['hashes']['mesh.ply']; bindings[str(path.parent/'result.json')]=sha(path.parent/'result.json')
        bindings[str(path)]=sha(path)
        mesh=o3d.io.read_triangle_mesh(str(path)); mesh.compute_triangle_normals()
        v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
        if label=='original': old_v=v.copy(); old_t=t.copy(); nt=len(t)
        else:
            np.testing.assert_array_equal(v[:len(old_v)],old_v); np.testing.assert_array_equal(t[:nt],old_t)
        models.append((label,scene_for(v,t),np.asarray(mesh.triangle_normals)))
    files=[]; stats=[]
    def panel(images,name):
        w,h=images[0][1].size; out=Image.new('RGB',(len(images)*w,h+24)); draw=ImageDraw.Draw(out)
        for j,(label,im) in enumerate(images): out.paste(im,(j*w,24)); draw.text((j*w+2,4),label,fill='white')
        path=dest/name; out.save(path); files.append(dict(path=str(path),sha256=sha(path)))
    rows=[i['camera'] for i in read(RGB/FRAME/'input.json')['inputs']]
    boxes={r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1}
    for prefix in ['G004_B','M004_B','E004_B']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix)); name=row['physical_camera']
        x0,y0,x1,y1=boxes[name]['native_review_box']; y1=min(1550,y1+100); w=x1-x0; h=y1-y0
        yy,xx=np.mgrid[y0:y1,x0:x1]; xy=portrait_to_native(np.c_[xx.ravel(),yy.ravel()])
        center=np.asarray(row['transform_matrix'])[:3,3]; direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        unit=direction/np.linalg.norm(direction,axis=1,keepdims=True)
        rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
        rp=RGB/FRAME/(name+'.png'); bindings[str(rp)]=sha(rp)
        images=[('train RGB',Image.open(rp).convert('RGB').crop((x0,y0-CROP[1],x1,y1-CROP[1])))]
        for label,scene,normals in models:
            hit=scene.cast_rays(rays); d=hit['t_hit'].numpy(); ids=hit['primitive_ids'].numpy(); ok=np.isfinite(d)
            color=np.full((len(xy),3),20,np.uint8); color[ok]=(70+170*abs(np.sum(normals[ids[ok]]*-unit[ok],axis=1)))[:,None]
            images.append((label,Image.fromarray(color.reshape(h,w,3))))
            if label=='original': base=ok
            stats.append(dict(camera=name,arm=label,newly_visible=int((ok&~base).sum()),
                              added_visible=int((ok&(ids>=nt)).sum())))
        panel(images,name+'.png')
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    cam=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME)
    x0,y0,x1,y1=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    images=[]
    for label,scene,normals in models:
        d,ids,_=camera_depth(scene,cam); ok=np.isfinite(d); color=np.full((*d.shape,3),20,np.uint8)
        color[ok]=(60+170*abs(normals[ids[ok]]@np.array([.3,.4,.866])))[:,None]
        images.append((label,Image.fromarray(np.rot90(color)).crop((x0-65,y0-65,x1+66,y1+66))))
        valid=np.rot90(ok)[y0:y1+1,x0:x1+1]
        if label=='original': missing=~valid
        stats.append(dict(camera='requested_hole_posthoc',arm=label,original_missing=int(missing.sum()),
                          remaining_missing=int((missing&~valid).sum())))
    panel(images,'requested_hole.png')
    for p in [parent,spots,RGB/'inference.json',RGB/FRAME/'input.json']: bindings[str(p)]=sha(p)
    write(dest/'result.json',dict(files=files,statistics=stats,input_hashes=bindings,script_sha256=sha(__file__),
        target_used_posthoc_only=True,visual_status='pending',production_accepted=False))
    print(stats[-4:],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_mhr_local_patch_admission'))
    review(p.parse_args().root)
