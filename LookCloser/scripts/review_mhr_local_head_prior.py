"""Native train-only RGB/clay review, including the neck, without geometry writes."""
import argparse
import numpy as np
from PIL import Image,ImageDraw
from study_mhr_local_head_prior import OUT,RGB,FRAME
from study_multiview_face_prior import read,save,sha,portrait_to_native,CROP

def main(arms):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from study_confidence_depth_prior import unproject
    proto=read(OUT/'protocol.json');source=read(RGB/FRAME/'input.json');rows=[i['camera'] for i in source['inputs']]
    old=o3d.io.read_triangle_mesh(proto['original_mesh']);old.compute_triangle_normals();v=np.asarray(old.vertices);t=np.asarray(old.triangles);models=[('original',scene_for(v,t),np.asarray(old.triangle_normals))]
    initial=np.load(OUT/'initial.npz')
    for arm in arms:
        data=initial if arm=='initial' else np.load(OUT/arm/'fit.npz');v=data['vertices'];t=initial['triangles'];mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_triangle_normals();models.append((arm,scene_for(v,t),np.asarray(mesh.triangle_normals)))
    predictions={r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1};dest=OUT/('review_'+'_'.join(arms));dest.mkdir(exist_ok=False);files=[]
    for prefix in ['G004_A','G004_B','M004_A','M004_B','E004_B','H004_C']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix));name=row['physical_camera'];x0,y0,x1,y1=predictions[name]['native_review_box'];y1=min(1550,y1+100);w=x1-x0;h=y1-y0
        yy,xx=np.mgrid[y0:y1,x0:x1];xy=portrait_to_native(np.column_stack((xx.ravel(),yy.ravel())));center=np.asarray(row['transform_matrix'])[:3,3];direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center;unit=direction/np.linalg.norm(direction,axis=1,keepdims=True)
        rays=o3d.core.Tensor(np.column_stack((np.broadcast_to(center,direction.shape),direction)).astype(np.float32));rgb=Image.open(RGB/FRAME/(name+'.png')).convert('RGB').crop((x0,y0-CROP[1],x1,y1-CROP[1]));images=[('train RGB',rgb)]
        for label,scene,norm in models:
            hit=scene.cast_rays(rays);depth=hit['t_hit'].numpy();tid=hit['primitive_ids'].numpy();ok=np.isfinite(depth);color=np.full((len(xy),3),20,np.uint8);color[ok]=(70+170*abs(np.sum(norm[tid[ok]]*-unit[ok],axis=1)))[:,None];images.append((label,Image.fromarray(color.reshape(h,w,3))))
        panel=Image.new('RGB',(len(images)*w,h+24));draw=ImageDraw.Draw(panel)
        for i,(label,im) in enumerate(images):panel.paste(im,(i*w,24));draw.text((i*w+2,4),label,fill='white')
        path=dest/(name+'.png');panel.save(path);files.append(dict(camera=name,path=str(path),sha256=sha(path)))
    save(dest/'manifest.json',dict(files=files,arms=arms,script_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),original_geometry_changed=False,prior_only=True,heldout_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['similarity','head20']);a=p.parse_args();main(a.arms)
