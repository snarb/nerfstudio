"""CPU native-train review of the frozen canonical fit, never geometry repair."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from study_canonical_face_prior import OUT,SOURCE,FRAME,CORE,JAW,CHEEK
from study_multiview_face_prior import read,save,sha,portrait_to_native,CROP
from triangulate_face_prior import projection_matrices,project,quantiles

def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from study_confidence_depth_prior import unproject
    protocol=read(OUT/'protocol.json');spec=read(OUT/'observations.json');rows=spec['cameras']
    dest=OUT/'review';dest.mkdir(exist_ok=False)
    old=o3d.io.read_triangle_mesh(protocol['original_mesh']);old.compute_triangle_normals()
    ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);oldscene=scene_for(ov,ot)
    canonical=np.load(OUT/'canonical.npz');tri=canonical['triangles']
    edges=np.sort(ot[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1);edges,count=np.unique(edges,axis=0,return_counts=True)
    boundary=ov[np.unique(edges[count==1])];boundarytree=cKDTree(boundary)
    records=[];fits={}
    for arm in ['similarity','regularized8']:
        v=np.load(OUT/arm/'fit.npz')['vertices'];mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_triangle_normals()
        fits[arm]=(v,scene_for(v,tri),np.asarray(mesh.triangle_normals))
        closest=oldscene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32)))['points'].numpy()
        distances=np.linalg.norm(v-closest,axis=1);bd=boundarytree.query(v)[0];lower=np.array(sorted(set(JAW+CHEEK)))
        records.append(dict(arm=arm,lower_face_vertices=len(lower),lower_face_surface_distance=quantiles(distances[lower]),
            lower_face_boundary_distance=quantiles(bd[lower]),vertices_meeting_locality=int(((distances[lower]<=.002)&(bd[lower]<=.003)).sum()),
            note='Vertex-only feasibility, not dense visibility or semantic acceptance. No additions generated.'))
    predictions={r['camera']:r for r in read(SOURCE/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1}
    paths=[]
    for prefix in ['G004_A','G004_B','M004_A','M004_B','E004_B']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix));name=row['physical_camera'];pred=predictions[name]
        box=pred['native_review_box'];x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
        yy,xx=np.mgrid[y0:y1,x0:x1];xy=portrait_to_native(np.column_stack((xx.ravel(),yy.ravel())))
        center=np.asarray(row['transform_matrix'])[:3,3];direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        rays=o3d.core.Tensor(np.column_stack((np.broadcast_to(center,direction.shape),direction)).astype(np.float32))
        unit=direction/np.linalg.norm(direction,axis=1,keepdims=True)
        images=[]
        for label,scene,norm in [('original',oldscene,np.asarray(old.triangle_normals))]+[(a,z[1],z[2]) for a,z in fits.items()]:
            hit=scene.cast_rays(rays);d=hit['t_hit'].numpy();tid=hit['primitive_ids'].numpy();valid=np.isfinite(d)
            rgb=np.full((len(d),3),20,np.uint8);shade=70+170*np.abs(np.sum(norm[tid[valid]]*-unit[valid],axis=1));rgb[valid]=shade[:,None]
            im=Image.fromarray(rgb.reshape(h,w,3));images.append((label,im))
        original_rgb=Image.open(SOURCE/FRAME/(name+'.png')).convert('RGB').crop((x0-CROP[0],y0-CROP[1],x1-CROP[0],y1-CROP[1]))
        images.insert(0,('train RGB',original_rgb));panel=Image.new('RGB',(w*4,h+24));draw=ImageDraw.Draw(panel)
        for i,(label,im) in enumerate(images):panel.paste(im,(i*w,24));draw.text((i*w+3,4),label,fill='white')
        path=dest/(name+'_clay.png');panel.save(path);paths.append(path)
        panel=Image.new('RGB',(w*2,h+24));draw=ImageDraw.Draw(panel)
        for i,(arm,(v,_,_)) in enumerate(fits.items()):
            im=original_rgb.copy();p=ImageDraw.Draw(im);uv,z=project(v,projection_matrices([row]));native=uv[:,0];portrait=np.column_stack((native[:,1],1919-native[:,0]))-np.array([x0,y0]);observed=np.asarray(pred['portrait_xy'])-np.array([x0,y0])
            for a,b in np.unique(np.sort(tri[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0):p.line([tuple(portrait[a]),tuple(portrait[b])],fill=(40,160,80),width=1)
            for k in CORE:
                a,b=observed[k];p.ellipse((a-1,b-1,a+1,b+1),fill='red');p.line([tuple(observed[k]),tuple(portrait[k])],fill='yellow',width=1)
            panel.paste(im,(i*w,24));draw.text((i*w+3,4),arm+' green; observed red',fill='white')
        path=dest/(name+'_projection.png');panel.save(path);paths.append(path)
    save(dest/'manifest.json',dict(frame=FRAME,review_cameras=[p.split('_clay')[0] for p in []],
        files=[dict(path=str(p),sha256=sha(p)) for p in paths],locality=records,protocol_sha256=sha(OUT/'protocol.json'),
        fit_summary_sha256=sha(OUT/'fit_summary.json'),script_sha256=sha(__file__),original_mesh_sha256=sha(protocol['original_mesh']),
        original_geometry_changed=False,prior_only_render_not_replacement=True,heldout_used=False))

if __name__=='__main__':main()
