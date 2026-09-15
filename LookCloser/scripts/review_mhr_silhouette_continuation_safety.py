"""Unchanged binary-mask gate and native localization of new unsafe facets."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from continue_mhr_silhouette_convergence import ROOT
from fit_mhr_silhouette_conformance import prepare,project_jacobian
from study_multiview_face_prior import read,save,sha,portrait_to_native
from study_jaw_repair_transfer import mask_votes
from study_confidence_depth_prior import unproject
from admit_mhr_local_patch_depth import Scene2


def main():
    import open3d as o3d
    from joint_temporal_texture import exr,display
    from calibrated_depth_witness import ROOT as COLOR
    dest=ROOT/'safety_review';dest.mkdir(exist_ok=False)
    _,rows,masks,names,evidence,_=prepare()
    a=np.load(ROOT/'fit.npz');q=np.load(ROOT/'locality/silhouette.npz')
    p=q['points'];pp=np.repeat(np.arange(len(p))[:,None],3,axis=1)
    support,outside=mask_votes(p,pp,rows,masks,names)
    assert ((support>=2)&(outside==0)).all()
    audit=read(ROOT/'audit.json');bad=np.asarray(audit['topology_localization']['new_strict_crossing']['triangle_ids'])
    reversed_ids=np.asarray(audit['topology_localization']['normal_reversed']['triangle_ids'])
    tri=a['triangles'];v=a['vertices'];mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tri));mesh.compute_triangle_normals()
    scene=Scene2(v,tri);normals=np.asarray(mesh.triangle_normals);flag=np.zeros(len(tri),np.uint8);flag[reversed_ids]=1;flag[bad]=2
    profile=read(COLOR/'camera_profiles.json');exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    bindings={str(p):sha(p) for p in [ROOT/'fit.npz',ROOT/'audit.json',ROOT/'locality/silhouette.npz',COLOR/'camera_profiles.json',COLOR/'exposure.json']}
    files=[];stats=[]
    centers=v[tri[bad]].mean(1)
    for prefix in ['C004_E','G004_B']:
        row=next(r for r in rows if r['physical_camera'].startswith(prefix));name=row['physical_camera']
        uv,z,_=project_jacobian(centers,row);portrait=np.c_[uv[:,1],1919-uv[:,0]]
        visible=(z>0)&(portrait[:,0]>0)&(portrait[:,0]<1080)&(portrait[:,1]>0)&(portrait[:,1]<1920)
        bounds=portrait[visible];assert len(bounds)
        x0,y0=np.maximum([0,0],np.floor(bounds.min(0)-90)).astype(int);x1,y1=np.minimum([1080,1920],np.ceil(bounds.max(0)+91)).astype(int)
        gain=np.asarray(profile['rgb_gain'][profile['physical_cameras'].index(name)])
        rgb=np.rint(display(exr(row['file_path'])*gain,exposure)*255).clip(0,255).astype(np.uint8)
        bindings[row['file_path']]=sha(row['file_path'])
        yy,xx=np.mgrid[y0:y1,x0:x1];xy=portrait_to_native(np.c_[xx.ravel(),yy.ravel()]);center=np.asarray(row['transform_matrix'])[:3,3]
        direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center;unit=direction/np.linalg.norm(direction,axis=1,keepdims=True)
        rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32));hit=scene.cast_rays(rays)
        d,ids=hit['t_hit'].numpy(),hit['primitive_ids'].numpy();ok=np.isfinite(d)
        clay=np.full((len(d),3),20,np.uint8);clay[ok]=(70+170*abs(np.sum(normals[ids[ok]]*-unit[ok],axis=1)))[:,None]
        marked=clay.copy();labels=np.zeros(len(d),np.uint8);labels[ok]=flag[ids[ok]];marked[labels==1]=[255,170,0];marked[labels==2]=[255,40,40]
        images=[Image.fromarray(np.rot90(rgb)[y0:y1,x0:x1]),Image.fromarray(clay.reshape(*yy.shape,3)),Image.fromarray(marked.reshape(*yy.shape,3))]
        w,h=images[0].size;panel=Image.new('RGB',(w*3,h+24));draw=ImageDraw.Draw(panel)
        for index,(title,im) in enumerate(zip(['train RGB','100-step prior','red crossing; amber reversal'],images)):
            panel.paste(im,(index*w,24));draw.text((index*w+2,4),title,fill='white')
        path=dest/(name+'.png');panel.save(path);files.append(dict(path=str(path),sha256=sha(path)))
        stats.append(dict(camera=name,portrait_box=[int(x0),int(y0),int(x1),int(y1)],visible_new_crossing_pixels=int((labels==2).sum()),visible_reversal_pixels=int((labels==1).sum())))
    np.savez_compressed(dest/'strict_mask.npz',support=support,outside=outside,points=p)
    save(dest/'result.json',dict(strict_unmodified_binary_mask_pass=int(((support>=2)&(outside==0)).sum()),queries=len(p),
        minimum_supporting_cameras=int(support.min()),maximum_disagreeing_cameras=int(outside.max()),
        independent_measured_mask_override_retained=True,mask_tolerance_pixels=0,files=files,statistics=stats,
        input_hashes=bindings,evidence=evidence,script_sha256=sha(__file__),
        target_used_only_for_posthoc44point_check=True,unsafe_crop_chosen_from_geometry_not_target=True,production_accepted=False))


if __name__=='__main__':main()
