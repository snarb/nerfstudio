"""Native train RGB/mask witnesses for rejected prior facets; no mask edits."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from build_train_hair_semantics import read, sha, write
from joint_temporal_texture import project
from study_mhr_local_head_prior import RGB, FRAME
from study_multiview_face_prior import CROP
from admit_mhr_local_patch_depth import inputs, CANDIDATES, PRIOR, OUT


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    _,rows,_,masks,names,binding=inputs(); dest=OUT/'mask_veto_witnesses'; dest.mkdir(exist_ok=False)
    arm='smooth400'; qp=PRIOR/'probe_smooth025_smooth100_smooth400'/(arm+'.npz')
    query=np.load(qp)['target_prior_points']; mesh=o3d.io.read_triangle_mesh(str(CANDIDATES/arm/'local_raw.ply'))
    v=np.asarray(mesh.vertices); pp=np.load(CANDIDATES/arm/'proposal_evidence.npz')['proposals']
    cp=scene_for(v,pp).compute_closest_points(o3d.core.Tensor(query.astype(np.float32)))
    exact=np.linalg.norm(query-cp['points'].numpy(),axis=1)<1e-6; ids=np.unique(cp['primitive_ids'].numpy()[exact])
    tri=v[pp[ids]]; points=np.concatenate([tri,tri.mean(1)[:,None]],axis=1).reshape(-1,3)
    cameras=[]; views={}
    for row in rows:
        uv,z=project(points,[row]); uv=uv[0]; z=z[0]; xy=np.rint(uv).astype(int)
        available=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
        inside=np.zeros(len(points),bool); idx=np.flatnonzero(available)
        mask=masks[names.index(row['physical_camera'])]; inside[idx]=mask[xy[idx,1],xy[idx,0]]
        bad=(available&~inside).reshape(-1,4).any(1); name=row['physical_camera']
        cameras.append(dict(camera=name,outside_facets=int(bad.sum()),available_facets=int(available.reshape(-1,4).all(1).sum())))
        views[name]=(uv,available&~inside,mask)
    cameras.sort(key=lambda x:(-x['outside_facets'],x['camera'])); files=[]; sources={}
    for item in [r for r in cameras if r['outside_facets']][:4]:
        name=item['camera']; uv,bad,mask=views[name]; xy=np.c_[uv[:,1],1919-uv[:,0]]
        center=np.median(xy[bad],axis=0).round().astype(int); x0=max(0,min(780,int(center[0])-150)); y0=max(CROP[1],min(CROP[3]-240,int(center[1])-120)); box=(x0,y0,x0+300,y0+240)
        path=RGB/FRAME/(name+'.png'); rgb=Image.open(path).convert('RGB').crop((box[0],box[1]-CROP[1],box[2],box[3]-CROP[1])); sources[str(path)]=sha(path)
        pm=np.rot90(mask)[box[1]:box[3],box[0]:box[2]]; color=np.asarray(rgb).copy(); color[~pm.astype(bool)]=(color[~pm.astype(bool)]*.25).astype(np.uint8)
        marked=Image.fromarray(color); draw=ImageDraw.Draw(marked)
        for p,is_bad in zip(xy,bad):
            x,y=p-[x0,y0]; draw.ellipse((x-2,y-2,x+2,y+2),fill='red' if is_bad else 'cyan')
        panel=Image.new('RGB',(600,264)); panel.paste(rgb,(0,24)); panel.paste(marked,(300,24)); draw=ImageDraw.Draw(panel)
        draw.text((2,4),name+' RGB / mask outside dark; red=veto samples',fill='white')
        out=dest/(name+'.png'); panel.save(out); files.append(dict(path=str(out),sha256=sha(out)))
    np.savez_compressed(dest/'queries.npz',proposal_ids=ids,points=points,matching_prior_points=query[exact])
    write(dest/'result.json',dict(arm=arm,facets=len(ids),cameras=cameras,files=files,source_rgb_hashes=sources,
        inputs=binding,query_sha256=sha(qp),evidence_sha256=sha(dest/'queries.npz'),script_sha256=sha(__file__),
        target_used_posthoc_only=True,masks_changed=False,production_accepted=False,visual_status='pending'))
    print(cameras[:6],flush=True)


if __name__=='__main__': main()
