"""Fixed-cohort, binary-mask, topology and native comparisons after fitting."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
import study_mhr_static_unsafe_freeze as study
from study_multiview_face_prior import read,save,sha
from study_confidence_depth_prior import unproject
from study_jaw_repair_transfer import mask_votes
from triangulate_face_prior import quantiles


def main():
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    study.configure();fit=study.fit
    a=np.load(study.FINAL/'fit.npz');t=a['triangles'];active=a['active']
    paths={'base':fit.SOURCE/'smooth100/fit.npz','margin2':study.zero.CONTROL/'fit.npz','zero':study.FAILED/'fit.npz','freeze':study.FINAL/'fit.npz'}
    variants={k:np.load(p)['vertices'] for k,p in paths.items()}
    _,rows,masks,names,evidence,validation=fit.prepare();sdfs=[]
    for row in rows:
        m=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~m)-distance_transform_edt(m)).astype(np.float32))
    grids={}
    for key,v in variants.items():
        grids[key]=np.full((62,int(active.sum())),np.nan)
        for ci,(ids,values,_) in enumerate(fit.silhouette_samples(v[active],rows,sdfs)):grids[key][ci,ids]=values
    common=np.logical_and.reduce([np.isfinite(g) for g in grids.values()]);cohorts={}
    for split,selected in [('fit',~validation),('reserved',validation)]:
        use=common&selected[:,None]
        cohorts[split]=dict(samples=int(use.sum()),arms={key:dict(outside=int((g[use]>0).sum()),
            mean_positive_sdf=float(np.maximum(g[use],0).mean()),positive_sdf=quantiles(np.maximum(g[use],0))) for key,g in grids.items()})
    source=Path('/mnt/data/dec5_mhr_production_patch_001193');residual=source/'residual_hole/evidence.npz'
    xy=np.load(residual)['portrait_xy'];camera_file=source/'admission/rgb/F004_E/baseline/frames/001193/result.json'
    camera=read(camera_file)['camera'];native=np.c_[1919-xy[:,1],xy[:,0]];center=np.asarray(camera['transform_matrix'])[:3,3]
    direction=unproject(camera,native[:,0],native[:,1],np.ones(len(native)),offset=.5)-center
    rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
    dest=study.ROOT/'comparison';dest.mkdir(exist_ok=False)
    stats={};details={};files=[];clays=[]
    def panel(items,path):
        w,h=items[0][1].size;out=Image.new('RGB',(len(items)*w,h+24));draw=ImageDraw.Draw(out)
        for i,(label,im) in enumerate(items):out.paste(im,(i*w,24));draw.text((i*w+3,4),label,fill='white')
        out.save(path);files.append(dict(path=str(path),sha256=sha(path)))
    for key,root in [('margin2',study.zero.CONTROL),('zero',study.FAILED),('freeze',study.FINAL)]:
        v=variants[key];scene=fit.Scene2(v,t);hit=scene.cast_rays(rays);d=hit['t_hit'].numpy();ok=np.isfinite(d)
        points=center+direction[ok]*d[ok,None];faces=hit['primitive_ids'].numpy()[ok]
        index=np.repeat(np.arange(len(points))[:,None],3,axis=1);support,outside=mask_votes(points,index,rows,masks,names)
        primary=np.load(root/'locality/silhouette.npz');p=primary['points'];pi=np.repeat(np.arange(len(p))[:,None],3,axis=1)
        ps,po=mask_votes(p,pi,rows,masks,names)
        topo=np.load(root/'review_v2/silhouette_topology.npz');base_topo=np.load(root/'review_v2/baseline_topology.npz')
        basepairs=set(map(tuple,base_topo['strict_pairs']));newpairs=[p for p in topo['strict_pairs'] if tuple(p) not in basepairs]
        unsafe=np.union1d(topo['reversed_triangles'],np.unique(topo['strict_pairs']))
        stats[key]=dict(primary_hits=len(p),primary_binary_pass=int(((ps>=2)&(po==0)).sum()),
            residual_hits=int(ok.sum()),residual_total=len(ok),residual_binary_pass=int(((support>=2)&(outside==0)).sum()),
            residual_unsafe_first_hit_facets=int(np.isin(faces,unsafe).sum()),
            primary_rim_distance=quantiles(np.linalg.norm(primary['rim_closest']-primary['rim_points'],axis=1)),
            primary_rim_signed=quantiles(primary['rim_signed_offset']),
            topology=dict(strict_pairs=len(topo['strict_pairs']),new_strict_pairs=len(newpairs),normal_changes_over90=len(topo['reversed_triangles'])),
            strict_global_gate_pass=not len(newpairs) and not len(topo['reversed_triangles']))
        details.update({key+'_points':points,key+'_depth':d,key+'_faces':faces,key+'_support':support,key+'_outside':outside})
        depth,tid,_=camera_depth(scene,camera);valid=np.isfinite(depth)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_triangle_normals()
        col=np.full((*depth.shape,3),20,np.uint8);col[valid]=(60+170*abs(np.asarray(mesh.triangle_normals)[tid[valid]]@np.array([.3,.4,.866])))[:,None]
        clays.append((key,Image.fromarray(np.rot90(col)).crop((635,1100,755,1210))))
    panel(clays,dest/'residual_prior_clay.png')
    for pattern in ['C004_E*_clay.png','G004_B*_clay.png','M004_B*_clay.png','requested_hole_prior_only.png']:
        name=next((study.FINAL/'review_v2').glob(pattern)).name;items=[]
        for key,root in [('zero',study.FAILED),('freeze',study.FINAL)]:
            im=Image.open(root/'review_v2'/name).convert('RGB');count=3 if name.startswith('requested') else 4
            items.append((key,im.crop((im.width*(count-1)//count,24,im.width,im.height))))
        panel(items,dest/name)
    np.savez_compressed(dest/'cohorts.npz',**grids,common=common,validation=validation)
    np.savez_compressed(dest/'queries.npz',**details,portrait_xy=xy)
    bindings=list(paths.values())+[residual,camera_file,study.FINAL/'protocol.json']
    save(dest/'result.json',dict(fixed_zero_sdf_cohorts=cohorts,posthoc=stats,
        input_hashes={str(p):sha(p) for p in bindings},files=files,script_sha256=sha(__file__),
        target_input_to_fit=False,production_modified=False,admission_permitted=stats['freeze']['strict_global_gate_pass']))
    print(stats,flush=True)


if __name__=='__main__':main()
